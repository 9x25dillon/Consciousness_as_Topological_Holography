# cr2bc.py — Coherence-Renewal Bi-Coupling (CR²BC) Engine
# Faithful, modular, drop-in Python implementation of the chaos-glyph spec.
# No fluff. No persona. Just math → code.

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np


class FrequencyBand(str, Enum):
    DELTA = "delta"
    THETA = "theta"
    ALPHA = "alpha"
    BETA = "beta"
    GAMMA = "gamma"

    @property
    def center_hz(self) -> float:
        return {
            FrequencyBand.DELTA: 2.0,
            FrequencyBand.THETA: 6.0,
            FrequencyBand.ALPHA: 10.0,
            FrequencyBand.BETA: 20.0,
            FrequencyBand.GAMMA: 40.0,
        }[self]


ALL_BANDS: Tuple[FrequencyBand, ...] = (
    FrequencyBand.DELTA,
    FrequencyBand.THETA,
    FrequencyBand.ALPHA,
    FrequencyBand.BETA,
    FrequencyBand.GAMMA,
)


@dataclass
class CoherenceSample:
    t: float
    kappa: Dict[FrequencyBand, float]
    phi: Dict[FrequencyBand, float]
    context: str = "unknown"


@dataclass
class AgentHints:
    agent_a: Optional[str] = None
    agent_b: Optional[str] = None


@dataclass
class InvariantState:
    vec: np.ndarray

    @staticmethod
    def zeros(dim: int) -> "InvariantState":
        return InvariantState(vec=np.zeros(dim, dtype=float))


@dataclass
class AuditState:
    score: float
    delta_kappa_norm: float
    spectral_deviation: float
    structural_break: float
    accepted: bool


@dataclass
class CR2BCConfig:
    grid_shape: Tuple[int, int] = (8, 8)
    band_scale_hz: float = 10.0
    temporal_tau0: float = 5.0
    window_size: int = 10
    alpha0: float = 0.5
    sigma_kappa: float = 0.1
    beta_max: float = 0.4
    theta: float = 0.6
    epsilon: float = 0.1
    lambda_T0: float = 0.1
    burst_sensitivity: float = 1.0
    invariant_dim: int = 16


class CR2BC:
    def __init__(self, config: Optional[CR2BCConfig] = None):
        self.config = config or CR2BCConfig()
        self.invariant = InvariantState.zeros(self.config.invariant_dim)
        self._grid_r = self._make_radial_grid(self.config.grid_shape)

    @staticmethod
    def _make_radial_grid(shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        ys, xs = np.mgrid[0:h, 0:w]
        ys = ys - (h - 1) / 2.0
        xs = xs - (w - 1) / 2.0
        r = np.sqrt(xs**2 + ys**2)
        r /= (np.max(r) + 1e-8)
        return r

    def encode_spatial_capsule(self, sample: CoherenceSample, band: FrequencyBand) -> np.ndarray:
        r = self._grid_r
        kappa = sample.kappa[band]
        phi = sample.phi[band]
        k_d = band.center_hz / 40.0
        psi = np.exp(-r**2)
        return psi * kappa * np.cos(phi - k_d * r)

    def encode_all_capsules(self, sample: CoherenceSample) -> Dict[FrequencyBand, np.ndarray]:
        return {b: self.encode_spatial_capsule(sample, b) for b in ALL_BANDS}

    def band_kernel(self) -> np.ndarray:
        B0 = self.config.band_scale_hz
        freqs = np.array([b.center_hz for b in ALL_BANDS], dtype=float)
        diff = np.abs(freqs[:, None] - freqs[None, :])
        return np.exp(-diff / B0)

    def temporal_kernel(self, times: List[float], t_ref: float) -> np.ndarray:
        tau0 = self.config.temporal_tau0
        deltas = np.maximum(0.0, t_ref - np.array(times, dtype=float))
        return np.exp(-deltas / tau0)

    def encode_hints(self, hints: Optional[AgentHints]) -> np.ndarray:
        dim = self.config.invariant_dim
        if not hints or (not hints.agent_a and not hints.agent_b):
            return np.zeros(dim, dtype=float)

        def hashed_vec(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.normal(size=dim)
            v /= np.linalg.norm(v) + 1e-8
            return v

        vec = np.zeros(dim, dtype=float)
        if hints.agent_a:
            vec += hashed_vec(hints.agent_a)
        if hints.agent_b:
            vec += hashed_vec(hints.agent_b)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def reconstruct(
        self,
        history: List[CoherenceSample],
        hints: Optional[AgentHints] = None,
    ) -> Tuple[CoherenceSample, InvariantState, AuditState]:
        if not history:
            raise ValueError("history must contain at least one sample")

        current = history[-1]
        times = [s.t for s in history]
        t_ref = current.t

        S_B = self.band_kernel()
        S_tau = self.temporal_kernel(times, t_ref)

        T, D = len(history), len(ALL_BANDS)
        K = np.zeros((T, D), dtype=float)
        for ti, s in enumerate(history):
            for di, b in enumerate(ALL_BANDS):
                K[ti, di] = s.kappa[b]

        w_time = S_tau / (S_tau.sum() + 1e-8)
        w_band = S_B / (S_B.sum(axis=1, keepdims=True) + 1e-8)

        K_recon = np.zeros(D, dtype=float)
        for di in range(D):
            contrib = 0.0
            for ti in range(T):
                for dj in range(D):
                    contrib += w_time[ti] * w_band[di, dj] * K[ti, dj]
            K_recon[di] = contrib

        phi_recon = np.array([current.phi[b] for b in ALL_BANDS], dtype=float)
        K_current = np.array([current.kappa[b] for b in ALL_BANDS], dtype=float)

        delta_kappa = K_recon - K_current
        delta_kappa_norm = float(np.linalg.norm(delta_kappa))
        K_mean = K.mean(axis=0)
        spectral_dev = float(np.linalg.norm(K_recon - K_mean))
        structural_break = float(np.linalg.norm(K[-1] - 2 * K[-2] + K[-3])) if T >= 3 else 0.0

        score = -(delta_kappa_norm + spectral_dev + structural_break)
        accepted = abs(score) < self.config.epsilon

        noise_level = float(K.std())
        alpha_t = self.config.alpha0 / (1.0 + self.config.sigma_kappa * noise_level)
        beta_t = self._beta_from_coherence(K_recon)

        kappa_renewed = K_current + alpha_t * (K_recon - K_current)
        kappa_renewed = np.clip(kappa_renewed, 0.0, 1.0)

        recon_sample = CoherenceSample(
            t=current.t,
            kappa={b: float(kappa_renewed[i]) for i, b in enumerate(ALL_BANDS)},
            phi={b: float(phi_recon[i]) for i, b in enumerate(ALL_BANDS)},
            context=current.context,
        )

        hint_vec = self.encode_hints(hints)
        new_invariant = self._update_invariant(self.invariant, recon_sample, beta_t, hint_vec)
        self.invariant = new_invariant

        audit = AuditState(
            score=float(score),
            delta_kappa_norm=delta_kappa_norm,
            spectral_deviation=spectral_dev,
            structural_break=structural_break,
            accepted=accepted,
        )
        return recon_sample, new_invariant, audit

    def _beta_from_coherence(self, kappa_recon: np.ndarray) -> float:
        kappa_mean = float(kappa_recon.mean())
        x = kappa_mean - self.config.theta
        sig = 1.0 / (1.0 + np.exp(-x))
        return float(self.config.beta_max * sig)

    def _update_invariant(
        self,
        prev: InvariantState,
        sample: CoherenceSample,
        beta_t: float,
        hint_vec: np.ndarray,
    ) -> InvariantState:
        band_values = np.array([sample.kappa[b] for b in ALL_BANDS], dtype=float)
        stats = np.array([band_values.mean(), band_values.std()], dtype=float)
        raw = np.concatenate([band_values, stats], axis=0)

        dim = self.config.invariant_dim
        rng = np.random.default_rng(42)
        proj = rng.normal(size=(dim, raw.shape[0]))
        proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
        encoded = proj @ raw
        encoded /= np.linalg.norm(encoded) + 1e-8

        new_vec = (1.0 - beta_t) * prev.vec + beta_t * encoded
        new_vec += 0.05 * hint_vec
        new_vec /= np.linalg.norm(new_vec) + 1e-8
        return InvariantState(vec=new_vec)


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    cfg = CR2BCConfig()
    engine = CR2BC(cfg)

    history: List[CoherenceSample] = []
    t = 0.0
    for k in np.linspace(0.3, 0.8, 5):
        kappa = {b: float(k + 0.05 * np.random.randn()) for b in ALL_BANDS}
        phi = {b: float(np.random.uniform(-np.pi, np.pi)) for b in ALL_BANDS}
        history.append(CoherenceSample(t=t, kappa=kappa, phi=phi, context="sim"))
        t += 1.0

    hints = AgentHints(agent_a="baseline", agent_b="task A")
    recon, inv, audit = engine.reconstruct(history, hints=hints)

    print("Reconstructed kappa:", recon.kappa)
    print("Invariant norm:", np.linalg.norm(inv.vec))
    print("Audit:", audit)
