# cr2bc.py — Coherence-Renewal Bi-Coupling (CR²BC) Engine
# Faithful, modular, drop-in Python implementation of the chaos-glyph spec.
# No fluff. No persona. Just math → code.

from __future__ import annotations
#!/usr/bin/env python3
"""
cr2bc.py
Coherence-Renewal Bi-Coupling (CR²BC) engine

Implements frequency-band coherence recovery with:
- Spatial capsule encoding C_t[d]
- Cross-band and temporal kernels
- Bi-coupling reconstruction κ̃_t
- Adaptive renewal κ̂_t = κ_t + α_t(κ̃_t - κ_t)
- Invariant prior mixing Π_t = (1-β_t)Π_{t-1} + β_t U[window]
- Audit gating with risk score s_t
"""

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
    """
    Single time-slice of coherence state.

    kappa: band → amplitude (0..1 or arbitrary units)
    phi:   band → phase in radians
    t:     timestamp (seconds or samples)
    context: optional tag ("baseline", "degraded", etc.)
    """
    t: float
    kappa: Dict[FrequencyBand, float]
    phi: Dict[FrequencyBand, float]
    context: str = "unknown"


@dataclass
class AgentHints:
    """
    High-level text / planning context for the two agents A, B.
    These never touch raw signals; they only bias the invariant update.
    """
    agent_a: Optional[str] = None
    agent_b: Optional[str] = None


@dataclass
class InvariantState:
    """
    Latent stable template Π_t.

    We treat it as a low-dimensional vector embedding summarising
    the last window of reconstructed coherence.
    """
    vec: np.ndarray

    @staticmethod
    def zeros(dim: int) -> "InvariantState":
        return InvariantState(vec=np.zeros(dim, dtype=float))


@dataclass
class AuditState:
    """
    Diagnostics & decision for the current step.
    """
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
    # Spatial capsule / grid
    grid_shape: Tuple[int, int] = (8, 8)

    # Cross-band & temporal kernels
    band_scale_hz: float = 10.0     # B0
    temporal_tau0: float = 5.0      # τ0 in seconds
    window_size: int = 10           # W: number of past samples

    # Renewal & mixing
    alpha0: float = 0.5             # base renewal rate
    sigma_kappa: float = 0.1        # noise sensitivity
    beta_max: float = 0.4           # max mixing rate
    theta: float = 0.6              # coherence threshold
    epsilon: float = 0.1            # audit tolerance

    # Regularization / smoothness (structural break sensitivity)
    lambda_T0: float = 0.1
    burst_sensitivity: float = 1.0

    # Invariant embedding dimension
    invariant_dim: int = 16


class CR2BC:
    def __init__(self, config: Optional[CR2BCConfig] = None):
        self.config = config or CR2BCConfig()
        self.invariant = InvariantState.zeros(self.config.invariant_dim)
    """
    Coherence-Renewal Bi-Coupling (CR²BC) engine.

    This is a faithful-but-practical implementation of the math spec:
    - builds spatial capsules C_t[d]
    - computes cross-band S_B and temporal S_tau kernels
    - performs bi-coupling reconstruction
    - updates invariant Π_t with auto-tuned β_t
    - emits audit vector + accept/hold decision
    """

    def __init__(self, config: Optional[CR2BCConfig] = None):
        self.config = config or CR2BCConfig()
        self.invariant = InvariantState.zeros(self.config.invariant_dim)

        # Pre-compute a simple radial grid for spatial capsules
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
    # -------------------------------
    # 1. Spatial capsules C_t[d]
    # -------------------------------
    def encode_spatial_capsule(
        self,
        sample: CoherenceSample,
        band: FrequencyBand,
    ) -> np.ndarray:
        """
        C_t[d](i,j) = ψ(r_ij) * κ_t[d] * cos( φ_t[d] - k_d r_ij )

        We use ψ(r) = exp(-r^2) and k_d = center_hz / K where K is an arbitrary scale.
        """
        r = self._grid_r
        kappa = sample.kappa[band]
        phi = sample.phi[band]
        k_d = band.center_hz / 40.0  # arbitrary, just sets spatial frequency

        psi = np.exp(-r**2)
        capsule = psi * kappa * np.cos(phi - k_d * r)
        return capsule

    def encode_all_capsules(
        self,
        sample: CoherenceSample,
    ) -> Dict[FrequencyBand, np.ndarray]:
        return {b: self.encode_spatial_capsule(sample, b) for b in ALL_BANDS}

    # -------------------------------
    # 2. Cross-band & temporal kernels
    # -------------------------------
    def band_kernel(self) -> np.ndarray:
        """
        S_B(d,d') = exp(-|B_d - B_d'| / B0)
        """
        B0 = self.config.band_scale_hz
        freqs = np.array([b.center_hz for b in ALL_BANDS], dtype=float)
        diff = np.abs(freqs[:, None] - freqs[None, :])
        return np.exp(-diff / B0)

    def temporal_kernel(self, times: List[float], t_ref: float) -> np.ndarray:
    def temporal_kernel(
        self,
        times: List[float],
        t_ref: float,
    ) -> np.ndarray:
        """
        S_tau(d, Δ) = exp(-Δ / τ0)  (we apply same kernel for all bands)

        Returns weights over the provided times (past window).
        """
        tau0 = self.config.temporal_tau0
        deltas = np.maximum(0.0, t_ref - np.array(times, dtype=float))
        return np.exp(-deltas / tau0)

    def encode_hints(self, hints: Optional[AgentHints]) -> np.ndarray:
        dim = self.config.invariant_dim
        if not hints or (not hints.agent_a and not hints.agent_b):
    # -------------------------------
    # 3. Agent hints → latent bias
    # -------------------------------
    def encode_hints(self, hints: Optional[AgentHints]) -> np.ndarray:
        """
        Map text hints into a fixed low-dim vector via seeded hashing.

        This is intentionally simple & deterministic. In a real system you would
        plug in an actual text encoder here.
        """
        dim = self.config.invariant_dim
        if hints is None or (hints.agent_a is None and hints.agent_b is None):
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

    # -------------------------------
    # 4. Reconstruction / bi-coupling
    # -------------------------------
    def reconstruct(
        self,
        history: List[CoherenceSample],
        hints: Optional[AgentHints] = None,
    ) -> Tuple[CoherenceSample, InvariantState, AuditState]:
        if not history:
            raise ValueError("history must contain at least one sample")

        """
        Core CR²BC step.

        history: list of samples X_{t-W:t}, newest last.
        """
        if not history:
            raise ValueError("history must contain at least one sample")

        # Use the last sample as "current" raw observation
        current = history[-1]
        times = [s.t for s in history]
        t_ref = current.t

        S_B = self.band_kernel()
        S_tau = self.temporal_kernel(times, t_ref)

        T, D = len(history), len(ALL_BANDS)
        # Kernels
        S_B = self.band_kernel()                    # [D, D]
        S_tau = self.temporal_kernel(times, t_ref)  # [T]

        # Stack kappa over time and band: K[t, d]
        T = len(history)
        D = len(ALL_BANDS)
        K = np.zeros((T, D), dtype=float)
        for ti, s in enumerate(history):
            for di, b in enumerate(ALL_BANDS):
                K[ti, di] = s.kappa[b]

        w_time = S_tau / (S_tau.sum() + 1e-8)
        w_band = S_B / (S_B.sum(axis=1, keepdims=True) + 1e-8)

        # Temporal weights per time step
        w_time = S_tau / (S_tau.sum() + 1e-8)

        # Cross-band weights, normalised row-wise
        w_band = S_B / (S_B.sum(axis=1, keepdims=True) + 1e-8)

        # Bi-coupled reconstruction:
        # κ̃_t[d] = sum_{t', d'} w_time[t'] * w_band[d, d'] * K[t', d']
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

        # Phases: for now, copy-through but could be regularised similarly
        phi_recon = np.array(
            [current.phi[b] for b in ALL_BANDS],
            dtype=float,
        )

        # Diagnostics
        K_current = np.array([current.kappa[b] for b in ALL_BANDS], dtype=float)
        delta_kappa = K_recon - K_current
        delta_kappa_norm = float(np.linalg.norm(delta_kappa))

        # Spectral deviation from moving average
        K_mean = K.mean(axis=0)
        spectral_dev = float(np.linalg.norm(K_recon - K_mean))

        # Structural break: second difference over time
        if T >= 3:
            sec_diff = K[-1] - 2 * K[-2] + K[-3]
            structural_break = float(np.linalg.norm(sec_diff))
        else:
            structural_break = 0.0

        # Risk-like score: more negative = more error
        score = -(delta_kappa_norm + spectral_dev + structural_break)

        # Acceptance: require low error magnitude
        accepted = abs(score) < self.config.epsilon

        # Auto-tune α_t, β_t, λ_T
        noise_level = float(K.std())
        alpha_t = self.config.alpha0 / (1.0 + self.config.sigma_kappa * noise_level)
        beta_t = self._beta_from_coherence(K_recon)
        lambda_T = self.config.lambda_T0 * (
            1.0 + self.config.burst_sensitivity * structural_break
        )

        # Renewal of kappa magnitudes (one Euler-like step)
        kappa_renewed = K_current + alpha_t * (K_recon - K_current)
        kappa_renewed = np.clip(kappa_renewed, 0.0, 1.0)

        # Build reconstructed sample
        recon_sample = CoherenceSample(
            t=current.t,
            kappa={b: float(kappa_renewed[i]) for i, b in enumerate(ALL_BANDS)},
            phi={b: float(phi_recon[i]) for i, b in enumerate(ALL_BANDS)},
            context=current.context,
        )

        hint_vec = self.encode_hints(hints)
        new_invariant = self._update_invariant(self.invariant, recon_sample, beta_t, hint_vec)
        # Update invariant Π_t
        hint_vec = self.encode_hints(hints)
        new_invariant = self._update_invariant(
            self.invariant,
            recon_sample,
            beta_t,
            hint_vec,
        )
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
    # -------------------------------
    # 5. Coherence → β_t schedule
    # -------------------------------
    def _beta_from_coherence(self, kappa_recon: np.ndarray) -> float:
        """
        β_t = β_max * sigmoid(κ̄ - θ)

        High average coherence → faster mixing with new pattern;
        low coherence → keep Π_t more stable.
        """
        kappa_mean = float(kappa_recon.mean())
        x = kappa_mean - self.config.theta
        sig = 1.0 / (1.0 + np.exp(-x))
        return float(self.config.beta_max * sig)

    # -------------------------------
    # 6. Invariant update Π_t
    # -------------------------------
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

        """
        Π_t = (1 - β_t) Π_{t-1} + β_t U[κ̃_t] + small hint bias

        Here U is a very simple encoder: concatenate band magnitudes and stats,
        then project / pad into invariant_dim.
        """
        band_values = np.array(
            [sample.kappa[b] for b in ALL_BANDS],
            dtype=float,
        )
        stats = np.array(
            [band_values.mean(), band_values.std()],
            dtype=float,
        )
        raw = np.concatenate([band_values, stats], axis=0)

        # Project / pad to invariant_dim via deterministic linear map
        dim = self.config.invariant_dim
        rng = np.random.default_rng(42)
        proj = rng.normal(size=(dim, raw.shape[0]))
        proj /= np.linalg.norm(proj, axis=1, keepdims=True) + 1e-8
        encoded = proj @ raw
        encoded /= np.linalg.norm(encoded) + 1e-8

        new_vec = (1.0 - beta_t) * prev.vec + beta_t * encoded
        new_vec += 0.05 * hint_vec

        encoded /= np.linalg.norm(encoded) + 1e-8

        # Blend invariant with new encoded pattern & hint
        new_vec = (1.0 - beta_t) * prev.vec + beta_t * encoded
        new_vec += 0.05 * hint_vec  # small bias from agent context
        new_vec /= np.linalg.norm(new_vec) + 1e-8
        return InvariantState(vec=new_vec)


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    cfg = CR2BCConfig()
    engine = CR2BC(cfg)

# Example usage & demos
# ---------------------------------------------------------------------

def demo_basic_reconstruction():
    """Demo: Basic CR²BC reconstruction"""
    print("=== CR²BC BASIC RECONSTRUCTION ===\n")

    cfg = CR2BCConfig(window_size=5, epsilon=0.5)
    engine = CR2BC(cfg)

    # Build fake history with gradually increasing coherence
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
        history.append(CoherenceSample(t=t, kappa=kappa, phi=phi, context="baseline"))
        t += 1.0

    hints = AgentHints(agent_a="baseline monitoring", agent_b="task A execution")

    recon, inv, audit = engine.reconstruct(history, hints=hints)

    print("Reconstructed kappa:")
    for band, value in recon.kappa.items():
        print(f"  {band.value}: {value:.4f}")
    print(f"\nInvariant norm: {np.linalg.norm(inv.vec):.4f}")
    print(f"\nAudit:")
    print(f"  Score: {audit.score:.4f}")
    print(f"  Δκ norm: {audit.delta_kappa_norm:.4f}")
    print(f"  Spectral deviation: {audit.spectral_deviation:.4f}")
    print(f"  Structural break: {audit.structural_break:.4f}")
    print(f"  Accepted: {audit.accepted}")
    print()


def demo_degradation_recovery():
    """Demo: Coherence degradation and recovery"""
    print("=== CR²BC DEGRADATION & RECOVERY ===\n")

    cfg = CR2BCConfig(window_size=10, alpha0=0.6, beta_max=0.5)
    engine = CR2BC(cfg)

    # Baseline phase
    print("Phase 1: Baseline")
    history: List[CoherenceSample] = []
    for t in range(5):
        kappa = {b: float(0.7 + 0.05 * np.random.randn()) for b in ALL_BANDS}
        phi = {b: float(np.random.uniform(-np.pi, np.pi)) for b in ALL_BANDS}
        history.append(CoherenceSample(t=float(t), kappa=kappa, phi=phi, context="baseline"))

    recon, inv, audit = engine.reconstruct(history)
    baseline_kappa = np.mean([recon.kappa[b] for b in ALL_BANDS])
    print(f"  Mean κ: {baseline_kappa:.4f}")
    print(f"  Accepted: {audit.accepted}")

    # Degradation phase
    print("\nPhase 2: Degradation")
    for t in range(5, 10):
        kappa = {b: float(0.3 + 0.1 * np.random.randn()) for b in ALL_BANDS}
        phi = {b: float(np.random.uniform(-np.pi, np.pi)) for b in ALL_BANDS}
        history.append(CoherenceSample(t=float(t), kappa=kappa, phi=phi, context="degraded"))

    recon, inv, audit = engine.reconstruct(history[-10:])
    degraded_kappa = np.mean([recon.kappa[b] for b in ALL_BANDS])
    print(f"  Mean κ: {degraded_kappa:.4f}")
    print(f"  Structural break: {audit.structural_break:.4f}")
    print(f"  Accepted: {audit.accepted}")

    # Recovery phase with agent hints
    print("\nPhase 3: Recovery (with agent hints)")
    for t in range(10, 15):
        kappa = {b: float(0.65 + 0.05 * np.random.randn()) for b in ALL_BANDS}
        phi = {b: float(np.random.uniform(-np.pi, np.pi)) for b in ALL_BANDS}
        history.append(CoherenceSample(t=float(t), kappa=kappa, phi=phi, context="recovery"))

    hints = AgentHints(agent_a="apply recovery protocol", agent_b="stabilize coherence")
    recon, inv, audit = engine.reconstruct(history[-10:], hints=hints)
    recovery_kappa = np.mean([recon.kappa[b] for b in ALL_BANDS])
    print(f"  Mean κ: {recovery_kappa:.4f}")
    print(f"  Δκ norm: {audit.delta_kappa_norm:.4f}")
    print(f"  Accepted: {audit.accepted}")
    print(f"\n→ Recovery factor: {(recovery_kappa - degraded_kappa) / (baseline_kappa - degraded_kappa):.2%}")
    print()


def demo_spatial_capsules():
    """Demo: Spatial capsule encoding"""
    print("=== CR²BC SPATIAL CAPSULES ===\n")

    cfg = CR2BCConfig(grid_shape=(16, 16))
    engine = CR2BC(cfg)

    # Create sample with known band structure
    sample = CoherenceSample(
        t=0.0,
        kappa={
            FrequencyBand.DELTA: 0.9,
            FrequencyBand.THETA: 0.7,
            FrequencyBand.ALPHA: 0.5,
            FrequencyBand.BETA: 0.3,
            FrequencyBand.GAMMA: 0.1,
        },
        phi={b: 0.0 for b in ALL_BANDS},
        context="test"
    )

    capsules = engine.encode_all_capsules(sample)

    print("Spatial capsule statistics:")
    for band, capsule in capsules.items():
        print(f"  {band.value}:")
        print(f"    Shape: {capsule.shape}")
        print(f"    Mean: {capsule.mean():.4f}")
        print(f"    Std: {capsule.std():.4f}")
        print(f"    Max: {capsule.max():.4f}")
    print()


def demo_cross_band_kernel():
    """Demo: Cross-band kernel structure"""
    print("=== CR²BC CROSS-BAND KERNEL ===\n")

    cfg = CR2BCConfig()
    engine = CR2BC(cfg)

    S_B = engine.band_kernel()

    print("Cross-band kernel S_B(d,d'):")
    print("       ", "  ".join([b.value[:3].upper() for b in ALL_BANDS]))
    for i, b1 in enumerate(ALL_BANDS):
        row = [f"{S_B[i, j]:.3f}" for j in range(len(ALL_BANDS))]
        print(f"{b1.value[:3].upper()}: {' '.join(row)}")
    print()

    print("Band coupling strengths (off-diagonal):")
    for i, b1 in enumerate(ALL_BANDS):
        for j, b2 in enumerate(ALL_BANDS):
            if i < j:
                print(f"  {b1.value} ↔ {b2.value}: {S_B[i, j]:.4f}")
    print()


def run_all_cr2bc_demos():
    """Run all CR²BC demos"""
    demo_basic_reconstruction()
    demo_degradation_recovery()
    demo_spatial_capsules()
    demo_cross_band_kernel()

    print("=" * 60)
    print("ALL CR²BC DEMOS COMPLETE")
    print("=" * 60)
    print("\nCoherence-Renewal Bi-Coupling operational.")
    print("κ̃_t reconstructed via spatial capsules and temporal kernels.")
    print("Π_t updated with adaptive β_t schedule.")
    print("Audit gating enforces structural stability.")
    print("\n📎 CR²BC ready for integration...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Coherence-Renewal Bi-Coupling (CR²BC)")
    parser.add_argument("--demo", type=str, default="all",
                       choices=["all", "basic", "recovery", "capsules", "kernel"],
                       help="Which demo to run")
    args = parser.parse_args()

    if args.demo == "all":
        run_all_cr2bc_demos()
    elif args.demo == "basic":
        demo_basic_reconstruction()
    elif args.demo == "recovery":
        demo_degradation_recovery()
    elif args.demo == "capsules":
        demo_spatial_capsules()
    elif args.demo == "kernel":
        demo_cross_band_kernel()
