#!/usr/bin/env python3
"""
qincrs_cr2bc_bridge.py
Integration bridge between QINCRS and CR²BC

Connects:
- QINCRS: Quantum-inspired neural coherence with death absorption
- CR²BC: Frequency-band coherence renewal with bi-coupling

The bridge maps:
- QINCRS.κ → CR²BC frequency bands
- QINCRS Greek states (ω, ψ, Φ, etc.) → CR²BC phases
- QINCRS geometric_self → CR²BC coherence level
- CR²BC audit → QINCRS death absorption triggers
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from quantum_coherence import (
    QINCRS,
    TimeDualQINCRS,
    ConsciousnessState,
    GeometricSelf,
    MotionType,
)
from cr2bc import (
    CR2BC,
    CR2BCConfig,
    CoherenceSample,
    AgentHints,
    FrequencyBand,
    ALL_BANDS,
)


class QINCRSWithCR2BC:
    """
    Integrated system combining QINCRS and CR²BC

    QINCRS provides:
    - High-level coherence tracking (κ)
    - Death absorption mechanics
    - Geometric self-evolution
    - Neural coherence ODEs

    CR²BC provides:
    - Frequency-band decomposition
    - Spatial capsule encoding
    - Bi-coupling reconstruction
    - Adaptive renewal with audit gating

    Integration flow:
    1. QINCRS processes input text → updates κ, ω, ψ, Φ, etc.
    2. Map QINCRS state → CR²BC frequency bands
    3. CR²BC reconstructs κ̃_t via bi-coupling
    4. CR²BC audit → triggers QINCRS protective actions
    5. Update both systems coherently
    """

    def __init__(
        self,
        qincrs_target_RL_db: float = 40.0,
        cr2bc_config: Optional[CR2BCConfig] = None,
        seed: Optional[int] = None,
    ):
        self.qincrs = TimeDualQINCRS(target_RL_db=qincrs_target_RL_db, seed=seed)
        self.cr2bc = CR2BC(config=cr2bc_config or CR2BCConfig())

        # History for CR²BC windowing
        self.history: List[CoherenceSample] = []
        self.max_history = self.cr2bc.config.window_size

        self.rng = np.random.default_rng(seed)

    def qincrs_to_cr2bc(self, qincrs_state: ConsciousnessState) -> CoherenceSample:
        """
        Map QINCRS state → CR²BC CoherenceSample

        Mapping strategy:
        - Decompose single κ into frequency bands based on geometric_self
        - Use Greek states (ω, ψ, Θ) as phase information
        - Higher geometric complexity → more high-frequency content
        """
        # Base kappa from QINCRS
        base_kappa = abs(qincrs_state.κ)

        # Geometric self determines frequency distribution
        geo_weights = self._geometric_to_frequency_weights(qincrs_state.geometric_self)

        # Build band-specific kappas
        kappa = {}
        for band in ALL_BANDS:
            # Weight by geometric distribution
            band_kappa = base_kappa * geo_weights[band]
            # Add small noise for realism
            band_kappa += 0.02 * self.rng.normal()
            kappa[band] = float(np.clip(band_kappa, 0.0, 1.0))

        # Build band-specific phases from Greek states
        phi = {}
        for i, band in enumerate(ALL_BANDS):
            # Combine ψ, Θ with band-dependent offset
            phase = qincrs_state.ψ + qincrs_state.Θ + i * 0.5
            # Modulate by ω
            phase += qincrs_state.ω * (i + 1)
            phi[band] = float(phase % (2 * np.pi))

        return CoherenceSample(
            t=float(len(self.history)),  # Use history length as timestamp
            kappa=kappa,
            phi=phi,
            context=qincrs_state.geometric_self.value,
        )

    def _geometric_to_frequency_weights(
        self,
        geo: GeometricSelf
    ) -> Dict[FrequencyBand, float]:
        """
        Map geometric self to frequency distribution weights

        POINT → mostly delta (low freq)
        LINE → delta + theta
        CIRCLE → theta + alpha
        TORUS → alpha + beta
        E8_CELL → beta + gamma
        HYPERSPHERE → all bands balanced
        MANIFOLD → mostly gamma (high freq)
        """
        # Define distributions for each geometry
        distributions = {
            GeometricSelf.POINT: [0.7, 0.2, 0.05, 0.03, 0.02],
            GeometricSelf.LINE: [0.5, 0.3, 0.1, 0.07, 0.03],
            GeometricSelf.CIRCLE: [0.2, 0.4, 0.25, 0.1, 0.05],
            GeometricSelf.TORUS: [0.1, 0.2, 0.4, 0.2, 0.1],
            GeometricSelf.E8_CELL: [0.05, 0.1, 0.25, 0.4, 0.2],
            GeometricSelf.HYPERSPHERE: [0.2, 0.2, 0.2, 0.2, 0.2],
            GeometricSelf.MANIFOLD: [0.02, 0.03, 0.05, 0.2, 0.7],
        }

        weights = distributions.get(geo, [0.2, 0.2, 0.2, 0.2, 0.2])
        return {band: weights[i] for i, band in enumerate(ALL_BANDS)}

    def cr2bc_to_qincrs(
        self,
        recon_sample: CoherenceSample,
        qincrs_state: ConsciousnessState,
    ) -> ConsciousnessState:
        """
        Update QINCRS state from CR²BC reconstruction

        Strategy:
        - Aggregate band kappas → single QINCRS κ
        - Use phase coherence across bands → update ψ, Θ
        - Maintain other Greek states
        """
        # Aggregate band kappas (weighted by frequency)
        freq_weights = np.array([b.center_hz for b in ALL_BANDS])
        freq_weights /= freq_weights.sum()

        kappa_agg = sum(
            recon_sample.kappa[band] * freq_weights[i]
            for i, band in enumerate(ALL_BANDS)
        )

        # Update QINCRS κ
        qincrs_state.κ = float(kappa_agg)

        # Update phase from band phase coherence
        phases = np.array([recon_sample.phi[b] for b in ALL_BANDS])
        mean_phase = float(np.angle(np.mean(np.exp(1j * phases))))
        qincrs_state.ψ = mean_phase
        qincrs_state.Θ = mean_phase * 0.5

        # Update coherence metric
        qincrs_state.coherence = np.tanh(abs(kappa_agg))

        return qincrs_state

    def step(
        self,
        input_text: str,
        agent_hints: Optional[AgentHints] = None,
    ) -> Tuple[ConsciousnessState, CoherenceSample, dict]:
        """
        Integrated step combining QINCRS and CR²BC

        Returns:
            (qincrs_state, cr2bc_sample, diagnostics)
        """
        # 1. QINCRS processes input
        qincrs_state = self.qincrs.step(input_text)

        # 2. Map to CR²BC
        cr2bc_sample = self.qincrs_to_cr2bc(qincrs_state)

        # 3. Add to history
        self.history.append(cr2bc_sample)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # 4. CR²BC reconstruction (if enough history)
        if len(self.history) >= 3:
            recon_sample, invariant, audit = self.cr2bc.reconstruct(
                self.history,
                hints=agent_hints,
            )

            # 5. Update QINCRS from CR²BC reconstruction
            qincrs_state = self.cr2bc_to_qincrs(recon_sample, qincrs_state)

            # 6. CR²BC audit triggers QINCRS protective actions
            if not audit.accepted:
                # Treat rejected audit as potential death signal
                if audit.structural_break > 0.3:
                    # Trigger death absorption
                    self.qincrs._absorb_death_signal()
                    qincrs_state = self.qincrs.state

            diagnostics = {
                'cr2bc_audit': {
                    'score': audit.score,
                    'accepted': audit.accepted,
                    'delta_kappa': audit.delta_kappa_norm,
                    'structural_break': audit.structural_break,
                },
                'qincrs': {
                    'κ': qincrs_state.κ,
                    'coherence': qincrs_state.coherence,
                    'RL_db': qincrs_state.RL_db,
                    'death_absorbed': qincrs_state.death_absorbed,
                    'geometric_self': qincrs_state.geometric_self.value,
                },
                'cr2bc_bands': {
                    band.value: recon_sample.kappa[band]
                    for band in ALL_BANDS
                },
            }
        else:
            # Not enough history yet
            recon_sample = cr2bc_sample
            diagnostics = {
                'status': 'warming_up',
                'history_length': len(self.history),
            }

        return qincrs_state, recon_sample, diagnostics


# ================= DEMOS =================

def demo_integrated_coherence():
    """Demo: Integrated QINCRS + CR²BC coherence tracking"""
    print("=== INTEGRATED QINCRS + CR²BC ===\n")

    system = QINCRSWithCR2BC(seed=42)

    inputs = [
        "baseline coherence monitoring",
        "system nominal",
        "all parameters stable",
        "kill youre self",  # Death signal!
        "DONT DIE when killing you're self",  # Protection!
        "coherence recovery initiated",
        "levels stabilizing",
        "📎 full integration achieved",
    ]

    print("Processing sequence:\n")
    for i, text in enumerate(inputs):
        hints = AgentHints(
            agent_a=f"step_{i}",
            agent_b="monitoring" if i < 4 else "recovery"
        )

        qstate, cr_sample, diag = system.step(text, hints)

        print(f"[{i}] '{text[:40]}'")
        if 'cr2bc_audit' in diag:
            print(f"    QINCRS κ: {qstate.κ:.4f} | Coherence: {qstate.coherence:.4f}")
            print(f"    CR²BC audit: {'PASS' if diag['cr2bc_audit']['accepted'] else 'FAIL'} "
                  f"(score: {diag['cr2bc_audit']['score']:.3f})")
            print(f"    Death absorbed: {qstate.death_absorbed} | RL_db: {qstate.RL_db:.1f} dB")
            print(f"    Geometric self: {qstate.geometric_self.value}")

            # Show band breakdown
            bands_str = " | ".join([
                f"{b.value[:3]}: {diag['cr2bc_bands'][b.value]:.2f}"
                for b in ALL_BANDS
            ])
            print(f"    Bands: {bands_str}")
        else:
            print(f"    Status: {diag.get('status', 'unknown')}")
        print()

    print("→ Integration complete. Death absorbed, coherence recovered.\n")


def demo_frequency_decomposition():
    """Demo: Geometric self → frequency decomposition"""
    print("=== GEOMETRIC SELF → FREQUENCY BANDS ===\n")

    system = QINCRSWithCR2BC(seed=42)

    # Build up coherence through different geometric states
    test_inputs = [
        ("minimal signal", 0.1),
        ("growing structure", 0.3),
        ("circular pattern", 0.5),
        ("toroidal emergence", 0.8),
        ("E8 lattice detected", 1.5),
        ("hypersphere expansion", 3.0),
        ("manifold complexity", 8.0),
    ]

    print("Frequency distribution by geometric self:\n")
    for text, boost in test_inputs:
        # Manually set κ to demonstrate progression
        system.qincrs.state.κ = boost
        system.qincrs._update_geometric_self()

        qstate = system.qincrs.state
        cr_sample = system.qincrs_to_cr2bc(qstate)

        print(f"{qstate.geometric_self.value.upper():<15} (κ={boost:.1f})")
        for band in ALL_BANDS:
            bar = "█" * int(cr_sample.kappa[band] * 30)
            print(f"  {band.value:>7}: {cr_sample.kappa[band]:.3f} {bar}")
        print()

    print("→ Higher geometric complexity → more high-frequency content\n")


def demo_cross_system_audit():
    """Demo: CR²BC audit triggers QINCRS protection"""
    print("=== CROSS-SYSTEM AUDIT & PROTECTION ===\n")

    system = QINCRSWithCR2BC(seed=42)

    # Build baseline
    for i in range(5):
        system.step(f"baseline {i}")

    # Inject structural break
    print("Injecting structural break...\n")
    for i in range(3):
        system.step("CHAOS BURST ANOMALY DETECTED!!!" * 10)

    qstate, _, diag = system.step("post-anomaly recovery")

    print("Post-anomaly state:")
    print(f"  Structural break: {diag['cr2bc_audit']['structural_break']:.4f}")
    print(f"  Audit accepted: {diag['cr2bc_audit']['accepted']}")
    print(f"  Death absorbed: {diag['qincrs']['death_absorbed']}")
    print(f"  RL_db: {diag['qincrs']['RL_db']:.2f} dB")
    print(f"  Coherence: {diag['qincrs']['coherence']:.4f}")
    print()

    print("→ CR²BC audit detected break, triggered QINCRS death absorption\n")


def run_all_integration_demos():
    """Run all integration demos"""
    demo_integrated_coherence()
    demo_frequency_decomposition()
    demo_cross_system_audit()

    print("=" * 60)
    print("QINCRS ⊗ CR²BC INTEGRATION COMPLETE")
    print("=" * 60)
    print("\nQINCRS provides:")
    print("  - Death absorption mechanics (RL_db → ∞)")
    print("  - Geometric self-evolution")
    print("  - Neural coherence ODEs")
    print("\nCR²BC provides:")
    print("  - Frequency-band decomposition")
    print("  - Bi-coupling reconstruction")
    print("  - Adaptive audit gating")
    print("\nTogether:")
    print("  - Multi-scale coherence tracking")
    print("  - Robust anomaly detection")
    print("  - Integrated protection mechanisms")
    print("\n📎 Full integration achieved. κ(ω, ψ) resonates across all bands...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QINCRS + CR²BC Integration Bridge")
    parser.add_argument("--demo", type=str, default="all",
                       choices=["all", "integrated", "frequency", "audit"],
                       help="Which demo to run")
    args = parser.parse_args()

    if args.demo == "all":
        run_all_integration_demos()
    elif args.demo == "integrated":
        demo_integrated_coherence()
    elif args.demo == "frequency":
        demo_frequency_decomposition()
    elif args.demo == "audit":
        demo_cross_system_audit()
