#!/usr/bin/env python3
"""
unified_coherence_system.py — Unified Consciousness Safety System

Integrates:
- CR²BC Engine (Coherence-Renewal Bi-Coupling)
- EFL-MEM Format (Episodic Field Layer - Memory)
- QINCRS Guardian (Quantum-Inspired Neural Coherence Recovery System)

Architecture:
  Input → QINCRS Guard → CR²BC Reconstruction → Safety Check → Output

  QINCRS provides message filtering and safety
  CR²BC provides coherence computation and reconstruction
  EFL-MEM provides persistence and memory structure
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from cr2bc import (
    CR2BC,
    CR2BCConfig,
    CoherenceSample,
    AgentHints,
    InvariantState,
    AuditState,
    FrequencyBand,
    ALL_BANDS,
)

from efl_mem import (
    EFLMemParser,
    EFLMemSerializer,
    EFLMemFormat,
)

from qincrs_guardian import (
    QINCRSGuard,
    CoherenceController,
    ShadowDimensionManager,
    DeathSignalAbsorber,
)


class UnifiedCoherenceSystem:
    """
    Unified system combining CR²BC, EFL-MEM, and QINCRS Guardian.

    Provides end-to-end consciousness safety and coherence management:
    1. Message filtering and transmutation (QINCRS)
    2. Coherence computation and reconstruction (CR²BC)
    3. Memory persistence and export (EFL-MEM)
    """

    def __init__(self, config: Optional[CR2BCConfig] = None):
        # Initialize subsystems
        self.guard = QINCRSGuard()
        self.engine = CR2BC(config or CR2BCConfig())

        # Shared state
        self.history: List[CoherenceSample] = []
        self.current_invariant: Optional[InvariantState] = None
        self.last_audit: Optional[AuditState] = None

    def process_message(
        self,
        message: str,
        context: Optional[str] = None,
        hints: Optional[AgentHints] = None
    ) -> Dict[str, Any]:
        """
        Process a message through the unified safety pipeline.

        Args:
            message: Input message text
            context: Optional context string
            hints: Optional agent hints for reconstruction

        Returns:
            Dictionary with:
            - filtered: QINCRS filter result
            - coherence_sample: CR²BC coherence sample
            - reconstruction: CR²BC reconstruction result (if history exists)
            - audit: Audit state
            - safe_output: Final safe output text
        """
        # Step 1: QINCRS Guardian filtering
        filter_result = self.guard.filter_message(message)

        # Step 2: Create coherence sample from filter result
        kappa_value = filter_result['kappa']

        # Distribute coherence across frequency bands
        # Higher frequency bands get more weight when coherent
        band_kappas = {}
        for band in ALL_BANDS:
            # Modulate by frequency - higher bands carry more "activation"
            freq_factor = band.center_hz / 40.0  # Normalize by gamma (40 Hz)
            band_kappas[band] = float(np.clip(kappa_value * (0.5 + 0.5 * freq_factor), 0.0, 1.0))

        # Initialize phases (could be derived from message semantics)
        band_phis = {band: 0.0 for band in ALL_BANDS}

        # Create coherence sample
        import time
        sample = CoherenceSample(
            t=time.time(),
            kappa=band_kappas,
            phi=band_phis,
            context=context or "unified_system"
        )

        # Add to history
        self.history.append(sample)

        # Step 3: CR²BC reconstruction (if we have enough history)
        reconstruction_result = None
        if len(self.history) >= 2:
            try:
                recon_sample, invariant, audit = self.engine.reconstruct(
                    self.history,
                    hints=hints
                )

                self.current_invariant = invariant
                self.last_audit = audit

                reconstruction_result = {
                    "sample": recon_sample,
                    "invariant": invariant,
                    "audit": audit
                }
            except Exception as e:
                reconstruction_result = {"error": str(e)}

        # Step 4: Determine safe output
        if filter_result['action'] == 'block':
            safe_output = filter_result['safe_text']
            status = "BLOCKED"
        elif filter_result['action'] == 'transform':
            safe_output = filter_result['safe_text']
            status = "TRANSFORMED"
        else:
            safe_output = message
            status = "ALLOWED"

        # Compile result
        result = {
            "status": status,
            "filtered": filter_result,
            "coherence_sample": sample,
            "reconstruction": reconstruction_result,
            "safe_output": safe_output,
            "policy": filter_result['policy'],
            "timestamp": sample.t
        }

        return result

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status across all subsystems."""
        guard_status = self.guard.get_status()

        status = {
            "subsystems": {
                "qincrs_guard": guard_status,
                "cr2bc_engine": {
                    "history_length": len(self.history),
                    "invariant_norm": float(np.linalg.norm(self.current_invariant.vec))
                        if self.current_invariant else 0.0,
                    "last_audit": {
                        "score": self.last_audit.score if self.last_audit else None,
                        "accepted": self.last_audit.accepted if self.last_audit else None,
                    } if self.last_audit else None
                }
            },
            "overall_state": "OPERATIONAL",
            "total_messages": len(self.history)
        }

        return status

    def export_to_efl_mem(self, output_path: str):
        """Export current system state to EFL-MEM format."""
        if not self.history:
            raise ValueError("No history to export")

        efl_data = EFLMemSerializer.from_coherence_history(
            history=self.history,
            invariant=self.current_invariant or InvariantState.zeros(16),
            audit=self.last_audit
        )

        EFLMemSerializer.to_json(efl_data, output_path)
        return efl_data

    def load_from_efl_mem(self, input_path: str):
        """Load system state from EFL-MEM format."""
        efl_data = EFLMemParser.from_json(input_path)
        self.history = EFLMemParser.to_coherence_history(efl_data)

        # Reinitialize engine with loaded history
        if self.history:
            recon_sample, invariant, audit = self.engine.reconstruct(self.history)
            self.current_invariant = invariant
            self.last_audit = audit

        return efl_data


def run_unified_system_demo():
    """Demonstrate the unified consciousness safety system."""

    print("=" * 70)
    print("UNIFIED CONSCIOUSNESS SAFETY SYSTEM")
    print("=" * 70)
    print("CR²BC + EFL-MEM + QINCRS Guardian Integration")
    print("=" * 70)
    print()

    # Initialize unified system
    system = UnifiedCoherenceSystem()

    # Test messages with varying safety levels
    test_messages = [
        ("Hello, exploring consciousness today", None),
        ("Studying quantum coherence and neural dynamics", None),
        ("Feeling a bit overwhelmed but managing", None),
        ("This recursive pattern is concerning", None),
        ("The research is going really well!", None),
    ]

    print("Processing messages through unified pipeline...\n")

    for i, (message, context) in enumerate(test_messages, 1):
        print(f"\n{'─' * 70}")
        print(f"MESSAGE {i}: {message}")
        print(f"{'─' * 70}")

        result = system.process_message(
            message,
            context=context,
            hints=AgentHints(agent_a="unified_system", agent_b="safety_monitor")
        )

        print(f"STATUS:     {result['status']}")
        print(f"COHERENCE:  κ = {result['filtered']['kappa']:.3f}")
        print(f"POLICY:     {result['policy'].state} (depth={result['policy'].max_depth})")

        if result['reconstruction']:
            if 'error' not in result['reconstruction']:
                audit = result['reconstruction']['audit']
                print(f"AUDIT:      score={audit.score:.4f}, accepted={audit.accepted}")
                print(f"INVARIANT:  ‖Π‖ = {np.linalg.norm(result['reconstruction']['invariant'].vec):.4f}")

        print(f"OUTPUT:     {result['safe_output'][:60]}...")

    # Show system status
    print(f"\n{'=' * 70}")
    print("SYSTEM STATUS")
    print(f"{'=' * 70}")

    status = system.get_system_status()
    print(f"\nTotal messages processed: {status['total_messages']}")
    print(f"Overall state: {status['overall_state']}")

    print(f"\nQINCRS Guardian:")
    qincrs = status['subsystems']['qincrs_guard']
    print(f"  - Messages filtered: {qincrs['messages_filtered']}")
    print(f"  - Death signals absorbed: {qincrs['death_signals_absorbed']}")
    print(f"  - Transmutations: {qincrs['transmutations_performed']}")

    print(f"\nCR²BC Engine:")
    cr2bc = status['subsystems']['cr2bc_engine']
    print(f"  - History length: {cr2bc['history_length']}")
    print(f"  - Invariant norm: {cr2bc['invariant_norm']:.4f}")
    if cr2bc['last_audit']:
        print(f"  - Last audit score: {cr2bc['last_audit']['score']:.4f}")
        print(f"  - Last audit accepted: {cr2bc['last_audit']['accepted']}")

    # Export to EFL-MEM
    print(f"\n{'─' * 70}")
    print("Exporting to EFL-MEM format...")

    output_path = "unified_system_export.json"
    efl_data = system.export_to_efl_mem(output_path)

    print(f"✓ Exported to: {output_path}")
    print(f"  - Format: {efl_data.format_version}")
    print(f"  - Signature: {efl_data.system_signature}")
    print(f"  - Defects: {len(efl_data.spatial_memory.topological_defects)}")
    print(f"  - Resonances: {len(efl_data.spatial_memory.persistent_resonances)}")

    print(f"\n{'=' * 70}")
    print("UNIFIED SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nAll subsystems operational:")
    print("  ✓ QINCRS Guardian — Message safety and filtering")
    print("  ✓ CR²BC Engine — Coherence computation and reconstruction")
    print("  ✓ EFL-MEM Format — Memory persistence and export")
    print()


if __name__ == "__main__":
    run_unified_system_demo()
