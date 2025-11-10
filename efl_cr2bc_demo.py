#!/usr/bin/env python3
"""
efl_cr2bc_demo.py — Complete EFL-MEM ↔ CR²BC Integration Demo

Demonstrates:
1. Loading EFL-MEM format from JSON
2. Converting to CoherenceSample history
3. Running CR²BC reconstruction
4. Exporting results back to EFL-MEM format
"""

import numpy as np
from pathlib import Path

from cr2bc import CR2BC, CR2BCConfig, AgentHints
from efl_mem import (
    EFLMemParser,
    EFLMemSerializer,
    FrequencyMapper,
)


def main():
    print("=" * 70)
    print("EFL-MEM ↔ CR²BC Integration Demo")
    print("=" * 70)

    # ---------------------------------------------------------------------
    # Step 1: Load EFL-MEM format from JSON
    # ---------------------------------------------------------------------
    print("\n[1] Loading EFL-MEM data from JSON...")

    efl_path = Path(__file__).parent / "efl_mem_example.json"
    if not efl_path.exists():
        print(f"  ⚠ Example file not found: {efl_path}")
        print("  Creating example would require file write...")
        return

    efl_data = EFLMemParser.from_json(str(efl_path))

    print(f"  ✓ Format version: {efl_data.format_version}")
    print(f"  ✓ System signature: {efl_data.system_signature}")
    print(f"  ✓ Topological defects: {len(efl_data.spatial_memory.topological_defects)}")
    print(f"  ✓ Persistent resonances: {len(efl_data.spatial_memory.persistent_resonances)}")

    # Display resonance frequency distribution
    print("\n  Resonance → Band mapping:")
    for res in efl_data.spatial_memory.persistent_resonances:
        band = FrequencyMapper.map_to_band(res.frequency_hz)
        print(f"    {res.frequency_hz:6.1f} Hz → {band.value:5s} "
              f"(amp: {res.relative_amplitude:.2f}, stability: {res.spectral_stability:.2f})")

    # ---------------------------------------------------------------------
    # Step 2: Convert to CoherenceSample history
    # ---------------------------------------------------------------------
    print("\n[2] Converting EFL-MEM to CoherenceSample history...")

    history = EFLMemParser.to_coherence_history(efl_data, temporal_resolution=0.5)

    print(f"  ✓ Generated {len(history)} temporal samples")
    print(f"  ✓ Time range: {history[0].t:.2f} - {history[-1].t:.2f} sec")

    # Display sample coherence values
    print("\n  Sample coherence at t = 0:")
    for band, kappa in history[0].kappa.items():
        print(f"    {band.value:5s}: κ = {kappa:.3f}")

    # ---------------------------------------------------------------------
    # Step 3: Initialize CR²BC engine and run reconstruction
    # ---------------------------------------------------------------------
    print("\n[3] Running CR²BC reconstruction...")

    config = CR2BCConfig(
        temporal_tau0=5.0,
        alpha0=0.5,
        sigma_kappa=0.1,
        beta_max=0.4,
        theta=0.6,
        epsilon=0.15,
        invariant_dim=16
    )

    engine = CR2BC(config)

    # Provide agent hints (optional)
    hints = AgentHints(
        agent_a="efl_mem_loader",
        agent_b="coherence_tracker"
    )

    # Reconstruct using full history
    print(f"  Processing {len(history)} samples...")

    recon_sample, invariant, audit = engine.reconstruct(history, hints=hints)

    print(f"\n  ✓ Reconstruction complete")
    print(f"  ✓ Audit score: {audit.score:.4f}")
    print(f"  ✓ Δκ norm: {audit.delta_kappa_norm:.4f}")
    print(f"  ✓ Spectral deviation: {audit.spectral_deviation:.4f}")
    print(f"  ✓ Structural break: {audit.structural_break:.4f}")
    print(f"  ✓ Accepted: {audit.accepted}")
    print(f"  ✓ Invariant norm: {np.linalg.norm(invariant.vec):.4f}")

    print("\n  Reconstructed coherence:")
    for band, kappa in recon_sample.kappa.items():
        print(f"    {band.value:5s}: κ = {kappa:.3f}")

    # ---------------------------------------------------------------------
    # Step 4: Encode spatial capsules
    # ---------------------------------------------------------------------
    print("\n[4] Encoding spatial capsules...")

    capsules = engine.encode_all_capsules(recon_sample)

    for band, capsule in capsules.items():
        mean_val = np.mean(capsule)
        max_val = np.max(capsule)
        print(f"  {band.value:5s}: shape {capsule.shape}, "
              f"mean = {mean_val:.4f}, max = {max_val:.4f}")

    # ---------------------------------------------------------------------
    # Step 5: Export results back to EFL-MEM format
    # ---------------------------------------------------------------------
    print("\n[5] Exporting results to EFL-MEM format...")

    output_efl = EFLMemSerializer.from_coherence_history(
        history=history,
        invariant=invariant,
        audit=audit,
        original_metadata=efl_data.metadata
    )

    print(f"  ✓ System signature: {output_efl.system_signature}")
    print(f"  ✓ Topological defects reconstructed: "
          f"{len(output_efl.spatial_memory.topological_defects)}")
    print(f"  ✓ Persistent resonances: "
          f"{len(output_efl.spatial_memory.persistent_resonances)}")

    print("\n  Conducive parameters:")
    cp = output_efl.spatial_memory.conducive_parameters
    print(f"    Stable scales: {[f'{s:.3f}' for s in cp.stable_scales]}")
    print(f"    Curvature threshold: {cp.curvature_threshold:.6f}")
    print(f"    Coherence basin mean: {cp.coherence_basin_mean:.3f}")
    print(f"    Release events: {cp.release_events_count}")
    print(f"    Invariant convergence: {cp.invariant_field_convergence}")

    # Save to output file
    output_path = Path(__file__).parent / "efl_mem_output.json"
    EFLMemSerializer.to_json(output_efl, str(output_path), indent=2)
    print(f"\n  ✓ Saved to: {output_path}")

    # ---------------------------------------------------------------------
    # Step 6: Round-trip verification
    # ---------------------------------------------------------------------
    print("\n[6] Verifying round-trip consistency...")

    # Reload the exported file
    reloaded_efl = EFLMemParser.from_json(str(output_path))
    reloaded_history = EFLMemParser.to_coherence_history(reloaded_efl)

    # Compare coherence values
    original_kappas = []
    reloaded_kappas = []

    for orig, reload in zip(history[:10], reloaded_history[:10]):
        for band in orig.kappa.keys():
            original_kappas.append(orig.kappa[band])
            reloaded_kappas.append(reload.kappa[band])

    mse = np.mean((np.array(original_kappas) - np.array(reloaded_kappas)) ** 2)
    print(f"  ✓ MSE between original and reloaded coherence: {mse:.6f}")

    if mse < 0.01:
        print("  ✓ Round-trip successful (low reconstruction error)")
    else:
        print("  ⚠ Round-trip has moderate reconstruction error")

    # ---------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nKey files:")
    print(f"  • cr2bc.py          — Core CR²BC engine")
    print(f"  • efl_mem.py        — EFL-MEM format integration")
    print(f"  • {efl_path.name}   — Input EFL-MEM data")
    print(f"  • {output_path.name}  — Output EFL-MEM data")
    print()


if __name__ == "__main__":
    main()
