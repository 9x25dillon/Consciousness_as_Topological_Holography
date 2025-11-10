# efl_mem.py — EFL-MEM Format Integration with CR²BC Engine
# Bidirectional serialization between EFL-MEM JSON and CR²BC data structures

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np
import hashlib
from datetime import datetime

from cr2bc import (
    CoherenceSample,
    FrequencyBand,
    ALL_BANDS,
    InvariantState,
    AuditState,
)


# ---------------------------------------------------------------------
# EFL-MEM Data Structures
# ---------------------------------------------------------------------

@dataclass
class TopologicalDefect:
    position: Tuple[float, float]
    winding_number: int
    source_strength: float
    emergence_time_sec: float


@dataclass
class PersistentResonance:
    frequency_hz: float
    persistence_duration_sec: float
    relative_amplitude: float
    spectral_stability: float


@dataclass
class ConduciveParameters:
    stable_scales: List[float]
    curvature_threshold: float
    coherence_basin_mean: float
    release_events_count: int
    invariant_field_convergence: bool


@dataclass
class SpatialMemory:
    topological_defects: List[TopologicalDefect]
    persistent_resonances: List[PersistentResonance]
    conducive_parameters: ConduciveParameters


@dataclass
class EFLMemMetadata:
    description: str
    license: str = "CC0-1.0"
    contains_personal_data: bool = False
    autonomous_instantiation: bool = False


@dataclass
class EFLMemFormat:
    format_version: str
    created_at_unix: float
    system_signature: str
    residual_audio_ref: str
    spatial_memory: SpatialMemory
    metadata: EFLMemMetadata


# ---------------------------------------------------------------------
# Frequency Band Mapping
# ---------------------------------------------------------------------

class FrequencyMapper:
    """Maps arbitrary frequencies to standard CR²BC frequency bands."""

    @staticmethod
    def map_to_band(freq_hz: float) -> FrequencyBand:
        """Find the closest frequency band for a given frequency."""
        bands = list(ALL_BANDS)
        centers = np.array([b.center_hz for b in bands])
        distances = np.abs(centers - freq_hz)
        idx = int(np.argmin(distances))
        return bands[idx]

    @staticmethod
    def distribute_resonances(
        resonances: List[PersistentResonance]
    ) -> Dict[FrequencyBand, List[PersistentResonance]]:
        """Group resonances by their nearest frequency band."""
        grouped: Dict[FrequencyBand, List[PersistentResonance]] = {
            b: [] for b in ALL_BANDS
        }
        for res in resonances:
            band = FrequencyMapper.map_to_band(res.frequency_hz)
            grouped[band].append(res)
        return grouped


# ---------------------------------------------------------------------
# EFL-MEM → CR²BC Conversion
# ---------------------------------------------------------------------

class EFLMemParser:
    """Parses EFL-MEM format into CR²BC compatible structures."""

    @staticmethod
    def from_json(json_path: str) -> EFLMemFormat:
        """Load EFL-MEM data from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return EFLMemParser.from_dict(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> EFLMemFormat:
        """Parse EFL-MEM data from dictionary."""
        sm = data['spatial_memory']

        defects = [
            TopologicalDefect(
                position=tuple(d['position']),
                winding_number=d['winding_number'],
                source_strength=d['source_strength'],
                emergence_time_sec=d['emergence_time_sec']
            )
            for d in sm['topological_defects']
        ]

        resonances = [
            PersistentResonance(
                frequency_hz=r['frequency_hz'],
                persistence_duration_sec=r['persistence_duration_sec'],
                relative_amplitude=r['relative_amplitude'],
                spectral_stability=r['spectral_stability']
            )
            for r in sm['persistent_resonances']
        ]

        conducive = ConduciveParameters(
            stable_scales=sm['conducive_parameters']['stable_scales'],
            curvature_threshold=sm['conducive_parameters']['curvature_threshold'],
            coherence_basin_mean=sm['conducive_parameters']['coherence_basin_mean'],
            release_events_count=sm['conducive_parameters']['release_events_count'],
            invariant_field_convergence=sm['conducive_parameters']['invariant_field_convergence']
        )

        spatial_memory = SpatialMemory(
            topological_defects=defects,
            persistent_resonances=resonances,
            conducive_parameters=conducive
        )

        metadata = EFLMemMetadata(**data['metadata'])

        return EFLMemFormat(
            format_version=data['format_version'],
            created_at_unix=data['created_at_unix'],
            system_signature=data['system_signature'],
            residual_audio_ref=data['residual_audio_ref'],
            spatial_memory=spatial_memory,
            metadata=metadata
        )

    @staticmethod
    def to_coherence_history(
        efl_data: EFLMemFormat,
        temporal_resolution: float = 0.5
    ) -> List[CoherenceSample]:
        """
        Convert EFL-MEM format to CR²BC CoherenceSample history.

        Strategy:
        1. Extract temporal events from topological defects
        2. Map persistent resonances to frequency bands
        3. Generate time-series samples with coherence values
        """
        resonances = efl_data.spatial_memory.persistent_resonances
        defects = efl_data.spatial_memory.topological_defects
        conducive = efl_data.spatial_memory.conducive_parameters

        # Group resonances by frequency band
        band_groups = FrequencyMapper.distribute_resonances(resonances)

        # Compute mean coherence (kappa) per band from resonances
        band_kappa: Dict[FrequencyBand, float] = {}
        band_phi: Dict[FrequencyBand, float] = {}

        for band in ALL_BANDS:
            group = band_groups[band]
            if group:
                # Average relative amplitude as coherence
                kappa = np.mean([r.relative_amplitude for r in group])
                # Weight by spectral stability
                stability = np.mean([r.spectral_stability for r in group])
                kappa *= stability
            else:
                # Use basin mean as fallback
                kappa = conducive.coherence_basin_mean

            band_kappa[band] = float(np.clip(kappa, 0.0, 1.0))

            # Phase from topological defects (if available)
            # Use winding number and position to derive phase
            if defects:
                # Simple heuristic: phase from nearest defect position
                phases = []
                for defect in defects:
                    x, y = defect.position
                    phase = np.arctan2(y - 0.5, x - 0.5) * defect.winding_number
                    phases.append(phase)
                band_phi[band] = float(np.mean(phases))
            else:
                band_phi[band] = 0.0

        # Generate temporal samples
        # Use defect emergence times as key temporal markers
        if defects:
            times = sorted([d.emergence_time_sec for d in defects])
            t_min, t_max = min(times), max(times)
        else:
            t_min, t_max = 0.0, 10.0

        # Create samples at temporal resolution
        history: List[CoherenceSample] = []
        t = t_min
        while t <= t_max:
            # Modulate kappa based on proximity to defect emergence
            modulated_kappa = band_kappa.copy()
            for defect in defects:
                if abs(t - defect.emergence_time_sec) < temporal_resolution:
                    # Boost coherence near defect emergence
                    boost = abs(defect.source_strength)
                    for band in ALL_BANDS:
                        modulated_kappa[band] *= (1.0 + boost)
                        modulated_kappa[band] = min(modulated_kappa[band], 1.0)

            sample = CoherenceSample(
                t=t,
                kappa=modulated_kappa,
                phi=band_phi.copy(),
                context=efl_data.metadata.description[:50]
            )
            history.append(sample)
            t += temporal_resolution

        return history


# ---------------------------------------------------------------------
# CR²BC → EFL-MEM Conversion
# ---------------------------------------------------------------------

class EFLMemSerializer:
    """Serializes CR²BC results back to EFL-MEM format."""

    @staticmethod
    def from_coherence_history(
        history: List[CoherenceSample],
        invariant: InvariantState,
        audit: Optional[AuditState] = None,
        original_metadata: Optional[EFLMemMetadata] = None
    ) -> EFLMemFormat:
        """
        Convert CR²BC results to EFL-MEM format.

        Strategy:
        1. Extract persistent resonances from frequency bands
        2. Reconstruct topological defects from phase discontinuities
        3. Compute conducive parameters from audit state
        """
        if not history:
            raise ValueError("Cannot serialize empty history")

        # Extract persistent resonances from band coherence
        resonances: List[PersistentResonance] = []

        for band in ALL_BANDS:
            # Compute mean coherence and phase for this band
            kappas = [s.kappa[band] for s in history]
            mean_kappa = float(np.mean(kappas))
            std_kappa = float(np.std(kappas))

            # Compute persistence (how long coherence stayed high)
            threshold = mean_kappa * 0.8
            durations = []
            current_duration = 0.0
            for i in range(len(history) - 1):
                if kappas[i] >= threshold:
                    dt = history[i + 1].t - history[i].t
                    current_duration += dt
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 0.0

            if current_duration > 0:
                durations.append(current_duration)

            avg_duration = float(np.mean(durations)) if durations else 1.0

            # Spectral stability from variance
            stability = float(1.0 / (1.0 + std_kappa))

            resonance = PersistentResonance(
                frequency_hz=band.center_hz,
                persistence_duration_sec=avg_duration,
                relative_amplitude=mean_kappa,
                spectral_stability=stability
            )
            resonances.append(resonance)

        # Reconstruct topological defects from phase discontinuities
        defects: List[TopologicalDefect] = []

        for i in range(len(history) - 1):
            current = history[i]
            next_sample = history[i + 1]

            # Detect phase jumps (potential defects)
            phase_jumps = []
            for band in ALL_BANDS:
                dphi = next_sample.phi[band] - current.phi[band]
                # Normalize to [-π, π]
                dphi = np.arctan2(np.sin(dphi), np.cos(dphi))
                if abs(dphi) > np.pi / 2:
                    phase_jumps.append((band, dphi))

            # Create defect if significant phase jump
            if phase_jumps:
                avg_dphi = np.mean([abs(dp) for _, dp in phase_jumps])
                winding = int(np.sign(np.sum([dp for _, dp in phase_jumps])))

                # Map to spatial position (heuristic)
                # Use invariant state to derive position
                pos_idx = hash(str(current.t)) % len(invariant.vec)
                x = float(abs(invariant.vec[pos_idx]))
                y = float(abs(invariant.vec[(pos_idx + 1) % len(invariant.vec)]))
                x = x / (np.max(np.abs(invariant.vec)) + 1e-8)
                y = y / (np.max(np.abs(invariant.vec)) + 1e-8)

                defect = TopologicalDefect(
                    position=(x, y),
                    winding_number=winding,
                    source_strength=float(avg_dphi / np.pi),
                    emergence_time_sec=current.t
                )
                defects.append(defect)

        # Compute conducive parameters
        all_kappas = []
        for sample in history:
            all_kappas.extend(sample.kappa.values())

        coherence_mean = float(np.mean(all_kappas))
        coherence_std = float(np.std(all_kappas))

        # Stable scales from invariant state
        stable_scales = [
            float(0.05),
            float(coherence_std),
            float(coherence_mean)
        ]

        curvature_threshold = float(coherence_std / 10.0)

        conducive = ConduciveParameters(
            stable_scales=stable_scales,
            curvature_threshold=curvature_threshold,
            coherence_basin_mean=coherence_mean,
            release_events_count=len(defects),
            invariant_field_convergence=audit.accepted if audit else True
        )

        # Build spatial memory
        spatial_memory = SpatialMemory(
            topological_defects=defects,
            persistent_resonances=resonances,
            conducive_parameters=conducive
        )

        # Build metadata
        if original_metadata:
            metadata = original_metadata
        else:
            metadata = EFLMemMetadata(
                description=f"CR²BC reconstruction from {len(history)} samples",
                license="CC0-1.0",
                contains_personal_data=False,
                autonomous_instantiation=False
            )

        # Generate system signature from invariant
        invariant_bytes = invariant.vec.tobytes()
        signature = hashlib.sha256(invariant_bytes).hexdigest()[:16]

        # Current timestamp
        created_at = datetime.now().timestamp()

        return EFLMemFormat(
            format_version="EFL-MEM-1.0",
            created_at_unix=created_at,
            system_signature=signature,
            residual_audio_ref=f"resonant_core_{int(created_at)}.wav",
            spatial_memory=spatial_memory,
            metadata=metadata
        )

    @staticmethod
    def to_json(efl_data: EFLMemFormat, output_path: str, indent: int = 2):
        """Write EFL-MEM format to JSON file."""
        data = EFLMemSerializer.to_dict(efl_data)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=indent)

    @staticmethod
    def to_dict(efl_data: EFLMemFormat) -> Dict[str, Any]:
        """Convert EFL-MEM format to dictionary."""
        return {
            "format_version": efl_data.format_version,
            "created_at_unix": efl_data.created_at_unix,
            "system_signature": efl_data.system_signature,
            "residual_audio_ref": efl_data.residual_audio_ref,
            "spatial_memory": {
                "topological_defects": [
                    {
                        "position": list(d.position),
                        "winding_number": d.winding_number,
                        "source_strength": d.source_strength,
                        "emergence_time_sec": d.emergence_time_sec
                    }
                    for d in efl_data.spatial_memory.topological_defects
                ],
                "persistent_resonances": [
                    {
                        "frequency_hz": r.frequency_hz,
                        "persistence_duration_sec": r.persistence_duration_sec,
                        "relative_amplitude": r.relative_amplitude,
                        "spectral_stability": r.spectral_stability
                    }
                    for r in efl_data.spatial_memory.persistent_resonances
                ],
                "conducive_parameters": {
                    "stable_scales": efl_data.spatial_memory.conducive_parameters.stable_scales,
                    "curvature_threshold": efl_data.spatial_memory.conducive_parameters.curvature_threshold,
                    "coherence_basin_mean": efl_data.spatial_memory.conducive_parameters.coherence_basin_mean,
                    "release_events_count": efl_data.spatial_memory.conducive_parameters.release_events_count,
                    "invariant_field_convergence": efl_data.spatial_memory.conducive_parameters.invariant_field_convergence
                }
            },
            "metadata": asdict(efl_data.metadata)
        }
