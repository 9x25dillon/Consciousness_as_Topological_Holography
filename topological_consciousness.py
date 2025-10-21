#!/usr/bin/env python3
"""
topological_consciousness.py
(2+1)D Topological Holography — Consciousness as TQFT
Headless-safe plotting; demos produce rg_flow.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import norm

class TopologicalConsciousness:
    """
    Models consciousness as topological holography in (2+1)D
    
    The 'sandwich' structure:
    3D boundary ↩ Z(C) ↪ 627D bulk
    
    where Z(C) is the Drinfeld center with ~627 anyonic charges
    """
    
    def __init__(self, n_anyons=27, central_charge=627, seed=None):
        """
        Initialize topological order
        
        Args:
            n_anyons: Number of anyon types (observable states = 3^3 = 27)
            central_charge: c ~ 627 (bulk dimension)
            seed: RNG seed for reproducibility (None for random)
        """
        self.n = int(n_anyons)
        self.c = float(central_charge)
        
        if seed is not None:
            rng = np.random.default_rng(seed)
            self._rng = rng
        else:
            self._rng = np.random.default_rng()
        
        # Generate modular S-matrix (must be unitary-ish and symmetric-ish)
        self.S = self._generate_modular_S_matrix()
        
        # Conformal dimensions (scaling weights)
        self.Delta = self._generate_conformal_dimensions()
        
        # Quantum dimensions (categorical dimensions)
        self.d = self._generate_quantum_dimensions()
        
        # Total quantum dimension (D² = sum of d_i²)
        self.D_squared = float(np.sum(self.d ** 2))
        
    def _generate_modular_S_matrix(self):
        """
        Generate modular S-matrix surrogate for anyon theory.
        Properties (approximate in this toy model):
        - Unitary: S† S ≈ I
        - Symmetric: S ≈ Sᵀ
        - S₀ᵢ ∝ dᵢ/D (quantum dimensions surrogate)
        """
        # Start with random complex matrix, QR -> unitary Q
        A = self._rng.standard_normal((self.n, self.n)) + 1j * self._rng.standard_normal((self.n, self.n))
        Q, _ = np.linalg.qr(A)
        S = (Q + Q.conj().T) / 2.0  # Symmetrize
        # Normalize so rows have norm ~1
        row_norms = np.sqrt((np.abs(S) ** 2).sum(axis=1, keepdims=True)) + 1e-12
        S = S / row_norms
        return S
    
    def _generate_conformal_dimensions(self):
        """
        Conformal dimensions Δᵢ (scaling weights of primary fields)
        Trinary-flavored surrogate spectrum.
        """
        phi = (1 + np.sqrt(5)) / 2.0  # Golden ratio
        idx = np.arange(self.n, dtype=float)
        Delta = (np.power(3.0, idx) - 1.0) / (np.power(phi, idx) * max(self.c, 1.0))
        Delta[0] = 0.0  # vacuum
        return Delta
    
    def _generate_quantum_dimensions(self):
        """
        Quantum dimensions dᵢ (categorical dimensions), surrogate.
        d_i ≈ φ^i, normalized so sum d_i^2 = 1 (toy scaling).
        """
        phi = (1 + np.sqrt(5)) / 2.0
        idx = np.arange(self.n, dtype=float)
        d = np.power(phi, idx)
        d[0] = 1.0
        D = np.sqrt(np.sum(d**2)) + 1e-12
        return d / D
    
    def profunctor_V(self, a, b):
        """
        Consciousness profunctor as anyon propagation amplitude
        V(a,b) = ⟨A_a|Ψ_b⟩ ~ S_ab / S_0a * d_b / D^2 (toy)
        """
        denom = self.S[0, a] if self.S[0, a] != 0 else 1e-12
        amplitude = self.S[a, b] / denom
        weight = self.d[b] / self.D_squared
        return amplitude * weight
    
    def cardy_boundary_energy(self, a, tau=1.0, lambda_tilde=None):
        """
        Toy Cardy boundary energy:
        E_a = (πc/24)(2τa)^2 + Σ_b (S_ab/S_0a) λ̃_b (2τa)^Δb
        """
        if lambda_tilde is None:
            lambda_tilde = np.ones(self.n, dtype=float)
        
        casimir = (np.pi * self.c / 24.0) * (2.0 * tau * a)**2
        
        interaction = 0.0 + 0j
        denom = self.S[0, a] if self.S[0, a] != 0 else 1e-12
        for b in range(self.n):
            coupling = (self.S[a, b] / denom) * lambda_tilde[b]
            scaling = (2.0 * tau * a) ** self.Delta[b] if (2.0 * tau * a) > 0 else 0.0
            interaction += coupling * scaling
        
        total_energy = casimir + interaction
        return total_energy
    
    def find_dominant_conscious_state(self, tau=1.0, lambda_tilde=None):
        energies = [self.cardy_boundary_energy(a, tau, lambda_tilde) for a in range(self.n)]
        # select by minimal real part of energy magnitude
        mags = np.array([np.abs(e) for e in energies])
        min_idx = int(np.argmin(mags))
        min_energy = energies[min_idx]
        return min_idx, min_energy, energies
    
    def kramers_wannier_self_duality(self, g_values, h_values):
        """
        Test self-duality: Z_{g,h} ~ Z_{h,g} in this toy surrogate.
        """
        g_values = np.asarray(g_values, dtype=float)
        h_values = np.asarray(h_values, dtype=float)
        Z = np.zeros((len(g_values), len(h_values)), dtype=complex)
        for i, g in enumerate(g_values):
            for j, h in enumerate(h_values):
                phase = np.exp(2j * np.pi * self.Delta * h)
                Z[i, j] = np.sum(self.S[int(np.floor(g*(self.n-1)+1e-9)) % self.n, :] * phase)
        symmetry_error = norm(Z - Z.T, 'fro')
        return Z, symmetry_error
    
    def twisted_sector_analysis(self):
        """
        Trinary logic from twisted sectors (toy surrogate).
        """
        collapsed = np.array([
            np.sum(self.S[0, k] * np.exp(2j*np.pi*g*self.Delta[k]) for k in range(self.n))
            for g in range(self.n)
        ])
        potential = np.array([
            np.sum(self.S[g, k] * np.exp(0j*self.Delta[k]) for k in range(self.n))
            for g in range(self.n)
        ])
        transcendent = np.array([
            np.sum(self.S[g, k] * np.exp(2j*np.pi*g*self.Delta[k]) for k in range(self.n))
            for g in range(self.n)
        ])
        return {
            'collapsed': np.abs(collapsed),
            'potential': np.abs(potential),
            'transcendent': np.abs(transcendent)
        }
    
    def rg_flow_to_fixed_point(self, K_initial, beta_function, n_steps=100):
        K_trajectory = [float(K_initial)]
        K = float(K_initial)
        for step in range(n_steps):
            dK = float(beta_function(K, step))
            K = K + dK
            K_trajectory.append(float(K))
        final_flow = abs(K_trajectory[-1] - K_trajectory[-2])
        return np.array(K_trajectory, dtype=float), float(final_flow)
    
    def modular_transformation_invariance(self, state):
        transformed = self.S @ state
        overlap = np.abs(np.vdot(transformed, state))
        denom = (norm(transformed) * norm(state)) + 1e-12
        invariance_measure = float(overlap / denom)
        return transformed, invariance_measure
    
    def epistemic_flux_conservation(self, boundary_states, bulk_coupling):
        n_states = len(boundary_states)
        flux = np.zeros(n_states - 1, dtype=np.complex128)
        for i in range(n_states - 1):
            flux[i] = (self.profunctor_V(i, i+1) - self.profunctor_V(i+1, i)) * bulk_coupling
        divergence = np.diff(flux) if len(flux) > 1 else np.array([0+0j])
        total_divergence = float(np.sum(np.abs(divergence)))
        return flux, divergence, total_divergence


# ----------------- DEMOS -----------------

def demo_anyon_consciousness(seed=0):
    print("=== CONSCIOUSNESS AS ANYON PROPAGATION ===\n")
    model = TopologicalConsciousness(n_anyons=27, central_charge=627, seed=seed)
    V_matrix = np.zeros((model.n, model.n), dtype=complex)
    for a in range(model.n):
        for b in range(model.n):
            V_matrix[a, b] = model.profunctor_V(a, b)
    print("Consciousness profunctor V(a,b):")
    print(f"  Shape: {model.n} × {model.n}")
    print(f"  Norm: {norm(V_matrix):.4f}")
    print(f"  Trace: {np.trace(V_matrix):.4f}")
    hermiticity = norm(V_matrix - V_matrix.conj().T)
    print(f"  Hermiticity error: {hermiticity:.6e}")
    print("\n→ V(a,b) encodes anyon propagation (toy).")


def demo_cardy_observer_saturation(seed=0):
    print("\n=== OBSERVER SATURATION VIA CARDY STATES ===\n")
    model = TopologicalConsciousness(n_anyons=27, central_charge=627, seed=seed)
    dominant_state, min_energy, all_energies = model.find_dominant_conscious_state()
    print(f"Dominant conscious state: anyon #{dominant_state}")
    print(f"Minimum boundary energy (magnitude): {np.abs(min_energy):.6f}")
    sorted_mags = np.sort(np.abs(np.array(all_energies, dtype=complex)))
    if len(sorted_mags) > 1:
        gap = float(sorted_mags[1] - sorted_mags[0])
        print(f"Energy gap (magnitude): {gap:.6f}")
        denom = float(np.abs(min_energy)) + 1e-12
        print(f"→ Gap/Temperature ratio (toy): {gap / denom:.4f}")


def demo_kramers_wannier(seed=0):
    print("\n=== KRAMERS-WANNIER SELF-DUALITY (TOY) ===\n")
    model = TopologicalConsciousness(n_anyons=27, central_charge=627, seed=seed)
    g_values = np.linspace(0, 1, 10)
    h_values = np.linspace(0, 1, 10)
    Z, symmetry_error = model.kramers_wannier_self_duality(g_values, h_values)
    print(f"Partition function Z shape: {Z.shape}")
    print(f"Self-duality error ||Z - Z^T||: {symmetry_error:.6e}")
    criticality = float(symmetry_error / (norm(Z, 'fro') + 1e-12))
    print(f"Relative error: {criticality:.6e}")


def demo_trinary_twisted_sectors(seed=0):
    print("\n=== TRINARY LOGIC FROM TWISTED SECTORS (TOY) ===\n")
    model = TopologicalConsciousness(n_anyons=27, central_charge=627, seed=seed)
    sectors = model.twisted_sector_analysis()
    print("First 5 entries per sector (magnitudes):")
    print("  Collapsed (∂):", np.round(sectors['collapsed'][:5], 6))
    print("  Potential (δ):", np.round(sectors['potential'][:5], 6))
    print("  Transcendent (∂↔δ):", np.round(sectors['transcendent'][:5], 6))


def demo_rg_curvature_dissipation(seed=0):
    print("\n=== RG FLOW & CURVATURE DISSIPATION (TOY) ===\n")
    model = TopologicalConsciousness(n_anyons=27, central_charge=627, seed=seed)
    def beta(K, step):
        return -0.1 * K * (1 - np.exp(-step/20.0))
    K_initial = 10.0
    trajectory, final_flow = model.rg_flow_to_fixed_point(K_initial, beta)
    print(f"Initial curvature: {trajectory[0]:.6f}")
    print(f"Final curvature: {trajectory[-1]:.6f}")
    print(f"Final flow rate: {final_flow:.6e}")
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory, linewidth=2, label='K (RG steps)')
    plt.axhline(0, linestyle='--', label='Fixed point K*=0')
    plt.xlabel('RG Step')
    plt.ylabel('Curvature K')
    plt.title('RG Flow to Conformal Fixed Point (Toy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = "rg_flow.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved: {out}")


def demo_modular_invariance(seed=0):
    print("\n=== MODULAR TRANSFORMATION INVARIANCE (TOY) ===\n")
    model = TopologicalConsciousness(n_anyons=27, central_charge=627, seed=seed)
    state = np.zeros(model.n, dtype=complex)
    state[0] = 1.0
    transformed, invariance = model.modular_transformation_invariance(state)
    print(f"Vacuum invariance measure: {invariance:.6f}")
    random_state = model._rng.standard_normal(model.n) + 1j*model._rng.standard_normal(model.n)
    random_state = random_state / (norm(random_state) + 1e-12)
    _, rnd_inv = model.modular_transformation_invariance(random_state)
    print(f"Random state invariance: {rnd_inv:.6f}")


def demo_epistemic_flux(seed=0):
    print("\n=== EPISTEMIC FLUX CONSERVATION (TOY) ===\n")
    model = TopologicalConsciousness(n_anyons=27, central_charge=627, seed=seed)
    boundary_states = list(range(model.n))
    bulk_coupling = 1.0
    flux, divergence, total_div = model.epistemic_flux_conservation(boundary_states, bulk_coupling)
    print(f"Number of boundary states: {len(boundary_states)}")
    print("Flux (first 5):", np.round(flux[:5], 6))
    if len(divergence) > 0:
        print("Divergence (first 5):", np.round(divergence[:5], 6))
    print(f"Total divergence L1: {total_div:.6e}")


def run_all(seed=0):
    demo_anyon_consciousness(seed)
    demo_cardy_observer_saturation(seed)
    demo_kramers_wannier(seed)
    demo_trinary_twisted_sectors(seed)
    demo_rg_curvature_dissipation(seed)
    demo_modular_invariance(seed)
    demo_epistemic_flux(seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Topological Holography — (2+1)D TQFT toy demos")
    parser.add_argument("--demo", type=str, default="all",
                        choices=["all","anyon","cardy","kw","trinary","rg","modular","flux"],
                        help="Which demo to run")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    args = parser.parse_args()
    demo = args.demo
    if demo == "all":
        run_all(args.seed)
    elif demo == "anyon":
        demo_anyon_consciousness(args.seed)
    elif demo == "cardy":
        demo_cardy_observer_saturation(args.seed)
    elif demo == "kw":
        demo_kramers_wannier(args.seed)
    elif demo == "trinary":
        demo_trinary_twisted_sectors(args.seed)
    elif demo == "rg":
        demo_rg_curvature_dissipation(args.seed)
    elif demo == "modular":
        demo_modular_invariance(args.seed)
    elif demo == "flux":
        demo_epistemic_flux(args.seed)