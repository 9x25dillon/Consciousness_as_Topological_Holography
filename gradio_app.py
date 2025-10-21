#!/usr/bin/env python3
"""
Gradio app: 3^627 Framework — Consciousness as Topological Holography

Interactive demos (headless-safe using Matplotlib):
- Phase coherence between two texts (cosine similarity → phase angle)
- Trinary vs binary quantization (toy MSE comparison)
- Cardy boundary energy spectrum (toy)
- Modular S-matrix heatmap (toy)
- Dimensional analysis: capacity vs dimensions with 627/994 markers
"""

from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr

from topological_consciousness import TopologicalConsciousness


# ------------------------ Utilities ------------------------

def _cosine_similarity_from_counts(text1: str, text2: str) -> float:
    """Lightweight cosine similarity using word-count dictionaries."""
    def counts(s: str):
        tokens = [t.lower() for t in s.split() if t.strip()]
        d = {}
        for t in tokens:
            d[t] = d.get(t, 0.0) + 1.0
        return d

    c1 = counts(text1)
    c2 = counts(text2)
    # Dot product over intersection
    dot = 0.0
    for k in c1.keys() & c2.keys():
        dot += c1[k] * c2[k]
    # Norms
    n1 = np.sqrt(sum(v * v for v in c1.values())) + 1e-12
    n2 = np.sqrt(sum(v * v for v in c2.values())) + 1e-12
    sim = float(dot / (n1 * n2))
    sim = max(-1.0, min(1.0, sim))
    return sim


def _simple_bar_plot(x_labels, values, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x_labels, values, color=["#4c78a8", "#72b7b2", "#f58518", "#e45756"][: len(values)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    return fig


# ------------------------ Demo Functions ------------------------

def demo_phase_coherence(text1: str, text2: str) -> Tuple[str, plt.Figure]:
    sim = _cosine_similarity_from_counts(text1 or "", text2 or "")
    phase = float(np.arccos(sim))  # radians
    deg = float(np.degrees(phase))

    summary = (
        f"**Phase coherence** (cosine similarity): {sim:.4f} | "
        f"**Phase angle**: {deg:.2f}°"
    )

    fig = _simple_bar_plot(["Similarity", "Angle (deg)"], [sim, deg], "Phase Coherence", "Value")
    return summary, fig


def demo_trinary_quantization(n_samples: int, noise_level: float) -> Tuple[str, plt.Figure]:
    rng = np.random.default_rng(0)
    w = rng.standard_normal(int(n_samples))
    noisy = w + noise_level * rng.standard_normal(int(n_samples))

    # Binary: {-1, +1}
    q2 = np.where(noisy >= 0.0, 1.0, -1.0)

    # Trinary: {-1, 0, +1} with threshold tau linked to noise_level
    tau = max(0.05, float(noise_level))
    q3 = np.zeros_like(noisy)
    q3[noisy > tau] = 1.0
    q3[noisy < -tau] = -1.0

    mse2 = float(np.mean((q2 - w) ** 2))
    mse3 = float(np.mean((q3 - w) ** 2))

    msg = (
        f"**Binary MSE**: {mse2:.4f}\n\n"
        f"**Trinary MSE**: {mse3:.4f}\n\n"
        f"**Winner**: {'Trinary' if mse3 < mse2 else 'Binary'}"
    )

    fig = _simple_bar_plot(["Binary", "Trinary"], [mse2, mse3], "Quantization MSE (toy)", "MSE vs original weights")
    return msg, fig


def demo_cardy_states(seed: int = 0) -> Tuple[str, plt.Figure]:
    model = TopologicalConsciousness(n_anyons=27, central_charge=627, seed=seed)
    energies = [model.cardy_boundary_energy(a) for a in range(model.n)]
    mags = np.array([np.abs(e) for e in energies], dtype=float)
    idx_min = int(np.argmin(mags))
    min_energy = float(mags[idx_min])

    msg = f"**Dominant state**: a={idx_min} | **Min |E|**: {min_energy:.6f}"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(mags, lw=2)
    ax.scatter([idx_min], [min_energy], color="#e45756", zorder=5, label="Minimum")
    ax.set_xlabel("Boundary state index a")
    ax.set_ylabel("|E_a|")
    ax.set_title("Cardy Boundary Energy Spectrum (toy)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return msg, fig


def demo_s_matrix(seed: int = 0) -> Tuple[str, plt.Figure]:
    model = TopologicalConsciousness(n_anyons=27, central_charge=627, seed=seed)
    S = np.abs(model.S)
    msg = f"**S-matrix**: shape {S.shape}, ‖S‖_F = {np.linalg.norm(S, 'fro'):.4f}"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(S, cmap="viridis", aspect="auto")
    ax.set_title("|S| heatmap (toy)")
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    fig.colorbar(im, ax=ax, shrink=0.85)
    return msg, fig


def demo_dimensional_sweep() -> Tuple[str, plt.Figure]:
    dims = np.arange(100, 1300, 50)
    bits = dims * np.log2(3.0)
    optimal_dim = 627
    target_bits = 994

    msg = (
        "## Why 627 Dimensions?\n\n"
        "For trinary encoding, capacity is I = d · log₂(3).\n\n"
        f"Setting I ≈ {target_bits} bits gives d ≈ {optimal_dim}."
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(dims, bits, lw=2, label="Information capacity (bits)")
    ax.axvline(optimal_dim, ls="--", color="#e45756", label="d = 627")
    ax.axhline(target_bits, ls="--", color="#4c78a8", label="994 bits")
    ax.set_xlabel("Dimensions d")
    ax.set_ylabel("Capacity I (bits)")
    ax.set_title("Information Capacity vs Dimension (trinary)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return msg, fig


# ------------------------ Gradio UI ------------------------

with gr.Blocks(title="3^627 Framework: Consciousness as Topological Holography") as demo:
    gr.Markdown(
        """
        ### 🌌 Consciousness as Topological Holography — Interactive Demos
        Explore toy visualizations inspired by a (2+1)D TQFT framework.
        """
    )

    with gr.Tabs():
        with gr.Tab("1️⃣ Phase Coherence"):
            gr.Markdown("High coherence suggests both observers access similar states.")
            with gr.Row():
                with gr.Column():
                    t1 = gr.Textbox(label="Observer 1", value="The capital of France is Paris.")
                    t2 = gr.Textbox(label="Observer 2", value="Paris is the capital city of France.")
                    btn1 = gr.Button("Calculate Phase Coherence", variant="primary")
                with gr.Column():
                    out_md1 = gr.Markdown()
                    out_plot1 = gr.Plot()
            btn1.click(demo_phase_coherence, inputs=[t1, t2], outputs=[out_md1, out_plot1])

        with gr.Tab("2️⃣ Trinary Quantization"):
            gr.Markdown("Compare binary {-1,+1} vs trinary {-1,0,+1} (toy).")
            with gr.Row():
                n_samples = gr.Slider(50, 5000, value=500, step=50, label="Number of weights")
                noise = gr.Slider(0.0, 0.8, value=0.1, step=0.05, label="Noise level")
            btn2 = gr.Button("Compare Quantization", variant="primary")
            with gr.Row():
                out_md2 = gr.Markdown()
                out_plot2 = gr.Plot()
            btn2.click(demo_trinary_quantization, inputs=[n_samples, noise], outputs=[out_md2, out_plot2])

        with gr.Tab("3️⃣ Cardy States"):
            gr.Markdown("Minimum boundary energy → dominant conscious state (toy).")
            btn3 = gr.Button("Generate Energy Spectrum", variant="primary")
            with gr.Row():
                out_md3 = gr.Markdown()
                out_plot3 = gr.Plot()
            btn3.click(lambda: demo_cardy_states(), inputs=[], outputs=[out_md3, out_plot3])

        with gr.Tab("4️⃣ Modular S-Matrix"):
            gr.Markdown("Heatmap of |S| (toy surrogate).")
            btn4 = gr.Button("Generate S-Matrix", variant="primary")
            with gr.Row():
                out_md4 = gr.Markdown()
                out_plot4 = gr.Plot()
            btn4.click(lambda: demo_s_matrix(), inputs=[], outputs=[out_md4, out_plot4])

        with gr.Tab("5️⃣ Why 627 Dimensions?"):
            gr.Markdown("Information-theoretic view of 627D.")
            btn5 = gr.Button("Show Analysis", variant="primary")
            with gr.Row():
                out_md5 = gr.Markdown()
                out_plot5 = gr.Plot()
            btn5.click(demo_dimensional_sweep, inputs=[], outputs=[out_md5, out_plot5])

    gr.Markdown("""
    ---
    Built with Gradio + NumPy + Matplotlib. Toy visuals; not physical predictions.
    """)


if __name__ == "__main__":
    # Launch the app if run directly
    demo.launch()