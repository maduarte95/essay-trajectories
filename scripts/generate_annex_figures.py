#!/usr/bin/env python3
"""
Generate figures for paper annex.

Creates:
1. Similarity matrix without diagonal (vmax set to off-diagonal max)
2. Displacement violin plot (already exists, just copy)
3. Summary statistics table as LaTeX
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine
from itertools import combinations

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "output" / "data"
PLOTS_DIR = BASE_DIR / "plots"
ANNEX_DIR = BASE_DIR / "annex_figures"

ANNEX_DIR.mkdir(exist_ok=True)


def load_embeddings():
    """Load all embeddings."""
    embeddings_dir = DATA_DIR / "embeddings"
    embeddings_dict = {}

    for file in sorted(embeddings_dir.glob("*.json")):
        if file.name in ["all_embeddings.npy", "embedding_metadata.json"]:
            continue

        with open(file, 'r') as f:
            data = json.load(f)
            student = data['student']
            source = data['source']
            embeddings = np.array(data['embeddings'])
            embeddings_dict[(student, source)] = {
                'embeddings': embeddings,
                'chunks': data['chunks']
            }

    return embeddings_dict


def interpolate_trajectory(embeddings: np.ndarray, n_points: int = 20) -> np.ndarray:
    """Interpolate trajectory to common number of points."""
    from scipy.interpolate import interp1d

    n_chunks, embedding_dim = embeddings.shape

    if n_chunks == 1:
        return np.tile(embeddings[0], (n_points, 1))

    original_positions = np.linspace(0, 1, n_chunks)
    target_positions = np.linspace(0, 1, n_points)

    interpolated = np.zeros((n_points, embedding_dim))
    for dim in range(embedding_dim):
        interp_func = interp1d(original_positions, embeddings[:, dim],
                               kind='linear', fill_value='extrapolate')
        interpolated[:, dim] = interp_func(target_positions)

    return interpolated


def compute_trajectory_similarity(traj1: np.ndarray, traj2: np.ndarray, n_points: int = 20) -> float:
    """Compute similarity between two trajectories."""
    interp1 = interpolate_trajectory(traj1, n_points)
    interp2 = interpolate_trajectory(traj2, n_points)

    flat1 = interp1.flatten()
    flat2 = interp2.flatten()

    similarity = 1 - cosine(flat1, flat2)
    return float(similarity)


def create_similarity_matrix_no_diagonal():
    """Create similarity matrix with diagonal excluded from color scale."""
    print("\nGenerating similarity matrix (no diagonal)...")

    embeddings_dict = load_embeddings()

    # Separate by source
    human_essays = {k: v for k, v in embeddings_dict.items() if k[1] == 'human'}
    ai_essays = {k: v for k, v in embeddings_dict.items() if k[1] == 'ai'}

    # All essays in order (human first, then AI)
    all_keys = sorted(human_essays.keys()) + sorted(ai_essays.keys())
    n_essays = len(all_keys)

    # Compute full pairwise matrix
    similarity_matrix = np.zeros((n_essays, n_essays))

    for i, key1 in enumerate(all_keys):
        for j, key2 in enumerate(all_keys):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                sim = compute_trajectory_similarity(
                    embeddings_dict[key1]['embeddings'],
                    embeddings_dict[key2]['embeddings']
                )
                similarity_matrix[i, j] = sim

    # Create labels
    labels = [f"{k[0]}-{k[1]}" for k in all_keys]

    # Mask upper triangle and diagonal
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 9))

    # Get off-diagonal max for vmax (only lower triangle, no diagonal)
    lower_tri_mask = np.tril(np.ones_like(similarity_matrix, dtype=bool), k=-1)
    vmax = similarity_matrix[lower_tri_mask].max()
    vmin = similarity_matrix[lower_tri_mask].min()

    # Create heatmap with mask
    sns.heatmap(similarity_matrix,
                mask=mask,  # Hide upper triangle and diagonal
                xticklabels=labels,
                yticklabels=labels,
                cmap='RdYlBu_r',
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'label': 'Cosine Similarity'},
                ax=ax)

    # Add dividing lines between human and AI
    n_human = len(human_essays)
    ax.axhline(y=n_human, color='black', linewidth=2)
    ax.axvline(x=n_human, color='black', linewidth=2)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.title('Pairwise Trajectory Similarity\n(Diagonal excluded from color scale)',
              fontsize=12, pad=15)
    plt.tight_layout()

    # Save
    output_path = ANNEX_DIR / 'similarity_matrix_no_diagonal.pdf'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    fig.savefig(ANNEX_DIR / 'similarity_matrix_no_diagonal.png', bbox_inches='tight', dpi=300)
    plt.close()

    print(f"  Saved to {output_path}")

    # Compute within-group statistics
    human_pairs = list(combinations(range(n_human), 2))
    ai_pairs = list(combinations(range(n_human, n_essays), 2))

    human_sims = [similarity_matrix[i, j] for i, j in human_pairs]
    ai_sims = [similarity_matrix[i, j] for i, j in ai_pairs]

    print(f"\nWithin-group similarity statistics:")
    print(f"  Human essays: mean={np.mean(human_sims):.3f}, std={np.std(human_sims):.3f}")
    print(f"  AI essays:    mean={np.mean(ai_sims):.3f}, std={np.std(ai_sims):.3f}")


def generate_latex_table():
    """Generate LaTeX table for summary statistics."""
    print("\nGenerating LaTeX summary statistics table...")

    # Load the CSV
    csv_path = PLOTS_DIR / "displacement_summary_stats.csv"
    df = pd.read_csv(csv_path)

    # Create LaTeX
    latex = r"""\begin{table}[h]
\centering
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Source} & \textbf{N} & \textbf{Mean Step} & \textbf{Step Var} & \textbf{Max Jump} \\
 & \textbf{Essays} & \textbf{$\mu$ (SD)} & \textbf{$\mu$ (SD)} & \textbf{$\mu$ (SD)} \\
\midrule
"""

    for _, row in df.iterrows():
        source = row['Source']
        n = int(row['N Essays'])
        mean_step_mu = float(row['Mean Step Size (μ)'])
        mean_step_sd = float(row['Mean Step Size (σ)'])
        var_mu = float(row['Step Variance (μ)'])
        var_sd = float(row['Step Variance (σ)'])
        max_mu = float(row['Max Jump (μ)'])
        max_sd = float(row['Max Jump (σ)'])

        latex += f"{source} & {n} & {mean_step_mu:.3f} ({mean_step_sd:.3f}) & {var_mu:.3f} ({var_sd:.3f}) & {max_mu:.3f} ({max_sd:.3f}) \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\caption{Summary statistics for step-to-step displacement profiles. Mean step size, step variance, and maximum jump computed for each essay trajectory in PCA-reduced embedding space (50 components). Values show mean and standard deviation across essays within each group.}
\label{tab:displacement-summary}
\end{table}
"""

    output_path = ANNEX_DIR / "displacement_summary_table.tex"
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"  Saved to {output_path}")
    print("\nLaTeX table preview:")
    print(latex)


def main():
    print("="*60)
    print("GENERATING ANNEX FIGURES")
    print("="*60)

    # 1. Similarity matrix without diagonal
    create_similarity_matrix_no_diagonal()

    # 2. Generate LaTeX table
    generate_latex_table()

    # 3. Copy violin plot (already exists)
    import shutil
    violin_src = PLOTS_DIR / "figure_displacement_violin.pdf"
    violin_dst = ANNEX_DIR / "displacement_violin.pdf"
    shutil.copy(violin_src, violin_dst)
    print(f"\nCopied violin plot to {violin_dst}")

    print("\n" + "="*60)
    print("ANNEX FIGURES COMPLETE")
    print("="*60)
    print(f"\nAll outputs in: {ANNEX_DIR}/")
    print("  - similarity_matrix_no_diagonal.pdf")
    print("  - displacement_violin.pdf")
    print("  - displacement_summary_table.tex")


if __name__ == "__main__":
    main()
