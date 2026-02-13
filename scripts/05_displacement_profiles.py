#!/usr/bin/env python3
"""
Step-to-Step Displacement Profile Analysis

Analyzes the "jumpiness" of essay trajectories by examining the distribution
of cosine distances between consecutive chunks.

Metrics:
- Mean step size: Average conceptual distance between consecutive chunks
- Step size variance: Uniformity of progression
- Max jump: Largest single transition

Visualizations:
- Violin plots comparing human vs AI distributions
- Individual essay displacement series
- Summary statistics
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy import stats
from sklearn.decomposition import PCA

# Set publication-ready style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8


def load_embeddings(data_dir: Path):
    """Load all embedding files."""
    embeddings = {}
    embedding_files = list(data_dir.glob("*.json"))

    for file in embedding_files:
        if file.name == "embedding_metadata.json":
            continue

        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        student = data['student']
        source = data['source']
        embeddings[(student, source)] = {
            'chunks': data['chunks'],
            'embeddings': np.array(data['embeddings']),
            'student': student,
            'source': source
        }

    return embeddings


def apply_pca_to_embeddings(embeddings_dict, n_components=50):
    """
    Apply PCA to all embeddings in a common subspace to reduce noise.

    Args:
        embeddings_dict: Dictionary of embeddings
        n_components: Number of PCA components to retain (default: 50)

    Returns:
        - Updated embeddings_dict with PCA-transformed embeddings
        - Fitted PCA object
        - Explained variance ratio
    """
    # Collect all embeddings for fitting PCA
    all_embeddings = []
    chunk_indices = []
    idx = 0

    for (student, source), data in sorted(embeddings_dict.items()):
        n_chunks = len(data['embeddings'])
        all_embeddings.append(data['embeddings'])
        chunk_indices.append((student, source, idx, idx + n_chunks))
        idx += n_chunks

    all_embeddings = np.vstack(all_embeddings)

    # Fit PCA on all data
    pca = PCA(n_components=n_components)
    all_pca = pca.fit_transform(all_embeddings)
    explained_var = pca.explained_variance_ratio_.sum()

    # Reconstruct individual trajectories in PCA space
    pca_embeddings_dict = {}
    for student, source, start, end in chunk_indices:
        pca_embeddings_dict[(student, source)] = {
            'chunks': embeddings_dict[(student, source)]['chunks'],
            'embeddings': all_pca[start:end],
            'student': student,
            'source': source
        }

    return pca_embeddings_dict, pca, explained_var


def compute_displacement_series(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine distance between consecutive embeddings."""
    n = len(embeddings)
    if n < 2:
        return np.array([])

    displacements = []
    for i in range(n - 1):
        dist = cosine(embeddings[i], embeddings[i + 1])
        displacements.append(dist)

    return np.array(displacements)


def create_violin_plot(displacement_data: pd.DataFrame, output_dir: Path):
    """
    Create violin plot comparing human vs AI displacement distributions.
    """
    print("\nGenerating Figure: Violin Plot...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colors = {'human': '#2E86AB', 'ai': '#E63946'}

    # Prepare data for each metric
    metrics = [
        ('mean_step', 'Mean Step Size\n(Avg. Cosine Distance)'),
        ('variance_step', 'Step Size Variance\n(Uniformity of Progression)'),
        ('max_jump', 'Maximum Jump\n(Largest Transition)')
    ]

    for ax, (metric, label) in zip(axes, metrics):
        # Get data
        human_vals = displacement_data[displacement_data['source'] == 'human'][metric].values
        ai_vals = displacement_data[displacement_data['source'] == 'ai'][metric].values

        # Create violin plot
        parts = ax.violinplot(
            [human_vals, ai_vals],
            positions=[1, 2],
            showmeans=True,
            showmedians=True,
            widths=0.7
        )

        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            source = 'human' if i == 0 else 'ai'
            pc.set_facecolor(colors[source])
            pc.set_alpha(0.6)

        # Style the other elements
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
            if partname in parts:
                parts[partname].set_edgecolor('black')
                parts[partname].set_linewidth(1.5)

        # Overlay individual points with jitter
        np.random.seed(42)
        jitter = 0.08

        human_x = np.random.normal(1, jitter, size=len(human_vals))
        ai_x = np.random.normal(2, jitter, size=len(ai_vals))

        ax.scatter(human_x, human_vals, color='white', s=50,
                  edgecolors=colors['human'], linewidths=2, zorder=3, alpha=0.9)
        ax.scatter(ai_x, ai_vals, color='white', s=50,
                  edgecolors=colors['ai'], linewidths=2, zorder=3, alpha=0.9)

        # Styling
        ax.set_ylabel(label, fontsize=10, fontweight='bold')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Human', 'AI'])
        ax.tick_params(labelsize=9)
        ax.set_xlim(0.5, 2.5)
        y_max = max(human_vals.max(), ai_vals.max())
        ax.set_ylim(0, y_max * 1.1)

        # Add n values
        ax.text(1, -0.05, f'n={len(human_vals)}',
               ha='center', va='top', fontsize=8, color='gray',
               transform=ax.get_xaxis_transform())
        ax.text(2, -0.05, f'n={len(ai_vals)}',
               ha='center', va='top', fontsize=8, color='gray',
               transform=ax.get_xaxis_transform())

        # Clean style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Step-to-Step Displacement Profiles: Human vs AI Essays',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    fig.savefig(output_dir / 'figure_displacement_violin.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure_displacement_violin.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved to {output_dir}/figure_displacement_violin.[png,pdf]")


def create_displacement_series_plot(embeddings_dict: dict, output_dir: Path):
    """
    Plot individual displacement series for each essay.
    Shows the actual step-by-step progression.
    """
    print("\nGenerating Figure: Displacement Series...")

    # Get list of students
    students = sorted(set(k[0] for k in embeddings_dict.keys()))

    n_cols = 3
    n_rows = (len(students) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if len(students) == 1 else axes

    colors = {'human': '#2E86AB', 'ai': '#E63946'}

    for i, student in enumerate(students):
        ax = axes[i]

        # Plot human series if available
        human_key = (student, 'human')
        if human_key in embeddings_dict:
            displacements = compute_displacement_series(embeddings_dict[human_key]['embeddings'])
            if len(displacements) > 0:
                steps = np.arange(1, len(displacements) + 1)
                ax.plot(steps, displacements, '-o', color=colors['human'],
                       alpha=0.7, linewidth=2, markersize=4, label='Human')

        # Plot AI series if available
        ai_key = (student, 'ai')
        if ai_key in embeddings_dict:
            displacements = compute_displacement_series(embeddings_dict[ai_key]['embeddings'])
            if len(displacements) > 0:
                steps = np.arange(1, len(displacements) + 1)
                ax.plot(steps, displacements, '-s', color=colors['ai'],
                       alpha=0.7, linewidth=2, markersize=4, label='AI')

        ax.set_title(f'Student {student}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Step Number', fontsize=9)
        ax.set_ylabel('Cosine Distance', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_ylim(0, 1.2)

        # Clean style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Add legend only to first subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    # Hide extra subplots
    for i in range(len(students), len(axes)):
        axes[i].axis('off')

    plt.suptitle('Step-to-Step Displacement Series by Student',
                fontsize=13, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save
    fig.savefig(output_dir / 'figure_displacement_series.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure_displacement_series.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved to {output_dir}/figure_displacement_series.[png,pdf]")


def create_pooled_distribution_plot(embeddings_dict: dict, output_dir: Path):
    """
    Plot pooled distribution of all step sizes across all essays.
    """
    print("\nGenerating Figure: Pooled Distribution...")

    # Collect all displacements
    human_displacements = []
    ai_displacements = []

    for (student, source), data in embeddings_dict.items():
        displacements = compute_displacement_series(data['embeddings'])
        if source == 'human':
            human_displacements.extend(displacements)
        else:
            ai_displacements.extend(displacements)

    human_displacements = np.array(human_displacements)
    ai_displacements = np.array(ai_displacements)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = {'human': '#2E86AB', 'ai': '#E63946'}

    # Histogram comparison
    ax = axes[0]
    bins = np.linspace(0, 1.0, 30)

    ax.hist(human_displacements, bins=bins, alpha=0.6, color=colors['human'],
           label=f'Human (n={len(human_displacements)} steps)', density=True)
    ax.hist(ai_displacements, bins=bins, alpha=0.6, color=colors['ai'],
           label=f'AI (n={len(ai_displacements)} steps)', density=True)

    ax.set_xlabel('Cosine Distance (Step Size)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Distribution of All Step Sizes', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.5, axis='y')

    # Cumulative distribution
    ax = axes[1]

    human_sorted = np.sort(human_displacements)
    ai_sorted = np.sort(ai_displacements)

    human_cdf = np.arange(1, len(human_sorted) + 1) / len(human_sorted)
    ai_cdf = np.arange(1, len(ai_sorted) + 1) / len(ai_sorted)

    ax.plot(human_sorted, human_cdf, '-', color=colors['human'],
           linewidth=2, label='Human', alpha=0.8)
    ax.plot(ai_sorted, ai_cdf, '-', color=colors['ai'],
           linewidth=2, label='AI', alpha=0.8)

    ax.set_xlabel('Cosine Distance (Step Size)', fontsize=10)
    ax.set_ylabel('Cumulative Probability', fontsize=10)
    ax.set_title('Cumulative Distribution Function', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # KS test
    plt.tight_layout()

    # Save
    fig.savefig(output_dir / 'figure_displacement_distribution.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure_displacement_distribution.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved to {output_dir}/figure_displacement_distribution.[png,pdf]")


def create_summary_table(displacement_data: pd.DataFrame, output_dir: Path):
    """Create a summary statistics table."""
    print("\nGenerating summary statistics table...")

    # Compute statistics
    summary_rows = []

    for source in ['human', 'ai']:
        data = displacement_data[displacement_data['source'] == source]

        summary_rows.append({
            'Source': source.capitalize(),
            'N Essays': len(data),
            'Mean Step Size (μ)': f"{data['mean_step'].mean():.4f}",
            'Mean Step Size (σ)': f"{data['mean_step'].std():.4f}",
            'Step Variance (μ)': f"{data['variance_step'].mean():.4f}",
            'Step Variance (σ)': f"{data['variance_step'].std():.4f}",
            'Max Jump (μ)': f"{data['max_jump'].mean():.4f}",
            'Max Jump (σ)': f"{data['max_jump'].std():.4f}",
        })

    summary_df = pd.DataFrame(summary_rows)

    # Save to CSV
    csv_path = output_dir / 'displacement_summary_stats.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved to {csv_path}")

    # Print to console
    print("\n" + "="*80)
    print("DISPLACEMENT PROFILE SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)


def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "output" / "data" / "embeddings"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("="*80)
    print("STEP-TO-STEP DISPLACEMENT PROFILE ANALYSIS")
    print("="*80)

    # Load embeddings
    print("\nLoading embeddings...")
    raw_embeddings_dict = load_embeddings(data_dir)
    print(f"  Loaded {len(raw_embeddings_dict)} essays")

    # Get original embedding dimension
    first_key = next(iter(raw_embeddings_dict.keys()))
    original_dim = raw_embeddings_dict[first_key]['embeddings'].shape[1]
    print(f"Original embedding dimension: {original_dim}")

    # Apply PCA to reduce noise
    print("\nApplying PCA to all embeddings in common subspace...")
    n_components = min(50, original_dim)
    embeddings_dict, pca, explained_var = apply_pca_to_embeddings(raw_embeddings_dict, n_components)
    print(f"PCA reduced to {n_components} components")
    print(f"Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")

    # Compute displacement statistics for each essay
    print("\nComputing displacement statistics in PCA subspace...")
    displacement_rows = []

    for (student, source), data in sorted(embeddings_dict.items()):
        displacements = compute_displacement_series(data['embeddings'])

        if len(displacements) > 0:
            displacement_rows.append({
                'student': student,
                'source': source,  # Keep lowercase
                'n_steps': len(displacements),
                'mean_step': np.mean(displacements),
                'variance_step': np.var(displacements),
                'max_jump': np.max(displacements),
            })

    displacement_data = pd.DataFrame(displacement_rows)

    # Generate visualizations
    create_violin_plot(displacement_data, plots_dir)
    create_displacement_series_plot(embeddings_dict, plots_dir)
    create_pooled_distribution_plot(embeddings_dict, plots_dir)
    create_summary_table(displacement_data, plots_dir)

    print("\n" + "="*80)
    print("DISPLACEMENT PROFILE ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {plots_dir}/")
    print("  - figure_displacement_violin.[png,pdf]")
    print("  - figure_displacement_series.[png,pdf]")
    print("  - figure_displacement_distribution.[png,pdf]")
    print("  - displacement_summary_stats.csv")
    print("="*80)


if __name__ == "__main__":
    main()
