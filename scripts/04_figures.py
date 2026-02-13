#!/usr/bin/env python3
"""
Step 4: Generate Figures

Generates publication-ready figures:
1. Trajectory visualization (2D PCA projection)
2. Metric comparison (paired bar/dot plots)
3. Divergence curve
4. Homogeneity matrix heatmap

All figures saved as both PDF and PNG (300 dpi).
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple

# Set publication-ready style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5


def load_embeddings(data_dir: Path) -> Dict[Tuple[str, str], Dict]:
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


def create_figure1_trajectories(embeddings_dict: Dict, output_dir: Path):
    """
    Figure 1: Trajectory Visualization (2D PCA projection)

    One subplot per student showing human vs AI trajectories in PCA space.
    """
    print("\nGenerating Figure 1: Trajectory Visualization...")

    # Collect all embeddings for PCA
    all_embeddings = []
    all_labels = []

    for (student, source), data in embeddings_dict.items():
        for i, emb in enumerate(data['embeddings']):
            all_embeddings.append(emb)
            all_labels.append((student, source, i))

    all_embeddings = np.array(all_embeddings)

    # Fit PCA on all data
    print("  Fitting PCA...")
    pca = PCA(n_components=2)
    all_pca = pca.fit_transform(all_embeddings)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Reconstruct trajectories in PCA space
    trajectories = {}
    idx = 0
    for (student, source), data in embeddings_dict.items():
        n_chunks = len(data['embeddings'])
        traj_pca = all_pca[idx:idx + n_chunks]
        trajectories[(student, source)] = traj_pca
        idx += n_chunks

    # Get list of students with at least one essay
    students = sorted(set(k[0] for k in embeddings_dict.keys()))

    # Create subplot grid
    n_students = len(students)
    n_cols = 3
    n_rows = (n_students + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_students == 1 else axes

    # Colors
    color_human = '#2E86AB'  # Blue
    color_ai = '#E63946'     # Red

    for i, student in enumerate(students):
        ax = axes[i]

        # Plot human trajectory if available
        human_key = (student, 'human')
        if human_key in trajectories:
            traj = trajectories[human_key]
            # Plot line
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color_human,
                   alpha=0.6, linewidth=1.5, label='Human')
            # Plot points with gradient
            colors_h = plt.cm.Blues(np.linspace(0.3, 0.9, len(traj)))
            ax.scatter(traj[:, 0], traj[:, 1], c=colors_h, s=30,
                      edgecolors='white', linewidths=0.5, zorder=3)
            # Mark start and end
            ax.scatter(traj[0, 0], traj[0, 1], marker='o', s=100,
                      color=color_human, edgecolors='white', linewidths=1.5,
                      zorder=4, label='_nolegend_')
            ax.scatter(traj[-1, 0], traj[-1, 1], marker='s', s=100,
                      color=color_human, edgecolors='white', linewidths=1.5,
                      zorder=4, label='_nolegend_')

        # Plot AI trajectory if available
        ai_key = (student, 'ai')
        if ai_key in trajectories:
            traj = trajectories[ai_key]
            # Plot line
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color_ai,
                   alpha=0.6, linewidth=1.5, label='AI')
            # Plot points with gradient
            colors_a = plt.cm.Reds(np.linspace(0.3, 0.9, len(traj)))
            ax.scatter(traj[:, 0], traj[:, 1], c=colors_a, s=30,
                      edgecolors='white', linewidths=0.5, zorder=3)
            # Mark start and end
            ax.scatter(traj[0, 0], traj[0, 1], marker='o', s=100,
                      color=color_ai, edgecolors='white', linewidths=1.5,
                      zorder=4, label='_nolegend_')
            ax.scatter(traj[-1, 0], traj[-1, 1], marker='s', s=100,
                      color=color_ai, edgecolors='white', linewidths=1.5,
                      zorder=4, label='_nolegend_')

        ax.set_title(f'Student {student}', fontsize=11, fontweight='bold')
        ax.set_xlabel('PC1', fontsize=9)
        ax.set_ylabel('PC2', fontsize=9)
        ax.tick_params(labelsize=8)

        # Add legend only to first subplot
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

        # Clean style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide extra subplots
    for i in range(n_students, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / 'figure1_trajectories.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure1_trajectories.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved to {output_dir}/figure1_trajectories.[png,pdf]")


def create_figure2_metric_comparison(metrics_df: pd.DataFrame, output_dir: Path):
    """
    Figure 2: Metric Comparison (aggregated human vs AI)

    Three subplots showing aggregate human vs AI for each metric with individual points.
    """
    print("\nGenerating Figure 2: Metric Comparison (Aggregate)...")

    metrics = ['mean_disp', 'tortuosity', 'momentum']
    metric_labels = ['Mean Displacement', 'Tortuosity', 'Momentum']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    colors = {'human': '#2E86AB', 'ai': '#E63946'}

    for ax, metric, label in zip(axes, metrics, metric_labels):
        # Separate human and AI values
        human_vals = metrics_df[metrics_df['source'] == 'human'][metric].values
        ai_vals = metrics_df[metrics_df['source'] == 'ai'][metric].values

        # Positions for bars
        positions = [1, 2]
        bar_width = 0.6

        # Calculate means and stds
        human_mean = np.mean(human_vals)
        ai_mean = np.mean(ai_vals)
        human_std = np.std(human_vals)
        ai_std = np.std(ai_vals)

        # Plot bars
        bars = ax.bar(positions, [human_mean, ai_mean], bar_width,
                     color=[colors['human'], colors['ai']],
                     alpha=0.7, edgecolor='white', linewidth=1)

        # Add error bars
        ax.errorbar(positions, [human_mean, ai_mean],
                   yerr=[human_std, ai_std],
                   fmt='none', color='black', capsize=5, linewidth=1.5)

        # Overlay individual points with jitter
        np.random.seed(42)
        jitter = 0.15
        human_x = np.random.normal(1, jitter, size=len(human_vals))
        ai_x = np.random.normal(2, jitter, size=len(ai_vals))

        ax.scatter(human_x, human_vals, color='white', s=40,
                  edgecolors=colors['human'], linewidths=1.5, zorder=3, alpha=0.8)
        ax.scatter(ai_x, ai_vals, color='white', s=40,
                  edgecolors=colors['ai'], linewidths=1.5, zorder=3, alpha=0.8)

        # Styling
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticks(positions)
        ax.set_xticklabels(['Human', 'AI'])
        ax.tick_params(labelsize=9)
        ax.set_xlim(0.5, 2.5)

        # Add n values
        ax.text(1, ax.get_ylim()[0], f'n={len(human_vals)}',
               ha='center', va='top', fontsize=8, color='gray')
        ax.text(2, ax.get_ylim()[0], f'n={len(ai_vals)}',
               ha='center', va='top', fontsize=8, color='gray')

        # Clean style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / 'figure2_metric_comparison.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure2_metric_comparison.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved to {output_dir}/figure2_metric_comparison.[png,pdf]")


def create_figure2b_metric_comparison_paired(metrics_df: pd.DataFrame, output_dir: Path):
    """
    Figure 2B: Metric Comparison (paired student view)

    Three subplots showing human vs AI for each metric with paired lines per student.
    """
    print("\nGenerating Figure 2B: Metric Comparison (Paired)...")

    # Get paired data
    students = sorted(set(metrics_df['student']))
    metrics = ['mean_disp', 'tortuosity', 'momentum']
    metric_labels = ['Mean Displacement', 'Tortuosity', 'Momentum']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    colors = {'human': '#2E86AB', 'ai': '#E63946'}

    for ax, metric, label in zip(axes, metrics, metric_labels):
        human_vals = []
        ai_vals = []
        valid_students = []

        for student in students:
            h = metrics_df[(metrics_df['student'] == student) &
                          (metrics_df['source'] == 'human')]
            a = metrics_df[(metrics_df['student'] == student) &
                          (metrics_df['source'] == 'ai')]

            if len(h) > 0 and len(a) > 0:
                human_vals.append(h[metric].values[0])
                ai_vals.append(a[metric].values[0])
                valid_students.append(student)

        x = np.arange(len(valid_students))
        width = 0.35

        # Plot paired lines
        for i, (h_val, a_val) in enumerate(zip(human_vals, ai_vals)):
            ax.plot([i - width/2, i + width/2], [h_val, a_val],
                   'k-', alpha=0.3, linewidth=1, zorder=1)

        # Plot bars
        ax.bar(x - width/2, human_vals, width, label='Human',
               color=colors['human'], alpha=0.7, edgecolor='white', linewidth=0.5)
        ax.bar(x + width/2, ai_vals, width, label='AI',
               color=colors['ai'], alpha=0.7, edgecolor='white', linewidth=0.5)

        ax.set_xlabel('Student', fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_students)
        ax.tick_params(labelsize=9)

        # Clean style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add legend only to first subplot
        if metric == metrics[0]:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / 'figure2b_metric_comparison_paired.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure2b_metric_comparison_paired.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved to {output_dir}/figure2b_metric_comparison_paired.[png,pdf]")


def create_figure3_divergence_curve(divergence_curves: Dict, output_dir: Path):
    """
    Figure 3: Divergence Curve

    Shows how human and AI trajectories diverge over normalized essay position.
    """
    print("\nGenerating Figure 3: Divergence Curve...")

    fig, ax = plt.subplots(figsize=(8, 5))

    # Normalized positions
    x = np.linspace(0, 1, 20)

    # Plot individual student curves
    all_curves = []
    for student, curve in sorted(divergence_curves.items()):
        curve_array = np.array(curve)
        all_curves.append(curve_array)
        ax.plot(x, curve_array, '-', alpha=0.3, linewidth=1.5,
               color='gray', label='_nolegend_')

    # Plot mean curve
    mean_curve = np.mean(all_curves, axis=0)
    ax.plot(x, mean_curve, '-', linewidth=3, color='#E63946',
           label=f'Mean (N={len(divergence_curves)})')

    # Add individual student labels in legend
    for student in sorted(divergence_curves.keys()):
        ax.plot([], [], '-', alpha=0.5, linewidth=1.5, color='gray',
               label=f'Student {student}')

    ax.set_xlabel('Normalized Essay Position', fontsize=11)
    ax.set_ylabel('Cosine Distance (Human vs AI)', fontsize=11)
    ax.set_title('Trajectory Divergence Over Essay Progression', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=10)

    # Clean style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    ax.legend(loc='upper left', fontsize=9, framealpha=0.9, ncol=2)

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / 'figure3_divergence_curve.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure3_divergence_curve.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved to {output_dir}/figure3_divergence_curve.[png,pdf]")


def create_figure4_homogeneity_matrix(similarity_df: pd.DataFrame,
                                     similarity_summary: Dict,
                                     output_dir: Path):
    """
    Figure 4: Homogeneity Matrix

    Heatmap showing pairwise trajectory similarity for all essays.
    """
    print("\nGenerating Figure 4: Homogeneity Matrix...")

    fig, ax = plt.subplots(figsize=(10, 9))

    # Create mask for upper triangle to show only lower triangle
    mask = np.triu(np.ones_like(similarity_df.values, dtype=bool), k=1)

    # Plot heatmap with mask
    sns.heatmap(similarity_df.values,
                mask=mask,
                xticklabels=similarity_df.columns,
                yticklabels=similarity_df.index,
                cmap='viridis',
                vmin=0, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Trajectory Similarity', 'shrink': 0.8},
                ax=ax)

    ax.set_title('Pairwise Trajectory Similarity', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(labelsize=9, rotation=45)

    # Add annotation with summary statistics
    mean_human = similarity_summary['mean_human_similarity']
    mean_ai = similarity_summary['mean_ai_similarity']
    diff = mean_ai - mean_human

    annotation = (f"Within-group similarity:\n"
                 f"Human-Human: {mean_human:.3f}\n"
                 f"AI-AI: {mean_ai:.3f}\n"
                 f"Difference: {diff:+.3f}")

    ax.text(0.02, 0.98, annotation,
           transform=ax.transAxes,
           fontsize=9,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))

    plt.tight_layout()

    # Save
    fig.savefig(output_dir / 'figure4_homogeneity_matrix.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'figure4_homogeneity_matrix.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved to {output_dir}/figure4_homogeneity_matrix.[png,pdf]")


def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "output" / "data"
    embeddings_dir = data_dir / "embeddings"
    metrics_dir = data_dir / "metrics"
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("="*70)
    print("GENERATING TRAJECTORY ANALYSIS FIGURES")
    print("="*70)

    # Load data
    print("\nLoading data...")
    embeddings_dict = load_embeddings(embeddings_dir)
    metrics_df = pd.read_csv(metrics_dir / "essay_metrics.csv")

    with open(metrics_dir / "divergence_curves.json", 'r') as f:
        divergence_curves = json.load(f)

    similarity_df = pd.read_csv(metrics_dir / "similarity_matrix.csv", index_col=0)

    with open(metrics_dir / "similarity_summary.json", 'r') as f:
        similarity_summary = json.load(f)

    print(f"  Loaded {len(embeddings_dict)} essays")
    print(f"  Loaded {len(divergence_curves)} divergence curves")

    # Generate figures
    create_figure1_trajectories(embeddings_dict, plots_dir)
    create_figure2_metric_comparison(metrics_df, plots_dir)
    create_figure2b_metric_comparison_paired(metrics_df, plots_dir)
    create_figure3_divergence_curve(divergence_curves, plots_dir)
    create_figure4_homogeneity_matrix(similarity_df, similarity_summary, plots_dir)

    print("\n" + "="*70)
    print("FIGURE GENERATION COMPLETE!")
    print("="*70)
    print(f"\nAll figures saved to: {plots_dir}/")
    print("  - figure1_trajectories.[png,pdf]")
    print("  - figure2_metric_comparison.[png,pdf] (aggregate view)")
    print("  - figure2b_metric_comparison_paired.[png,pdf] (paired view)")
    print("  - figure3_divergence_curve.[png,pdf]")
    print("  - figure4_homogeneity_matrix.[png,pdf]")
    print("\nAll figures saved as both PNG (300 dpi) and PDF for publication.")
    print("="*70)


if __name__ == "__main__":
    main()
