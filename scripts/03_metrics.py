#!/usr/bin/env python3
"""
Step 3: Compute Trajectory Metrics

Computes trajectory metrics for each essay:
- Step-to-step displacement profile (mean, variance, max)
- Tortuosity (path efficiency)
- Momentum (directional consistency)
- Matched-pair divergence curves
- Inter-essay homogeneity (pairwise similarity)

All analyses are performed in PCA-reduced space to reduce noise.
All distances use cosine distance unless otherwise noted.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine
from scipy.interpolate import interp1d
from itertools import combinations
from sklearn.decomposition import PCA


def load_embeddings(data_dir: Path) -> Dict[Tuple[str, str], Dict]:
    """
    Load all embedding files.

    Returns:
        Dict mapping (student, source) -> {chunks, embeddings, student, source}
    """
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


def apply_pca_to_embeddings(embeddings_dict: Dict[Tuple[str, str], Dict],
                            n_components: int = 50) -> Tuple[Dict, PCA, float]:
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


def compute_displacements(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine distances between consecutive embeddings.

    Args:
        embeddings: Array of shape (n_chunks, embedding_dim)

    Returns:
        Array of shape (n_chunks - 1,) with cosine distances
    """
    n = len(embeddings)
    if n < 2:
        return np.array([])

    displacements = []
    for i in range(n - 1):
        dist = cosine(embeddings[i], embeddings[i + 1])
        displacements.append(dist)

    return np.array(displacements)


def compute_displacement_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute displacement profile metrics.

    Returns:
        Dict with mean_disp, var_disp, max_disp
    """
    displacements = compute_displacements(embeddings)

    if len(displacements) == 0:
        return {
            'mean_disp': np.nan,
            'var_disp': np.nan,
            'max_disp': np.nan
        }

    return {
        'mean_disp': float(np.mean(displacements)),
        'var_disp': float(np.var(displacements)),
        'max_disp': float(np.max(displacements))
    }


def compute_tortuosity(embeddings: np.ndarray) -> float:
    """
    Compute path efficiency (tortuosity).

    tortuosity = path_length / endpoint_distance

    A value of 1.0 means perfectly direct. Higher values mean more wandering.
    """
    if len(embeddings) < 2:
        return np.nan

    displacements = compute_displacements(embeddings)
    path_length = np.sum(displacements)

    endpoint_distance = cosine(embeddings[0], embeddings[-1])

    if endpoint_distance == 0:
        return np.inf

    return float(path_length / endpoint_distance)


def compute_momentum(embeddings: np.ndarray) -> float:
    """
    Compute directional consistency (momentum).

    For each step, compute direction vector and then cosine similarity
    between consecutive direction vectors.

    Returns average momentum across all steps.
    """
    n = len(embeddings)
    if n < 3:
        return np.nan

    # Compute direction vectors
    directions = []
    for i in range(n - 1):
        direction = embeddings[i + 1] - embeddings[i]
        directions.append(direction)

    # Compute cosine similarity between consecutive directions
    momentums = []
    for i in range(len(directions) - 1):
        dir1 = directions[i]
        dir2 = directions[i + 1]

        # Normalize
        norm1 = np.linalg.norm(dir1)
        norm2 = np.linalg.norm(dir2)

        if norm1 == 0 or norm2 == 0:
            continue

        # Cosine similarity = dot product of normalized vectors
        similarity = np.dot(dir1, dir2) / (norm1 * norm2)
        momentums.append(similarity)

    if len(momentums) == 0:
        return np.nan

    return float(np.mean(momentums))


def interpolate_trajectory(embeddings: np.ndarray, n_points: int = 20) -> np.ndarray:
    """
    Interpolate trajectory to a common number of points.

    Args:
        embeddings: Array of shape (n_chunks, embedding_dim)
        n_points: Number of points to interpolate to

    Returns:
        Array of shape (n_points, embedding_dim)
    """
    n_chunks, embedding_dim = embeddings.shape

    if n_chunks == 1:
        # Edge case: repeat the single point
        return np.tile(embeddings[0], (n_points, 1))

    # Create normalized positions for original chunks
    original_positions = np.linspace(0, 1, n_chunks)

    # Create target positions
    target_positions = np.linspace(0, 1, n_points)

    # Interpolate each dimension independently
    interpolated = np.zeros((n_points, embedding_dim))
    for dim in range(embedding_dim):
        interp_func = interp1d(original_positions, embeddings[:, dim],
                               kind='linear', fill_value='extrapolate')
        interpolated[:, dim] = interp_func(target_positions)

    return interpolated


def compute_divergence_curve(human_embeddings: np.ndarray,
                             ai_embeddings: np.ndarray,
                             n_points: int = 20) -> np.ndarray:
    """
    Compute divergence curve between matched human and AI trajectories.

    Returns array of shape (n_points,) with cosine distances at each position.
    """
    human_interp = interpolate_trajectory(human_embeddings, n_points)
    ai_interp = interpolate_trajectory(ai_embeddings, n_points)

    divergence = []
    for i in range(n_points):
        dist = cosine(human_interp[i], ai_interp[i])
        divergence.append(dist)

    return np.array(divergence)


def compute_trajectory_similarity(traj1: np.ndarray, traj2: np.ndarray,
                                  n_points: int = 20) -> float:
    """
    Compute similarity between two trajectories.

    Interpolates both to n_points, flattens, and computes cosine similarity.
    """
    interp1 = interpolate_trajectory(traj1, n_points)
    interp2 = interpolate_trajectory(traj2, n_points)

    # Flatten to single vectors
    flat1 = interp1.flatten()
    flat2 = interp2.flatten()

    # Cosine similarity = 1 - cosine distance
    similarity = 1 - cosine(flat1, flat2)

    return float(similarity)


def compute_pairwise_similarities(embeddings_dict: Dict[Tuple[str, str], Dict],
                                  n_points: int = 20) -> Dict[str, any]:
    """
    Compute pairwise trajectory similarities within human and AI groups.

    Returns:
        Dict with:
        - pairwise_matrix: Full similarity matrix
        - mean_human_similarity: Mean similarity among human essays
        - mean_ai_similarity: Mean similarity among AI essays
        - similarity_data: List of dicts for saving to CSV
    """
    # Separate by source
    human_essays = {k: v for k, v in embeddings_dict.items() if k[1] == 'human'}
    ai_essays = {k: v for k, v in embeddings_dict.items() if k[1] == 'ai'}

    # All essays in order
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
                    embeddings_dict[key2]['embeddings'],
                    n_points
                )
                similarity_matrix[i, j] = sim

    # Compute within-group similarities
    human_pairs = list(combinations(human_essays.keys(), 2))
    human_sims = []
    for key1, key2 in human_pairs:
        sim = compute_trajectory_similarity(
            human_essays[key1]['embeddings'],
            human_essays[key2]['embeddings'],
            n_points
        )
        human_sims.append(sim)

    ai_pairs = list(combinations(ai_essays.keys(), 2))
    ai_sims = []
    for key1, key2 in ai_pairs:
        sim = compute_trajectory_similarity(
            ai_essays[key1]['embeddings'],
            ai_essays[key2]['embeddings'],
            n_points
        )
        ai_sims.append(sim)

    return {
        'similarity_matrix': similarity_matrix,
        'essay_keys': all_keys,
        'mean_human_similarity': float(np.mean(human_sims)) if human_sims else np.nan,
        'mean_ai_similarity': float(np.mean(ai_sims)) if ai_sims else np.nan,
        'human_similarities': human_sims,
        'ai_similarities': ai_sims
    }


def main():
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "output" / "data" / "embeddings"
    metrics_dir = base_dir / "output" / "data" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embeddings...")
    raw_embeddings_dict = load_embeddings(data_dir)
    print(f"Loaded {len(raw_embeddings_dict)} essays")

    # Get original embedding dimension
    first_key = next(iter(raw_embeddings_dict.keys()))
    original_dim = raw_embeddings_dict[first_key]['embeddings'].shape[1]
    print(f"Original embedding dimension: {original_dim}")

    # Apply PCA to reduce noise
    print("\nApplying PCA to all embeddings in common subspace...")
    n_components = min(50, original_dim)  # Use 50 components or less if original dim is smaller
    embeddings_dict, pca, explained_var = apply_pca_to_embeddings(raw_embeddings_dict, n_components)
    print(f"PCA reduced to {n_components} components")
    print(f"Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")

    # Compute per-essay metrics
    print("\nComputing per-essay metrics in PCA subspace...")
    metrics_rows = []

    for (student, source), data in sorted(embeddings_dict.items()):
        print(f"  Processing {student}-{source}...")

        embeddings = data['embeddings']
        n_chunks = len(embeddings)

        # Displacement metrics
        disp_metrics = compute_displacement_metrics(embeddings)

        # Tortuosity
        tortuosity = compute_tortuosity(embeddings)

        # Momentum
        momentum = compute_momentum(embeddings)

        metrics_rows.append({
            'student': student,
            'source': source,
            'n_chunks': n_chunks,
            'mean_disp': disp_metrics['mean_disp'],
            'var_disp': disp_metrics['var_disp'],
            'max_disp': disp_metrics['max_disp'],
            'tortuosity': tortuosity,
            'momentum': momentum
        })

    # Save per-essay metrics
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(['student', 'source'])
    metrics_csv_path = metrics_dir / "essay_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False, float_format='%.4f')
    print(f"\nSaved essay metrics to {metrics_csv_path}")

    # Compute divergence curves
    print("\nComputing divergence curves...")
    students = sorted(set(k[0] for k in embeddings_dict.keys()))
    divergence_curves = {}

    for student in students:
        human_key = (student, 'human')
        ai_key = (student, 'ai')

        if human_key not in embeddings_dict or ai_key not in embeddings_dict:
            print(f"  WARNING: Missing pair for student {student}")
            continue

        print(f"  Processing {student}...")
        curve = compute_divergence_curve(
            embeddings_dict[human_key]['embeddings'],
            embeddings_dict[ai_key]['embeddings']
        )
        divergence_curves[student] = curve.tolist()

    # Save divergence curves
    divergence_path = metrics_dir / "divergence_curves.json"
    with open(divergence_path, 'w') as f:
        json.dump(divergence_curves, f, indent=2)
    print(f"Saved divergence curves to {divergence_path}")

    # Compute pairwise similarities
    print("\nComputing pairwise similarities...")
    similarity_results = compute_pairwise_similarities(embeddings_dict)

    # Save similarity matrix
    similarity_matrix = similarity_results['similarity_matrix']
    essay_keys = similarity_results['essay_keys']

    # Create labeled dataframe
    labels = [f"{student}-{source}" for student, source in essay_keys]
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=labels,
        columns=labels
    )
    similarity_matrix_path = metrics_dir / "similarity_matrix.csv"
    similarity_df.to_csv(similarity_matrix_path, float_format='%.4f')
    print(f"Saved similarity matrix to {similarity_matrix_path}")

    # Save summary statistics
    summary = {
        'mean_human_similarity': similarity_results['mean_human_similarity'],
        'mean_ai_similarity': similarity_results['mean_ai_similarity'],
        'n_human_pairs': len(similarity_results['human_similarities']),
        'n_ai_pairs': len(similarity_results['ai_similarities'])
    }
    summary_path = metrics_dir / "similarity_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved similarity summary to {summary_path}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nPer-essay metrics computed for {len(metrics_rows)} essays")
    print(f"\nDivergence curves computed for {len(divergence_curves)} student pairs")
    print(f"\nPairwise similarity statistics:")
    print(f"  Mean human-human similarity: {summary['mean_human_similarity']:.4f}")
    print(f"  Mean AI-AI similarity: {summary['mean_ai_similarity']:.4f}")
    print(f"  Difference (AI - Human): {summary['mean_ai_similarity'] - summary['mean_human_similarity']:.4f}")

    print("\n" + "="*60)
    print("Metric computation complete!")
    print("="*60)
    print(f"\nOutputs saved to: {metrics_dir}/")
    print("  - essay_metrics.csv")
    print("  - divergence_curves.json")
    print("  - similarity_matrix.csv")
    print("  - similarity_summary.json")


if __name__ == "__main__":
    main()
