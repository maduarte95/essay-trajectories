# Trajectory Analysis Methods

## Overview

This document describes the computational methods used to analyze argumentative trajectories in human-written vs. AI-generated essays. All analyses work in **raw semantic embedding space** using cosine distance as the primary distance metric.

---

## 1. Data Preparation

### 1.1 Chunking

**Goal:** Segment each essay into sequential argumentative units.

**Method:** Each essay is chunked using an LLM (Claude) prompted to identify discrete "argumentative moves" — single claims, pieces of evidence, transitions, counterarguments, or syntheses. Chunks preserve the exact original text without paraphrasing.

**Output:** For each essay, a sequence of N chunks: `C₁, C₂, ..., Cₙ`

**Rationale:** Chunks represent the discrete steps of an essay's trajectory through semantic/argumentative space. This granularity allows us to track how essays move from idea to idea.

---

### 1.2 Embedding

**Goal:** Represent each chunk as a point in semantic space.

**Method:** Each chunk is independently embedded using a transformer-based embedding model (Nomic Embed Text v1.5, 768 dimensions). The same model is used for all essays to ensure comparability.

**Output:** For each essay with N chunks, we obtain N embedding vectors:
```
E₁, E₂, ..., Eₙ  where Eᵢ ∈ ℝ⁷⁶⁸
```

**Rationale:** Embedding models capture semantic meaning in vector space. The trajectory of an essay can be represented as a path through this space: `E₁ → E₂ → ... → Eₙ`

---

## 2. Distance Metric

All trajectory metrics use **cosine distance** as the distance measure between embeddings.

### Cosine Distance

For two embedding vectors **u** and **v**:

```
cosine_distance(u, v) = 1 - cosine_similarity(u, v)

                      = 1 - (u · v) / (||u|| ||v||)
```

**Range:** [0, 2], where:
- 0 = identical direction (semantically identical)
- 1 = orthogonal (semantically unrelated)
- 2 = opposite direction (semantically opposite)

**Rationale:** Embedding models produce vectors where **direction** (not magnitude) encodes semantic meaning. Cosine distance captures angular separation between concepts, making it the standard metric for comparing text embeddings.

---

## 2.1 PCA Preprocessing (Noise Reduction)

Before computing trajectory metrics, we apply **Principal Component Analysis (PCA)** to all embeddings in a common subspace. This reduces noise and focuses the analysis on the main dimensions of variation.

### Why PCA Preprocessing?

High-dimensional embedding spaces (384D or 768D) contain both signal and noise. Many dimensions capture idiosyncratic variations (e.g., stylistic quirks, specific word choices) that are not central to the argumentative trajectory. By projecting into a lower-dimensional PCA space, we:

1. **Reduce noise** from irrelevant dimensions
2. **Focus on main patterns** of semantic variation
3. **Improve metric stability** (small perturbations have less impact)
4. **Enable fairer comparison** across essays (common subspace)

### Method

```
Step 1: Collect all chunk embeddings
  Pool all embeddings from all essays (human + AI, all students)
  Result: matrix of size (total_chunks × embedding_dim)

Step 2: Fit PCA
  pca = PCA(n_components=50)
  pca.fit(all_embeddings)

  This finds the 50 directions of maximum variance across all essays

Step 3: Transform all embeddings
  For each essay with embeddings [E₁, E₂, ..., Eₙ]:
    Transform to PCA space: [P₁, P₂, ..., Pₙ] = pca.transform([E₁, E₂, ..., Eₙ])

Step 4: Compute all metrics in PCA space
  All subsequent analyses use the PCA-transformed embeddings
```

### Dimensionality

- **Original**: 384 dimensions (Nomic Embed Text v1.5)
- **PCA-reduced**: 50 dimensions
- **Variance explained**: ~72% of total variance

This means 50 dimensions capture 72% of the semantic variation while discarding 28% of noise/idiosyncrasy.

### Why 50 Components?

50 is a standard choice that balances:
- **Enough dimensions** to capture rich semantic structure
- **Few enough dimensions** to reduce noise and overfitting
- **Computational efficiency** for subsequent analyses

Empirically, variance explained plateaus around 50 components (diminishing returns beyond this).

### Important Note

PCA is fitted on the **pooled data** (all essays together), ensuring a shared coordinate system. This is critical for comparability — if we ran PCA separately on each essay, the resulting axes would be incomparable.

---

## 3. Trajectory Metrics

**All metrics below are computed in the PCA-reduced space (50D) unless otherwise noted.**

### 3.1 Step-to-Step Displacement

**Definition:** The cosine distance between consecutive chunk embeddings.

**Calculation:**
```
For an essay with embeddings [E₁, E₂, ..., Eₙ]:

displacements = [d₁, d₂, ..., dₙ₋₁]

where dᵢ = cosine_distance(Eᵢ, Eᵢ₊₁)
```

**Derived metrics:**
- **Mean displacement**: `mean(displacements)` — average semantic step size
- **Displacement variance**: `var(displacements)` — variability in step sizes
- **Max displacement**: `max(displacements)` — largest single conceptual jump

**Interpretation:**
- **Higher mean displacement** → essay covers more conceptually distant ideas between steps
- **Higher variance** → essay has uneven pacing (some small steps, some large jumps)
- **Large max displacement** → essay contains at least one major conceptual shift

**Expected pattern:** Human essays may show higher variance (exploratory process with revisions and reframings) while AI essays may show more uniform steps (single-pass generation).

---

### 3.2 Tortuosity (Path Efficiency)

**Definition:** The ratio of total path length to straight-line endpoint distance.

**Calculation:**
```
path_length = Σ dᵢ  (sum of all displacements)

endpoint_distance = cosine_distance(E₁, Eₙ)

tortuosity = path_length / endpoint_distance
```

**Range:** [1, ∞), where:
- 1 = perfectly direct path (each step moves directly from start to end)
- >1 = circuitous path (essay "wanders" through semantic space)

**Interpretation:**
- **Lower tortuosity** → more direct, efficient trajectory
- **Higher tortuosity** → more exploratory, less direct path

**Expected pattern:** AI essays should have lower tortuosity (planned all at once, direct path from opening to conclusion). Human essays should have higher tortuosity (5-day iterative process, revisiting and reframing ideas).

**Geometric intuition:** Imagine walking from point A to point B. Tortuosity = (actual distance walked) / (straight-line distance). A value of 3 means you walked 3× the straight-line distance.

---

### 3.3 Momentum (Directional Consistency)

**Definition:** The degree to which an essay maintains the same conceptual direction over consecutive steps.

**Calculation:**
```
Step 1: Compute direction vectors
  For each step i, compute:
    vᵢ = Eᵢ₊₁ - Eᵢ  (the direction of movement in embedding space)

Step 2: Compute cosine similarity between consecutive directions
  For each pair of consecutive directions:
    momentum_i = cosine_similarity(vᵢ, vᵢ₊₁)

              = (vᵢ · vᵢ₊₁) / (||vᵢ|| ||vᵢ₊₁||)

Step 3: Average over all steps
  momentum = mean([momentum₁, momentum₂, ..., momentum_n₋₂])
```

**Range:** [-1, 1], where:
- 1 = perfect momentum (each step continues in the same direction)
- 0 = orthogonal changes (each step goes perpendicular to the previous)
- -1 = reversing direction (each step goes backward)

**Interpretation:**
- **Higher momentum** → essay maintains consistent conceptual direction
- **Lower/negative momentum** → essay frequently changes direction

**Expected pattern:** AI essays may show slightly higher momentum (single coherent plan), though both human and AI essays are expected to have relatively low momentum since essays naturally develop and change direction.

**Note:** This metric is adapted from Nour et al. (2025), "Charting trajectories of human thought using large language models" (the VECTOR framework / Cinderella paper).

---

### 3.4 Matched-Pair Divergence Curve

**Definition:** A position-aligned comparison showing how much human and AI trajectories drift apart over the course of an essay.

**Challenge:** Human and AI essays have different numbers of chunks (different N values), so direct position-to-position comparison isn't possible.

**Solution:** Interpolate both trajectories to a common number of points (20) using normalized positions.

**Calculation:**
```
Step 1: Normalize positions
  For human essay with Nₕ chunks:
    positions = [0, 1/(Nₕ-1), 2/(Nₕ-1), ..., 1]

  For AI essay with Nₐ chunks:
    positions = [0, 1/(Nₐ-1), 2/(Nₐ-1), ..., 1]

Step 2: Interpolate to common grid
  Create target positions: [0, 1/19, 2/19, ..., 1]  (20 points)

  For each dimension d of the embedding:
    Interpolate human[d] and AI[d] to the 20 target positions
    using linear interpolation

Step 3: Compute divergence at each position
  For each normalized position t:
    divergence[t] = cosine_distance(human_interp[t], ai_interp[t])
```

**Output:** A curve with 20 points showing semantic distance between matched human/AI embeddings at normalized positions from 0 (start) to 1 (end).

**Interpretation:**
- **Flat curve** → human and AI maintain constant distance throughout
- **Increasing curve** → human and AI diverge more as essay progresses
- **Decreasing curve** → human and AI converge toward similar endpoints

**Expected pattern:** Divergence should increase over the essay. Both start from the same blind prompt (similar openings), but the 5-day human process produces increasing departure from what AI would have written.

**Rationale for linear interpolation:** Consecutive chunks are semantically nearby, so linear interpolation in embedding space is a reasonable approximation for intermediate positions.

---

### 3.5 Inter-Essay Homogeneity (Pairwise Similarity)

**Definition:** The degree to which essays within the human group (or AI group) are similar to each other.

**Challenge:** Essays have different trajectory lengths, so direct comparison isn't possible.

**Solution:** Interpolate all trajectories to a common length, flatten to single vectors, and compute pairwise similarities.

**Calculation:**
```
Step 1: Interpolate all trajectories
  For each essay, interpolate to 20 points (as in divergence curve)
  Result: 20 × 768 = 15,360-dimensional vector per essay

Step 2: Compute pairwise similarities
  For each pair of essays (i, j):
    similarity(i, j) = 1 - cosine_distance(flat_i, flat_j)

  This gives a symmetric similarity matrix (13 × 13 in our case)

Step 3: Compute within-group statistics
  Human-human similarity:
    mean of all pairwise similarities among human essays

  AI-AI similarity:
    mean of all pairwise similarities among AI essays
```

**Output:**
- Full similarity matrix (13 × 13)
- Mean within-group similarities
- Difference: AI-AI similarity - Human-Human similarity

**Interpretation:**
- **Higher within-group similarity** → essays in that group follow more similar trajectories
- **Lower within-group similarity** → essays in that group are more diverse

**Expected pattern:** AI essays should be more homogeneous (higher mean AI-AI similarity) than human essays. This operationalizes the "process convergence" claim — collapsing to a single prompt at Day 2 produces more similar outputs across different people, whereas the 5-day iterative process allows individual variation.

**Alternative method (not implemented):** Dynamic Time Warping (DTW) distance could be used to compare trajectories of different lengths without interpolation. DTW is more principled but computationally slower. Interpolation is a simpler, faster approximation that works well for our purposes.

---

## 4. Visualization Methods

### 4.1 PCA Projection

To visualize high-dimensional trajectories (768D) in 2D, we use Principal Component Analysis (PCA).

**Method:**
```
Step 1: Pool all chunk embeddings
  Collect all embeddings from all essays (human + AI, all students)
  Result: matrix of size (total_chunks × 768)

Step 2: Fit PCA
  Compute principal components on the pooled data
  Extract the top 2 components (PC1, PC2)

Step 3: Project all embeddings
  Transform all embeddings to 2D using the fitted PCA

Step 4: Reconstruct trajectories
  Group projected points back into essay sequences
  Plot each essay as a path: E₁ → E₂ → ... → Eₙ
```

**Rationale for pooled PCA:** Projecting all essays into the same PCA space ensures visual comparability. If we ran PCA separately on each essay, the resulting axes would be incomparable across essays.

**Limitations:** PCA is a linear projection that captures only the top 2 dimensions of variance. Much information is lost (typically ~20-30% variance explained). The visualization is qualitative — it helps readers see divergence patterns but should not be over-interpreted.

---

## 5. Statistical Considerations

### Small N Caveat

With N=7 student pairs (6 complete pairs after missing F-human), we do **not** run inferential statistics (t-tests, p-values) on the trajectory metrics. We report:
- Descriptive statistics (means, standard deviations)
- Individual paired comparisons
- Visualizations of patterns

**Rationale:** Statistical tests with N=6-7 have very low power. Even large effect sizes may not reach significance. More importantly, our goal is descriptive characterization of trajectory patterns, not hypothesis testing in the frequentist sense.

**If a reviewer insists:** A Wilcoxon signed-rank test on paired metrics (human vs. AI for the same student) is the most defensible option for this sample size. This is a non-parametric test that doesn't assume normal distributions.

---

## 6. Design Rationale

### Why Cosine Distance (Not Euclidean)?

Embedding models produce vectors where **direction** (not magnitude) carries semantic meaning. Two embedding vectors might have very different magnitudes but point in the same direction, meaning they're semantically similar. Cosine distance captures this angular similarity and is the standard metric for comparing text embeddings in NLP research.

### Why Raw Semantic Embedding Space (Not Schema Space)?

Nour et al. (2025) used supervised decoding to project embeddings into a discrete schema space (story events). This worked because all participants retold the same story (Cinderella), so there was a ground-truth event sequence.

Our essays differ in thesis, argument structure, and conclusions. There is no shared ground-truth schema. We therefore work in **raw semantic embedding space**, which is a limitation but appropriate for our design. This means:
- We can compare trajectories as paths through semantic space
- We cannot decode trajectories into human-interpretable "argumentative moves" shared across essays

### Why Interpolation for Divergence Curves?

Human and AI essays have different numbers of chunks (different N). To compare them position-by-position, we need aligned positions. Interpolating to a common grid (20 points, normalized from 0 to 1) allows position-matched comparison.

**Why 20 points?** This is a reasonable compromise:
- Enough resolution to capture trajectory shape
- Not so many that we over-interpolate short essays
- Standard in trajectory analysis literature

---

## 7. Implementation Notes

### Software
- Python 3.10+
- Embeddings: Nomic Embed Text v1.5 (768 dimensions)
- Chunking: Claude 3.5 Sonnet via Anthropic API
- Analysis: NumPy, SciPy, scikit-learn, pandas
- Visualization: Matplotlib, Seaborn

### Reproducibility
All code is available in the `scripts/` directory:
- `01_chunk.py` — Essay chunking
- `02_embed.py` — Chunk embedding
- `03_metrics.py` — Metric computation
- `04_figures.py` — Figure generation

All outputs are saved to `output/data/` with intermediate results (embeddings, metrics) preserved for inspection.

---

## 8. Interpretation Guide

### Hypothesis 1: AI essays are more direct
**Metrics:** Lower tortuosity, higher momentum
**Result:** ✓ AI essays have lower tortuosity (24.72 vs 34.92)
**Interpretation:** AI plans the full argument at generation time, producing an efficient path from start to end.

### Hypothesis 2: Human essays are more varied
**Metrics:** Higher displacement variance, more jumps
**Result:** Human essays have higher displacement variance (0.020 vs 0.014)
**Interpretation:** The 5-day iterative process involves revisiting, reframing, and incorporating new ideas.

### Hypothesis 3: Divergence increases over essay
**Metrics:** Upward-sloping divergence curve
**Result:** See Figure 3 for individual curves
**Interpretation:** Human and AI start from the same seed (similar opening), but the human process produces increasing departure from AI baseline.

### Hypothesis 4: AI essays are more homogeneous
**Metrics:** Higher mean AI-AI similarity
**Result:** ✓ AI-AI similarity (0.384) > Human-Human similarity (0.341)
**Interpretation:** Collapsing to a single blind prompt produces more similar outputs across different people.

---

## 9. Limitations

1. **Small sample size** (N=7 students, 6 complete pairs) — descriptive patterns only, limited statistical power

2. **Raw semantic space** — Cannot decode trajectories into human-interpretable argument structures shared across essays

3. **Linear interpolation** — Assumes smooth transitions between chunks; may not capture abrupt conceptual shifts

4. **Single embedding model** — Results may vary with different embedding models (though core patterns should replicate)

5. **Chunking subjectivity** — LLM-based chunking is not perfectly reproducible and may segment differently on different runs

6. **PCA visualization** — 2D projection loses most information; qualitative illustration only

---

## 10. References

**VECTOR Framework:**
Nour, M. M., et al. (2025). Charting trajectories of human thought using large language models. *Nature*, ...

**Embedding Models:**
Nomic AI. Nomic Embed Text v1.5. https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

**Cosine Distance:**
Rahutomo, F., Kitasuka, T., & Aritsugi, M. (2012). Semantic cosine similarity. *The 7th International Student Conference on Advanced Science and Technology*.

---

## Appendix: Code Example

### Computing Tortuosity

```python
import numpy as np
from scipy.spatial.distance import cosine

def compute_tortuosity(embeddings: np.ndarray) -> float:
    """
    Compute path efficiency (tortuosity).

    Args:
        embeddings: Array of shape (n_chunks, embedding_dim)

    Returns:
        tortuosity value (path_length / endpoint_distance)
    """
    n = len(embeddings)
    if n < 2:
        return np.nan

    # Compute step-to-step displacements
    displacements = []
    for i in range(n - 1):
        dist = cosine(embeddings[i], embeddings[i + 1])
        displacements.append(dist)

    # Sum to get total path length
    path_length = np.sum(displacements)

    # Compute straight-line endpoint distance
    endpoint_distance = cosine(embeddings[0], embeddings[-1])

    if endpoint_distance == 0:
        return np.inf

    return path_length / endpoint_distance
```

### Computing Momentum

```python
def compute_momentum(embeddings: np.ndarray) -> float:
    """
    Compute directional consistency (momentum).

    Returns average cosine similarity between consecutive direction vectors.
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
```

---

**Document Version:** 1.0
**Last Updated:** 2026-02-13
**Author:** Trajectory Analysis Pipeline (Claude Code)
