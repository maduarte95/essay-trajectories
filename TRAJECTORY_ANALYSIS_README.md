# Essay Trajectory Analysis: Human vs. AI Blind-Prompt Comparison

## Context

This analysis accompanies a paper on post-AGI thinking. Seven doctoral students (A–G) in cognitive science wrote essays about thinking over a 5-day intensive. On Day 2, each student captured their essay intentions in a "blind prompt." On Day 5, that frozen prompt was submitted to an LLM with a standardizing metaprompt (see below), producing an AI-generated essay from the same seed. Students also produced their own essays through 5 days of structured thinking.

We want to compare the **internal argumentative trajectories** of human essays vs. AI essays. Simple whole-essay semantic similarity is confounded by shared topic. Instead, we chunk each essay into sequential argumentative steps, embed those chunks, and analyze the resulting trajectories through semantic space.

### Reference methodology
This approach is inspired by Nour et al. (2025), "Charting trajectories of human thought using large language models" (the VECTOR framework / Cinderella paper). Key differences from that work:
- We do NOT have a shared ground-truth schema (essays differ in thesis and argument)
- We therefore work in **raw semantic embedding space**, not a decoded schema space
- Our N is small (7 paired essays), so we focus on descriptive trajectory metrics and visualizations rather than inferential statistics

---

## Directory Structure

```
project/
├── FINAL/          # Human-written essays (the process output)
│   ├── A.md        # or .txt, .docx — one file per student
│   ├── B.md
│   ├── C.md
│   ├── D.md
│   ├── E.md
│   ├── F.md
│   └── G.md
├── META/           # AI-generated essays from blind prompts
│   ├── A.md        # same naming convention — matched pairs
│   ├── B.md
│   ├── C.md
│   ├── D.md
│   ├── E.md
│   ├── F.md
│   └── G.md
├── output/         # All outputs go here
│   ├── figures/
│   ├── data/       # Intermediate CSVs/JSONs (embeddings, chunks, metrics)
│   └── summary.md  # Human-readable summary of results
└── scripts/
    ├── 01_chunk.py
    ├── 02_embed.py
    ├── 03_metrics.py
    └── 04_figures.py
```

---

## Pipeline

### Step 1: Chunking (`01_chunk.py`)

**Goal:** Segment each essay into sequential argumentative units (chunks).

**Method:** Use an LLM (Claude via Anthropic API, or OpenAI) to segment each essay. Each chunk should be one "argumentative move" — a single claim, a piece of evidence, a transition, a counterargument, etc. These are the discrete steps of the essay's trajectory.

**Prompt for the chunking LLM:**
```
Below is an academic essay. Segment it into sequential argumentative chunks. 
Each chunk should represent ONE argumentative move — a single claim, a piece 
of supporting evidence, a counterargument, a transition to a new topic, or a 
concluding synthesis. 

Rules:
- Preserve the EXACT original text. Do not paraphrase or summarize.
- Every word in the essay must appear in exactly one chunk.
- Chunks should be roughly paragraph-sized but can be shorter if a paragraph 
  contains multiple distinct moves.
- Do NOT split mid-sentence.
- Return the result as a JSON array of strings, where each string is one chunk 
  in order.

Essay:
{essay_text}
```

**Output:** For each essay, save a JSON file:
```json
{
  "student": "A",
  "source": "human",  // or "ai"
  "chunks": [
    "First argumentative chunk text...",
    "Second argumentative chunk text...",
    ...
  ]
}
```

Save to `output/data/chunks/`.

**Important notes:**
- File formats in FINAL/ and META/ may vary (.md, .txt, .docx). Handle all gracefully. For .docx, use `python-docx` to extract text.
- If any file is missing or empty, log a warning and skip that pair.
- Log the number of chunks per essay for inspection.

---

### Step 2: Embedding (`02_embed.py`)

**Goal:** Embed each chunk as a vector in semantic space.

**Method:** Use OpenAI's `text-embedding-3-small` (1536 dimensions) or another embedding model. Embed each chunk independently.

**Output:** For each essay, save:
```json
{
  "student": "A",
  "source": "human",
  "chunks": ["chunk1 text", "chunk2 text", ...],
  "embeddings": [[0.012, -0.034, ...], [0.008, 0.051, ...], ...]
}
```

Save to `output/data/embeddings/`. Also save a single consolidated NumPy file or CSV for easy downstream use.

**Notes:**
- Use the same embedding model for ALL essays (human and AI). Do not mix models.
- If using the Anthropic API isn't possible for embeddings, `sentence-transformers` (e.g., `all-MiniLM-L6-v2`) from HuggingFace is a good local alternative — free, no API key needed. Prefer this if API access is uncertain.

---

### Step 3: Compute Trajectory Metrics (`03_metrics.py`)

Compute the following metrics for each essay. All distances are **cosine distances** unless otherwise noted.

#### 3a. Step-to-step displacement profile
For each essay with N chunks, compute the cosine distance between consecutive chunk embeddings:
```
displacements = [cosine_distance(embed[i], embed[i+1]) for i in range(N-1)]
```
From this derive:
- **Mean displacement**: average step size
- **Displacement variance**: how uniform the steps are
- **Max displacement**: largest single jump

#### 3b. Tortuosity (path efficiency)
```
path_length = sum(displacements)
endpoint_distance = cosine_distance(embed[0], embed[-1])
tortuosity = path_length / endpoint_distance
```
A value of 1.0 means perfectly direct (each step goes straight from start to end). Higher values mean the essay wanders. The hypothesis: AI essays are more direct (lower tortuosity) because they are planned all at once.

#### 3c. Momentum (directional consistency)
For each step, compute the direction vector:
```
direction[i] = embed[i+1] - embed[i]
```
Then compute cosine similarity between consecutive direction vectors:
```
momentum[i] = cosine_similarity(direction[i], direction[i+1])
```
Average over all steps to get a single momentum score per essay. High momentum = the essay keeps moving in the same conceptual direction. Low momentum = frequent changes of direction. (This metric is adapted from Nour et al., 2025.)

#### 3d. Matched-pair divergence curve
For each student (A–G), align the human and AI trajectories by normalized position. Interpolate both to a common number of points (e.g., 20) using linear interpolation over the embedding sequences. Then compute the cosine distance between the human and AI embedding at each normalized position:
```
divergence_curve[t] = cosine_distance(human_interp[t], ai_interp[t])
```
This gives a curve per student showing how much the two trajectories drift apart over the course of the essay.

#### 3e. Inter-essay homogeneity (pairwise similarity)
Compute pairwise trajectory similarity for all essays within the human group and within the AI group. To compare trajectories of different lengths, use one of:
- **Option A (preferred for simplicity):** Interpolate all trajectories to 20 points, flatten to a single vector (20 × embedding_dim), compute pairwise cosine similarity.
- **Option B:** Dynamic Time Warping (DTW) distance between chunk embedding sequences (use `dtw-python` or `tslearn`). More principled but slower.

Compute:
- Mean pairwise similarity among AI essays
- Mean pairwise similarity among human essays
- Test: are AI essays more similar to each other than human essays are to each other?

**Output:** Save a CSV with one row per essay:

| student | source | n_chunks | mean_disp | var_disp | max_disp | tortuosity | momentum |
|---------|--------|----------|-----------|----------|----------|------------|----------|
| A       | human  | 12       | 0.34      | 0.02     | 0.61     | 3.2        | 0.15     |
| A       | ai     | 10       | 0.28      | 0.01     | 0.42     | 2.1        | 0.22     |
| ...     | ...    | ...      | ...       | ...      | ...      | ...        | ...      |

Also save the divergence curves and pairwise similarity matrices as separate files.

---

### Step 4: Figures (`04_figures.py`)

Generate the following figures. Use a clean, publication-ready style (white background, no gridlines, legible fonts). Use `matplotlib` and/or `seaborn`. Save as both PDF and PNG (300 dpi).

#### Figure 1: Trajectory Visualization (2D PCA projection)
- Run PCA on ALL chunk embeddings (human + AI, all students pooled) to get a shared 2D space.
- Plot each essay as a trajectory: points connected by lines, colored by source (e.g., blue = human, red/orange = AI).
- One subplot per student (A–G), so a 2×4 or 3×3 grid (leave extra panels blank).
- Points are chunks in order; use arrows or alpha gradient (faint → dark) to show direction.
- Title each subplot with the student letter.
- This is the main qualitative figure — it lets readers SEE the divergence.

#### Figure 2: Metric Comparison (paired bar/dot plot)
- For each metric (mean displacement, tortuosity, momentum), plot human vs. AI as paired points connected by lines (one line per student).
- Three subplots side by side.
- This shows per-student paired differences.
- Alternatively: a simple grouped bar chart with error bars if paired dots look too cluttered.

#### Figure 3: Divergence Curve
- X-axis: normalized essay position (0 to 1).
- Y-axis: cosine distance between matched human/AI chunk embeddings.
- One thin line per student (A–G), plus a thick line for the mean.
- This is the key figure for the blind prompt method — it shows whether human process produces increasing divergence from the AI baseline.

#### Figure 4: Homogeneity Matrix
- A heatmap showing pairwise trajectory similarity for all 14 essays (7 human + 7 AI).
- Order: group by source (all human first, then all AI) so block structure is visible.
- Annotate with a summary: mean within-AI similarity vs. mean within-human similarity.
- Colormap: a diverging or sequential colormap (e.g., `viridis` or `coolwarm`).

---

## Technical Requirements

- **Python 3.10+**
- **Required packages:** `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`
- **For embeddings (choose one):**
  - `openai` (requires API key in env var `OPENAI_API_KEY`) — use `text-embedding-3-small`
  - `sentence-transformers` (local, no API key) — use `all-MiniLM-L6-v2` as fallback
- **For chunking:** `anthropic` or `openai` (requires API key)
- **Optional:** `python-docx` (if any essays are .docx), `dtw-python` (if using DTW for homogeneity)

If no API keys are available, the chunking step can fall back to a simple heuristic: split on paragraph breaks (double newlines), then split any paragraph longer than ~150 words at sentence boundaries. This is less precise but workable.

---

## Design Decisions & Rationale

**Why cosine distance, not Euclidean?** Embedding models produce vectors where direction (not magnitude) carries semantic meaning. Cosine distance is standard for comparing text embeddings.

**Why not a schema space (as in Nour et al.)?** The Cinderella paper could use supervised decoding because all participants retold the same known story. Our essays share a topic domain but differ in thesis, argument structure, and conclusions. There is no ground-truth event sequence to decode into. We therefore work in raw semantic embedding space, which is a limitation but appropriate for our design.

**Why interpolation for the divergence curve?** Human and AI essays will have different numbers of chunks. Interpolating to a common grid (e.g., 20 points) allows position-matched comparison. Linear interpolation in embedding space is a reasonable approximation since consecutive chunks are semantically nearby.

**Why PCA on the pooled set?** Projecting all essays into the same PCA space ensures visual comparability. PCA on individual essays would yield incomparable axes.

**Small N caveat:** With N=7 pairs, we do NOT run inferential statistics (t-tests, p-values) on the trajectory metrics. We report descriptive patterns and visualize individual pairs. If a reviewer insists on a test, a Wilcoxon signed-rank test on paired metrics is the most defensible option, but we should be upfront about the power limitation.

---

## Expected Hypotheses (for interpreting results)

1. **AI essays are more direct:** Lower tortuosity, higher momentum. AI plans the full argument at generation time, so the trajectory moves efficiently from start to end.
2. **Human essays are more varied:** Higher displacement variance, more "jumps." The 5-day process involves revisiting, reframing, and incorporating new ideas.
3. **Divergence increases over the essay:** The divergence curve should trend upward. Human and AI start from the same seed (similar opening), but the human process produces increasing departure from what AI would have written.
4. **AI essays are more homogeneous:** Higher pairwise similarity among AI essays than among human essays. This operationalizes the paper's "process convergence" claim — collapsing to a single prompt produces more similar outputs across different people.

---

## Appendix: The Standardizing Metaprompt

This was appended to each student's Day 2 blind prompt to generate the AI essay:

> The text above is a student's description of an essay they want to write for a course called "Thinking: From Humans to Animals to Machines." Your job is to write the best possible version of this essay. Treat the student's input as the seed. Do not limit yourself to what they explicitly state. Bring relevant evidence, frameworks, counterarguments, and examples. If vague, interpret generously and develop the strongest version of what they seem to reach toward. If detailed, honor their claims and build on them. Write 1000–1500 words. Make a clear, defensible argument. Cite real academic sources from training knowledge. Address the strongest objection. Clear, confident academic prose.

## Appendix: Student–Essay Pairings

| Student | Essay Topic |
|---------|-------------|
| A | Whose thoughts are an LLM's thoughts? |
| B | Multiscalar architecture of cognition |
| C | Social construction of thinking |
| D | Sense of self and thinking |
| E | Split-brain and parallel thought |
| F | Machine thinking / AI cognition |
| G | Machine thinking / AI cognition |
