# Chunking Output Format

This document describes the structure of the JSON files produced by `01_chunk.py` to help with downstream processing (embedding, metrics, visualization).

## Output Directory Structure

```
output/data/chunks/
├── student-A-human.json
├── student-A-ai.json
├── student-B-human.json
├── student-B-ai.json
├── student-C-human.json
├── student-C-ai.json
├── student-D-human.json
├── student-D-ai.json
├── student-E-human.json
├── student-E-ai.json
├── student-F-ai.json          # Only AI essay for F
├── student-G-human.json
├── student-G-ai.json
└── errors.json                 # Only if errors occurred
```

## Individual Essay JSON Format

Each essay file follows this structure:

```json
{
  "student": "A",
  "source": "human",
  "chunks": [
    "First argumentative chunk text preserved exactly as it appeared in the original essay...",
    "Second argumentative chunk text...",
    "Third argumentative chunk text...",
    ...
  ]
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `student` | `string` | Student identifier: "A", "B", "C", "D", "E", "F", or "G" |
| `source` | `string` | Essay source: "human" (from FINAL/) or "ai" (from META/) |
| `chunks` | `array[string]` | Sequential array of text chunks, each representing one argumentative move |

### Key Properties

1. **Order Preserved**: Chunks appear in the exact order they appeared in the original essay
2. **No Overlap**: Each word appears in exactly one chunk
3. **Exact Text**: Text is preserved character-for-character (no paraphrasing or summarization)
4. **Complete Coverage**: All text from the original essay is included
5. **Sentence Integrity**: No chunk splits mid-sentence

## Chunk Characteristics

Based on the chunking prompt, each chunk represents **one argumentative move**:

- A single claim
- A piece of supporting evidence
- A counterargument
- A transition to a new topic
- A concluding synthesis
- An introduction or framing statement

Typical chunk characteristics:
- **Length**: Roughly paragraph-sized (but can be shorter)
- **Count**: Varies by essay (typically 8-20 chunks for 1000-1500 word essays)
- **Boundaries**: Always at sentence boundaries, never mid-sentence

## Error Format (if errors.json exists)

```json
[
  {
    "custom_id": "student-B-ai",
    "error": "JSON parsing error: Expecting value: line 1 column 1 (char 0)",
    "content": "First 500 characters of the raw response that failed to parse..."
  },
  {
    "custom_id": "student-F-human",
    "error": "Request expired"
  }
]
```

## Using the Output in Downstream Scripts

### Step 2: Embedding (`02_embed.py`)

For each chunk file, you'll:
1. Load the JSON
2. Extract the `chunks` array
3. Embed each chunk independently
4. Save embeddings alongside the chunk text

Example loading code:
```python
import json
from pathlib import Path

chunks_dir = Path("output/data/chunks")

for chunk_file in chunks_dir.glob("*.json"):
    if chunk_file.name == "errors.json":
        continue

    with open(chunk_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    student = data["student"]
    source = data["source"]
    chunks = data["chunks"]

    # Embed each chunk
    embeddings = [embed_text(chunk) for chunk in chunks]

    # Save with embeddings
    output = {
        "student": student,
        "source": source,
        "chunks": chunks,
        "embeddings": embeddings
    }
```

### Step 3: Metrics (`03_metrics.py`)

For trajectory analysis, you'll need to:
1. Load embedding files
2. Match human/AI pairs by student ID
3. Compute metrics on the embedding sequences

Example pairing code:
```python
import json
from pathlib import Path

embeddings_dir = Path("output/data/embeddings")

# Group by student
essays_by_student = {}

for emb_file in embeddings_dir.glob("*.json"):
    with open(emb_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    student = data["student"]
    source = data["source"]

    if student not in essays_by_student:
        essays_by_student[student] = {}

    essays_by_student[student][source] = data

# Compute paired metrics
for student, essays in essays_by_student.items():
    if "human" in essays and "ai" in essays:
        human_embeddings = essays["human"]["embeddings"]
        ai_embeddings = essays["ai"]["embeddings"]

        # Compute divergence curve, etc.
```

### Step 4: Visualization (`04_figures.py`)

The structure makes it easy to:
- Plot individual trajectories (one line per essay)
- Color by source (human vs AI)
- Group by student for paired comparisons
- Pool all essays for PCA/dimensionality reduction

## Naming Convention

The filename format is: `{student}-{source}.json`

Examples:
- `student-A-human.json` → Student A's human-written essay
- `student-A-ai.json` → Student A's AI-generated essay

This makes it easy to:
1. Parse student and source from filename
2. Match pairs (same student, different source)
3. Sort/filter by either dimension

## Expected Output Summary

Given current directory contents:

**FINAL (human):** 6 essays (A, B, C, D, E, G)
**META (ai):** 7 essays (A, B, C, D, E, F, G)

**Paired essays:** 6 pairs (A, B, C, D, E, G)
**Unpaired:** 1 AI-only essay (F)

### For Trajectory Analysis

- **Paired trajectories**: 6 pairs where you can compute divergence curves
- **Homogeneity analysis**:
  - 6 human essays → 15 pairwise comparisons
  - 7 AI essays → 21 pairwise comparisons

## Data Validation

Before proceeding to embedding, verify:

```python
import json
from pathlib import Path

chunks_dir = Path("output/data/chunks")

for chunk_file in chunks_dir.glob("*.json"):
    if chunk_file.name == "errors.json":
        continue

    with open(chunk_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate structure
    assert "student" in data
    assert "source" in data
    assert "chunks" in data
    assert isinstance(data["chunks"], list)
    assert len(data["chunks"]) > 0
    assert all(isinstance(chunk, str) for chunk in data["chunks"])

    print(f"✓ {chunk_file.name}: {len(data['chunks'])} chunks")
```

## Next Steps After Chunking

1. **Validate output**: Check chunk counts and spot-check a few essays
2. **Embedding**: Generate embeddings for all chunks
3. **Metrics**: Compute trajectory metrics (tortuosity, momentum, divergence)
4. **Visualization**: Create figures for the paper

Each downstream script can rely on this consistent JSON structure!
