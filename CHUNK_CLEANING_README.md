# Chunk Cleaning

## Purpose

The raw chunks from Step 1 (chunking) contain full paragraphs with:
- Specific citations and references (e.g., "Craik, 1943")
- Detailed examples and quotes
- Semantic elaboration and supporting evidence
- Author names, dates, and bibliographic details

This creates noise when comparing essay trajectories. Two essays might discuss similar logical structures but diverge semantically due to different references or examples.

**Solution:** Summarize each chunk into a single concise bullet point that captures ONLY the logical/argumentative move.

## What This Does

The cleaning script:
1. Reads each chunk from `output/data/chunks/`
2. Uses Claude to summarize each chunk into a short logical bullet point
3. Removes references, citations, examples, and detailed content
4. Focuses on the argumentative structure: what claim is being made, what logical move this represents
5. Saves cleaned chunks to `output/data/chunks_cleaned/`

## Example Transformation

**Before (raw chunk):**
> Kenneth Craik, writing in 1943, proposed that what makes organisms adaptive is their capacity to build internal models that function as "distance receptors in time" — neural processes that let an agent respond to situations before they arrive (Craik, 1943). Tolman formalized a version of this as "cognitive maps": internal spatial models that rats used to navigate mazes, rather than just chaining stimulus-response pairs (Tolman, 1948).

**After (cleaned):**
> Thinking involves building internal models that allow prediction and planning

## Usage

```bash
uv run python scripts/clean_chunks.py
```

This will process all chunk files and create cleaned versions in `output/data/chunks_cleaned/`.

## Output Format

Each cleaned file contains:
```json
{
  "student": "A",
  "source": "human",
  "original_chunks": ["full original text...", ...],
  "cleaned_chunks": [
    "Short logical bullet point 1",
    "Short logical bullet point 2",
    ...
  ]
}
```

The original chunks are preserved for reference, but downstream analysis (embedding, metrics, visualization) should use the `cleaned_chunks`.

## Next Steps

After cleaning chunks:

1. **Update Step 2 (embedding)** to use `cleaned_chunks` instead of raw `chunks`
2. **Run embeddings** on the cleaned bullet points
3. **Continue with metrics and visualization** as planned

The cleaned chunks should produce trajectories that better reflect the *logical structure* of arguments rather than surface-level semantic similarity from shared references.

## Why This Matters

Two essays about "thinking" might:
- Cite the same papers (Craik, Tolman, Dennett) → high semantic similarity
- But make **different logical moves** with those citations

Cleaning isolates the logic:
- Essay A: "Internal models enable prediction"
- Essay B: "Internal models are insufficient without embodiment"

Even if both cite Craik, their logical trajectories differ. The cleaned chunks make this visible.
