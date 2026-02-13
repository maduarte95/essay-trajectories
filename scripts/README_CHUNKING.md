# Essay Chunking Script

This script uses Anthropic's Batch API to chunk essays into argumentative units for trajectory analysis.

## Setup

1. **Set your Anthropic API key** in `.env`:
   ```bash
   echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
   ```

2. **Dependencies are already installed** via `uv sync`

## Usage

Run the chunking script from the project root:

```bash
uv run python scripts/01_chunk.py
```

## What it does

1. **Reads all essays** from `FINAL/` (human) and `META/` (AI) directories
2. **Creates a batch request** to Anthropic's API with all essays
3. **Waits for completion** (polls every 60 seconds)
4. **Saves results** to `output/data/chunks/` as JSON files

## Output Format

Each essay produces a JSON file named `{student}-{source}.json`:

```json
{
  "student": "A",
  "source": "human",
  "chunks": [
    "First argumentative chunk text...",
    "Second argumentative chunk text...",
    ...
  ]
}
```

## Pricing

Using `claude-sonnet-4-5` with Batch API:
- **50% discount** on standard pricing
- Batch input: $1.50 / MTok
- Batch output: $7.50 / MTok

For a typical 1000-1500 word essay (~2K tokens), expect:
- Input: ~$0.003 per essay
- Output: ~$0.015-0.030 per essay (chunked response)
- **Total cost for ~10 essays: $0.20-0.40**

## Processing Time

- Batches typically complete within **1 hour**
- May take up to 24 hours depending on load
- Script polls every 60 seconds for status

## Error Handling

If any essays fail to process:
- Errors are saved to `output/data/chunks/errors.json`
- Check for:
  - JSON parsing errors (model didn't return valid JSON)
  - Invalid requests (malformed input)
  - Expired requests (took >24 hours)

## Next Steps

After chunking completes, proceed to:
- `02_embed.py` - Generate embeddings for chunks
- `03_metrics.py` - Calculate trajectory metrics
- `04_figures.py` - Visualize results
