# Essay Chunking Setup - Quick Start Guide

This guide will help you chunk the essays using Anthropic's Batch API.

## Prerequisites

✅ All dependencies installed via `uv sync`
✅ Directory structure created
✅ Script ready at `scripts/01_chunk.py`

## Step 1: Set Your API Key

You need an Anthropic API key. Add it to the `.env` file:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

Get your API key from: https://console.anthropic.com/settings/keys

## Step 2: Test API Connection (Optional)

```bash
uv run python scripts/test_api.py
```

You should see:
```
✓ Anthropic client initialized successfully
✓ API key found: sk-ant-a...
```

## Step 3: Run the Chunking Script

From the project root:

```bash
uv run python scripts/01_chunk.py
```

## What Happens

1. **Reads all essays** from FINAL/ and META/ directories
   - Currently: 3 human essays + 7 AI essays = 10 total

2. **Creates a batch request** with all essays
   - Each essay gets chunked into argumentative units
   - Uses `claude-sonnet-4-5` model (cost-effective)

3. **Polls for completion** every 60 seconds
   - Typical completion time: < 1 hour
   - Max wait time: 24 hours

4. **Saves results** to `output/data/chunks/`
   - Format: `{student}-{source}.json`
   - Example: `A-human.json`, `A-ai.json`

## Output Structure

Each JSON file contains:

```json
{
  "student": "A",
  "source": "human",
  "chunks": [
    "First chunk of text preserving exact original...",
    "Second chunk of text...",
    ...
  ]
}
```

## Cost Estimate

**Batch API = 50% discount on standard pricing**

For ~10 essays (1000-1500 words each):
- Input tokens: ~20K total → ~$0.03
- Output tokens: ~40K total → ~$0.30
- **Total: ~$0.33 for all essays**

## Troubleshooting

### "No API key found"
- Check that `.env` exists and contains `ANTHROPIC_API_KEY=...`
- Make sure there are no quotes around the key

### "No essays found"
- Verify files are in `FINAL/` and `META/` directories
- Script looks for `.md` and `.txt` files

### Processing takes too long
- Batches can take up to 24 hours
- Script will keep polling automatically
- You can safely Ctrl+C and restart later (batch continues processing)

### JSON parsing errors
- Check `output/data/chunks/errors.json` for details
- Model might not have returned valid JSON
- Can re-run script for failed essays

## Next Steps

After chunking completes:

1. **Verify output**: Check `output/data/chunks/*.json`
2. **Count chunks**: See how many chunks per essay
3. **Proceed to embedding**: Run `02_embed.py` (to be created)

## Files Created

```
essay-trajectories/
├── scripts/
│   ├── 01_chunk.py              ← Main chunking script
│   ├── test_api.py              ← API key test
│   └── README_CHUNKING.md       ← Detailed documentation
├── output/
│   └── data/
│       └── chunks/              ← Output directory (will contain JSON files)
├── .env                         ← Your API key (YOU need to create this)
└── CHUNKING_SETUP.md           ← This file
```

## Full Documentation

See [scripts/README_CHUNKING.md](scripts/README_CHUNKING.md) for complete details on:
- Prompt used for chunking
- Error handling
- Pricing breakdown
- Next pipeline steps
