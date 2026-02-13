#!/usr/bin/env python3
"""
Retry failed essay chunks

This script re-processes any essays that failed in the original batch.
"""

import json
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

load_dotenv()

# Configuration
FINAL_DIR = Path("FINAL")
META_DIR = Path("META")
OUTPUT_DIR = Path("output/data/chunks")
ERRORS_FILE = OUTPUT_DIR / "errors.json"
MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 8192  # Double the tokens to ensure full response

# Updated chunking prompt with explicit JSON formatting instruction
CHUNKING_PROMPT = """Below is an academic essay. Segment it into sequential argumentative chunks.
Each chunk should represent ONE argumentative move — a single claim, a piece
of supporting evidence, a counterargument, a transition to a new topic, or a
concluding synthesis.

Rules:
- Preserve the EXACT original text. Do not paraphrase or summarize.
- Every word in the essay must appear in exactly one chunk.
- Chunks should be roughly paragraph-sized but can be shorter if a paragraph
  contains multiple distinct moves.
- Do NOT split mid-sentence.
- Return ONLY a valid JSON array of strings.
- Properly escape all quotes and special characters in the JSON.
- Do not include any markdown formatting or code blocks.
- Ensure the entire essay is included - do not truncate.

Essay:
{essay_text}

Return format: ["chunk 1 text", "chunk 2 text", ...]"""


def read_essay(file_path: Path) -> str:
    """Read essay text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def retry_failed_essays():
    """Retry essays that failed in the original batch."""

    # Load errors
    if not ERRORS_FILE.exists():
        print("No errors.json file found - all essays succeeded!")
        return

    with open(ERRORS_FILE, 'r') as f:
        errors = json.load(f)

    if not errors:
        print("No errors to retry!")
        return

    print(f"Found {len(errors)} failed essays")

    # Initialize client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗ No API key found")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Process each failed essay
    for error in errors:
        custom_id = error["custom_id"]
        parts = custom_id.split('-')
        student = parts[0]
        source = parts[1]

        print(f"\n{'='*60}")
        print(f"Retrying: {custom_id}")
        print(f"{'='*60}")

        # Find the essay file
        if source == "human":
            essay_dir = FINAL_DIR
        else:
            essay_dir = META_DIR

        # Find matching file
        essay_files = list(essay_dir.glob(f"student-{student}-*.md")) + \
                     list(essay_dir.glob(f"student-{student}-*.txt"))

        if not essay_files:
            print(f"✗ Could not find essay file for {custom_id}")
            continue

        essay_path = essay_files[0]
        essay_text = read_essay(essay_path)

        print(f"Found essay: {essay_path.name}")
        print(f"Essay length: {len(essay_text)} characters")

        # Create a direct (non-batch) request for debugging
        print("\nSending request to API...")
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": CHUNKING_PROMPT.format(essay_text=essay_text)
                    }
                ]
            )

            content = message.content[0].text

            # Try to parse
            print(f"\nReceived response ({len(content)} chars)")

            # Clean markdown if present
            content_cleaned = content.strip()
            if content_cleaned.startswith("```"):
                lines = content_cleaned.split('\n')
                content_cleaned = '\n'.join(lines[1:-1]) if len(lines) > 2 else content_cleaned

            # Save raw response for debugging
            debug_file = OUTPUT_DIR / f"{custom_id}_raw_response.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(content_cleaned)
            print(f"Saved raw response to: {debug_file}")

            # Try to parse
            try:
                chunks = json.loads(content_cleaned)

                if not isinstance(chunks, list):
                    print(f"✗ Response is not a list: {type(chunks)}")
                    continue

                # Validate
                if not all(isinstance(c, str) for c in chunks):
                    print(f"✗ Not all chunks are strings")
                    continue

                # Success! Save it
                result_data = {
                    "student": student,
                    "source": source,
                    "chunks": chunks
                }

                output_file = OUTPUT_DIR / f"{custom_id}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)

                print(f"✓ Successfully chunked: {len(chunks)} chunks")
                print(f"✓ Saved to: {output_file}")

            except json.JSONDecodeError as e:
                print(f"✗ JSON parsing failed: {e}")
                print(f"  Check {debug_file} for the raw response")

                # Try to identify the issue
                print("\nAttempting to diagnose JSON error...")
                try:
                    # Find where it breaks
                    for i in range(0, len(content_cleaned), 1000):
                        chunk_to_test = content_cleaned[:i]
                        try:
                            json.loads(chunk_to_test + ']')  # Try to close it
                        except:
                            pass
                    print(f"Error occurs around character {e.pos}")
                    context_start = max(0, e.pos - 100)
                    context_end = min(len(content_cleaned), e.pos + 100)
                    print(f"Context: ...{content_cleaned[context_start:context_end]}...")
                except:
                    pass

        except Exception as e:
            print(f"✗ API request failed: {e}")


if __name__ == "__main__":
    retry_failed_essays()
