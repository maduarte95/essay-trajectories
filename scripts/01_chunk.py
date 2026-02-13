#!/usr/bin/env python3
"""
Essay Chunking Script using Anthropic Batch API

This script reads essays from FINAL/ and META/ directories, creates batch requests
to chunk each essay into argumentative units, and saves the results.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# Load environment variables from .env
load_dotenv()

# Configuration
FINAL_DIR = Path("FINAL")
META_DIR = Path("META")
OUTPUT_DIR = Path("output/data/chunks")
MODEL = "claude-sonnet-4-5"  # Using Sonnet for cost-effectiveness
MAX_TOKENS = 4096

# Chunking prompt
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
- Return the result as a JSON array of strings, where each string is one chunk
  in order.
- Do not wrap the text in markdown.

Essay:
{essay_text}

Return ONLY the JSON array, no other text."""


def read_essay(file_path: Path) -> str:
    """Read essay text from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def find_essay_files(directory: Path) -> List[Path]:
    """Find all essay files in a directory (supports .md, .txt)."""
    extensions = ['.md', '.txt']
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
    return sorted(files)


def extract_student_id(filename: str) -> str:
    """Extract student ID from filename (e.g., 'student-A-...' -> 'A')."""
    parts = filename.split('-')
    if len(parts) >= 2 and parts[0] == 'student':
        return parts[1]
    return filename


def create_batch_requests(essays: List[Dict[str, Any]]) -> List[Request]:
    """Create batch requests for all essays."""
    requests = []

    for essay in essays:
        custom_id = f"{essay['student']}-{essay['source']}"

        request = Request(
            custom_id=custom_id,
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {
                        "role": "user",
                        "content": CHUNKING_PROMPT.format(essay_text=essay['text'])
                    }
                ]
            )
        )
        requests.append(request)

    return requests


def wait_for_batch(client: anthropic.Anthropic, batch_id: str) -> Any:
    """Poll batch status until completion."""
    print(f"\nWaiting for batch {batch_id} to complete...")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status

        if status == "ended":
            print(f"✓ Batch completed!")
            print(f"  Succeeded: {batch.request_counts.succeeded}")
            print(f"  Errored: {batch.request_counts.errored}")
            print(f"  Expired: {batch.request_counts.expired}")
            print(f"  Canceled: {batch.request_counts.canceled}")
            return batch

        print(f"  Status: {status} (processing: {batch.request_counts.processing})")
        time.sleep(60)  # Poll every 60 seconds


def process_batch_results(client: anthropic.Anthropic, batch_id: str, output_dir: Path) -> tuple:
    """Retrieve and save batch results."""
    print("\nProcessing batch results...")

    results_by_id = {}
    errors = []

    try:
        # Stream results using the correct API method
        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id

            # Check result type using the correct attribute access
            if hasattr(result.result, 'type'):
                result_type = result.result.type
            else:
                # Fallback: check which attribute exists
                if hasattr(result.result, 'message'):
                    result_type = "succeeded"
                elif hasattr(result.result, 'error'):
                    result_type = "errored"
                else:
                    print(f"⚠ {custom_id}: Unknown result type, skipping")
                    continue

            if result_type == "succeeded":
                # Extract the JSON array from the response
                message = result.result.message

                # Handle both string and list content
                if isinstance(message.content, list):
                    content = message.content[0].text
                else:
                    content = message.content

                try:
                    # Parse the JSON response - strip markdown code blocks if present
                    content_cleaned = content.strip()
                    if content_cleaned.startswith("```"):
                        # Remove markdown code blocks
                        lines = content_cleaned.split('\n')
                        content_cleaned = '\n'.join(lines[1:-1]) if len(lines) > 2 else content_cleaned

                    chunks = json.loads(content_cleaned)

                    # Validate it's a list
                    if not isinstance(chunks, list):
                        raise ValueError(f"Expected list, got {type(chunks)}")

                    # Extract student and source from custom_id
                    parts = custom_id.split('-')
                    student = parts[0]
                    source = parts[1]

                    results_by_id[custom_id] = {
                        "student": student,
                        "source": source,
                        "chunks": chunks
                    }

                    print(f"✓ {custom_id}: {len(chunks)} chunks")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"✗ {custom_id}: JSON parsing error - {e}")
                    errors.append({
                        "custom_id": custom_id,
                        "error": f"JSON parsing error: {e}",
                        "content": content[:500]  # Only save first 500 chars
                    })

            elif result_type == "errored":
                error_info = result.result.error
                print(f"✗ {custom_id}: Error - {error_info.type if hasattr(error_info, 'type') else 'unknown'}")
                errors.append({"custom_id": custom_id, "error": str(error_info)})

            elif result_type == "expired":
                print(f"✗ {custom_id}: Expired")
                errors.append({"custom_id": custom_id, "error": "Request expired"})

            elif result_type == "canceled":
                print(f"✗ {custom_id}: Canceled")
                errors.append({"custom_id": custom_id, "error": "Request canceled"})

    except Exception as e:
        print(f"✗ Error retrieving batch results: {e}")
        errors.append({"error": f"Batch retrieval error: {e}"})

    # Save results
    for custom_id, data in results_by_id.items():
        output_file = output_dir / f"{custom_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved: {output_file}")

    # Save errors if any
    if errors:
        error_file = output_dir / "errors.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"\n⚠ Errors saved to: {error_file}")

    return len(results_by_id), len(errors)


def main():
    """Main execution function."""
    print("=" * 60)
    print("Essay Chunking with Anthropic Batch API")
    print("=" * 60)

    # Verify API key is loaded
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗ Error: ANTHROPIC_API_KEY not found in environment")
        print("  Make sure .env file exists with: ANTHROPIC_API_KEY=your_key_here")
        return

    # Initialize client (will automatically use env var)
    try:
        client = anthropic.Anthropic(api_key=api_key)
        print(f"✓ API key loaded: {api_key[:12]}...{api_key[-4:]}")
    except Exception as e:
        print(f"✗ Error initializing Anthropic client: {e}")
        return

    # Collect all essays
    essays = []

    # Read FINAL essays (human)
    print(f"\nReading essays from {FINAL_DIR}/...")
    final_files = find_essay_files(FINAL_DIR)
    for file_path in final_files:
        student_id = extract_student_id(file_path.stem)
        text = read_essay(file_path)
        essays.append({
            "student": student_id,
            "source": "human",
            "text": text,
            "file": str(file_path)
        })
        print(f"  ✓ {student_id} (human): {len(text)} chars")

    # Read META essays (AI)
    print(f"\nReading essays from {META_DIR}/...")
    meta_files = find_essay_files(META_DIR)
    for file_path in meta_files:
        student_id = extract_student_id(file_path.stem)
        text = read_essay(file_path)
        essays.append({
            "student": student_id,
            "source": "ai",
            "text": text,
            "file": str(file_path)
        })
        print(f"  ✓ {student_id} (ai): {len(text)} chars")

    print(f"\nTotal essays to process: {len(essays)}")

    if not essays:
        print("No essays found!")
        return

    # Create batch requests
    print("\nCreating batch requests...")
    requests = create_batch_requests(essays)
    print(f"Created {len(requests)} batch requests")

    # Create batch
    print("\nSubmitting batch to Anthropic API...")
    message_batch = client.messages.batches.create(requests=requests)
    batch_id = message_batch.id
    print(f"✓ Batch created: {batch_id}")
    print(f"  Expires at: {message_batch.expires_at}")

    # Wait for completion
    batch = wait_for_batch(client, batch_id)

    # Process results
    success_count, error_count = process_batch_results(client, batch_id, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total essays: {len(essays)}")
    print(f"Successfully chunked: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
