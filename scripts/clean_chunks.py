#!/usr/bin/env python3
"""
Clean chunks by summarizing each into a short logical bullet point.

This script reads the raw chunks from output/data/chunks/ and creates
cleaned versions that focus on the logical structure rather than specific
references, citations, or detailed semantic content.

The goal is to isolate the logical/argumentative structure of essays
to enable better trajectory comparison.
"""

import json
import os
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Anthropic client
client = Anthropic()

CHUNKS_DIR = Path("output/data/chunks")
CLEANED_DIR = Path("output/data/chunks_cleaned")

# Create cleaned directory if it doesn't exist
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

CLEANING_PROMPT = """You are helping to analyze the logical structure of academic essays.

Below is a chunk from an essay (one argumentative step). Your task is to summarize it into a single, concise bullet point that captures ONLY the logical or argumentative move being made.

Guidelines:
- Focus on the LOGIC: what claim is being made, what move in the argument this represents
- Remove all specific references, citations, author names, dates
- Remove detailed examples, quotes, and semantic elaboration
- Keep it to ONE short sentence (like a bullet point in an outline)
- Use present tense and active voice
- Think of this as extracting the skeleton of the argument

Example:
Input: "Kenneth Craik, writing in 1943, proposed that what makes organisms adaptive is their capacity to build internal models that function as 'distance receptors in time' — neural processes that let an agent respond to situations before they arrive (Craik, 1943). Tolman formalized a version of this as 'cognitive maps': internal spatial models that rats used to navigate mazes..."

Output: "Thinking involves building internal models that allow prediction and planning"

Now summarize this chunk:

{chunk}

Respond with ONLY the summary bullet point, nothing else."""


def clean_chunk(chunk_text: str) -> str:
    """Use Claude to summarize a chunk into a logical bullet point."""
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": CLEANING_PROMPT.format(chunk=chunk_text)
            }
        ]
    )

    # Extract the text response
    summary = message.content[0].text.strip()

    # Remove any leading bullet points or dashes that the model might add
    summary = summary.lstrip('•-* ').strip()

    return summary


def process_file(filepath: Path) -> dict:
    """Process a single chunk file and return cleaned version."""
    print(f"Processing {filepath.name}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    student = data['student']
    source = data['source']
    chunks = data['chunks']

    print(f"  Found {len(chunks)} chunks for student {student} ({source})")

    # Clean each chunk
    cleaned_chunks = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  Cleaning chunk {i}/{len(chunks)}...", end='\r')
        cleaned = clean_chunk(chunk)
        cleaned_chunks.append(cleaned)

    print(f"  Completed {len(cleaned_chunks)} chunks" + " " * 20)

    return {
        "student": student,
        "source": source,
        "original_chunks": chunks,  # Keep originals for reference
        "cleaned_chunks": cleaned_chunks
    }


def main():
    """Process all chunk files."""
    # Get all JSON files except errors.json
    chunk_files = [
        f for f in CHUNKS_DIR.glob("*.json")
        if f.name != "errors.json"
    ]

    print(f"Found {len(chunk_files)} chunk files to process\n")

    for filepath in sorted(chunk_files):
        try:
            cleaned_data = process_file(filepath)

            # Save cleaned version
            output_path = CLEANED_DIR / filepath.name
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

            print(f"  Saved to {output_path}\n")

        except Exception as e:
            print(f"  ERROR processing {filepath.name}: {e}\n")
            continue

    print("=" * 60)
    print("Cleaning complete!")
    print(f"Cleaned files saved to: {CLEANED_DIR}")


if __name__ == "__main__":
    main()
