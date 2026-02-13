#!/usr/bin/env python3
"""
Clean chunks using Anthropic's Batch API for efficient processing.

This script reads raw chunks and creates cleaned versions that focus on
logical structure using batched API requests for speed and cost efficiency.
"""

import json
import time
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Anthropic client
client = Anthropic()

CHUNKS_DIR = Path("output/data/chunks")
CLEANED_DIR = Path("output/data/chunks_cleaned")
BATCH_REQUESTS_FILE = Path("output/data/batch_requests.jsonl")
BATCH_RESULTS_FILE = Path("output/data/batch_results.jsonl")

# Create directories
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


def create_batch_requests():
    """Create a JSONL file with all batch requests."""
    print("Creating batch requests...")

    # Get all chunk files
    chunk_files = [
        f for f in CHUNKS_DIR.glob("*.json")
        if f.name != "errors.json"
    ]

    all_requests = []
    chunk_metadata = []  # Track which request corresponds to which chunk

    for filepath in sorted(chunk_files):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        student = data['student']
        source = data['source']
        chunks = data['chunks']

        print(f"  {filepath.name}: {len(chunks)} chunks")

        for chunk_idx, chunk_text in enumerate(chunks):
            # Create unique custom_id for this chunk
            custom_id = f"{student}-{source}-{chunk_idx}"

            # Create batch request
            request = {
                "custom_id": custom_id,
                "params": {
                    "model": "claude-sonnet-4-5-20250929",
                    "max_tokens": 150,
                    "messages": [
                        {
                            "role": "user",
                            "content": CLEANING_PROMPT.format(chunk=chunk_text)
                        }
                    ]
                }
            }

            all_requests.append(request)

            # Track metadata
            chunk_metadata.append({
                "custom_id": custom_id,
                "student": student,
                "source": source,
                "chunk_idx": chunk_idx,
                "original_text": chunk_text
            })

    # Write requests to JSONL
    with open(BATCH_REQUESTS_FILE, 'w', encoding='utf-8') as f:
        for request in all_requests:
            f.write(json.dumps(request) + '\n')

    # Save metadata
    metadata_file = Path("output/data/batch_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)

    print(f"\nCreated {len(all_requests)} batch requests")
    print(f"Saved to: {BATCH_REQUESTS_FILE}")
    print(f"Metadata saved to: {metadata_file}")

    return len(all_requests)


def submit_batch():
    """Submit the batch job to Anthropic."""
    print("\nSubmitting batch job...")

    # Read all requests into memory
    requests = []
    with open(BATCH_REQUESTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            requests.append(json.loads(line))

    # Submit batch
    batch = client.messages.batches.create(
        requests=requests
    )

    print(f"Batch submitted!")
    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  Request counts: {batch.request_counts}")

    return batch.id


def check_batch_status(batch_id: str):
    """Check the status of a batch job."""
    batch = client.messages.batches.retrieve(batch_id)

    print(f"\nBatch Status: {batch.processing_status}")
    print(f"  Request counts: {batch.request_counts}")

    if batch.processing_status == "ended":
        print(f"  Results URL: {batch.results_url}")

    return batch


def download_results(batch_id: str):
    """Download batch results."""
    print("\nDownloading results...")

    # Get batch info
    batch = client.messages.batches.retrieve(batch_id)

    if batch.processing_status != "ended":
        print(f"Batch not ready yet. Status: {batch.processing_status}")
        return False

    # Download results
    results = client.messages.batches.results(batch_id)

    # Save results to file
    with open(BATCH_RESULTS_FILE, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result.to_dict()) + '\n')

    print(f"Results saved to: {BATCH_RESULTS_FILE}")
    return True


def process_results():
    """Process batch results and create cleaned chunk files."""
    print("\nProcessing results...")

    # Load metadata
    metadata_file = Path("output/data/batch_metadata.json")
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Create lookup by custom_id
    metadata_lookup = {item['custom_id']: item for item in metadata}

    # Load results
    results_by_id = {}
    with open(BATCH_RESULTS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            result = json.loads(line)
            custom_id = result['custom_id']

            # Extract cleaned text
            if result['result']['type'] == 'succeeded':
                message = result['result']['message']
                cleaned_text = message['content'][0]['text'].strip()
                cleaned_text = cleaned_text.lstrip('•-* ').strip()
                results_by_id[custom_id] = cleaned_text
            else:
                # Handle errors
                error = result['result'].get('error', {})
                print(f"  ERROR for {custom_id}: {error}")
                results_by_id[custom_id] = "[ERROR]"

    # Organize by file
    files_data = {}
    for custom_id, cleaned_text in results_by_id.items():
        meta = metadata_lookup[custom_id]
        student = meta['student']
        source = meta['source']
        chunk_idx = meta['chunk_idx']
        original_text = meta['original_text']

        file_key = f"{student}-{source}"
        if file_key not in files_data:
            files_data[file_key] = {
                'student': student,
                'source': source,
                'original_chunks': [],
                'cleaned_chunks': []
            }

        files_data[file_key]['original_chunks'].append((chunk_idx, original_text))
        files_data[file_key]['cleaned_chunks'].append((chunk_idx, cleaned_text))

    # Sort chunks by index and write files
    for file_key, data in files_data.items():
        # Sort by chunk index
        data['original_chunks'].sort(key=lambda x: x[0])
        data['cleaned_chunks'].sort(key=lambda x: x[0])

        # Remove indices
        data['original_chunks'] = [text for _, text in data['original_chunks']]
        data['cleaned_chunks'] = [text for _, text in data['cleaned_chunks']]

        # Write file
        output_file = CLEANED_DIR / f"{file_key}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  Created {output_file.name}")

    print(f"\nAll cleaned files saved to: {CLEANED_DIR}")


def wait_for_batch(batch_id: str, check_interval: int = 30):
    """Wait for batch to complete, checking periodically."""
    print(f"\nWaiting for batch to complete (checking every {check_interval}s)...")
    print("You can safely stop this script and resume later with:")
    print(f"  python scripts/clean_chunks_batch.py --resume {batch_id}")
    print()

    while True:
        batch = check_batch_status(batch_id)

        if batch.processing_status == "ended":
            print("\nBatch completed!")
            return True

        time.sleep(check_interval)


def main():
    """Main execution flow."""
    import sys

    # Check for resume mode
    if len(sys.argv) > 1 and sys.argv[1] == "--resume":
        if len(sys.argv) < 3:
            print("Usage: --resume <batch_id>")
            return

        batch_id = sys.argv[2]
        print(f"Resuming batch: {batch_id}")

        # Check status
        batch = check_batch_status(batch_id)

        if batch.processing_status == "ended":
            if download_results(batch_id):
                process_results()
        else:
            wait_for_batch(batch_id)
            if download_results(batch_id):
                process_results()

        return

    # Normal flow: create and submit batch
    print("=" * 60)
    print("CHUNK CLEANING - BATCH MODE")
    print("=" * 60)

    # Step 1: Create batch requests
    num_requests = create_batch_requests()

    # Step 2: Submit batch
    batch_id = submit_batch()

    # Save batch ID for reference
    batch_id_file = Path("output/data/batch_id.txt")
    with open(batch_id_file, 'w') as f:
        f.write(batch_id)
    print(f"\nBatch ID saved to: {batch_id_file}")

    # Step 3: Wait for completion
    wait_for_batch(batch_id)

    # Step 4: Download and process results
    if download_results(batch_id):
        process_results()

    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
