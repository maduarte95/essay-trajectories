#!/usr/bin/env python3
"""
Step 2: Embedding

Generate embeddings for each chunk using sentence-transformers all-MiniLM-L6-v2.
This script reads the chunked essays from output/data/chunks/ and produces
embedding files in output/data/embeddings/.

Usage:
    python scripts/02_embed.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks(chunks_dir: Path) -> List[Dict]:
    """Load all chunk files from the chunks directory."""
    chunk_files = sorted(chunks_dir.glob("*.json"))
    chunk_files = [f for f in chunk_files if f.name != "errors.json"]

    all_chunks = []
    for chunk_file in chunk_files:
        with open(chunk_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            all_chunks.append(data)

    print(f"Loaded {len(all_chunks)} chunk files")
    return all_chunks


def embed_chunks(model: SentenceTransformer, chunks_data: List[Dict], output_dir: Path):
    """Generate embeddings for all chunks and save to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for chunk_data in chunks_data:
        student = chunk_data["student"]
        source = chunk_data["source"]
        # Use cleaned_chunks if available, otherwise fall back to chunks
        chunks = chunk_data.get("cleaned_chunks", chunk_data.get("chunks", []))

        print(f"Embedding {student}-{source} ({len(chunks)} chunks)...", end=" ", flush=True)

        # Generate embeddings for all chunks
        # The model returns numpy arrays
        embeddings = model.encode(
            chunks,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        # Prepare output data
        output_data = {
            "student": student,
            "source": source,
            "chunks": chunks,
            "embeddings": embeddings.tolist()  # Convert to list for JSON serialization
        }

        # Save to JSON file
        output_file = output_dir / f"{student}-{source}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Saved to {output_file.name}")

    print(f"\nAll embeddings saved to {output_dir}")


def save_consolidated_embeddings(chunks_data: List[Dict], embeddings_dir: Path, output_dir: Path):
    """Save a consolidated numpy array of all embeddings for easy downstream use."""
    # Load all embeddings
    all_embeddings = []
    metadata = []

    for chunk_data in chunks_data:
        student = chunk_data["student"]
        source = chunk_data["source"]

        # Load the embeddings from the JSON file
        emb_file = embeddings_dir / f"{student}-{source}.json"
        with open(emb_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            embeddings = np.array(data["embeddings"])

            for i, emb in enumerate(embeddings):
                all_embeddings.append(emb)
                metadata.append({
                    "student": student,
                    "source": source,
                    "chunk_index": i,
                    "chunk_text": data["chunks"][i][:100] + "..."  # First 100 chars
                })

    # Save as numpy array
    all_embeddings_array = np.array(all_embeddings)
    np.save(output_dir / "all_embeddings.npy", all_embeddings_array)

    # Save metadata as JSON
    with open(output_dir / "embedding_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nConsolidated embeddings saved:")
    print(f"  - {output_dir / 'all_embeddings.npy'} (shape: {all_embeddings_array.shape})")
    print(f"  - {output_dir / 'embedding_metadata.json'} ({len(metadata)} entries)")


def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    chunks_dir = project_root / "output" / "data" / "chunks_cleaned"
    embeddings_dir = project_root / "output" / "data" / "embeddings"

    # Check that chunks exist
    if not chunks_dir.exists():
        print(f"Error: Chunks directory not found at {chunks_dir}")
        print("Please run 01_chunk.py first.")
        return

    # Load the embedding model
    print("Loading sentence-transformers model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(f"Model loaded. Embedding dimension: {model.get_sentence_embedding_dimension()}")
    print()

    # Load chunks
    chunks_data = load_chunks(chunks_dir)

    # Generate embeddings
    embed_chunks(model, chunks_data, embeddings_dir)

    # Save consolidated version
    save_consolidated_embeddings(chunks_data, embeddings_dir, embeddings_dir)

    print("\n✓ Embedding step complete!")


if __name__ == "__main__":
    main()
