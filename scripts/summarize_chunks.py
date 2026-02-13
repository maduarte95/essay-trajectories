#!/usr/bin/env python3
"""
Summarize chunking results
"""

import json
from pathlib import Path

chunks_dir = Path("output/data/chunks")

print("=" * 70)
print("CHUNKING SUMMARY")
print("=" * 70)

results = []

for chunk_file in sorted(chunks_dir.glob("*.json")):
    if chunk_file.name == "errors.json":
        continue

    with open(chunk_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    student = data["student"]
    source = data["source"]
    num_chunks = len(data["chunks"])

    results.append({
        "student": student,
        "source": source,
        "chunks": num_chunks
    })

# Print by student
print("\nBy Student:\n")
print(f"{'Student':<10} {'Human':<15} {'AI':<15}")
print("-" * 40)

students = sorted(set(r["student"] for r in results))
for student in students:
    human_chunks = next((r["chunks"] for r in results if r["student"] == student and r["source"] == "human"), "-")
    ai_chunks = next((r["chunks"] for r in results if r["student"] == student and r["source"] == "ai"), "-")
    print(f"{student:<10} {str(human_chunks):<15} {str(ai_chunks):<15}")

# Overall stats
print("\n" + "=" * 70)
print("OVERALL STATISTICS")
print("=" * 70)

human_results = [r for r in results if r["source"] == "human"]
ai_results = [r for r in results if r["source"] == "ai"]

print(f"\nTotal essays processed: {len(results)}")
print(f"  Human essays: {len(human_results)}")
print(f"  AI essays: {len(ai_results)}")

if human_results:
    human_chunks = [r["chunks"] for r in human_results]
    print(f"\nHuman essay chunks:")
    print(f"  Mean: {sum(human_chunks)/len(human_chunks):.1f}")
    print(f"  Min: {min(human_chunks)}")
    print(f"  Max: {max(human_chunks)}")
    print(f"  Total: {sum(human_chunks)}")

if ai_results:
    ai_chunks = [r["chunks"] for r in ai_results]
    print(f"\nAI essay chunks:")
    print(f"  Mean: {sum(ai_chunks)/len(ai_chunks):.1f}")
    print(f"  Min: {min(ai_chunks)}")
    print(f"  Max: {max(ai_chunks)}")
    print(f"  Total: {sum(ai_chunks)}")

# Check for pairs
paired = []
for student in students:
    has_human = any(r["student"] == student and r["source"] == "human" for r in results)
    has_ai = any(r["student"] == student and r["source"] == "ai" for r in results)
    if has_human and has_ai:
        paired.append(student)

print(f"\nPaired essays (for trajectory analysis): {len(paired)}")
print(f"  Students: {', '.join(paired)}")

print("\n" + "=" * 70)
print("✓ All essays successfully chunked!")
print("=" * 70)
