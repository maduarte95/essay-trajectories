#!/usr/bin/env python3
"""
Small test to verify batch API works correctly with a minimal example
"""

import os
import time
import json
from dotenv import load_dotenv
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

load_dotenv()

def test_small_batch():
    """Test with a tiny batch to verify the API works."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("✗ No API key found")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Create a tiny test request
    print("Creating small test batch...")
    test_text = "This is a test essay. It has two sentences."

    batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id="test-1",
                params=MessageCreateParamsNonStreaming(
                    model="claude-sonnet-4-5",
                    max_tokens=200,
                    messages=[
                        {
                            "role": "user",
                            "content": f'Split this into sentences as a JSON array: "{test_text}"'
                        }
                    ]
                )
            )
        ]
    )

    batch_id = batch.id
    print(f"✓ Batch created: {batch_id}")
    print(f"  Status: {batch.processing_status}")

    # Poll for completion (should be very fast)
    print("\nWaiting for completion...")
    max_wait = 300  # 5 minutes max
    start = time.time()

    while time.time() - start < max_wait:
        batch_status = client.messages.batches.retrieve(batch_id)
        if batch_status.processing_status == "ended":
            print(f"✓ Batch completed in {int(time.time() - start)} seconds")
            break
        print(f"  Status: {batch_status.processing_status}")
        time.sleep(10)
    else:
        print("✗ Batch did not complete in time")
        return

    # Retrieve results
    print("\nRetrieving results...")
    for result in client.messages.batches.results(batch_id):
        print(f"\n--- Result for {result.custom_id} ---")
        print(f"Result type: {result.result.type}")

        if result.result.type == "succeeded":
            message = result.result.message
            print(f"Message model: {message.model}")
            print(f"Stop reason: {message.stop_reason}")
            print(f"Content type: {type(message.content)}")
            print(f"Content length: {len(message.content)}")

            if isinstance(message.content, list):
                content_text = message.content[0].text
            else:
                content_text = message.content

            print(f"\nRaw response:\n{content_text}")

            # Try to parse as JSON (with markdown cleaning)
            try:
                content_cleaned = content_text.strip()
                if content_cleaned.startswith("```"):
                    # Remove markdown code blocks
                    lines = content_cleaned.split('\n')
                    content_cleaned = '\n'.join(lines[1:-1]) if len(lines) > 2 else content_cleaned

                print(f"\nCleaned content:\n{content_cleaned}")

                parsed = json.loads(content_cleaned)
                print(f"\n✓ Successfully parsed JSON: {parsed}")
            except json.JSONDecodeError as e:
                print(f"\n✗ JSON parse error: {e}")

        else:
            print(f"✗ Result was not successful: {result.result}")

if __name__ == "__main__":
    test_small_batch()
