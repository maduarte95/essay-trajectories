#!/usr/bin/env python3
"""
Quick test to verify Anthropic API key is set up correctly
"""

import os
from dotenv import load_dotenv
import anthropic

# Load environment variables from .env
load_dotenv()

def test_api():
    """Test API connection."""
    # Check if API key exists in environment
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("✗ No API key found in environment")
        print("\nMake sure .env file exists with:")
        print("  ANTHROPIC_API_KEY=your_key_here")
        return False

    print(f"✓ API key loaded from .env: {api_key[:12]}...{api_key[-4:]}")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        print("✓ Anthropic client initialized successfully")

        # Try a minimal API call to verify the key works
        print("\nTesting API with a minimal request...")
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✓ API test successful! Response: {message.content[0].text}")
        return True

    except anthropic.AuthenticationError:
        print("✗ Authentication failed - check your API key")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_api()
    exit(0 if success else 1)
