#!/usr/bin/env python3
"""
Test script to verify Hugging Face API connection for Fashion Trend Intelligence project.
"""

import os
from huggingface_hub import login, whoami, hf_hub_download
from dotenv import load_dotenv


def main():
    """Main entry point for testing Hugging Face connection."""
    print("🧪 Testing Hugging Face Connection")
    print("=" * 40)

    # Charger les variables depuis .env
    load_dotenv()
    token = os.getenv("HUGGINGFACE_API_KEY")

    if not token or token == "your_huggingface_api_key_here":
        print("❌ API token not configured!")
        print("🔧 Please set your HUGGINGFACE_API_KEY in the .env file")
        print("📖 Get your token from: https://huggingface.co/settings/tokens")
        return False

    try:
        # Login
        print("🔐 Logging in to Hugging Face...")
        login(token=token)

        # Vérifie l'identité
        info = whoami()
        print(f"✅ Connected to Hugging Face as: {info['name']}")
        print("\n🎉 Hugging Face connection test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("🔧 Please check your token and internet connection")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
