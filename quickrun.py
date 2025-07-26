#!/usr/bin/env python3
"""
Quick run script - zero config needed, just API keys.
Usage: python quickrun.py
"""

import os
import subprocess
import sys

def check_api_keys():
    """Check which API keys are available."""
    keys = {
        'anthropic': os.getenv('API_KEY_ANTHROPIC'),
        'openai': os.getenv('API_KEY_OPENAI'),
        'groq': os.getenv('GROQ_API_KEY'),
        'google': os.getenv('API_KEY_GOOGLE')
    }
    
    available = {k: v for k, v in keys.items() if v}
    
    if not available:
        print("❌ No API keys found!")
        print("\nPlease set one of these environment variables:")
        print("  export API_KEY_ANTHROPIC=your_key")
        print("  export API_KEY_OPENAI=your_key")
        print("  export GROQ_API_KEY=your_key")
        print("  export API_KEY_GOOGLE=your_key")
        print("\nOr add them to a .env file")
        sys.exit(1)
    
    return available

def select_model(available_keys):
    """Select best available model."""
    # Priority order
    if 'anthropic' in available_keys:
        return 'claude-3-5-sonnet'
    elif 'openai' in available_keys:
        return 'gpt-4o-mini'
    elif 'groq' in available_keys:
        return 'llama-3.1-8b'
    elif 'google' in available_keys:
        return 'gemini-flash'
    
def main():
    print("🚀 Quick Run - Reasoning Faithfulness Experiment\n")
    
    # Load .env if exists
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
    
    # Check API keys
    available = check_api_keys()
    print(f"✅ Found API keys for: {', '.join(available.keys())}")
    
    # Select model
    model = select_model(available)
    print(f"📊 Using model: {model}")
    
    # Run test
    print("\n🧪 Running quick test (3 questions)...")
    cmd = f"make test MODEL={model}"
    
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("\n✅ Test completed successfully!")
        print(f"\nTo run full experiment:")
        print(f"  make run MODEL={model}")
    except subprocess.CalledProcessError:
        print("\n❌ Test failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()