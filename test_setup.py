#!/usr/bin/env python3
"""
Test script to verify the Code Contests dataset setup and basic functionality.
Run this before running the full experiment.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    try:
        import openai
        print("  ✓ openai")
    except ImportError:
        print("  ✗ openai - run: pip install openai")
        return False
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - run: pip install numpy")
        return False
    
    try:
        import dotenv
        print("  ✓ python-dotenv")
    except ImportError:
        print("  ✗ python-dotenv - run: pip install python-dotenv")
        return False
    
    try:
        import codebleu
        print("  ✓ codebleu")
    except ImportError:
        print("  ✗ codebleu - run: pip install codebleu")
        return False
    
    try:
        import datasets
        print("  ✓ datasets")
    except ImportError:
        print("  ✗ datasets - run: pip install datasets")
        return False
    
    return True


def test_env_file():
    """Test that .env file exists and has API key"""
    print("\nTesting environment configuration...")
    env_file = Path(".env")
    
    if not env_file.exists():
        print("  ✗ .env file not found")
        print("    Create a .env file with: OPENAI_API_KEY=your-api-key-here")
        return False
    
    print("  ✓ .env file exists")
    
    import dotenv
    dotenv.load_dotenv()
    
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("  ✗ OPENAI_API_KEY not set in .env file")
        return False
    
    print(f"  ✓ OPENAI_API_KEY is set (length: {len(api_key)})")
    return True


def test_dataset_loading():
    """Test that the dataset can be loaded"""
    print("\nTesting dataset loading...")
    
    try:
        from datasets import load_dataset
        print("  Attempting to load Code Contests dataset...")
        print("  (This may take a while on first run - downloading ~2GB)")
        
        # Load just a small sample to test
        dataset = load_dataset("deepmind/code_contests", split="train", streaming=True)
        
        # Get first example
        first_example = next(iter(dataset))
        
        print(f"  ✓ Dataset loaded successfully")
        print(f"  ✓ Sample problem: {first_example.get('name', 'N/A')[:50]}")
        
        # Check for required columns
        required_cols = ['incorrect_solutions', 'solutions', 'public_tests', 'private_tests']
        for col in required_cols:
            if col in first_example:
                print(f"  ✓ Column '{col}' found")
            else:
                print(f"  ✗ Column '{col}' NOT found")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error loading dataset: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Code Contests Setup Verification")
    print("="*60)
    print()
    
    tests = [
        ("Package Imports", test_imports),
        ("Environment File", test_env_file),
        ("Dataset Loading", test_dataset_loading),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Test failed with exception: {e}")
            results.append((name, False))
        print()
    
    print("="*60)
    print("Summary")
    print("="*60)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    
    if all(result for _, result in results):
        print("✓ All tests passed! You're ready to run the experiment.")
        print("\nRun: python3 calculateLogits.py")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
