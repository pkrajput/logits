"""Test the validation function to ensure it catches explanatory text"""

# Test cases
test_cases = [
    ("print(2)", True, "Simple code should pass"),
    ("def foo():\n    return 2", True, "Function definition should pass"),
    ("The code you provided is just a number (2).", False, "Explanation should fail"),
    ("I'm sorry, but there is no context", False, "Apology should fail"),
    ("Your question doesn't provide enough context", False, "Question reference should fail"),
    ("2", True, "Just a number should pass"),
    ("Here's the fixed code:\nprint(2)", False, "Prefixed explanation should fail"),
    ("```python\nprint(2)\n```", True, "Markdown code block should pass after cleaning"),
]

def validate_code_only_response(response_text: str):
    """Simplified version for testing"""
    from typing import Tuple
    
    # Strip whitespace
    cleaned = response_text.strip()
    
    # Remove markdown code blocks if present
    if cleaned.startswith("```"):
        lines = cleaned.split('\n')
        # Remove first line with ```python or ```
        if lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line with ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = '\n'.join(lines).strip()
    
    # Check for common non-code phrases that indicate explanation text
    non_code_indicators = [
        "your question",
        "here is",
        "here's",
        "the correct code",
        "the fixed code",
        "to fix",
        "you should",
        "this code",
        "explanation:",
        "solution:",
        "the issue",
        "the problem",
        "the bug",
        "i'm sorry",
        "i am sorry",
        "there is no",
        "there's no",
        "could you",
        "can you",
        "without context",
        "no context",
        "please provide",
        "not clear",
        "what this code",
        "what the code",
        "supposed to do",
    ]
    
    cleaned_lower = cleaned.lower()
    for indicator in non_code_indicators:
        if indicator in cleaned_lower[:300]:  # Check first 300 chars
            return False, cleaned
    
    # If the response is very short and doesn't look like code, it's probably an explanation
    if len(cleaned) < 5:
        return False, cleaned
    
    # Check if it starts with natural language (not code syntax)
    first_chars = cleaned[:50].strip()
    if first_chars and not any([
        first_chars[0] in '#/"\'-@',  # Comment or string start
        first_chars.startswith(('def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'return ', 'print(', 'with ', 'try:', 'except', 'finally')),
        first_chars[0].isspace(),
        first_chars[0].isdigit(),  # Could be a number
        first_chars.split()[0] in ['public', 'private', 'protected', 'static', 'void', 'int', 'String', 'using', 'namespace', 'include']
    ]):
        # Check if it looks like a sentence (capital letter + spaces)
        if first_chars[0].isupper() and ' ' in first_chars[:40]:
            # Additional check: does it contain punctuation typical of prose?
            if any(p in first_chars[:100] for p in ['. ', '? ', '! ', ', but', ', and']):
                return False, cleaned
    
    return True, cleaned

# Run tests
print("Testing validation function:\n")
for text, expected_valid, description in test_cases:
    is_valid, cleaned = validate_code_only_response(text)
    status = "✓" if is_valid == expected_valid else "✗"
    print(f"{status} {description}")
    print(f"   Input: {text[:50]}...")
    print(f"   Expected: {expected_valid}, Got: {is_valid}")
    if is_valid != expected_valid:
        print(f"   FAILED!")
    print()
