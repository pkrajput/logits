from openai import OpenAI
import numpy as np
import os
import dotenv
from pathlib import Path
import json
from typing import List, Dict, Tuple
import ast
from codebleu import calc_codebleu

dotenv.load_dotenv()

try:
    DEFAULT_REQUEST_TIMEOUT = float(os.environ.get("OPENAI_REQUEST_TIMEOUT", "120"))
except ValueError:
    DEFAULT_REQUEST_TIMEOUT = 120.0

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    max_retries=3,
    timeout=DEFAULT_REQUEST_TIMEOUT,
)

# Dataset configuration
DATASET_REPO_ID = "deepmind/code_contests"
DATASET_LOCAL_DIR = Path(__file__).parent / "Datasets" / "code_contests"

def check_and_download_dataset():
    """
    Check if the Code Contests dataset exists locally, and download it if not.
    
    Returns:
        Path: Path to the local dataset directory
    """
    if not DATASET_LOCAL_DIR.exists() or not any(DATASET_LOCAL_DIR.iterdir()):
        try:
            from datasets import load_dataset
            print(f"Downloading Code Contests dataset to {DATASET_LOCAL_DIR}...")
            dataset = load_dataset(DATASET_REPO_ID, cache_dir=str(DATASET_LOCAL_DIR))
            # Save dataset locally
            DATASET_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(DATASET_LOCAL_DIR / "dataset"))
            print("Dataset downloaded successfully!")
        except ImportError:
            raise ImportError("Please install the datasets library: pip install datasets")
        except Exception as e:
            raise Exception(f"Error downloading dataset: {e}")
    
    return DATASET_LOCAL_DIR

# Check and download dataset on module import
dataset_path = check_and_download_dataset()


def load_code_contests_dataset(split: str = "train"):
    """
    Load the Code Contests dataset from the local directory.
    
    Args:
        split: Dataset split to load ('train', 'valid', or 'test')
    
    Returns:
        Dataset: HuggingFace dataset with code contest problems
    """
    from datasets import load_from_disk
    
    dataset_dir = DATASET_LOCAL_DIR / "dataset"
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}. Please run check_and_download_dataset() first.")
    
    dataset = load_from_disk(str(dataset_dir))
    return dataset[split] if split in dataset else dataset


def detect_language(code: str) -> str:
    """
    Simple heuristic to detect programming language from code.
    
    Args:
        code: Source code string
        
    Returns:
        str: Detected language ('python', 'cpp', 'java', or 'unknown')
    """
    code_lower = code.lower().strip()
    
    # Python indicators
    if 'def ' in code or 'import ' in code or code.startswith('from '):
        return 'python'
    
    # C++ indicators
    if '#include' in code or 'std::' in code or 'cout' in code or 'cin' in code:
        return 'cpp'
    
    # Java indicators
    if 'public class' in code or 'public static void main' in code:
        return 'java'
    
    # Default to python if unsure
    return 'python'


def looks_like_code(text: str) -> bool:
    """Heuristic check to see if a string looks like source code rather than prose.
    Returns True if it likely contains code constructs.
    """
    if not isinstance(text, str):
        return False
    s = text.strip()
    if not s:
        return False
    # Very short single token like '2' may be code (a literal), allow digits
    if s.isdigit():
        return True
    indicators = [
        'def ', 'class ', 'import ', 'from ', 'return ', 'if ', 'for ', 'while ', 'try:', 'except', 'finally',
        '#include', 'std::', ';', '{', '}', 'public class', 'static void main', 'System.out',
        '=', '(', ')'
    ]
    return any(tok in s for tok in indicators)


def extract_any_solution(field) -> Tuple[str | None, str | None]:
    """Robustly extract a code string and (optional) language from a dataset field.
    Handles shapes like:
    - dict { 'language': 'python', 'solution': '<code>' }
    - list of such dicts
    - dict mapping language -> list[str] or str
    - list[str]
    - plain str
    Returns (code, language) where either can be None if unavailable.
    """
    if not field:
        return None, None
    # Plain string
    if isinstance(field, str):
        return field, None
    # Dict case
    if isinstance(field, dict):
        # Shape: { 'language': 'python', 'solution': '<code>' }
        if 'solution' in field:
            code = field.get('solution')
            lang = field.get('language')
            return (str(code) if code is not None else None), (str(lang) if lang is not None else None)
        # Otherwise treat as language -> solutions mapping
        for lang, sols in field.items():
            if isinstance(sols, list) and sols:
                code = sols[0]
                return (str(code) if code is not None else None), str(lang)
            if isinstance(sols, str):
                return sols, str(lang)
        return None, None
    # List case
    if isinstance(field, list):
        for item in field:
            if isinstance(item, dict):
                if 'solution' in item:
                    code = item.get('solution')
                    lang = item.get('language')
                    return (str(code) if code is not None else None), (str(lang) if lang is not None else None)
                # Fall back to any string-like value in dict
                for k, v in item.items():
                    if isinstance(v, str):
                        return v, k if k != 'code' else None
            elif isinstance(item, str):
                return item, None
        return None, None
    # Unknown type
    try:
        return str(field), None
    except Exception:
        return None, None


def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4",
    max_tokens=2000,
    temperature=0.7,
    stop=None,
    tools=None,
    logprobs=None,
    top_logprobs=None,
    request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
):
    """Get completion from OpenAI API - handles both chat/completions and responses endpoints"""

    # Only use responses endpoint for o1 models, not regular gpt-4
    uses_responses_endpoint = model.startswith("o1")
    api_client = client if request_timeout is None else client.with_options(timeout=request_timeout)

    if uses_responses_endpoint:
        input_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            input_messages.append({
                "role": role,
                "content": content,
            })

        params = {
            "model": model,
            "input": input_messages,
            "temperature": temperature,
        }

        if max_tokens is not None:
            params["max_output_tokens"] = max_tokens

        # Note: responses endpoint may not support logprobs the same way
        # Skipping logprobs for responses endpoint for now

        completion = api_client.responses.create(**params)

    else:
        params = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stop": stop,
            "logprobs": logprobs,
            "top_logprobs": top_logprobs,
        }

        if model.startswith("o1") or ("gpt-4" in model):
            params["max_completion_tokens"] = max_tokens
        elif ("codex" not in model):
            params["max_tokens"] = max_tokens
        else:
            params["gpt-4"] = True

        if tools:
            params["tools"] = tools

        completion = api_client.chat.completions.create(**params)

    return completion


def extract_response_content(response, model: str) -> str:
    """Extract text content from response, handling both endpoints"""
    uses_responses_endpoint = model.startswith("o1")
    
    if uses_responses_endpoint:
        # Responses endpoint format
        return getattr(response, "output_text", "")
    else:
        # Chat completions endpoint format
        return response.choices[0].message.content


def extract_logprobs(response, model: str):
    """Extract token logprobs as floats from response regardless of endpoint"""
    uses_responses_endpoint = model.startswith("o1")

    logprob_values: List[float] = []

    if uses_responses_endpoint:
        for output in getattr(response, "output", []):
            if getattr(output, "type", None) != "message":
                continue
            for content in getattr(output, "content", []):
                if getattr(content, "type", None) != "output_text":
                    continue
                logprobs_info = getattr(content, "logprobs", None)
                if logprobs_info is None:
                    continue

                token_logprobs = getattr(logprobs_info, "token_logprobs", None)
                if token_logprobs is not None:
                    logprob_values.extend(token_logprobs)
                    continue

                tokens = getattr(logprobs_info, "tokens", [])
                for token in tokens:
                    logprob = getattr(token, "logprob", None)
                    if logprob is not None:
                        logprob_values.append(logprob)
    else:
        logprob_content = getattr(response.choices[0].logprobs, "content", None)
        if logprob_content:
            for token in logprob_content:
                logprob = getattr(token, "logprob", None)
                if logprob is not None:
                    logprob_values.append(logprob)

    return logprob_values


def validate_code_only_response(response_text: str) -> Tuple[bool, str]:
    """
    Validate that the response contains only code, no explanations.
    
    Args:
        response_text: The response from the LLM
        
    Returns:
        Tuple[bool, str]: (is_valid, cleaned_code)
            is_valid: True if response appears to be code only
            cleaned_code: The cleaned/extracted code
    """
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


def repair_code_direct(buggy_function: str, model: str = "gpt-4", max_retries: int = 2) -> str:
    """
    Method 1: Direct code repair (baseline)
    
    Args:
        buggy_function: The buggy code to repair
        model: Model to use
        max_retries: Number of times to retry if response contains explanations
        
    Returns:
        str: Repaired code
    """
    system_prompt = """You are a code repair assistant. Your task is to fix buggy code.

CRITICAL INSTRUCTIONS:
- Output ONLY the repaired code
- Do NOT include any explanations, descriptions, or commentary
- Do NOT use markdown code blocks (no ```)
- Do NOT add phrases like "Here's the fix" or "The corrected code is"
- Do NOT add comments explaining what you changed
- Just output the raw code that fixes the bug"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Fix this code:\n\n{buggy_function}"}
    ]
    
    for attempt in range(max_retries + 1):
        response = get_completion(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=2000
        )
        
        repaired_code = extract_response_content(response, model)
        is_valid, cleaned_code = validate_code_only_response(repaired_code)
        
        if is_valid:
            return cleaned_code
        
        # If not valid and we have retries left, add a follow-up message
        if attempt < max_retries:
            messages.append({"role": "assistant", "content": repaired_code})
            messages.append({
                "role": "user", 
                "content": "You included explanatory text. Output ONLY the code, nothing else. No explanations, no markdown, just the raw code."
            })
    
    # If all retries failed, return the cleaned version of the last attempt
    return cleaned_code


def repair_code_with_selection(buggy_function: str, model: str = "gpt-4", num_candidates: int = 5, max_retries: int = 2) -> Tuple[str, List[Dict]]:
    """
    Method 2: Generate multiple candidates and select the one with lowest surprisal
    
    Args:
        buggy_function: The buggy code to repair
        model: Model to use
        num_candidates: Number of candidates to generate
        max_retries: Number of times to retry if response contains explanations
        
    Returns:
        Tuple[str, List[Dict]]: Best candidate and list of all candidates with their perplexities
    """
    system_prompt = """You are a code repair assistant. Your task is to fix buggy code.

CRITICAL INSTRUCTIONS:
- Output ONLY the repaired code
- Do NOT include any explanations, descriptions, or commentary
- Do NOT use markdown code blocks (no ```)
- Do NOT add phrases like "Here's the fix" or "The corrected code is"
- Do NOT add comments explaining what you changed
- Just output the raw code that fixes the bug"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Fix this code:\n\n{buggy_function}"}
    ]
    
    candidates = []
    
    for i in range(num_candidates):
        # Try to get a valid code-only response
        for attempt in range(max_retries + 1):
            response = get_completion(
                messages=messages,
                model=model,
                temperature=0.7,
                max_tokens=2000,
                logprobs=True,
                top_logprobs=5
            )
            
            repaired_code = extract_response_content(response, model)
            is_valid, cleaned_code = validate_code_only_response(repaired_code)
            
            if is_valid:
                repaired_code = cleaned_code
                break
            
            # If not valid and we have retries left, add a follow-up message
            if attempt < max_retries:
                messages.append({"role": "assistant", "content": repaired_code})
                messages.append({
                    "role": "user", 
                    "content": "You included explanatory text. Output ONLY the code, nothing else. No explanations, no markdown, just the raw code."
                })
                # Remove the follow-up for next candidate
                if attempt == max_retries - 1:
                    messages = messages[:2]  # Reset to original prompt
            else:
                repaired_code = cleaned_code
        
        # Reset messages for next candidate
        messages = messages[:2]
        
        # Calculate perplexity for this candidate
        logprob_values = extract_logprobs(response, model)

        if logprob_values:
            mean_logprob = float(np.mean(logprob_values))
            perplexity = float(np.exp(-mean_logprob))
        else:
            mean_logprob = float('-inf')
            perplexity = float('inf')
        
        candidates.append({
            "candidate_id": i + 1,
            "code": repaired_code,
            "perplexity": perplexity,
            "mean_logprob": mean_logprob
        })
    
    # Select the candidate with lowest perplexity (lowest surprisal)
    best_candidate = min(candidates, key=lambda x: x['perplexity'])
    
    return best_candidate['code'], candidates


def run_tests_on_solution(solution_code: str, public_tests: Dict, private_tests: Dict, language: str = None) -> Dict:
    """
    Run public and private tests on a solution to check correctness.
    
    Args:
        solution_code: The code solution to test
        public_tests: Dictionary with 'input' and 'output' lists
        private_tests: Dictionary with 'input' and 'output' lists
        language: Programming language (auto-detected if None)
        
    Returns:
        dict: Test results including pass rates and details
    """
    import subprocess
    import tempfile
    
    # Auto-detect language if not specified
    if language is None:
        language = detect_language(solution_code)
    
    results = {
        "language": language,
        "public_tests": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        },
        "private_tests": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": []
        },
        "all_passed": False
    }
    
    def run_single_test(code: str, test_input: str, expected_output: str, lang: str) -> Tuple[bool, str]:
        """Run a single test case"""
        try:
            # Determine file extension and command based on language
            if lang == 'python':
                ext = '.py'
                cmd = ['python3']
            elif lang == 'cpp':
                ext = '.cpp'
                # For C++, we'd need to compile first - skip for now
                return False, "C++ testing not yet implemented"
            elif lang == 'java':
                ext = '.java'
                # For Java, we'd need to compile first - skip for now
                return False, "Java testing not yet implemented"
            else:
                return False, f"Unsupported language: {lang}"
            
            with tempfile.NamedTemporaryFile(mode='w', suffix=ext, delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            print("CODE: ", temp_file)

            try:
                result = subprocess.run(
                    cmd + [temp_file],
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                print(result)
                
                actual_output = result.stdout.strip()
                expected = expected_output.strip()
                
                if actual_output == expected:
                    return True, ""
                else:
                    return False, f"Expected: {expected[:100]}, Got: {actual_output[:100]}"
            finally:
                os.unlink(temp_file)
        except subprocess.TimeoutExpired:
            return False, "Timeout"
        except Exception as e:
            return False, str(e)
    
    # Run public tests
    if public_tests and 'input' in public_tests and 'output' in public_tests:
        for i, (test_in, test_out) in enumerate(zip(public_tests['input'], public_tests['output'])):
            results["public_tests"]["total"] += 1
            passed, error = run_single_test(solution_code, test_in, test_out, language)
            if passed:
                results["public_tests"]["passed"] += 1
            else:
                results["public_tests"]["failed"] += 1
                results["public_tests"]["errors"].append(f"Test {i}: {error}")
    
    # Run private tests
    if private_tests and 'input' in private_tests and 'output' in private_tests:
        for i, (test_in, test_out) in enumerate(zip(private_tests['input'], private_tests['output'])):
            results["private_tests"]["total"] += 1
            passed, error = run_single_test(solution_code, test_in, test_out, language)
            if passed:
                results["private_tests"]["passed"] += 1
            else:
                results["private_tests"]["failed"] += 1
                results["private_tests"]["errors"].append(f"Test {i}: {error}")
    
    # Check if all tests passed
    total_tests = results["public_tests"]["total"] + results["private_tests"]["total"]
    total_passed = results["public_tests"]["passed"] + results["private_tests"]["passed"]
    results["all_passed"] = (total_tests > 0 and total_passed == total_tests)
    
    return results


def check_correctness(repaired_code: str, ground_truth) -> Dict:
    """
    Check if the repaired code matches the ground truth using multiple metrics:
    1. Exact string match (after normalization)
    2. AST equivalence (for Python code)
    3. CodeBLEU score
    
    Args:
        repaired_code: The repaired code
        ground_truth: The correct fixed code (can be str or other types from dataset)
        
    Returns:
        dict: Dictionary with correctness metrics including:
            - exact_match: bool
            - ast_match: bool (None if AST parse fails)
            - codebleu_score: float (0-1)
            - is_correct: bool (True if exact match OR AST match)
    """
    # Convert ground_truth to string if it's not already
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth) if ground_truth is not None else ""
    
    # Strip whitespace for comparison
    repaired_normalized = repaired_code.strip()
    truth_normalized = ground_truth.strip()
    
    # 1. Exact string match
    exact_match = repaired_normalized == truth_normalized
    
    # 2. AST equivalence check (for Python code)
    ast_match = None
    try:
        repaired_ast = ast.parse(repaired_normalized)
        truth_ast = ast.parse(truth_normalized)
        # Compare AST dumps (this is a simple structural comparison)
        ast_match = ast.dump(repaired_ast) == ast.dump(truth_ast)
    except (SyntaxError, ValueError):
        # If either code has syntax errors, AST comparison is not possible
        ast_match = None
    
    # 3. CodeBLEU score
    codebleu_score = 0.0
    try:
        result = calc_codebleu(
            references=[ground_truth],
            predictions=[repaired_code],
            lang="python",
            weights=(0.25, 0.25, 0.25, 0.25)
        )
        codebleu_score = result.get('codebleu', 0.0)
    except Exception as e:
        # If CodeBLEU calculation fails, default to 0
        codebleu_score = 0.0
    
    # Consider correct if exact match OR AST match
    is_correct = exact_match or (ast_match is True)
    
    return {
        "exact_match": exact_match,
        "ast_match": ast_match,
        "codebleu_score": codebleu_score,
        "is_correct": is_correct
    }


def run_accuracy_experiment(num_samples: int = 100, model: str = "gpt-4", start_idx: int = 0, split: str = "train"):
    """
    Run accuracy comparison experiment between two methods on Code Contests dataset:
    1. Direct repair (baseline)
    2. Generate 5 candidates and select lowest surprisal
    
    Uses incorrect_solutions as input and validates using public/private tests.
    Accuracy is determined solely by test passage (not by comparison to correct solutions).
    Correct solutions are used for CodeBLEU comparison when available, but not required.
    
    Args:
        num_samples: Number of code snippets to test
        model: Model to use
        start_idx: Starting index in dataset
        split: Dataset split to use ('train', 'valid', or 'test')
        
    Returns:
        dict: Results with accuracy metrics for both methods
    """
    dataset = load_code_contests_dataset(split=split)
    
    results = {
        "experiment_config": {
            "num_samples": num_samples,
            "model": model,
            "start_idx": start_idx,
            "split": split,
            "num_candidates_method2": 5
        },
        "method1_direct": {
            "correct": 0,
            "total": 0,
            "results": []
        },
        "method2_selection": {
            "correct": 0,
            "total": 0,
            "results": []
        },
        "comparison": []
    }
    
    end_idx = min(start_idx + num_samples, len(dataset))
    
    print(f"Running accuracy experiment on {end_idx - start_idx} samples from Code Contests dataset...")
    print(f"Model: {model}")
    print(f"Split: {split}")
    print(f"Starting from index: {start_idx}\n")
    
    for i in range(start_idx, end_idx):
        entry = dataset[i]
        
        # Skip entries without incorrect solutions
        if not entry.get('incorrect_solutions') or len(entry['incorrect_solutions']) == 0:
            print(f"Skipping sample {i}: No incorrect solutions available")
            continue
        
        # Extract buggy solution from the dictionary (keys are language names)
        buggy_function = None
        fixed_function = None
        selected_lang = None
        
        # Get the first language that has incorrect solutions
        for lang in entry['incorrect_solutions'].keys():
            if len(entry['incorrect_solutions'][lang]) > 0:
                buggy_solution = entry['incorrect_solutions'][lang][0]
                # Convert to string if needed
                buggy_function = str(buggy_solution) if not isinstance(buggy_solution, str) else buggy_solution
                selected_lang = lang
                # Try to get correct solution if available (for CodeBLEU comparison)
                if entry.get('solutions') and lang in entry['solutions'] and len(entry['solutions'][lang]) > 0:
                    correct_solution = entry['solutions'][lang][0]
                    fixed_function = str(correct_solution) if not isinstance(correct_solution, str) else correct_solution
                break
        
        if buggy_function is None:
            print(f"Skipping sample {i}: No usable incorrect solutions")
            continue
        
        # Debug: verify buggy input looks like code and log diagnostics
        try:
            lang_guess = detect_language(buggy_function)
            is_codeish = looks_like_code(buggy_function)
            preview = buggy_function.strip().replace("\n", "\\n")[:160]
            incorrect_langs = list(entry.get('incorrect_solutions', {}).keys())
            correct_langs = list(entry.get('solutions', {}).keys()) if entry.get('solutions') else []
            print(f"Debug buggy input -> type={type(buggy_function).__name__}, selected_lang={selected_lang}, lang_guess={lang_guess}, looks_like_code={is_codeish}, len={len(buggy_function)}")
            print(f"Debug langs -> incorrect={incorrect_langs}, correct={correct_langs}")
            print(f"Debug preview -> {preview}")
            # Optionally dump to a temp file for inspection
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as df:
                df.write(buggy_function)
                print(f"Debug saved buggy input to: {df.name}")
        except Exception as _dbg_e:
            print(f"Debug logging error: {_dbg_e}")

        # Get test cases
        public_tests = entry.get('public_tests', {})
        private_tests = entry.get('private_tests', {})
        
        sample_num = i - start_idx + 1
        print(f"Processing sample {sample_num}/{end_idx - start_idx} (dataset index {i})...", end=" ")
        
        try:
            # Method 1: Direct repair
            repaired_direct = repair_code_direct(buggy_function, model)
            
            # Test the repaired code
            test_results_direct = run_tests_on_solution(
                repaired_direct, 
                public_tests, 
                private_tests
            )
            
            # Check code similarity only if we have a correct solution
            correctness_direct = {}
            if fixed_function:
                correctness_direct = check_correctness(repaired_direct, fixed_function)
            correctness_direct['test_results'] = test_results_direct
            correctness_direct['tests_passed'] = test_results_direct['all_passed']
            
            # Method 2: Generate 5 candidates and select best
            repaired_selection, candidates = repair_code_with_selection(buggy_function, model, num_candidates=5)
            
            # Test the selected code
            test_results_selection = run_tests_on_solution(
                repaired_selection,
                public_tests,
                private_tests
            )
            
            # Check code similarity only if we have a correct solution
            correctness_selection = {}
            if fixed_function:
                correctness_selection = check_correctness(repaired_selection, fixed_function)
            correctness_selection['test_results'] = test_results_selection
            correctness_selection['tests_passed'] = test_results_selection['all_passed']
            
            # Update counts - consider correct if tests pass
            if test_results_direct['all_passed']:
                results["method1_direct"]["correct"] += 1
            if test_results_selection['all_passed']:
                results["method2_selection"]["correct"] += 1
            
            results["method1_direct"]["total"] += 1
            results["method2_selection"]["total"] += 1
            
            # Store individual results
            sample_result = {
                "dataset_index": i,
                "problem_name": entry.get('name', 'unknown'),
                "buggy_function": buggy_function,
                "fixed_function": fixed_function,  # May be None
                "has_correct_solution": fixed_function is not None,
                "method1_direct": {
                    "repaired_code": repaired_direct,
                    **correctness_direct
                },
                "method2_selection": {
                    "repaired_code": repaired_selection,
                    **correctness_selection,
                    "candidates": candidates,
                    "selected_perplexity": min(c['perplexity'] for c in candidates)
                }
            }
            
            results["method1_direct"]["results"].append({
                "dataset_index": i,
                "is_correct": test_results_direct['all_passed']
            })
            
            results["method2_selection"]["results"].append({
                "dataset_index": i,
                "is_correct": test_results_selection['all_passed']
            })
            
            results["comparison"].append(sample_result)
            
            # Print progress
            status = []
            if test_results_direct['all_passed']:
                status.append("M1:✓")
            else:
                status.append("M1:✗")
            if test_results_selection['all_passed']:
                status.append("M2:✓")
            else:
                status.append("M2:✗")
            
            print(" | ".join(status))
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            results["comparison"].append({
                "dataset_index": i,
                "error": str(e)
            })
    
    # Calculate final accuracy
    if results["method1_direct"]["total"] > 0:
        results["method1_direct"]["accuracy"] = results["method1_direct"]["correct"] / results["method1_direct"]["total"]
    else:
        results["method1_direct"]["accuracy"] = 0.0
    
    if results["method2_selection"]["total"] > 0:
        results["method2_selection"]["accuracy"] = results["method2_selection"]["correct"] / results["method2_selection"]["total"]
    else:
        results["method2_selection"]["accuracy"] = 0.0
    
    return results


def print_summary(results: dict):
    """Print a summary of the experiment results"""
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70)
    print(f"\nModel: {results['experiment_config']['model']}")
    print(f"Dataset Split: {results['experiment_config'].get('split', 'N/A')}")
    print(f"Number of samples: {results['experiment_config']['num_samples']}")
    
    # Calculate additional metrics from detailed results
    m1_tests_passed = sum(1 for r in results['comparison'] if r.get('method1_direct', {}).get('tests_passed', False))
    m1_exact = sum(1 for r in results['comparison'] if r.get('method1_direct', {}).get('exact_match', False))
    m1_ast = sum(1 for r in results['comparison'] if r.get('method1_direct', {}).get('ast_match', False))
    
    # Calculate average CodeBLEU only for entries with correct solutions
    m1_codebleu_scores = [r.get('method1_direct', {}).get('codebleu_score', 0) 
                          for r in results['comparison'] 
                          if r.get('method1_direct', {}).get('codebleu_score') is not None]
    m1_avg_codebleu = np.mean(m1_codebleu_scores) if m1_codebleu_scores else 0
    
    m2_tests_passed = sum(1 for r in results['comparison'] if r.get('method2_selection', {}).get('tests_passed', False))
    m2_exact = sum(1 for r in results['comparison'] if r.get('method2_selection', {}).get('exact_match', False))
    m2_ast = sum(1 for r in results['comparison'] if r.get('method2_selection', {}).get('ast_match', False))
    
    m2_codebleu_scores = [r.get('method2_selection', {}).get('codebleu_score', 0) 
                          for r in results['comparison'] 
                          if r.get('method2_selection', {}).get('codebleu_score') is not None]
    m2_avg_codebleu = np.mean(m2_codebleu_scores) if m2_codebleu_scores else 0
    
    # Count how many samples have correct solutions
    samples_with_correct = sum(1 for r in results['comparison'] if r.get('has_correct_solution', False))
    
    print(f"\nSamples with correct solutions available: {samples_with_correct}/{len(results['comparison'])}")
    
    print(f"\nMethod 1 - Direct Repair (Baseline):")
    print(f"  Tests Passed (Primary Metric): {m1_tests_passed}/{results['method1_direct']['total']}")
    print(f"  Accuracy: {results['method1_direct']['accuracy']:.2%}")
    if samples_with_correct > 0:
        print(f"  Exact matches: {m1_exact}")
        print(f"  AST matches: {m1_ast}")
        print(f"  Avg CodeBLEU: {m1_avg_codebleu:.4f}")
    
    print(f"\nMethod 2 - Generate 5 & Select Lowest Surprisal:")
    print(f"  Tests Passed (Primary Metric): {m2_tests_passed}/{results['method2_selection']['total']}")
    print(f"  Accuracy: {results['method2_selection']['accuracy']:.2%}")
    if samples_with_correct > 0:
        print(f"  Exact matches: {m2_exact}")
        print(f"  AST matches: {m2_ast}")
        print(f"  Avg CodeBLEU: {m2_avg_codebleu:.4f}")
    
    print(f"\nImprovement:")
    diff = results['method2_selection']['accuracy'] - results['method1_direct']['accuracy']
    print(f"  Absolute: {diff:+.2%}")
    if results['method1_direct']['accuracy'] > 0:
        relative = (diff / results['method1_direct']['accuracy']) * 100
        print(f"  Relative: {relative:+.1f}%")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Run experiment on Code Contests dataset
    # Using incorrect_solutions as input and validating with public/private tests
    results = run_accuracy_experiment(
        num_samples=10,
        model="gpt-4",
        start_idx=0,
        split="train"  # Options: 'train', 'valid', 'test'
    )
    
    # Save results to file
    output_file = "accuracy_experiment_results_code_contests.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary
    print_summary(results)
