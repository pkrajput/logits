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
DATASET_REPO_ID = "zxliu/ReAPR-Automatic-Program-Repair-via-Retrieval-Augmented-Large-Language-Models"
DATASET_LOCAL_DIR = Path(__file__).parent / "ReAPR-dataset"

def check_and_download_dataset():
    """
    Check if the ReAPR dataset exists locally, and download it if not.
    
    Returns:
        Path: Path to the local dataset directory
    """
    if not DATASET_LOCAL_DIR.exists() or not any(DATASET_LOCAL_DIR.iterdir()):
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                local_dir=str(DATASET_LOCAL_DIR)
            )
        except ImportError:
            raise
        except Exception as e:
            raise
    
    return DATASET_LOCAL_DIR

# Check and download dataset on module import
dataset_path = check_and_download_dataset()


def load_reapr_dataset():
    """
    Load the ReAPR dataset from the local directory.
    
    Returns:
        list: List of dictionaries containing buggy_function and fixed_function
    """
    dataset_file = DATASET_LOCAL_DIR / "new_merged_file.json"
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_file}")
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    return data


def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4-mini",
    max_tokens=2000,
    temperature=0.7,
    stop=None,
    tools=None,
    logprobs=None,
    top_logprobs=None,
    request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT,
):
    """Get completion from OpenAI API - handles both chat/completions and responses endpoints"""

    uses_responses_endpoint = "gpt-4-mini" in model
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

        if logprobs:
            params["logprobs"] = True
            if top_logprobs is not None:
                params["top_logprobs"] = top_logprobs

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

        if model.startswith("o1") or ("gpt-4-mini" in model):
            params["max_completion_tokens"] = max_tokens
        elif ("codex" not in model):
            params["max_tokens"] = max_tokens
        else:
            params["gpt-4-mini"] = True

        if tools:
            params["tools"] = tools

        completion = api_client.chat.completions.create(**params)

    return completion


def extract_response_content(response, model: str) -> str:
    """Extract text content from response, handling both endpoints"""
    uses_responses_endpoint = "gpt-4-mini" in model
    
    if uses_responses_endpoint:
        # Responses endpoint format
        return getattr(response, "output_text", "")
    else:
        # Chat completions endpoint format
        return response.choices[0].message.content


def extract_logprobs(response, model: str):
    """Extract token logprobs as floats from response regardless of endpoint"""
    uses_responses_endpoint = "gpt-4-mini" in model

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


def repair_code_direct(buggy_function: str, model: str = "gpt-4-mini") -> str:
    """
    Method 1: Direct code repair (baseline)
    
    Args:
        buggy_function: The buggy code to repair
        model: Model to use
        
    Returns:
        str: Repaired code
    """
    messages = [
        {"role": "system", "content": "You are a code repair assistant. Only output code. Do not include any explanations, comments, or natural language text in your response. Only generate the code itself. Do not print code comment markers or quotes at the beginning or end of the code block."},
        {"role": "user", "content": f"Repair this buggy code:\n\n{buggy_function}"}
    ]
    
    response = get_completion(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=2000
    )
    
    return extract_response_content(response, model)


def repair_code_with_selection(buggy_function: str, model: str = "gpt-4-mini", num_candidates: int = 5) -> Tuple[str, List[Dict]]:
    """
    Method 2: Generate multiple candidates and select the one with lowest surprisal
    
    Args:
        buggy_function: The buggy code to repair
        model: Model to use
        num_candidates: Number of candidates to generate
        
    Returns:
        Tuple[str, List[Dict]]: Best candidate and list of all candidates with their perplexities
    """
    messages = [
        {"role": "system", "content": "You are a code repair assistant. Only output code. Do not include any explanations, comments, or natural language text in your response. Only generate the code itself. Do not print code comment markers or quotes at the beginning or end of the code block."},
        {"role": "user", "content": f"Repair this buggy code:\n\n{buggy_function}"}
    ]
    
    candidates = []
    
    for i in range(num_candidates):
        response = get_completion(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=2000,
            logprobs=True,
            top_logprobs=5
        )
        
        repaired_code = extract_response_content(response, model)
        
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


def check_correctness(repaired_code: str, ground_truth: str) -> Dict:
    """
    Check if the repaired code matches the ground truth using multiple metrics:
    1. Exact string match (after normalization)
    2. AST equivalence (for Python code)
    3. CodeBLEU score
    
    Args:
        repaired_code: The repaired code
        ground_truth: The correct fixed code
        
    Returns:
        dict: Dictionary with correctness metrics including:
            - exact_match: bool
            - ast_match: bool (None if AST parse fails)
            - codebleu_score: float (0-1)
            - is_correct: bool (True if exact match OR AST match)
    """
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


def run_accuracy_experiment(num_samples: int = 100, model: str = "gpt-4-mini", start_idx: int = 0):
    """
    Run accuracy comparison experiment between two methods:
    1. Direct repair (baseline)
    2. Generate 5 candidates and select lowest surprisal
    
    Args:
        num_samples: Number of code snippets to test
        model: Model to use
        start_idx: Starting index in dataset
        
    Returns:
        dict: Results with accuracy metrics for both methods
    """
    dataset = load_reapr_dataset()
    
    results = {
        "experiment_config": {
            "num_samples": num_samples,
            "model": model,
            "start_idx": start_idx,
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
    
    print(f"Running accuracy experiment on {end_idx - start_idx} samples...")
    print(f"Model: {model}")
    print(f"Starting from index: {start_idx}\n")
    
    for i in range(start_idx, end_idx):
        entry = dataset[i]
        buggy_function = entry['buggy_function']
        fixed_function = entry['fixed_function']
        
        sample_num = i - start_idx + 1
        print(f"Processing sample {sample_num}/{end_idx - start_idx} (dataset index {i})...", end=" ")
        
        try:
            # Method 1: Direct repair
            repaired_direct = repair_code_direct(buggy_function, model)
            correctness_direct = check_correctness(repaired_direct, fixed_function)
            
            # Method 2: Generate 5 candidates and select best
            repaired_selection, candidates = repair_code_with_selection(buggy_function, model, num_candidates=5)
            correctness_selection = check_correctness(repaired_selection, fixed_function)
            
            # Update counts
            if correctness_direct["is_correct"]:
                results["method1_direct"]["correct"] += 1
            if correctness_selection["is_correct"]:
                results["method2_selection"]["correct"] += 1
            
            results["method1_direct"]["total"] += 1
            results["method2_selection"]["total"] += 1
            
            # Store individual results
            sample_result = {
                "dataset_index": i,
                "buggy_function": buggy_function,
                "fixed_function": fixed_function,
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
                "is_correct": correctness_direct["is_correct"]
            })
            
            results["method2_selection"]["results"].append({
                "dataset_index": i,
                "is_correct": correctness_selection["is_correct"]
            })
            
            results["comparison"].append(sample_result)
            
            # Print progress
            status = []
            if correctness_direct["is_correct"]:
                status.append("M1:✓")
            else:
                status.append("M1:✗")
            if correctness_selection["is_correct"]:
                status.append("M2:✓")
            else:
                status.append("M2:✗")
            
            print(" | ".join(status))
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
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
    print(f"Number of samples: {results['experiment_config']['num_samples']}")
    
    # Calculate additional metrics from detailed results
    m1_exact = sum(1 for r in results['comparison'] if r.get('method1_direct', {}).get('exact_match', False))
    m1_ast = sum(1 for r in results['comparison'] if r.get('method1_direct', {}).get('ast_match', False))
    m1_avg_codebleu = np.mean([r.get('method1_direct', {}).get('codebleu_score', 0) for r in results['comparison']])
    
    m2_exact = sum(1 for r in results['comparison'] if r.get('method2_selection', {}).get('exact_match', False))
    m2_ast = sum(1 for r in results['comparison'] if r.get('method2_selection', {}).get('ast_match', False))
    m2_avg_codebleu = np.mean([r.get('method2_selection', {}).get('codebleu_score', 0) for r in results['comparison']])
    
    print(f"\nMethod 1 - Direct Repair (Baseline):")
    print(f"  Correct (Exact or AST match): {results['method1_direct']['correct']}/{results['method1_direct']['total']}")
    print(f"  Accuracy: {results['method1_direct']['accuracy']:.2%}")
    print(f"  Exact matches: {m1_exact}")
    print(f"  AST matches: {m1_ast}")
    print(f"  Avg CodeBLEU: {m1_avg_codebleu:.4f}")
    
    print(f"\nMethod 2 - Generate 5 & Select Lowest Surprisal:")
    print(f"  Correct (Exact or AST match): {results['method2_selection']['correct']}/{results['method2_selection']['total']}")
    print(f"  Accuracy: {results['method2_selection']['accuracy']:.2%}")
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
    # Run experiment on 100 code snippets
    results = run_accuracy_experiment(
        num_samples=10,
        model="gpt-4-mini",
        start_idx=0
    )
    
    # Save results to file
    output_file = "accuracy_experiment_results_5_codex.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Print summary
    print_summary(results)
