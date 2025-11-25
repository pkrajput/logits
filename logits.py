from openai import OpenAI
import os
import sys
import time
import dotenv
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

# Optional deps used at runtime
try:
    import pandas as pd
except Exception:
    pd = None  # Will validate later

try:
    from datasets import load_from_disk
except Exception:
    load_from_disk = None  # Will validate later

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


# -----------------------------
# Core API Functions (from calculateLogits.py)
# -----------------------------

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
        return getattr(response, "output_text", "")
    else:
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


# -----------------------------
# Helpers for dataset + prompts
# -----------------------------

REPO_ROOT = Path(__file__).parent
DEFAULT_DATASET_DIR = REPO_ROOT / "Datasets" / "code_contests" / "dataset"


def ensure_runtime_deps():
    missing = []
    if pd is None:
        missing.append("pandas")
    if load_from_disk is None:
        missing.append("datasets")
    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: " + ", ".join(missing) + 
            "\nPlease install them (e.g., pip install pandas datasets)."
        )


def load_code_contests_local(split: str = "train", fallback_path: str | None = None):
    """Load Code Contests dataset from disk."""
    ensure_runtime_deps()

    paths_to_try: List[Path] = []

    if fallback_path:
        paths_to_try.append(Path(fallback_path).expanduser().resolve())

    paths_to_try.append(DEFAULT_DATASET_DIR)

    last_err: Exception | None = None
    for base in paths_to_try:
        try:
            if base.is_dir():
                ds = load_from_disk(str(base))
                if isinstance(ds, dict) and split in ds:
                    return ds[split]
                return ds if hasattr(ds, "features") else ds[split]
        except Exception as e:
            last_err = e

    raise FileNotFoundError(
        f"Unable to load dataset. Tried: {', '.join(str(p) for p in paths_to_try)}\n"
        + (f"Last error: {last_err}" if last_err else "")
    )


def strip_code_fences(text: str) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def build_prompt(entry: Dict[str, Any]) -> List[Dict[str, str]]:
    """Create a concise prompt for the model to generate a solution."""
    name = entry.get("name", "")
    desc = entry.get("description", "")
    public_tests = entry.get("public_tests", {}) or {}
    sample_snippets: List[str] = []
    try:
        ins = public_tests.get("input", []) or []
        outs = public_tests.get("output", []) or []
        for i in range(min(3, len(ins), len(outs))):
            sample_snippets.append(
                f"# Sample Input {i+1}\n{ins[i]}\n# Expected Output {i+1}\n{outs[i]}"
            )
    except Exception:
        pass

    system = (
        "You are a competitive programming assistant. "
        "Produce ONLY runnable Python 3 code solving the problem. "
        "Use standard input and output. Do not include explanations, comments, or markdown."
    )

    user_parts = [
        f"Problem: {name}",
        "\nDescription:\n" + str(desc).strip(),
    ]
    if sample_snippets:
        user_parts.append("\nExamples (for guidance only):\n" + "\n\n".join(sample_snippets))

    user = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# -----------------------------
# Method 1: Direct single generation
# -----------------------------

def generate_solution_direct(messages: List[Dict[str, str]], model: str = "gpt-4o-mini", max_tokens: int = 1024) -> Tuple[str, Dict[str, Any]]:
    """
    Method 1: Generate a single solution.
    
    Returns:
        (code, metadata) where metadata includes timing info
    """
    start_time = time.time()
    
    response = get_completion(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=max_tokens,
    )
    
    code = extract_response_content(response, model)
    code = strip_code_fences(code)
    
    elapsed = time.time() - start_time
    
    metadata = {
        "method": "direct",
        "elapsed_seconds": elapsed,
        "num_candidates": 1,
    }
    
    return code, metadata


# -----------------------------
# Method 2: Generate 5 candidates and select lowest surprisal
# -----------------------------

def generate_solution_with_selection(
    messages: List[Dict[str, str]], 
    model: str = "gpt-4o-mini", 
    num_candidates: int = 5,
    max_tokens: int = 1024
) -> Tuple[str, Dict[str, Any]]:
    """
    Method 2: Generate multiple candidates and select the one with lowest surprisal (perplexity).
    
    Returns:
        (best_code, metadata) where metadata includes candidate info
    """
    start_time = time.time()
    
    candidates = []
    
    for i in range(num_candidates):
        response = get_completion(
            messages=messages,
            model=model,
            temperature=0.7,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=5
        )
        
        code = extract_response_content(response, model)
        code = strip_code_fences(code)
        
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
            "code": code,
            "perplexity": perplexity,
            "mean_logprob": mean_logprob,
            "num_tokens": len(logprob_values)
        })
    
    # Select the candidate with lowest perplexity (lowest surprisal)
    best_candidate = min(candidates, key=lambda x: x['perplexity'])
    
    elapsed = time.time() - start_time
    
    metadata = {
        "method": "selection",
        "elapsed_seconds": elapsed,
        "num_candidates": num_candidates,
        "selected_candidate_id": best_candidate['candidate_id'],
        "selected_perplexity": best_candidate['perplexity'],
        "selected_mean_logprob": best_candidate['mean_logprob'],
        "selected_num_tokens": best_candidate['num_tokens'],
        "all_candidates": candidates
    }
    
    return best_candidate['code'], metadata


# -----------------------------
# Comparison experiment
# -----------------------------

def run_method_comparison(
    n: int = 100,
    split: str = "train",
    dataset_path_hint: str | None = None,
    model: str = "gpt-4o-mini",
    num_candidates: int = 5,
    save_basename: str = "method_comparison_results",
):
    """
    Compare two methods:
    1. Direct single generation
    2. Generate 5 candidates and select lowest surprisal
    
    Saves results as DataFrame with both solutions and metadata.
    """
    ensure_runtime_deps()

    ds = load_code_contests_local(split=split, fallback_path=dataset_path_hint)
    head = ds.select(range(min(n, len(ds))))

    # Convert to DataFrame with original columns
    df = head.to_pandas()

    # Storage for results
    method1_codes = []
    method1_metadata = []
    method2_codes = []
    method2_metadata = []
    
    start_time = time.time()
    
    for idx, row in enumerate(head):
        messages = build_prompt(row)
        problem_name = row.get('name', f'problem_{idx}')
        
        print(f"\n{'='*80}")
        print(f"Processing {idx+1}/{len(head)}: {problem_name[:60]}")
        print(f"{'='*80}")
        
        # Method 1: Direct generation
        print("  Method 1 (Direct): Generating single solution...", end=' ')
        try:
            code1, meta1 = generate_solution_direct(messages, model=model)
            method1_codes.append(code1)
            method1_metadata.append(meta1)
            print(f"✓ ({meta1['elapsed_seconds']:.1f}s)")
        except Exception as e:
            error_code = f"# ERROR: {e}"
            method1_codes.append(error_code)
            method1_metadata.append({"method": "direct", "error": str(e)})
            print(f"✗ Error: {e}")
        
        # Method 2: Generate and select
        print(f"  Method 2 (Selection): Generating {num_candidates} candidates...", end=' ')
        try:
            code2, meta2 = generate_solution_with_selection(
                messages, 
                model=model, 
                num_candidates=num_candidates
            )
            method2_codes.append(code2)
            method2_metadata.append(meta2)
            print(f"✓ Selected #{meta2['selected_candidate_id']} (perplexity={meta2['selected_perplexity']:.2f}, {meta2['elapsed_seconds']:.1f}s)")
        except Exception as e:
            error_code = f"# ERROR: {e}"
            method2_codes.append(error_code)
            method2_metadata.append({"method": "selection", "error": str(e)})
            print(f"✗ Error: {e}")
        
        # Progress update
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(head) - idx - 1)
            print(f"\n  Progress: {idx+1}/{len(head)} | Elapsed: {elapsed:.1f}s | Est. remaining: {remaining:.1f}s")
    
    # Add results to dataframe
    df['method1_solution'] = method1_codes
    df['method1_metadata'] = [json.dumps(m) for m in method1_metadata]
    df['method2_solution'] = method2_codes
    df['method2_metadata'] = [json.dumps(m) for m in method2_metadata]
    
    # Calculate summary statistics
    df['method1_elapsed'] = [m.get('elapsed_seconds', 0) for m in method1_metadata]
    df['method2_elapsed'] = [m.get('elapsed_seconds', 0) for m in method2_metadata]
    df['method1_has_error'] = df['method1_solution'].str.startswith('# ERROR:', na=False)
    df['method2_has_error'] = df['method2_solution'].str.startswith('# ERROR:', na=False)
    df['solutions_differ'] = df['method1_solution'] != df['method2_solution']
    
    # Save results
    results_dir = REPO_ROOT / "Results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = results_dir / f"{save_basename}.csv"
    parquet_path = results_dir / f"{save_basename}.parquet"
    
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        pass
    
    total_elapsed = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total problems processed: {len(df)}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"\nMethod 1 (Direct):")
    print(f"  Avg time per problem: {df['method1_elapsed'].mean():.1f}s")
    print(f"  Errors: {df['method1_has_error'].sum()}/{len(df)}")
    print(f"\nMethod 2 (Selection with {num_candidates} candidates):")
    print(f"  Avg time per problem: {df['method2_elapsed'].mean():.1f}s")
    print(f"  Errors: {df['method2_has_error'].sum()}/{len(df)}")
    print(f"  Avg selected perplexity: {np.mean([m.get('selected_perplexity', float('inf')) for m in method2_metadata if 'selected_perplexity' in m]):.2f}")
    print(f"\nComparison:")
    print(f"  Solutions differ: {df['solutions_differ'].sum()}/{len(df)} ({100*df['solutions_differ'].mean():.1f}%)")
    print(f"  Time ratio (M2/M1): {df['method2_elapsed'].mean() / df['method1_elapsed'].mean():.2f}x")
    print(f"\nResults saved:")
    print(f"  - {csv_path}")
    if parquet_path.exists():
        print(f"  - {parquet_path}")
    print(f"{'='*80}\n")
    
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two methods: direct generation vs. generate-5-and-select-lowest-surprisal"
    )
    parser.add_argument("--limit", type=int, default=10, help="Number of problems to process")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"], help="Dataset split")
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional explicit path to load_from_disk()")
    parser.add_argument("--save-name", type=str, default=None, help="Base filename for results")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name to use")
    parser.add_argument("--num-candidates", type=int, default=5, help="Number of candidates for Method 2")
    args = parser.parse_args()

    save_base = args.save_name or f"method_comparison_{args.model.replace('-', '_')}_{args.split}_n{args.limit}"
    
    run_method_comparison(
        n=args.limit,
        split=args.split,
        dataset_path_hint=args.dataset_path,
        model=args.model,
        num_candidates=args.num_candidates,
        save_basename=save_base,
    )

