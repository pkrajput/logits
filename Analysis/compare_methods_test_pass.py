"""
Compare Method 1 (Direct) vs Method 2 (Selection) on test pass rates.
Loads results from logits.py experiment output and runs test execution for both solutions.
Generates summary statistics and prints comparison table.
"""
import pandas as pd
import json
import os
from pathlib import Path
import sys

# Import test execution utilities from previous analysis
sys.path.append(str(Path(__file__).parent))
from test_execution_analysis import run_python_solution, parse_test_data

RESULTS_DIR = Path(__file__).parent.parent / "Results"
DEFAULT_RESULTS_FILE = RESULTS_DIR / "experiment_100.parquet"


def load_results(results_path=DEFAULT_RESULTS_FILE):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    return pd.read_parquet(results_path)


def evaluate_solution(row, solution_col):
    """
    Run all available public tests for a given solution.
    Returns: (num_passed, num_total)
    """
    code = row[solution_col]
    public_tests = row.get("public_tests", {}) or {}
    test_data = parse_test_data(public_tests)
    inputs = test_data.get('input', [])
    outputs = test_data.get('output', [])
    num_passed = 0
    for test_in, expected_out in zip(inputs, outputs):
        success, actual_out, error = run_python_solution(code, str(test_in), timeout=2)
        # Consider a test passed if output matches expected
        if success and actual_out.strip() == str(expected_out).strip():
            num_passed += 1
    return num_passed, len(inputs)


def main():
    df = load_results()
    print(f"Loaded {len(df)} problems from {DEFAULT_RESULTS_FILE}")

    method1_results = []
    method2_results = []

    for idx, row in df.iterrows():
        # Method 1
        try:
            m1_passed, m1_total = evaluate_solution(row, "method1_solution")
        except Exception:
            m1_passed, m1_total = 0, 0
        method1_results.append((m1_passed, m1_total))

        # Method 2
        try:
            m2_passed, m2_total = evaluate_solution(row, "method2_solution")
        except Exception:
            m2_passed, m2_total = 0, 0
        method2_results.append((m2_passed, m2_total))

        if (idx + 1) % 10 == 0:
            print(f"Evaluated {idx+1}/{len(df)} problems...")

    df["method1_passed"] = [x[0] for x in method1_results]
    df["method1_total"] = [x[1] for x in method1_results]
    df["method2_passed"] = [x[0] for x in method2_results]
    df["method2_total"] = [x[1] for x in method2_results]

    # Summary statistics
    m1_total_tests = df["method1_total"].sum()
    m2_total_tests = df["method2_total"].sum()
    m1_total_passed = df["method1_passed"].sum()
    m2_total_passed = df["method2_passed"].sum()

    print("\nTest Pass Rate Comparison:")
    print("=========================")
    print(f"Method 1 (Direct): {m1_total_passed}/{m1_total_tests} passed ({100*m1_total_passed/m1_total_tests:.2f}%)")
    print(f"Method 2 (Selection): {m2_total_passed}/{m2_total_tests} passed ({100*m2_total_passed/m2_total_tests:.2f}%)")

    # Per-problem comparison
    better_m2 = ((df["method2_passed"] > df["method1_passed"]).sum())
    better_m1 = ((df["method1_passed"] > df["method2_passed"]).sum())
    equal = ((df["method1_passed"] == df["method2_passed"]).sum())
    print(f"\nProblems where Method 2 passed more tests: {better_m2}/{len(df)}")
    print(f"Problems where Method 1 passed more tests: {better_m1}/{len(df)}")
    print(f"Problems where both methods passed equal tests: {equal}/{len(df)}")

    # Save comparison table
    out_csv = RESULTS_DIR / "method_comparison_test_pass.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nDetailed results saved to: {out_csv}")

if __name__ == "__main__":
    main()
