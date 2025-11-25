"""
Test Execution Analysis for GPT-4o-mini Code Contests results.

This script:
1. Runs generated solutions against public and private test cases
2. Calculates pass rates
3. Generates visualizations of test performance
"""

import pandas as pd
import numpy as np
import ast
import json
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import time

# Configuration
RESULTS_PATH = Path(__file__).parent.parent / "Results" / "gpt4o_mini_code_contests_train_head100.parquet"
ANALYSIS_DIR = Path(__file__).parent / "analysis_outputs"
OUTPUT_DIR = ANALYSIS_DIR / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Test execution timeout
TEST_TIMEOUT = 2  # Reduced from 3 to speed up
MAX_TESTS_PER_PROBLEM = 5  # Reduced from 10 to speed up analysis
MAX_PROBLEMS_TO_TEST = 50  # Only test first 50 problems for speed


def parse_test_data(test_field) -> Dict[str, List[str]]:
    """Parse test data from parquet dict structure."""
    if test_field is None or (isinstance(test_field, float) and pd.isna(test_field)):
        return {'input': [], 'output': []}
    
    try:
        # When loaded from parquet, test_field should be a dict
        if isinstance(test_field, dict):
            inputs = test_field.get('input', [])
            outputs = test_field.get('output', [])
            
            # Convert numpy arrays to lists if needed
            if hasattr(inputs, 'tolist'):
                inputs = inputs.tolist()
            if hasattr(outputs, 'tolist'):
                outputs = outputs.tolist()
            
            # Ensure we have lists
            if not isinstance(inputs, list):
                inputs = [inputs] if inputs else []
            if not isinstance(outputs, list):
                outputs = [outputs] if outputs else []
            
            return {'input': inputs, 'output': outputs}
        
        return {'input': [], 'output': []}
    except Exception as e:
        # If parsing fails, return empty
        return {'input': [], 'output': []}


def run_python_solution(code: str, test_input: str, timeout: int = TEST_TIMEOUT) -> Tuple[bool, str, str]:
    """
    Run a Python solution with given input and return success status and output.
    
    Returns:
        (success, output, error_message)
    """
    if not isinstance(code, str) or code.strip().startswith('# ERROR:'):
        return False, "", "Code contains generation error"
    
    try:
        # Create temporary file for the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Run the code with the test input
            result = subprocess.run(
                ['python3', temp_file],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = result.stdout.strip()
            
            # Check if there was a runtime error
            if result.returncode != 0:
                error_msg = result.stderr.strip()[:200]
                return False, output, f"Runtime error: {error_msg}"
            
            return True, output, ""
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
                
    except subprocess.TimeoutExpired:
        return False, "", "Timeout exceeded"
    except Exception as e:
        return False, "", f"Execution error: {str(e)[:200]}"


def test_solution(code: str, public_tests: Dict, private_tests: Dict) -> Dict[str, Any]:
    """
    Test a solution against public and private test cases.
    
    Returns dict with test results.
    """
    results = {
        'public_total': 0,
        'public_passed': 0,
        'public_failed': 0,
        'public_errors': 0,
        'private_total': 0,
        'private_passed': 0,
        'private_failed': 0,
        'private_errors': 0,
        'total_tests': 0,
        'total_passed': 0,
        'pass_rate': 0.0,
        'public_pass_rate': 0.0,
        'private_pass_rate': 0.0,
        'has_runtime_error': False,
        'has_timeout': False,
    }
    
    # Run public tests
    public_inputs = public_tests.get('input', [])
    public_outputs = public_tests.get('output', [])
    
    # Limit number of tests
    if len(public_inputs) > MAX_TESTS_PER_PROBLEM:
        public_inputs = public_inputs[:MAX_TESTS_PER_PROBLEM]
        public_outputs = public_outputs[:MAX_TESTS_PER_PROBLEM]
    
    for test_in, expected_out in zip(public_inputs, public_outputs):
        results['public_total'] += 1
        
        success, actual_out, error = run_python_solution(code, str(test_in))
        
        if not success:
            results['public_errors'] += 1
            if 'Timeout' in error:
                results['has_timeout'] = True
            elif 'Runtime error' in error:
                results['has_runtime_error'] = True
        elif actual_out.strip() == str(expected_out).strip():
            results['public_passed'] += 1
        else:
            results['public_failed'] += 1
    
    # Run private tests
    private_inputs = private_tests.get('input', [])
    private_outputs = private_tests.get('output', [])
    
    # Limit number of tests
    if len(private_inputs) > MAX_TESTS_PER_PROBLEM:
        private_inputs = private_inputs[:MAX_TESTS_PER_PROBLEM]
        private_outputs = private_outputs[:MAX_TESTS_PER_PROBLEM]
    
    for test_in, expected_out in zip(private_inputs, private_outputs):
        results['private_total'] += 1
        
        success, actual_out, error = run_python_solution(code, str(test_in))
        
        if not success:
            results['private_errors'] += 1
            if 'Timeout' in error:
                results['has_timeout'] = True
            elif 'Runtime error' in error:
                results['has_runtime_error'] = True
        elif actual_out.strip() == str(expected_out).strip():
            results['private_passed'] += 1
        else:
            results['private_failed'] += 1
    
    # Calculate totals and rates
    results['total_tests'] = results['public_total'] + results['private_total']
    results['total_passed'] = results['public_passed'] + results['private_passed']
    
    if results['total_tests'] > 0:
        results['pass_rate'] = results['total_passed'] / results['total_tests']
    
    if results['public_total'] > 0:
        results['public_pass_rate'] = results['public_passed'] / results['public_total']
    
    if results['private_total'] > 0:
        results['private_pass_rate'] = results['private_passed'] / results['private_total']
    
    return results


def run_all_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run tests for all solutions in the dataframe."""
    print("Running test execution analysis...")
    print(f"Testing up to {MAX_PROBLEMS_TO_TEST} solutions (limited for speed)...")
    
    test_results = []
    
    # Limit to first N problems
    problems_to_test = min(MAX_PROBLEMS_TO_TEST, len(df))
    
    for idx in range(problems_to_test):
        row = df.iloc[idx]
        print(f"Testing {idx + 1}/{problems_to_test}: {row['name'][:50]}...", end=' ')
        
        # Parse test data
        public_tests = parse_test_data(row['public_tests'])
        private_tests = parse_test_data(row['private_tests'])
        
        # Skip if no tests available
        if not public_tests['input'] and not private_tests['input']:
            print("No tests available")
            test_results.append({
                'public_total': 0,
                'public_passed': 0,
                'public_failed': 0,
                'public_errors': 0,
                'private_total': 0,
                'private_passed': 0,
                'private_failed': 0,
                'private_errors': 0,
                'total_tests': 0,
                'total_passed': 0,
                'pass_rate': 0.0,
                'public_pass_rate': 0.0,
                'private_pass_rate': 0.0,
                'has_runtime_error': False,
                'has_timeout': False,
            })
            continue
        
        # Run tests
        code = row['gpt_4_nano_solution']
        results = test_solution(code, public_tests, private_tests)
        test_results.append(results)
        
        print(f"Pass rate: {results['pass_rate']:.1%} ({results['total_passed']}/{results['total_tests']})")
    
    # For remaining problems not tested, add empty results
    for idx in range(problems_to_test, len(df)):
        test_results.append({
            'public_total': 0,
            'public_passed': 0,
            'public_failed': 0,
            'public_errors': 0,
            'private_total': 0,
            'private_passed': 0,
            'private_failed': 0,
            'private_errors': 0,
            'total_tests': 0,
            'total_passed': 0,
            'pass_rate': 0.0,
            'public_pass_rate': 0.0,
            'private_pass_rate': 0.0,
            'has_runtime_error': False,
            'has_timeout': False,
        })
    
    # Add results to dataframe
    test_df = pd.DataFrame(test_results)
    for col in test_df.columns:
        df[f'test_{col}'] = test_df[col]
    
    return df


def generate_test_visualizations(df: pd.DataFrame):
    """Generate visualizations for test execution results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\nGenerating test execution visualizations...")
    
    # Filter to only problems with tests
    df_with_tests = df[df['test_total_tests'] > 0].copy()
    
    if len(df_with_tests) == 0:
        print("No problems with test cases found!")
        return
    
    print(f"Analyzing {len(df_with_tests)} problems with test cases")
    
    # Add difficulty_name and source_name if not present
    if 'difficulty_name' not in df_with_tests.columns:
        difficulty_names = [
            "UNKNOWN_DIFFICULTY", "EASY", "MEDIUM", "HARD", "HARDER", "HARDEST", "EXTERNAL",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"
        ]
        df_with_tests['difficulty_parsed'] = df_with_tests['difficulty'].apply(lambda x: int(x) if pd.notna(x) else -1)
        df_with_tests['difficulty_name'] = df_with_tests['difficulty_parsed'].apply(
            lambda x: difficulty_names[x] if 0 <= x < len(difficulty_names) else "UNKNOWN"
        )
    
    if 'source_name' not in df_with_tests.columns:
        source_names = ["UNKNOWN_SOURCE", "CODECHEF", "CODEFORCES", "HACKEREARTH", "CODEJAM", "ATCODER", "AIZU"]
        df_with_tests['source_parsed'] = df_with_tests['source'].apply(lambda x: int(x) if pd.notna(x) else -1)
        df_with_tests['source_name'] = df_with_tests['source_parsed'].apply(
            lambda x: source_names[x] if 0 <= x < len(source_names) else "UNKNOWN"
        )
    
    # Create main visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall pass rate distribution
    ax1 = fig.add_subplot(gs[0, :2])
    pass_rates = df_with_tests['test_pass_rate'] * 100
    ax1.hist(pass_rates, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(pass_rates.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {pass_rates.mean():.1f}%')
    ax1.axvline(pass_rates.median(), color='green', linestyle='--', linewidth=2,
                label=f'Median: {pass_rates.median():.1f}%')
    ax1.set_xlabel('Pass Rate (%)', fontsize=11)
    ax1.set_ylabel('Number of Problems', fontsize=11)
    ax1.set_title('Distribution of Test Pass Rates', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Pass rate summary pie chart
    ax2 = fig.add_subplot(gs[0, 2])
    perfect = (df_with_tests['test_pass_rate'] == 1.0).sum()
    partial = ((df_with_tests['test_pass_rate'] > 0) & (df_with_tests['test_pass_rate'] < 1.0)).sum()
    zero = (df_with_tests['test_pass_rate'] == 0.0).sum()
    
    sizes = [perfect, partial, zero]
    labels = [f'100% Pass\n({perfect})', f'Partial Pass\n({partial})', f'0% Pass\n({zero})']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Pass Rate Categories', fontsize=13, fontweight='bold')
    
    # 3. Public vs Private test pass rates
    ax3 = fig.add_subplot(gs[1, 0])
    public_rates = df_with_tests[df_with_tests['test_public_total'] > 0]['test_public_pass_rate'] * 100
    private_rates = df_with_tests[df_with_tests['test_private_total'] > 0]['test_private_pass_rate'] * 100
    
    bp = ax3.boxplot([public_rates, private_rates], labels=['Public Tests', 'Private Tests'],
                      patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax3.set_ylabel('Pass Rate (%)', fontsize=11)
    ax3.set_title('Public vs Private Test Performance', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Pass rate by difficulty
    ax4 = fig.add_subplot(gs[1, 1:])
    difficulty_stats = df_with_tests.groupby('difficulty_name').agg({
        'test_pass_rate': ['mean', 'count']
    }).reset_index()
    difficulty_stats.columns = ['difficulty', 'mean_pass_rate', 'count']
    difficulty_stats = difficulty_stats[difficulty_stats['count'] >= 2]  # At least 2 problems
    difficulty_stats = difficulty_stats.sort_values('mean_pass_rate', ascending=False).head(12)
    
    bars = ax4.barh(range(len(difficulty_stats)), difficulty_stats['mean_pass_rate'] * 100)
    ax4.set_yticks(range(len(difficulty_stats)))
    ax4.set_yticklabels([f"{d} (n={c})" for d, c in 
                         zip(difficulty_stats['difficulty'], difficulty_stats['count'])])
    ax4.set_xlabel('Mean Pass Rate (%)', fontsize=11)
    ax4.set_title('Pass Rate by Difficulty Level', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # Color bars by performance
    for i, (bar, rate) in enumerate(zip(bars, difficulty_stats['mean_pass_rate'])):
        if rate > 0.8:
            bar.set_color('#2ecc71')
        elif rate > 0.5:
            bar.set_color('#f39c12')
        else:
            bar.set_color('#e74c3c')
    
    # 5. Pass rate by source
    ax5 = fig.add_subplot(gs[2, 0])
    source_stats = df_with_tests.groupby('source_name').agg({
        'test_pass_rate': ['mean', 'count']
    }).reset_index()
    source_stats.columns = ['source', 'mean_pass_rate', 'count']
    source_stats = source_stats.sort_values('mean_pass_rate', ascending=True)
    
    bars = ax5.barh(range(len(source_stats)), source_stats['mean_pass_rate'] * 100)
    ax5.set_yticks(range(len(source_stats)))
    ax5.set_yticklabels([f"{s} (n={c})" for s, c in 
                         zip(source_stats['source'], source_stats['count'])])
    ax5.set_xlabel('Mean Pass Rate (%)', fontsize=11)
    ax5.set_title('Pass Rate by Source', fontsize=12, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    
    # Color bars
    for bar, rate in zip(bars, source_stats['mean_pass_rate']):
        if rate > 0.8:
            bar.set_color('#2ecc71')
        elif rate > 0.5:
            bar.set_color('#f39c12')
        else:
            bar.set_color('#e74c3c')
    
    # 6. Error analysis
    ax6 = fig.add_subplot(gs[2, 1])
    error_stats = {
        'Runtime Error': df_with_tests['test_has_runtime_error'].sum(),
        'Timeout': df_with_tests['test_has_timeout'].sum(),
        'Wrong Output': (df_with_tests['test_total_tests'] - 
                        df_with_tests['test_total_passed'] - 
                        df_with_tests['test_public_errors'] - 
                        df_with_tests['test_private_errors']).sum(),
    }
    
    ax6.bar(error_stats.keys(), error_stats.values(), color=['#e74c3c', '#f39c12', '#3498db'])
    ax6.set_ylabel('Number of Occurrences', fontsize=11)
    ax6.set_title('Error Type Distribution', fontsize=12, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    for i, (k, v) in enumerate(error_stats.items()):
        ax6.text(i, v + 1, str(int(v)), ha='center', fontweight='bold')
    
    # 7. Test count statistics
    ax7 = fig.add_subplot(gs[2, 2])
    test_count_stats = {
        'Public Tests': df_with_tests['test_public_total'].sum(),
        'Private Tests': df_with_tests['test_private_total'].sum(),
    }
    
    ax7.bar(test_count_stats.keys(), test_count_stats.values(), 
            color=['lightblue', 'lightcoral'], alpha=0.7)
    ax7.set_ylabel('Total Test Cases', fontsize=11)
    ax7.set_title('Test Case Distribution', fontsize=12, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    for i, (k, v) in enumerate(test_count_stats.items()):
        ax7.text(i, v + 50, f'{int(v)}\n({v/df_with_tests["test_total_tests"].sum()*100:.1f}%)', 
                ha='center', fontweight='bold')
    
    plt.suptitle('GPT-4o-mini Code Contests: Test Execution Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(OUTPUT_DIR / 'test_execution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {OUTPUT_DIR / 'test_execution_analysis.png'}")
    plt.close()


def generate_summary_report(df: pd.DataFrame) -> str:
    """Generate a text summary of test execution results."""
    df_with_tests = df[df['test_total_tests'] > 0]
    
    if len(df_with_tests) == 0:
        return "No test cases found in dataset."
    
    report = []
    report.append("=" * 80)
    report.append("TEST EXECUTION ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Problems with tests: {len(df_with_tests)}/{len(df)}")
    report.append(f"Total test cases executed: {df_with_tests['test_total_tests'].sum():.0f}")
    report.append(f"  - Public tests: {df_with_tests['test_public_total'].sum():.0f}")
    report.append(f"  - Private tests: {df_with_tests['test_private_total'].sum():.0f}")
    report.append(f"Total tests passed: {df_with_tests['test_total_passed'].sum():.0f}")
    report.append(f"Overall pass rate: {df_with_tests['test_pass_rate'].mean():.1%}")
    report.append("")
    
    # Pass rate distribution
    report.append("PASS RATE DISTRIBUTION")
    report.append("-" * 80)
    perfect = (df_with_tests['test_pass_rate'] == 1.0).sum()
    partial = ((df_with_tests['test_pass_rate'] > 0) & (df_with_tests['test_pass_rate'] < 1.0)).sum()
    zero = (df_with_tests['test_pass_rate'] == 0.0).sum()
    
    report.append(f"100% pass rate: {perfect}/{len(df_with_tests)} ({100*perfect/len(df_with_tests):.1f}%)")
    report.append(f"Partial pass (1-99%): {partial}/{len(df_with_tests)} ({100*partial/len(df_with_tests):.1f}%)")
    report.append(f"0% pass rate: {zero}/{len(df_with_tests)} ({100*zero/len(df_with_tests):.1f}%)")
    report.append(f"Mean pass rate: {df_with_tests['test_pass_rate'].mean():.1%}")
    report.append(f"Median pass rate: {df_with_tests['test_pass_rate'].median():.1%}")
    report.append("")
    
    # Public vs Private
    report.append("PUBLIC VS PRIVATE TESTS")
    report.append("-" * 80)
    public_df = df_with_tests[df_with_tests['test_public_total'] > 0]
    private_df = df_with_tests[df_with_tests['test_private_total'] > 0]
    
    report.append(f"Public test mean pass rate: {public_df['test_public_pass_rate'].mean():.1%}")
    report.append(f"Private test mean pass rate: {private_df['test_private_pass_rate'].mean():.1%}")
    report.append("")
    
    # Error analysis
    report.append("ERROR ANALYSIS")
    report.append("-" * 80)
    report.append(f"Runtime errors: {df_with_tests['test_has_runtime_error'].sum()}")
    report.append(f"Timeouts: {df_with_tests['test_has_timeout'].sum()}")
    total_failed = (df_with_tests['test_total_tests'] - df_with_tests['test_total_passed']).sum()
    total_errors = (df_with_tests['test_public_errors'] + df_with_tests['test_private_errors']).sum()
    report.append(f"Wrong answers: {total_failed - total_errors:.0f}")
    report.append("")
    
    # Top performers
    report.append("TOP 5 PERFORMERS")
    report.append("-" * 80)
    top_5 = df_with_tests.nlargest(5, 'test_pass_rate')[['name', 'test_pass_rate', 'test_total_passed', 'test_total_tests']]
    for _, row in top_5.iterrows():
        report.append(f"  {row['name'][:50]}: {row['test_pass_rate']:.1%} ({row['test_total_passed']:.0f}/{row['test_total_tests']:.0f})")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def main():
    """Main execution pipeline."""
    print("=" * 80)
    print("GPT-4o-mini Code Contests: Test Execution Analysis")
    print("=" * 80)
    
    # Load data from parquet (preserves dict structures)
    print(f"\nLoading data from {RESULTS_PATH}...")
    df = pd.read_parquet(RESULTS_PATH, engine='pyarrow')
    print(f"Loaded {len(df)} problems")
    
    # Run tests
    start_time = time.time()
    df = run_all_tests(df)
    elapsed = time.time() - start_time
    print(f"\nTest execution completed in {elapsed:.1f} seconds")
    
    # Generate visualizations
    generate_test_visualizations(df)
    
    # Generate report
    report = generate_summary_report(df)
    print("\n" + report)
    
    # Save enhanced dataset
    output_csv = ANALYSIS_DIR / "analyzed_results_with_tests.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved enhanced dataset to {output_csv}")
    
    # Save report
    report_path = ANALYSIS_DIR / "test_execution_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved report to {report_path}")
    
    print("\n" + "=" * 80)
    print("Test execution analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
