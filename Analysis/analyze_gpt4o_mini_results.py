"""
Analysis script for GPT-4o-mini Code Contests results.

This script analyzes the generated solutions from gpt-4o-mini on the Code Contests dataset,
providing insights into:
- Solution generation success rates
- Code quality metrics
- Problem difficulty distribution
- Language detection and syntax validation
- Comparison with ground truth solutions
"""

import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re
from collections import Counter


# Configuration
RESULTS_PATH = Path(__file__).parent.parent / "Results" / "gpt4o_mini_code_contests_train_head100.csv"
OUTPUT_DIR = Path(__file__).parent / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load the results CSV."""
    print(f"Loading data from {RESULTS_PATH}...")
    df = pd.read_csv(RESULTS_PATH)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def parse_json_field(field_value: Any) -> Any:
    """Safely parse JSON-serialized fields."""
    if pd.isna(field_value):
        return None
    if isinstance(field_value, str):
        try:
            return json.loads(field_value)
        except json.JSONDecodeError:
            return field_value
    return field_value


def detect_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Detect solutions that contain error messages."""
    df['has_error'] = df['gpt_4_nano_solution'].str.contains('# ERROR:', na=False)
    df['error_type'] = df['gpt_4_nano_solution'].apply(
        lambda x: x.split('# ERROR:')[1].strip() if isinstance(x, str) and '# ERROR:' in x else None
    )
    return df


def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """Check if code is valid Python syntax."""
    if not isinstance(code, str) or code.strip().startswith('# ERROR:'):
        return False, "Error in generation"
    
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg}"
    except Exception as e:
        return False, f"ParseError: {str(e)}"


def analyze_syntax_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Validate Python syntax for all generated solutions."""
    print("\nValidating Python syntax...")
    
    syntax_results = df['gpt_4_nano_solution'].apply(validate_python_syntax)
    df['is_valid_syntax'] = syntax_results.apply(lambda x: x[0])
    df['syntax_error'] = syntax_results.apply(lambda x: x[1])
    
    valid_count = df['is_valid_syntax'].sum()
    total_count = len(df)
    print(f"Valid Python syntax: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
    
    return df


def extract_solution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Extract statistics about the generated solutions."""
    print("\nExtracting solution statistics...")
    
    def get_code_stats(code: str) -> Dict[str, Any]:
        if not isinstance(code, str) or code.strip().startswith('# ERROR:'):
            return {
                'line_count': 0,
                'char_count': 0,
                'has_imports': False,
                'has_functions': False,
                'has_classes': False,
                'has_loops': False,
                'has_conditionals': False,
            }
        
        lines = code.split('\n')
        return {
            'line_count': len(lines),
            'char_count': len(code),
            'has_imports': bool(re.search(r'\bimport\b|\bfrom\b', code)),
            'has_functions': bool(re.search(r'\bdef\b', code)),
            'has_classes': bool(re.search(r'\bclass\b', code)),
            'has_loops': bool(re.search(r'\bfor\b|\bwhile\b', code)),
            'has_conditionals': bool(re.search(r'\bif\b', code)),
        }
    
    stats = df['gpt_4_nano_solution'].apply(get_code_stats)
    stats_df = pd.DataFrame(stats.tolist())
    
    for col in stats_df.columns:
        df[f'solution_{col}'] = stats_df[col]
    
    return df


def analyze_difficulty_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze problem difficulty distribution and success rates."""
    print("\nAnalyzing difficulty distribution...")
    
    # Parse difficulty field if it's a number (ClassLabel index)
    df['difficulty_parsed'] = df['difficulty'].apply(
        lambda x: int(x) if pd.notna(x) else -1
    )
    
    difficulty_names = [
        "UNKNOWN_DIFFICULTY", "EASY", "MEDIUM", "HARD", "HARDER", "HARDEST", "EXTERNAL",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V"
    ]
    
    df['difficulty_name'] = df['difficulty_parsed'].apply(
        lambda x: difficulty_names[x] if 0 <= x < len(difficulty_names) else "UNKNOWN"
    )
    
    difficulty_stats = df.groupby('difficulty_name').agg({
        'is_valid_syntax': ['count', 'sum', 'mean'],
        'has_error': 'sum'
    }).round(3)
    
    return difficulty_stats


def analyze_source_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze problem source distribution."""
    print("\nAnalyzing source distribution...")
    
    source_names = ["UNKNOWN_SOURCE", "CODECHEF", "CODEFORCES", "HACKEREARTH", "CODEJAM", "ATCODER", "AIZU"]
    
    df['source_parsed'] = df['source'].apply(lambda x: int(x) if pd.notna(x) else -1)
    df['source_name'] = df['source_parsed'].apply(
        lambda x: source_names[x] if 0 <= x < len(source_names) else "UNKNOWN"
    )
    
    source_stats = df.groupby('source_name').agg({
        'is_valid_syntax': ['count', 'sum', 'mean'],
        'has_error': 'sum'
    }).round(3)
    
    return source_stats


def compare_with_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Compare generated solutions with ground truth solutions."""
    print("\nComparing with ground truth solutions...")
    
    def extract_python_solution(solutions_field: Any) -> str | None:
        """Extract a Python solution from the solutions field."""
        if pd.isna(solutions_field):
            return None
        
        try:
            solutions = parse_json_field(solutions_field)
            if not solutions:
                return None
            
            # Solutions format: {'language': [...], 'solution': [...]}
            if isinstance(solutions, dict):
                languages = solutions.get('language', [])
                solution_codes = solutions.get('solution', [])
                
                # Look for Python solutions (language codes 1=PYTHON, 3=PYTHON3)
                for i, lang in enumerate(languages):
                    if lang in [1, 3] and i < len(solution_codes):
                        return solution_codes[i]
            
            return None
        except Exception:
            return None
    
    df['ground_truth_python'] = df['solutions'].apply(extract_python_solution)
    df['has_ground_truth'] = df['ground_truth_python'].notna()
    
    print(f"Problems with Python ground truth: {df['has_ground_truth'].sum()}/{len(df)}")
    
    return df


def calculate_similarity_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic similarity metrics between generated and ground truth."""
    print("\nCalculating similarity metrics...")
    
    def simple_similarity(gen: str, truth: str) -> float:
        """Simple character-based similarity."""
        if not isinstance(gen, str) or not isinstance(truth, str):
            return 0.0
        if gen.strip().startswith('# ERROR:'):
            return 0.0
        
        gen_clean = gen.strip().lower()
        truth_clean = truth.strip().lower()
        
        if not gen_clean or not truth_clean:
            return 0.0
        
        # Simple Jaccard similarity on character 3-grams
        def get_ngrams(s: str, n: int = 3) -> set:
            return set(s[i:i+n] for i in range(len(s) - n + 1))
        
        gen_ngrams = get_ngrams(gen_clean)
        truth_ngrams = get_ngrams(truth_clean)
        
        if not gen_ngrams or not truth_ngrams:
            return 0.0
        
        intersection = len(gen_ngrams & truth_ngrams)
        union = len(gen_ngrams | truth_ngrams)
        
        return intersection / union if union > 0 else 0.0
    
    # Only calculate for rows with ground truth
    mask = df['has_ground_truth']
    df.loc[mask, 'similarity_score'] = df[mask].apply(
        lambda row: simple_similarity(row['gpt_4_nano_solution'], row['ground_truth_python']),
        axis=1
    )
    
    return df


def generate_summary_report(df: pd.DataFrame, difficulty_stats, source_stats) -> str:
    """Generate a comprehensive summary report."""
    total = len(df)
    errors = df['has_error'].sum()
    valid_syntax = df['is_valid_syntax'].sum()
    has_ground_truth = df['has_ground_truth'].sum()
    
    report = []
    report.append("=" * 80)
    report.append("GPT-4o-mini CODE CONTESTS ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall Statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total problems processed: {total}")
    report.append(f"Generation errors: {errors} ({100*errors/total:.1f}%)")
    report.append(f"Valid Python syntax: {valid_syntax}/{total} ({100*valid_syntax/total:.1f}%)")
    report.append(f"Problems with ground truth: {has_ground_truth}/{total} ({100*has_ground_truth/total:.1f}%)")
    report.append("")
    
    # Solution Characteristics
    report.append("SOLUTION CHARACTERISTICS")
    report.append("-" * 80)
    valid_solutions = df[df['is_valid_syntax']]
    if len(valid_solutions) > 0:
        report.append(f"Average lines of code: {valid_solutions['solution_line_count'].mean():.1f}")
        report.append(f"Average characters: {valid_solutions['solution_char_count'].mean():.1f}")
        report.append(f"Solutions with imports: {valid_solutions['solution_has_imports'].sum()} ({100*valid_solutions['solution_has_imports'].mean():.1f}%)")
        report.append(f"Solutions with functions: {valid_solutions['solution_has_functions'].sum()} ({100*valid_solutions['solution_has_functions'].mean():.1f}%)")
        report.append(f"Solutions with classes: {valid_solutions['solution_has_classes'].sum()} ({100*valid_solutions['solution_has_classes'].mean():.1f}%)")
        report.append(f"Solutions with loops: {valid_solutions['solution_has_loops'].sum()} ({100*valid_solutions['solution_has_loops'].mean():.1f}%)")
        report.append(f"Solutions with conditionals: {valid_solutions['solution_has_conditionals'].sum()} ({100*valid_solutions['solution_has_conditionals'].mean():.1f}%)")
    report.append("")
    
    # Difficulty Distribution
    report.append("DIFFICULTY DISTRIBUTION")
    report.append("-" * 80)
    report.append(str(difficulty_stats))
    report.append("")
    
    # Source Distribution
    report.append("SOURCE DISTRIBUTION")
    report.append("-" * 80)
    report.append(str(source_stats))
    report.append("")
    
    # Similarity Metrics
    if 'similarity_score' in df.columns:
        similarity_data = df[df['similarity_score'].notna()]
        if len(similarity_data) > 0:
            report.append("SIMILARITY TO GROUND TRUTH")
            report.append("-" * 80)
            report.append(f"Mean similarity: {similarity_data['similarity_score'].mean():.3f}")
            report.append(f"Median similarity: {similarity_data['similarity_score'].median():.3f}")
            report.append(f"Std similarity: {similarity_data['similarity_score'].std():.3f}")
            report.append("")
    
    # Error Analysis
    if errors > 0:
        report.append("ERROR ANALYSIS")
        report.append("-" * 80)
        error_types = df[df['has_error']]['error_type'].value_counts()
        for error, count in error_types.items():
            if error:
                report.append(f"  {error[:100]}: {count}")
        report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)


def save_detailed_results(df: pd.DataFrame):
    """Save detailed analysis results."""
    print("\nSaving detailed results...")
    
    # Save enhanced DataFrame
    output_csv = OUTPUT_DIR / "analyzed_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved enhanced DataFrame to {output_csv}")
    
    # Save examples of successful and failed generations
    if df['is_valid_syntax'].sum() > 0:
        success_examples = df[df['is_valid_syntax']].head(5)[
            ['name', 'difficulty_name', 'source_name', 'gpt_4_nano_solution']
        ]
        success_path = OUTPUT_DIR / "success_examples.json"
        success_examples.to_json(success_path, orient='records', indent=2)
        print(f"Saved success examples to {success_path}")
    
    if df['has_error'].sum() > 0:
        error_examples = df[df['has_error']].head(5)[
            ['name', 'difficulty_name', 'source_name', 'gpt_4_nano_solution', 'error_type']
        ]
        error_path = OUTPUT_DIR / "error_examples.json"
        error_examples.to_json(error_path, orient='records', indent=2)
        print(f"Saved error examples to {error_path}")


def main():
    """Main analysis pipeline."""
    print("Starting GPT-4o-mini Code Contests Analysis")
    print("=" * 80)
    
    # Load data
    df = load_data()
    
    # Run analyses
    df = detect_errors(df)
    df = analyze_syntax_validity(df)
    df = extract_solution_stats(df)
    difficulty_stats = analyze_difficulty_distribution(df)
    source_stats = analyze_source_distribution(df)
    df = compare_with_ground_truth(df)
    df = calculate_similarity_metrics(df)
    
    # Generate report
    report = generate_summary_report(df, difficulty_stats, source_stats)
    print("\n" + report)
    
    # Save results
    report_path = OUTPUT_DIR / "analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")
    
    save_detailed_results(df)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
