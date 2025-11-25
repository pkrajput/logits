"""
Quick reference script for exploring GPT-4o-mini Code Contests results.

Use this for interactive data exploration and custom queries.
"""

import pandas as pd
from pathlib import Path

# Load the analyzed results
ANALYSIS_DIR = Path(__file__).parent / "analysis_outputs"
df = pd.read_csv(ANALYSIS_DIR / "analyzed_results.csv")

print("=" * 80)
print("GPT-4o-mini Code Contests Results - Quick Reference")
print("=" * 80)
print(f"\nDataset: {len(df)} problems")
print(f"Columns: {len(df.columns)}")
print(f"\nAvailable columns:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

print("\n" + "=" * 80)
print("Quick Stats")
print("=" * 80)
print(f"Valid syntax: {df['is_valid_syntax'].sum()}/{len(df)} ({100*df['is_valid_syntax'].mean():.1f}%)")
print(f"Has errors: {df['has_error'].sum()}/{len(df)} ({100*df['has_error'].mean():.1f}%)")
print(f"Avg lines of code: {df[df['is_valid_syntax']]['solution_line_count'].mean():.1f}")

print("\n" + "=" * 80)
print("Example Queries")
print("=" * 80)

# Example 1: Find longest solution
print("\n1. Longest solution:")
longest = df.loc[df['solution_line_count'].idxmax()]
print(f"   Problem: {longest['name']}")
print(f"   Lines: {longest['solution_line_count']}")
print(f"   Difficulty: {longest['difficulty_name']}")

# Example 2: Find problems without imports
print("\n2. Solutions without imports:")
no_imports = df[~df['solution_has_imports'] & df['is_valid_syntax']]
print(f"   Count: {len(no_imports)}")
if len(no_imports) > 0:
    print(f"   Examples: {', '.join(no_imports['name'].head(3).tolist())}")

# Example 3: Most complex solutions (functions + classes + imports)
print("\n3. Most complex solutions (with functions, loops, and conditionals):")
complex_mask = (df['solution_has_functions'] & 
                df['solution_has_loops'] & 
                df['solution_has_conditionals'] &
                df['is_valid_syntax'])
complex_solutions = df[complex_mask].nlargest(3, 'solution_line_count')
for idx, row in complex_solutions.iterrows():
    print(f"   - {row['name']} ({row['solution_line_count']} lines, {row['difficulty_name']})")

# Example 4: Problems by difficulty
print("\n4. Top difficulty levels:")
difficulty_counts = df['difficulty_name'].value_counts().head(5)
for diff, count in difficulty_counts.items():
    print(f"   {diff}: {count}")

# Example 5: Failed syntax validation
print("\n5. Invalid syntax cases:")
invalid = df[~df['is_valid_syntax']]
if len(invalid) > 0:
    for idx, row in invalid.iterrows():
        print(f"   - {row['name']}: {row['syntax_error'][:60]}...")
else:
    print("   None found!")

print("\n" + "=" * 80)
print("Interactive Usage")
print("=" * 80)
print("""
To explore interactively, run:

    from pathlib import Path
    import pandas as pd
    
    df = pd.read_csv(Path('Analysis/analysis_outputs/analyzed_results.csv'))
    
Then use pandas commands:
    
    # View specific problem
    df[df['name'] == 'problem_name']
    
    # Filter by difficulty
    df[df['difficulty_name'] == 'HARD']
    
    # Filter by source
    df[df['source_name'] == 'CODEFORCES']
    
    # View solution code
    print(df.iloc[0]['gpt_4_nano_solution'])
    
    # Custom analysis
    df.groupby('difficulty_name')['solution_line_count'].describe()
""")

print("\n" + "=" * 80)
print("Useful Columns for Analysis")
print("=" * 80)
print("""
Core Info:
  - name: Problem identifier
  - description: Problem description text
  - difficulty_name: Difficulty level
  - source_name: Platform (CODEFORCES, ATCODER, etc.)

Solution Quality:
  - gpt_4_nano_solution: Generated code
  - is_valid_syntax: Boolean, code is valid Python
  - syntax_error: Error message if invalid
  - has_error: Boolean, generation failed

Code Metrics:
  - solution_line_count: Number of lines
  - solution_char_count: Number of characters
  - solution_has_imports: Boolean
  - solution_has_functions: Boolean
  - solution_has_classes: Boolean
  - solution_has_loops: Boolean
  - solution_has_conditionals: Boolean

Ground Truth:
  - ground_truth_python: Reference solution (if available)
  - has_ground_truth: Boolean
  - similarity_score: Similarity to ground truth (if available)
""")

print("=" * 80)
