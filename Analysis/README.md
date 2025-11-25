# GPT-4o-mini Code Contests Analysis

This directory contains analysis scripts for evaluating GPT-4o-mini's performance on the Code Contests dataset.

## Scripts

### 1. `analyze_gpt4o_mini_results.py`
Performs comprehensive statistical analysis on the generated solutions.

**Features:**
- Syntax validation for all generated Python code
- Error detection and categorization
- Code complexity metrics (lines, characters, features)
- Difficulty and source distribution analysis
- Ground truth comparison (when available)
- Similarity scoring

**Output:**
- `analysis_outputs/analysis_report.txt` - Comprehensive text report
- `analysis_outputs/analyzed_results.csv` - Enhanced CSV with all metrics
- `analysis_outputs/success_examples.json` - Examples of successful generations
- `analysis_outputs/error_examples.json` - Examples of failed generations

**Usage:**
```bash
python Analysis/analyze_gpt4o_mini_results.py
```

### 2. `visualize_gpt4o_mini_results.py`
Creates visual representations of the analysis results.

**Features:**
- Overall summary dashboard
- Success rates by difficulty level
- Success rates by problem source
- Code complexity distributions
- Detailed statistics table

**Output:**
All visualizations saved to `analysis_outputs/visualizations/`:
- `overall_summary.png` - 4-panel overview
- `difficulty_analysis.png` - Performance by difficulty
- `source_analysis.png` - Performance by source platform
- `code_complexity.png` - Code metrics distributions
- `statistics_table.png` - Formatted statistics table

**Usage:**
```bash
python Analysis/visualize_gpt4o_mini_results.py
```

### 3. `test_execution_analysis.py`
**NEW!** Runs generated solutions against actual test cases and visualizes pass rates.

**Features:**
- Executes solutions against public and private test cases
- Calculates pass rates per problem
- Analyzes error types (runtime errors, timeouts, wrong outputs)
- Compares public vs private test performance
- Pass rate breakdown by difficulty and source

**Output:**
- `analysis_outputs/test_execution_report.txt` - Detailed test results
- `analysis_outputs/analyzed_results_with_tests.csv` - Data with test metrics
- `analysis_outputs/visualizations/test_execution_analysis.png` - Comprehensive test visualization

**Usage:**
```bash
python Analysis/test_execution_analysis.py
```

**Note:** Tests first 50 problems with max 5 test cases each for speed. Modify `MAX_PROBLEMS_TO_TEST` and `MAX_TESTS_PER_PROBLEM` constants to adjust.

## Analysis Results Summary

Based on the first 100 problems from the Code Contests training set:

### Key Findings

✅ **Syntax Validation: 99%**
- 99 out of 100 solutions have valid Python syntax
- 0 generation errors
- High consistency across different difficulty levels

🎯 **Functional Correctness: 48.7%**
- **NEW!** Tested 50 problems against actual unit tests
- 17/50 (34%) achieved 100% test pass rate
- 17/50 (34%) achieved partial pass (1-99%)
- 16/50 (32%) failed all tests
- Mean pass rate: 48.7%, Median: 36.7%

### Test Execution Results (50 problems sampled)

- **Total test cases executed:** 275 (104 public, 171 private)
- **Tests passed:** 133/275 (48.4%)
- **Public vs Private:** Nearly identical performance (~46% each)
- **Error breakdown:**
  - Wrong answers: 125 cases
  - Runtime errors: 3 problems
  - Timeouts: 2 problems

### Solution Characteristics

- **Average code length:** 27.5 lines
- **Common patterns:**
  - 90.9% use loops
  - 86.9% use conditionals
  - 63.6% import modules
  - 39.4% define functions
  - Only 1% use classes

### Performance by Difficulty

All difficulty levels achieved 97-100% success rate:
- **Competitive programming levels (A-J):** Excellent performance
- **General difficulty (EASY, MEDIUM, EXTERNAL):** 100% success
- **UNKNOWN_DIFFICULTY:** 97% success (32/33)

### Performance by Source

- **CODEFORCES:** 100% (61/61 problems) - Best performance
- **ATCODER:** 100% (10/10 problems)
- **HACKEREARTH:** 100% (10/10 problems)
- **CODECHEF:** 100% (6/6 problems)
- **AIZU:** 92.3% (12/13 problems) - One syntax error

## Running the Full Analysis Pipeline

To analyze a new results file:

1. **Run the main data generation:**
   ```bash
   python logits.py --limit 100 --split train --model gpt-4o-mini
   ```

2. **Analyze the results:**
   ```bash
   python Analysis/analyze_gpt4o_mini_results.py
   ```

3. **Generate visualizations:**
   ```bash
   python Analysis/visualize_gpt4o_mini_results.py
   ```

## Dependencies

The analysis scripts require:
- pandas
- numpy
- matplotlib
- seaborn

Install with:
```bash
pip install pandas numpy matplotlib seaborn
```

## Notes

- **Ground Truth Limitation:** The current dataset analysis found 0/100 problems with Python ground truth solutions. This is expected as the Code Contests dataset primarily contains C++ and other language solutions. Future analysis could incorporate test execution for validation.

- **Syntax vs. Correctness:** The 99% success rate refers to syntactically valid Python code. Functional correctness would require running the code against test cases.

- **Model Choice:** While the script defaults to `gpt-4-nano`, we used `gpt-4o-mini` as a fallback. Update the model parameter as needed.

## Future Enhancements

Potential improvements for deeper analysis:
1. Execute solutions against test cases to measure functional correctness
2. Compare generated solutions with correct solutions (when available)
3. Analyze common error patterns in failed generations
4. Benchmark execution time and memory usage
5. Perform CodeBLEU scoring against ground truth
6. Analyze prompt engineering effectiveness

## File Structure

```
Analysis/
├── analyze_gpt4o_mini_results.py    # Main analysis script
├── visualize_gpt4o_mini_results.py  # Visualization script
├── README.md                          # This file
└── analysis_outputs/                  # Generated outputs
    ├── analysis_report.txt
    ├── analyzed_results.csv
    ├── success_examples.json
    ├── error_examples.json
    └── visualizations/
        ├── overall_summary.png
        ├── difficulty_analysis.png
        ├── source_analysis.png
        ├── code_complexity.png
        └── statistics_table.png
```
