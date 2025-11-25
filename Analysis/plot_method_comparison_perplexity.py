"""
Plot Method Comparison: Perplexity vs. Test Pass Rate

- Green: Baseline (Method 1 direct)
- Orange: Selected candidate from Method 2 (lowest perplexity)
- Blue: All Method 2 candidates

X axis: Normalized perplexity
Y axis: Percent of public tests passed
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
from test_execution_analysis import run_python_solution, parse_test_data

RESULTS_DIR = Path(__file__).parent.parent / "Results"
RESULTS_FILE = RESULTS_DIR / "experiment_100.parquet"

# Helper to extract candidate info from metadata

def extract_candidates(row):
    meta = row.get('method2_metadata')
    if isinstance(meta, str):
        meta = json.loads(meta)
    candidates = meta.get('all_candidates', [])
    return candidates


def percent_passed(code, public_tests):
    test_data = parse_test_data(public_tests)
    inputs = test_data.get('input', [])
    outputs = test_data.get('output', [])
    if not inputs or not outputs:
        return 0.0
    num_passed = 0
    for test_in, expected_out in zip(inputs, outputs):
        success, actual_out, error = run_python_solution(code, str(test_in), timeout=2)
        if success and actual_out.strip() == str(expected_out).strip():
            num_passed += 1
    return num_passed / len(inputs) if len(inputs) > 0 else 0.0

def remove_outliers_iqr(perplexities, multiplier=1.5):
    """Remove outliers using IQR method"""
    if len(perplexities) < 4:
        return perplexities
    
    q1 = np.percentile(perplexities, 25)
    q3 = np.percentile(perplexities, 75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return [p for p in perplexities if lower_bound <= p <= upper_bound]

def main():
    df = pd.read_parquet(RESULTS_FILE)

    baseline_x = []
    baseline_y = []
    selected_x = []
    selected_y = []
    candidate_x = []
    candidate_y = []
    
    # Track selected perplexities separately for global normalization
    selected_perplexities_raw = []
    selected_data = []
    
    outliers_removed = 0
    total_points = 0
    
    print("Processing problems with grouped normalization and outlier removal...")
    
    # First pass: collect selected perplexities for separate global normalization
    for idx, row in df.iterrows():
        meta2 = row['method2_metadata']
        if isinstance(meta2, str):
            meta2 = json.loads(meta2)
        
        selected_perplexity = meta2.get('selected_perplexity', None)
        if selected_perplexity is not None:
            selected_perplexities_raw.append(selected_perplexity)
            selected_data.append({
                'idx': idx,
                'perplexity': selected_perplexity,
                'code': row['method2_solution'],
                'public_tests': row.get('public_tests', {}),
                'pass_rate': row['method2_passed'] / row['method2_total'] if 'method2_total' in row and row['method2_total'] > 0 else None
            })
    
    # Remove outliers from selected perplexities and normalize globally
    if selected_perplexities_raw:
        filtered_selected_perps = remove_outliers_iqr(selected_perplexities_raw, multiplier=1.5)
        if len(filtered_selected_perps) >= 2:
            selected_min = min(filtered_selected_perps)
            selected_max = max(filtered_selected_perps)
            selected_range = selected_max - selected_min
            
            # Normalize selected points globally
            for sd in selected_data:
                if sd['perplexity'] in filtered_selected_perps:
                    norm_sel = (sd['perplexity'] - selected_min) / (selected_range + 1e-8)
                    selected_x.append(norm_sel)
                    if sd['pass_rate'] is not None:
                        selected_y.append(sd['pass_rate'])
                    else:
                        selected_y.append(percent_passed(sd['code'], sd['public_tests']))
    
    # Second pass: Process candidates and baseline with grouped normalization
    for idx, row in df.iterrows():
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/100 problems...")
        
        meta2 = row['method2_metadata']
        if isinstance(meta2, str):
            meta2 = json.loads(meta2)
        
        public_tests = row.get('public_tests', {})
        baseline_code = row['method1_solution']
        selected_candidate_id = meta2.get('selected_candidate_id', None)
        all_candidates = meta2.get('all_candidates', [])
        
        # Use cached test results if available
        baseline_pass_rate = None
        if 'method1_total' in row and row['method1_total'] > 0:
            baseline_pass_rate = row['method1_passed'] / row['method1_total']
        
        # Collect all perplexities for this problem
        candidate_perplexities = [c['perplexity'] for c in all_candidates if 'perplexity' in c]
        
        if not candidate_perplexities:
            continue
        
        total_points += len(candidate_perplexities)
        
        # Remove outliers from this group
        filtered_perplexities = remove_outliers_iqr(candidate_perplexities, multiplier=1.5)
        outliers_removed += len(candidate_perplexities) - len(filtered_perplexities)
        
        if len(filtered_perplexities) < 2:
            # Not enough data after outlier removal, skip normalization
            continue
        
        # Normalize within this group based on filtered perplexities
        min_p = min(filtered_perplexities)
        max_p = max(filtered_perplexities)
        range_p = max_p - min_p
        
        if range_p < 1e-8:
            # No variation, skip this problem
            continue
        
        # For baseline, use mean of filtered perplexities
        baseline_perp = np.mean(filtered_perplexities)
        norm_baseline = (baseline_perp - min_p) / range_p
        baseline_x.append(norm_baseline)
        if baseline_pass_rate is not None:
            baseline_y.append(baseline_pass_rate)
        else:
            baseline_y.append(percent_passed(baseline_code, public_tests))
        
        # Process all candidates (grouped normalization)
        for c in all_candidates:
            if 'perplexity' not in c or 'code' not in c:
                continue
            
            perp = c['perplexity']
            
            # Only include if within filtered range
            if perp in filtered_perplexities:
                norm_p = (perp - min_p) / range_p
                candidate_x.append(norm_p)
                
                # Compute pass rate for candidates
                candidate_y.append(percent_passed(c['code'], public_tests))
    
    print(f"\nOutliers removed: {outliers_removed}/{total_points} ({100*outliers_removed/total_points:.1f}%)")
    print(f"Selected points: {len(selected_x)}, Candidate points: {len(candidate_x)}")
    
    # Bin the data to 0.01 intervals and compute means for regression
    def bin_data(x, y, bin_size=0.025):
        bins = np.arange(0, 1.0 + bin_size, bin_size)
        binned_x = []
        binned_y = []
        for i in range(len(bins) - 1):
            mask = (np.array(x) >= bins[i]) & (np.array(x) < bins[i+1])
            if np.sum(mask) > 0:
                binned_x.append((bins[i] + bins[i+1]) / 2)
                binned_y.append(np.mean(np.array(y)[mask]))
        return np.array(binned_x), np.array(binned_y)
    
    # Bin candidate and selected datasets
    cand_bin_x, cand_bin_y = bin_data(candidate_x, candidate_y) if len(candidate_x) > 1 else (np.array([]), np.array([]))
    sel_bin_x, sel_bin_y = bin_data(selected_x, selected_y) if len(selected_x) > 1 else (np.array([]), np.array([]))
    
    print(f"Binned points: Candidate={len(cand_bin_x)}, Selected={len(sel_bin_x)}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot binned means as scatter points
    if len(cand_bin_x) > 0:
        ax.scatter(cand_bin_x, cand_bin_y, color='cornflowerblue', alpha=0.6, label='Candidate (Binned Mean)', s=50, zorder=3)
    if len(sel_bin_x) > 0:
        ax.scatter(sel_bin_x, sel_bin_y, color='orange', s=100, label='Selected (Binned Mean)', edgecolors='black', linewidths=0.5, zorder=5)
    
    # Compute and plot regression lines on binned data
    if len(cand_bin_x) > 1:
        z = np.polyfit(cand_bin_x, cand_bin_y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        r2 = np.corrcoef(cand_bin_x, cand_bin_y)[0,1]**2
        ax.plot(x_line, p(x_line), color='cornflowerblue', linewidth=2.5, 
               label=f'Candidate Fit (R²={r2:.3f}, slope={z[0]:.3f})', 
               linestyle='-', alpha=0.9)
    
    if len(sel_bin_x) > 1:
        z = np.polyfit(sel_bin_x, sel_bin_y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        r2 = np.corrcoef(sel_bin_x, sel_bin_y)[0,1]**2
        ax.plot(x_line, p(x_line), color='orange', linewidth=2.5, 
               label=f'Selected Fit (R²={r2:.3f}, slope={z[0]:.3f})', 
               linestyle='-', alpha=0.9)
    
    ax.set_xlabel('Normalized Perplexity (Candidates: Grouped, Selected: Global)')
    ax.set_ylabel('Percent of Public Tests Passed')
    ax.set_title('Method Comparison: Perplexity vs. Test Pass Rate')
    ax.legend(loc='best')
    plt.tight_layout()
    out_path = RESULTS_DIR / 'method_comparison_perplexity_vs_passrate.png'
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    main()
