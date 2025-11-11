"""
Recalculate CodeBLEU scores for existing accuracy_experiment_results.json
"""
import json
import numpy as np
from codebleu import calc_codebleu
from logprob import check_correctness

# Load existing results
print("Loading existing results from accuracy_experiment_results.json...")
with open('accuracy_experiment_results.json', 'r') as f:
    results = json.load(f)

print(f"Found {len(results['comparison'])} samples to reprocess\n")

# Recalculate accuracy with CodeBLEU
updated_comparison = []
method1_correct = 0
method2_correct = 0
method1_exact = 0
method1_ast = 0
method2_exact = 0
method2_ast = 0
method1_codebleu_scores = []
method2_codebleu_scores = []

for i, sample in enumerate(results['comparison']):
    if 'error' in sample:
        updated_comparison.append(sample)
        continue
    
    print(f"Processing sample {i+1}/{len(results['comparison'])}...", end=" ")
    
    # Get the codes and ground truth
    repaired_direct = sample['method1_direct']['repaired_code']
    repaired_selection = sample['method2_selection']['repaired_code']
    ground_truth = sample['fixed_function']
    
    # Recalculate correctness with all metrics
    correctness_direct = check_correctness(repaired_direct, ground_truth)
    correctness_selection = check_correctness(repaired_selection, ground_truth)
    
    # Update counts
    if correctness_direct["is_correct"]:
        method1_correct += 1
    if correctness_selection["is_correct"]:
        method2_correct += 1
    
    if correctness_direct["exact_match"]:
        method1_exact += 1
    if correctness_direct["ast_match"]:
        method1_ast += 1
    if correctness_selection["exact_match"]:
        method2_exact += 1
    if correctness_selection["ast_match"]:
        method2_ast += 1
    
    method1_codebleu_scores.append(correctness_direct["codebleu_score"])
    method2_codebleu_scores.append(correctness_selection["codebleu_score"])
    
    # Update sample with new metrics
    updated_sample = {
        "dataset_index": sample["dataset_index"],
        "buggy_function": sample["buggy_function"],
        "fixed_function": sample["fixed_function"],
        "method1_direct": {
            "repaired_code": repaired_direct,
            **correctness_direct
        },
        "method2_selection": {
            "repaired_code": repaired_selection,
            **correctness_selection,
            "candidates": sample['method2_selection']['candidates'],
            "selected_perplexity": sample['method2_selection']['selected_perplexity']
        }
    }
    
    updated_comparison.append(updated_sample)
    
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

# Calculate updated accuracies
total_samples = len([s for s in updated_comparison if 'error' not in s])
method1_accuracy = method1_correct / total_samples if total_samples > 0 else 0.0
method2_accuracy = method2_correct / total_samples if total_samples > 0 else 0.0
method1_avg_codebleu = np.mean(method1_codebleu_scores) if method1_codebleu_scores else 0.0
method2_avg_codebleu = np.mean(method2_codebleu_scores) if method2_codebleu_scores else 0.0

# Update results
results['comparison'] = updated_comparison
results['method1_direct']['correct'] = method1_correct
results['method1_direct']['total'] = total_samples
results['method1_direct']['accuracy'] = method1_accuracy
results['method2_selection']['correct'] = method2_correct
results['method2_selection']['total'] = total_samples
results['method2_selection']['accuracy'] = method2_accuracy

# Save updated results
output_file = "accuracy_experiment_results_updated.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Updated results saved to {output_file}")

# Print summary
print("\n" + "="*70)
print("RECALCULATED RESULTS SUMMARY")
print("="*70)
print(f"\nModel: {results['experiment_config']['model']}")
print(f"Number of samples: {total_samples}")

print(f"\nMethod 1 - Direct Repair (Baseline):")
print(f"  Correct (Exact or AST match): {method1_correct}/{total_samples}")
print(f"  Accuracy: {method1_accuracy:.2%}")
print(f"  Exact matches: {method1_exact}")
print(f"  AST matches: {method1_ast}")
print(f"  Avg CodeBLEU: {method1_avg_codebleu:.4f}")

print(f"\nMethod 2 - Generate 5 & Select Lowest Surprisal:")
print(f"  Correct (Exact or AST match): {method2_correct}/{total_samples}")
print(f"  Accuracy: {method2_accuracy:.2%}")
print(f"  Exact matches: {method2_exact}")
print(f"  AST matches: {method2_ast}")
print(f"  Avg CodeBLEU: {method2_avg_codebleu:.4f}")

print(f"\nImprovement:")
diff = method2_accuracy - method1_accuracy
print(f"  Absolute: {diff:+.2%}")
if method1_accuracy > 0:
    relative = (diff / method1_accuracy) * 100
    print(f"  Relative: {relative:+.1f}%")

print(f"\nCodeBLEU Improvement:")
codebleu_diff = method2_avg_codebleu - method1_avg_codebleu
print(f"  Absolute: {codebleu_diff:+.4f}")
if method1_avg_codebleu > 0:
    codebleu_relative = (codebleu_diff / method1_avg_codebleu) * 100
    print(f"  Relative: {codebleu_relative:+.1f}%")

print("\n" + "="*70)
