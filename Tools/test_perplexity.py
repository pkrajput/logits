"""
Test script to demonstrate perplexity calculation on ReAPR dataset.
This script shows how to use the logprob module to calculate perplexity 
for buggy functions from the ReAPR dataset.
"""

import logits.tokenLogProb as tokenLogProb

# Example 1: Load and inspect a single entry
print("="*80)
print("Example 1: Loading the dataset")
print("="*80)
dataset = tokenLogProb.load_reapr_dataset()
print(f"\nFirst buggy function:\n{dataset[0]['buggy_function'][:300]}...\n")

# Example 2: Calculate perplexity for a single buggy function
print("\n" + "="*80)
print("Example 2: Calculate perplexity for one buggy function")
print("="*80)
result = tokenLogProb.calculate_buggy_function_perplexity(
    dataset[0]['buggy_function'],
    model="gpt-4o-mini",
    verbose=True
)
print(f"\nPerplexity: {result['perplexity']:.4f}")
print(f"Number of tokens: {result['num_tokens']}")

# Example 3: Process multiple samples (uncomment to run)
# print("\n" + "="*80)
# print("Example 3: Process multiple samples")
# print("="*80)
# results = logprob.process_reapr_dataset(
#     num_samples=3,
#     model="gpt-4o-mini",
#     start_idx=0
# )
