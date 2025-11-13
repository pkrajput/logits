"""
Quick script to visualize existing perplexity results
"""
import json
from tokenLogProb import visualize_token_surprisal

# Load existing results
with open('perplexity_results.json', 'r') as f:
    results = json.load(f)

# Visualize the first result
if results and len(results) > 0:
    first_result = results[0]
    if 'tokens' in first_result and 'avg_logprobs' in first_result and 'std_logprobs' in first_result:
        print(f"Visualizing {len(first_result['tokens'])} tokens...")
        print(f"Mean perplexity: {first_result['mean_perplexity']:.4f}")
        print(f"Number of runs: {first_result.get('num_runs', 'N/A')}")
        
        # Get the original buggy function if available
        original_code = first_result.get('buggy_function', None)
        
        visualize_token_surprisal(
            first_result['tokens'],
            first_result['avg_logprobs'],
            first_result['std_logprobs'],
            original_code=original_code,
            output_html="token_surprisal_visualization.html"
        )
        print("\n✓ Visualization created! Open 'token_surprisal_visualization.html' in your browser.")
        print("Yellow = Surprising tokens (high perplexity)")
        print("Red border = High disagreement across runs (variance in predictions)")
    else:
        print("No token data available for visualization")
        print("Available keys:", first_result.keys())
else:
    print("No results found in perplexity_results.json")
