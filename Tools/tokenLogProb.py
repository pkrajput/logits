from openai import OpenAI
from math import exp
import numpy as np
from IPython.display import display, HTML
import os
import dotenv
from pathlib import Path

dotenv.load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Dataset configuration
DATASET_REPO_ID = "zxliu/ReAPR-Automatic-Program-Repair-via-Retrieval-Augmented-Large-Language-Models"
DATASET_LOCAL_DIR = Path(__file__).parent / "ReAPR-dataset"

def check_and_download_dataset():
    """
    Check if the ReAPR dataset exists locally, and download it if not.
    
    Returns:
        Path: Path to the local dataset directory
    """
    if not DATASET_LOCAL_DIR.exists() or not any(DATASET_LOCAL_DIR.iterdir()):
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=DATASET_REPO_ID,
                repo_type="dataset",
                local_dir=str(DATASET_LOCAL_DIR)
            )
        except ImportError:
            raise
        except Exception as e:
            raise
    
    return DATASET_LOCAL_DIR

# Check and download dataset on module import
dataset_path = check_and_download_dataset()

def get_completion(
    messages: list[dict[str, str]],
    model: str = "gpt-4-nano",
    max_tokens=500,
    temperature=0,
    stop=None,
    seed=123,
    tools=None,
    logprobs=None,  # whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the content of message..
    top_logprobs=None,
) -> str:
    params = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    completion = client.chat.completions.create(**params)
    return completion


def get_perplexity(prompt, API_RESPONSE):
    logprobs = [token.logprob for token in API_RESPONSE.choices[0].logprobs.content]
    response_text = API_RESPONSE.choices[0].message.content
    response_text_tokens = [token.token for token in API_RESPONSE.choices[0].logprobs.content]
    max_starter_length = max(len(s) for s in ["Prompt:", "Response:", "Tokens:", "Logprobs:", "Perplexity:"])
    max_token_length = max(len(s) for s in response_text_tokens)
    

    formatted_response_tokens = [s.rjust(max_token_length) for s in response_text_tokens]
    formatted_lps = [f"{lp:.2f}".rjust(max_token_length) for lp in logprobs]

    perplexity_score = np.exp(-np.mean(logprobs))

    return perplexity_score


def visualize_token_surprisal(tokens, avg_logprobs, std_logprobs, original_code=None, output_html="token_visualization.html"):
    """
    Visualize tokens with dual color highlighting:
    - Yellow: based on average log probability (surprisal)
    - Red border: based on standard deviation (disagreement across runs)
    
    Args:
        tokens (list): List of token strings
        avg_logprobs (list): List of average log probabilities for each token
        std_logprobs (list): List of standard deviations of log probabilities
        original_code (str, optional): The original buggy code to display below
        output_html (str): Output HTML filename
    
    Returns:
        str: HTML string with highlighted tokens
    """
    import colorsys
    
    # Use percentile-based scaling for better variation
    # This helps show more nuanced differences in the data
    avg_logprobs_arr = np.array(avg_logprobs)
    std_logprobs_arr = np.array(std_logprobs)
    
    # For average logprobs, use percentiles to avoid extreme values dominating
    p10_avg = np.percentile(avg_logprobs_arr, 10)
    p90_avg = np.percentile(avg_logprobs_arr, 90)
    
    # For std, use percentiles as well
    p10_std = np.percentile(std_logprobs_arr, 10)
    p90_std = np.percentile(std_logprobs_arr, 90)
    
    html_parts = ['<!DOCTYPE html>\n<html>\n<head>\n<style>']
    html_parts.append('body { font-family: "Courier New", monospace; font-size: 14px; line-height: 1.6; padding: 20px; background: #1e1e1e; }')
    html_parts.append('.token { display: inline; padding: 2px 1px; white-space: pre; border-radius: 2px; }')
    html_parts.append('.legend { margin: 20px 0; padding: 15px; background: #2d2d2d; border-radius: 5px; color: white; }')
    html_parts.append('.legend-item { display: inline-block; margin-right: 20px; margin-bottom: 10px; }')
    html_parts.append('.stats { margin: 20px 0; padding: 15px; background: #2d2d2d; border-radius: 5px; color: white; }')
    html_parts.append('.color-examples { margin: 20px 0; padding: 15px; background: #2d2d2d; border-radius: 5px; color: white; }')
    html_parts.append('.code-section { margin: 20px 0; }')
    html_parts.append('.code-section h3 { color: white; margin-bottom: 10px; }')
    html_parts.append('.generated-code { background: #f5f5f5; padding: 20px; border-radius: 5px; border-left: 4px solid #4CAF50; font-family: "Courier New", monospace; line-height: 1.5; }')
    html_parts.append('.generated-code pre { margin: 0; white-space: pre; color: #333; font-family: "Courier New", monospace; line-height: 1.5; }')
    html_parts.append('.original-code { background: #f5f5f5; padding: 20px; border-radius: 5px; border-left: 4px solid #888; margin-top: 20px; }')
    html_parts.append('.original-code pre { margin: 0; white-space: pre; color: #333; font-family: "Courier New", monospace; line-height: 1.5; }')
    html_parts.append('h2 { color: white; }')
    html_parts.append('</style>\n</head>\n<body>')
    html_parts.append('<h2>Token Surprisal & Disagreement Visualization</h2>')
    
    # Add legend with examples
    html_parts.append('<div class="legend">')
    html_parts.append('<strong>Color Coding:</strong><br>')
    html_parts.append('<strong>Yellow Background</strong> = Surprisal (perplexity):<br>')
    html_parts.append('<span class="legend-item"><span style="background: rgba(255, 255, 0, 0.1); color: black; padding: 3px 6px; border-radius: 2px;">Light Yellow = Low surprisal</span></span>')
    html_parts.append('<span class="legend-item"><span style="background: rgba(255, 255, 0, 0.5); color: black; padding: 3px 6px; border-radius: 2px;">Medium Yellow = Medium surprisal</span></span>')
    html_parts.append('<span class="legend-item"><span style="background: rgba(255, 255, 0, 1.0); color: black; padding: 3px 6px; border-radius: 2px;">Bright Yellow = High surprisal</span></span><br>')
    html_parts.append('<strong>Red Border</strong> = Disagreement across runs:<br>')
    html_parts.append('<span class="legend-item"><span style="background: white; color: black; padding: 3px 6px; border: 1px solid #ccc; border-radius: 2px;">No border = Low disagreement</span></span>')
    html_parts.append('<span class="legend-item"><span style="background: white; color: black; padding: 3px 6px; border: 2px solid rgba(255, 0, 0, 0.6); border-radius: 2px;">Medium border = Medium disagreement</span></span>')
    html_parts.append('<span class="legend-item"><span style="background: white; color: black; padding: 3px 6px; border: 3px solid rgba(255, 0, 0, 1.0); border-radius: 2px;">Thick border = High disagreement</span></span>')
    html_parts.append('</div>')
    
    # Add color combination examples
    html_parts.append('<div class="color-examples">')
    html_parts.append('<strong>Common Combinations:</strong><br>')
    html_parts.append('<span style="display: inline-block; background: rgba(255, 255, 0, 0.2); padding: 3px 6px; border: 1px solid #ccc; margin: 5px; border-radius: 2px;">Low surprisal + Low disagreement = Confident, expected</span>')
    html_parts.append('<span style="display: inline-block; background: rgba(255, 255, 0, 1.0); padding: 3px 6px; border: 1px solid #ccc; margin: 5px; border-radius: 2px;">High surprisal + Low disagreement = Consistently surprising</span>')
    html_parts.append('<span style="display: inline-block; background: rgba(255, 255, 0, 0.2); padding: 3px 6px; border: 3px solid rgba(255, 0, 0, 1.0); margin: 5px; border-radius: 2px;">Low surprisal + High disagreement = Uncertain but expected</span>')
    html_parts.append('<span style="display: inline-block; background: rgba(255, 255, 0, 1.0); padding: 3px 6px; border: 3px solid rgba(255, 0, 0, 1.0); margin: 5px; border-radius: 2px; color: black;">High surprisal + High disagreement = Very uncertain!</span>')
    html_parts.append('</div>')
    
    # Add statistics
    html_parts.append('<div class="stats">')
    html_parts.append(f'<strong>Statistics:</strong><br>')
    html_parts.append(f'Total tokens: {len(tokens)}<br>')
    html_parts.append(f'Mean avg logprob: {np.mean(avg_logprobs):.4f}<br>')
    html_parts.append(f'Mean perplexity: {np.exp(-np.mean(avg_logprobs)):.4f}<br>')
    html_parts.append(f'Mean std logprob (disagreement): {np.mean(std_logprobs):.4f}<br>')
    html_parts.append(f'Min/Max std: {np.min(std_logprobs):.4f} / {np.max(std_logprobs):.4f}<br>')
    html_parts.append(f'10th/90th percentile avg logprob: {p10_avg:.4f} / {p90_avg:.4f}<br>')
    html_parts.append(f'10th/90th percentile std: {p10_std:.4f} / {p90_std:.4f}')
    html_parts.append('</div>')
    
    # Generated/Repaired code section
    html_parts.append('<div class="code-section">')
    html_parts.append('<h3>Model Generated/Repaired Code (with highlighting):</h3>')
    html_parts.append('<div class="generated-code">')
    html_parts.append('<pre>')
    
    for token, avg_lp, std_lp in zip(tokens, avg_logprobs, std_logprobs):
        # Use percentile-based normalization for better scaling
        # Clamp values to percentile range for more variation in the middle
        if p90_avg != p10_avg:
            normalized_avg = (avg_lp - p10_avg) / (p90_avg - p10_avg)
            normalized_avg = np.clip(normalized_avg, 0, 1)
        else:
            normalized_avg = 0.5
        
        # Apply power transformation to increase sensitivity
        # This makes medium values more distinguishable
        yellow_alpha = (1.0 - normalized_avg) ** 0.7  # Power < 1 increases variation
        
        # Normalize std with percentile-based scaling
        if p90_std != p10_std:
            normalized_std = (std_lp - p10_std) / (p90_std - p10_std)
            normalized_std = np.clip(normalized_std, 0, 1)
        else:
            normalized_std = 0.0
        
        # Apply power transformation for better visual separation
        red_intensity = normalized_std ** 0.6
        
        # Yellow background for surprisal
        bg_color = f'rgba(255, 255, 0, {yellow_alpha:.3f})'
        
        # Red border for disagreement - thickness and opacity both scale
        border_width = int(1 + red_intensity * 3)  # 1-4px border
        border_color = f'rgba(255, 0, 0, {red_intensity:.3f})'
        
        # Add slight shadow for high disagreement to make it more visible
        shadow = ''
        if red_intensity > 0.5:
            shadow = f'box-shadow: 0 0 3px rgba(255, 0, 0, {red_intensity:.3f});'
        
        # Escape HTML special characters in token
        token_escaped = token.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        
        html_parts.append(f'<span class="token" style="background-color: {bg_color}; border: {border_width}px solid {border_color}; {shadow}" title="avg logprob: {avg_lp:.4f}, std: {std_lp:.4f}, perplexity: {np.exp(-avg_lp):.2f}">{token_escaped}</span>')
    
    html_parts.append('</pre>')
    html_parts.append('</div>')
    html_parts.append('</div>')
    
    # Original code section (if provided)
    if original_code:
        html_parts.append('<div class="code-section">')
        html_parts.append('<h3>Original Buggy Code (for reference):</h3>')
        html_parts.append('<div class="original-code">')
        original_escaped = original_code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_parts.append(f'<pre>{original_escaped}</pre>')
        html_parts.append('</div>')
        html_parts.append('</div>')
    
    html_parts.append('</body>\n</html>')
    
    html_content = ''.join(html_parts)
    
    # Save to file
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Visualization saved to {output_html}")
    
    return html_content


def load_reapr_dataset():
    """
    Load the ReAPR dataset from the local directory.
    
    Returns:
        list: List of dictionaries containing buggy_function and fixed_function
    """
    import json
    dataset_file = DATASET_LOCAL_DIR / "new_merged_file.json"
    
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_file}")
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    return data


def calculate_buggy_code_token_perplexity(buggy_function, model="gpt-4o-mini", temperature=1, num_runs=5):
    """
    Calculate token-level perplexity for the buggy code itself.
    Runs multiple times to capture variance in predictions.
    
    Args:
        buggy_function (str): The buggy function code
        model (str): The model to use for calculation
        temperature (float): Temperature for sampling (default: 1)
        num_runs (int): Number of times to run the calculation (default: 5)
    
    Returns:
        dict: Dictionary containing token-level perplexity data with variance across runs
    """
    all_runs = []
    
    for run in range(num_runs):
        # Instruct the model to only generate code, no natural language explanations
        messages = [
            {"role": "system", "content": "You are a code repair assistant. Only output code. Do not include any explanations, comments, or natural language text in your response. Only generate the code itself. Do not print code comment markers or quotes at the beginning or end of the code block."},
            {"role": "user", "content": f"Repair this buggy code:\n\n{buggy_function}"}
        ]
        
        # Get completion with logprobs - no seed for variance
        response = get_completion(
            messages=messages,
            model=model,
            logprobs=True,
            top_logprobs=5,
            max_tokens=2000,
            temperature=temperature,
            seed=None  # Allow variance across runs
        )
        
        # Extract token-level information including top 5 alternatives
        logprobs_data = response.choices[0].logprobs.content
        tokens = []
        logprobs = []
        token_perplexities = []
        top_logprobs_per_token = []
        
        for token_data in logprobs_data:
            tokens.append(token_data.token)
            logprobs.append(token_data.logprob)
            token_perplexities.append(np.exp(-token_data.logprob))
            
            # Extract top 5 logprobs for this token
            top_5 = []
            if token_data.top_logprobs:
                for alt_token in token_data.top_logprobs:
                    top_5.append({
                        "token": alt_token.token,
                        "logprob": alt_token.logprob,
                        "perplexity": np.exp(-alt_token.logprob)
                    })
            top_logprobs_per_token.append(top_5)
        
        all_runs.append({
            "run": run + 1,
            "tokens": tokens,
            "logprobs": logprobs,
            "token_perplexities": token_perplexities,
            "top_5_logprobs_per_token": top_logprobs_per_token,
            "response_text": response.choices[0].message.content,
            "num_tokens": len(tokens)
        })
    
    # Align tokens across runs and calculate statistics
    # Use the first run as reference for token positions
    reference_tokens = all_runs[0]['tokens']
    num_tokens = len(reference_tokens)
    
    # Calculate average and std for each token position
    avg_logprobs = []
    std_logprobs = []
    avg_perplexities = []
    std_perplexities = []
    
    for i in range(num_tokens):
        token_logprobs_at_pos = []
        for run in all_runs:
            if i < len(run['logprobs']):
                token_logprobs_at_pos.append(run['logprobs'][i])
        
        if token_logprobs_at_pos:
            avg_lp = np.mean(token_logprobs_at_pos)
            std_lp = np.std(token_logprobs_at_pos)
            avg_logprobs.append(avg_lp)
            std_logprobs.append(std_lp)
            avg_perplexities.append(np.exp(-avg_lp))
            std_perplexities.append(std_lp)  # Standard deviation of logprobs as disagreement measure
    
    return {
        "runs": all_runs,
        "num_runs": num_runs,
        "tokens": reference_tokens,
        "avg_logprobs": avg_logprobs,
        "std_logprobs": std_logprobs,
        "avg_perplexities": avg_perplexities,
        "std_perplexities": std_perplexities,
        "mean_perplexity": np.exp(-np.mean(avg_logprobs)),
        "num_tokens": num_tokens
    }


def process_reapr_dataset(num_samples=1, model="gpt-4o-mini", start_idx=0, temperature=1, num_runs=5):
    """
    Process the ReAPR dataset and calculate token-level perplexity for buggy functions.
    
    Args:
        num_samples (int): Number of samples to process (default: 1)
        model (str): The model to use for calculation
        start_idx (int): Starting index in the dataset
        temperature (float): Temperature for sampling (default: 1)
        num_runs (int): Number of runs per sample (default: 5)
    
    Returns:
        list: List of results with token-level perplexity scores
    """
    dataset = load_reapr_dataset()
    results = []
    
    end_idx = min(start_idx + num_samples, len(dataset))
    
    for i in range(start_idx, end_idx):
        entry = dataset[i]
        buggy_function = entry['buggy_function']
        
        try:
            result = calculate_buggy_code_token_perplexity(
                buggy_function,
                model=model,
                temperature=temperature,
                num_runs=num_runs
            )
            
            results.append({
                "index": i,
                "buggy_function": buggy_function,
                "fixed_function": entry['fixed_function'],
                **result
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "error": str(e),
                "mean_perplexity": None
            })
    
    return results


if __name__ == "__main__":
    # Process a single sample from the ReAPR dataset
    # Run 5 times to capture variance in predictions
    
    results = process_reapr_dataset(
        num_samples=1,
        model="gpt-4o-mini",
        start_idx=0,
        temperature=1,
        num_runs=5
    )
    
    # Save results to a file
    import json
    output_file = "perplexity_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization for the first result
    if results and len(results) > 0:
        first_result = results[0]
        if 'tokens' in first_result and 'avg_logprobs' in first_result and 'std_logprobs' in first_result:
            # Get the original buggy function if available
            original_code = first_result.get('buggy_function', None)
            
            visualize_token_surprisal(
                first_result['tokens'],
                first_result['avg_logprobs'],
                first_result['std_logprobs'],
                original_code=original_code,
                output_html="token_surprisal_visualization.html"
            )
            print(f"\nVisualization created! Open 'token_surprisal_visualization.html' in your browser to view.")
        else:
            print("No token data available for visualization")