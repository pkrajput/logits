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


def calculate_buggy_code_token_perplexity(buggy_function, model="gpt-4o-mini", temperature=1):
    """
    Calculate token-level perplexity for the buggy code itself.
    Saves the top 5 logprobs for each token.
    
    Args:
        buggy_function (str): The buggy function code
        model (str): The model to use for calculation
        temperature (float): Temperature for sampling (default: 1)
    
    Returns:
        dict: Dictionary containing token-level perplexity data with top 5 logprobs
    """
    # Ask the model to echo/complete the buggy code to get token logprobs
    messages = [
        {"role": "user", "content": buggy_function}
    ]
    
    # Get completion with logprobs
    response = get_completion(
        messages=messages,
        model=model,
        logprobs=True,
        top_logprobs=5,
        max_tokens=2000,
        temperature=temperature,
        seed=123
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
    
    return {
        "tokens": tokens,
        "logprobs": logprobs,
        "token_perplexities": token_perplexities,
        "top_5_logprobs_per_token": top_logprobs_per_token,
        "mean_perplexity": np.exp(-np.mean(logprobs)),
        "response_text": response.choices[0].message.content,
        "num_tokens": len(tokens)
    }


def process_reapr_dataset(num_samples=1, model="gpt-4o-mini", start_idx=0, temperature=1):
    """
    Process the ReAPR dataset and calculate token-level perplexity for buggy functions.
    
    Args:
        num_samples (int): Number of samples to process (default: 1)
        model (str): The model to use for calculation
        start_idx (int): Starting index in the dataset
        temperature (float): Temperature for sampling (default: 1)
    
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
                temperature=temperature
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
    # Calculate token-level perplexity with top 5 logprobs for each token
    
    results = process_reapr_dataset(
        num_samples=1,
        model="gpt-4o-mini",
        start_idx=0,
        temperature=1
    )
    
    # Save results to a file
    import json
    output_file = "perplexity_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)