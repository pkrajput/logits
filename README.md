# Perplexity Analysis for ReAPR Dataset

This module calculates the perplexity of buggy functions from the ReAPR (Automatic Program Repair via Retrieval-Augmented Large Language Models) dataset using OpenAI's API.

## Features

- **Automatic dataset management**: Downloads the ReAPR dataset from Hugging Face if not present locally
- **Perplexity calculation**: Uses OpenAI's API with logprobs to calculate perplexity for buggy code
- **Batch processing**: Process multiple samples from the dataset
- **Statistical analysis**: Provides summary statistics for multiple samples

## Setup

1. Install required packages:
```bash
pip install openai numpy python-dotenv huggingface_hub datasets
```

2. Set up your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### As a script

Run the main script to process 5 samples from the dataset:

```bash
python logprob.py
```

This will:
- Check for the ReAPR dataset locally and download if needed
- Process 5 buggy functions from the dataset
- Calculate perplexity for each using GPT-4o-mini
- Save results to `perplexity_results.json`

### As a module

```python
import logprob

# Load the dataset
dataset = logprob.load_reapr_dataset()

# Calculate perplexity for a single buggy function
result = logprob.calculate_buggy_function_perplexity(
    dataset[0]['buggy_function'],
    model="gpt-4o-mini",
    verbose=True
)
print(f"Perplexity: {result['perplexity']}")

# Process multiple samples
results = logprob.process_reapr_dataset(
    num_samples=10,
    model="gpt-4o-mini",
    start_idx=0
)
```

### Test script

Run the test script to see examples:

```bash
python test_perplexity.py
```

## Functions

### `check_and_download_dataset()`
Checks if the ReAPR dataset exists locally and downloads it if not.

### `load_reapr_dataset()`
Loads the ReAPR dataset from the local directory. Returns a list of dictionaries with `buggy_function` and `fixed_function` fields.

### `get_completion(messages, model, ...)`
Makes an API call to OpenAI with the specified parameters, including logprobs.

### `get_perplexity(prompt, API_RESPONSE)`
Calculates and displays the perplexity score from an API response with logprobs.

### `calculate_buggy_function_perplexity(buggy_function, model, verbose)`
Calculates the perplexity of a single buggy function.

**Parameters:**
- `buggy_function` (str): The buggy function code
- `model` (str): The OpenAI model to use (default: "gpt-4o-mini")
- `verbose` (bool): Whether to print detailed output (default: True)

**Returns:**
- Dictionary with `perplexity`, `response`, and `num_tokens`

### `process_reapr_dataset(num_samples, model, start_idx)`
Process multiple samples from the ReAPR dataset.

**Parameters:**
- `num_samples` (int): Number of samples to process (default: 10)
- `model` (str): The OpenAI model to use (default: "gpt-4o-mini")
- `start_idx` (int): Starting index in the dataset (default: 0)

**Returns:**
- List of result dictionaries with perplexity scores and statistics

## Dataset

The ReAPR dataset contains 364,063 pairs of buggy and fixed functions. Each entry has:
- `buggy_function`: The original buggy code
- `fixed_function`: The corrected version of the code

## Output

Results include:
- Perplexity score for each buggy function
- Token-level logprobs
- Response from the model
- Summary statistics (mean, median, min, max perplexity)

## Notes

- The perplexity calculation uses the model's logprobs from analyzing the buggy code
- Lower perplexity typically indicates the model finds the code more predictable/familiar
- Higher perplexity might indicate unusual or buggy patterns
