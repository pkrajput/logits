# Code Repair with Surprisal-Based Selection

This project evaluates automated code repair methods using the **DeepMind Code Contests dataset**. It compares a baseline direct repair approach against a novel method that generates multiple repair candidates and selects the one with the lowest surprisal (perplexity).

## 🎯 Hypothesis

**Lower surprisal correlates with higher correctness** — When an LLM is more "confident" (lower perplexity) in a repair, it's more likely to be correct.

## 📊 Methodology

### Two Repair Methods

**Method 1: Direct Repair (Baseline)**
- Single LLM call to repair buggy code
- Temperature: 0.7
- Simple, fast approach

**Method 2: Candidate Selection with Surprisal**
1. Generate 5 repair candidates (temp=0.7)
2. Calculate perplexity for each using logprobs
3. Select the candidate with lowest surprisal
4. Hypothesis: Lower surprisal → Higher correctness

### Validation Strategy

**Primary Metric: Test Execution** (Functional Correctness)
- Runs public and private test cases from the dataset
- Solution is correct if ALL tests pass
- Most reliable indicator of actual correctness

**Secondary Metrics: Code Similarity** (Structural Correctness)
- Exact string match with ground truth
- AST (Abstract Syntax Tree) equivalence
- CodeBLEU score (0.0 - 1.0)

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- OpenAI API key
- ~2GB free disk space (for dataset)
- Internet connection (first run only)

### Installation

**Option 1: Automated (Recommended)**
```bash
./install.sh
```

**Option 2: Manual**
```bash
# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Copy environment template:**
```bash
cp .env.example .env
```

2. **Edit `.env` and add your OpenAI API key:**
```
OPENAI_API_KEY=sk-proj-your-actual-key-here
OPENAI_REQUEST_TIMEOUT=120
```

### Verify Setup

Run the test script to ensure everything is configured:
```bash
python3 test_setup.py
```

Expected output:
```
✓ All tests passed! You're ready to run the experiment.
```

### Run Your First Experiment

```bash
python3 calculateLogits.py
```

**What happens:**
1. Downloads Code Contests dataset (~2GB, first run only)
2. Loads incorrect solutions as buggy input
3. Repairs code using both methods
4. Validates with test execution + code similarity
5. Saves results to `accuracy_experiment_results_code_contests.json`

## ⚙️ Configuration

Edit the `__main__` block in `calculateLogits.py`:

```python
results = run_accuracy_experiment(
    num_samples=10,      # Number of problems to test (start small!)
    model="gpt-4-mini",  # OpenAI model: "gpt-4", "gpt-3.5-turbo", etc.
    start_idx=0,         # Skip first N problems
    split="train"        # Dataset split: "train", "valid", or "test"
)
```

**Sample Configurations:**

```python
# Quick test (5 minutes, ~$0.02)
num_samples=5, model="gpt-3.5-turbo"

# Standard run (20-30 minutes, ~$0.10)
num_samples=20, model="gpt-4-mini"

# Large scale (2-3 hours, ~$0.30)
num_samples=100, model="gpt-4-mini"
```

## 📁 Dataset

Uses the [DeepMind Code Contests](https://huggingface.co/datasets/deepmind/code_contests) dataset:

| Column | Description | Usage |
|--------|-------------|-------|
| `incorrect_solutions` | Buggy code submissions | **Input** to repair |
| `solutions` | Correct code submissions | **Ground truth** reference |
| `public_tests` | Public test cases (input/output) | **Validation** |
| `private_tests` | Private test cases (input/output) | **Validation** |
| `name` | Problem name | Metadata |
| `description` | Problem statement | Context |
| `difficulty` | Problem difficulty rating | Filtering |
| `cf_rating` | Codeforces rating | Filtering |

**Dataset features:**
- Programming competition problems
- Multiple languages (Python, C++, Java)
- Real-world buggy solutions
- Comprehensive test suites

## 🔄 Workflow

```
┌─────────────────────────────────────┐
│  Code Contests Dataset (HuggingFace) │
│  • incorrect_solutions → INPUT       │
│  • solutions → GROUND TRUTH          │
│  • public/private_tests → VALIDATION │
└─────────────────────────────────────┘
                  │
                  ▼
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│  Method 1:   │    │  Method 2:   │
│ Direct Repair│    │ Multi-Cand.  │
│              │    │ Selection    │
│ 1 LLM call   │    │ 5 LLM calls  │
│              │    │ + logprobs   │
└──────────────┘    └──────────────┘
        │                   │
        │                   │ Select lowest
        │                   │ perplexity
        │                   │
        └─────────┬─────────┘
                  │
                  ▼
        ┌─────────────────┐
        │   VALIDATION    │
        └─────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ Test Exec.   │    │ Code Simil.  │
│ (Primary)    │    │ (Secondary)  │
│              │    │              │
│ • Run tests  │    │ • Exact      │
│ • ALL pass?  │    │ • AST        │
│              │    │ • CodeBLEU   │
└──────────────┘    └──────────────┘
        │                   │
        └─────────┬─────────┘
                  │
                  ▼
        ┌─────────────────┐
        │ Compare Methods │
        │                 │
        │ Which is better?│
        └─────────────────┘
```

## 📊 Understanding Results

### Console Output

During execution:
```
Processing sample 1/10 (dataset index 0)... M1:✓ | M2:✓
Processing sample 2/10 (dataset index 1)... M1:✗ | M2:✓
Processing sample 3/10 (dataset index 2)... M1:✓ | M2:✓
```

- `M1:✓` = Method 1 passed all tests
- `M1:✗` = Method 1 failed some tests
- `M2:✓` = Method 2 passed all tests
- `M2:✗` = Method 2 failed some tests

### Summary Report

```
======================================================================
EXPERIMENT RESULTS SUMMARY
======================================================================

Model: gpt-4-mini
Dataset Split: train
Number of samples: 10

Method 1 - Direct Repair (Baseline):
  Tests Passed (Primary Metric): 7/10
  Accuracy: 70.00%
  Exact matches: 5
  AST matches: 6
  Avg CodeBLEU: 0.7234

Method 2 - Generate 5 & Select Lowest Surprisal:
  Tests Passed (Primary Metric): 9/10
  Accuracy: 90.00%
  Exact matches: 7
  AST matches: 8
  Avg CodeBLEU: 0.8456

Improvement:
  Absolute: +20.00%
  Relative: +28.6%
======================================================================
```

### Results File

Detailed results saved in JSON format:
```bash
cat accuracy_experiment_results_code_contests.json | python3 -m json.tool | less
```

## 🛠️ Main Functions

### `check_and_download_dataset()`
Automatically downloads the Code Contests dataset from HuggingFace if not present locally. Dataset is cached for future runs.

### `load_code_contests_dataset(split='train')`
Loads the dataset from local cache. Supports 'train', 'valid', and 'test' splits.

**Returns:** HuggingFace Dataset object

### `detect_language(code)`
Auto-detects programming language from source code (Python/C++/Java).

**Parameters:**
- `code` (str): Source code string

**Returns:** str ('python', 'cpp', 'java', or 'unknown')

### `repair_code_direct(buggy_function, model='gpt-4-mini')`
Method 1: Direct repair approach (baseline).

**Parameters:**
- `buggy_function` (str): The buggy code to repair
- `model` (str): OpenAI model to use

**Returns:** str (repaired code)

### `repair_code_with_selection(buggy_function, model='gpt-4-mini', num_candidates=5)`
Method 2: Generate multiple candidates and select lowest surprisal.

**Parameters:**
- `buggy_function` (str): The buggy code to repair
- `model` (str): OpenAI model to use
- `num_candidates` (int): Number of candidates to generate

**Returns:** 
- `best_code` (str): Selected repair with lowest perplexity
- `candidates` (list): All candidates with their perplexity scores

### `run_tests_on_solution(solution_code, public_tests, private_tests, language=None)`
Executes test cases against repaired code.

**Parameters:**
- `solution_code` (str): Code to test
- `public_tests` (dict): Public test cases with 'input' and 'output'
- `private_tests` (dict): Private test cases with 'input' and 'output'
- `language` (str): Programming language (auto-detected if None)

**Returns:** dict with test results and pass/fail statistics

### `check_correctness(repaired_code, ground_truth)`
Checks code similarity using multiple metrics.

**Parameters:**
- `repaired_code` (str): The repaired code
- `ground_truth` (str): The correct solution

**Returns:** dict with exact_match, ast_match, codebleu_score, is_correct

### `run_accuracy_experiment(num_samples, model, start_idx, split)`
Main experiment runner comparing both repair methods.

**Parameters:**
- `num_samples` (int): Number of problems to test
- `model` (str): OpenAI model to use
- `start_idx` (int): Starting index in dataset
- `split` (str): Dataset split ('train', 'valid', 'test')

**Returns:** dict with comprehensive results and metrics

## 💰 Cost Estimation

Using **gpt-4-mini** (recommended, cheapest):
- Input: ~$0.15 per 1M tokens
- Output: ~$0.60 per 1M tokens
- Per sample: ~2000 input + 1000 output tokens

| Samples | Method 1 Cost | Method 2 Cost* | Total |
|---------|---------------|----------------|-------|
| 10      | ~$0.01        | ~$0.05         | ~$0.06 |
| 50      | ~$0.05        | ~$0.25         | ~$0.30 |
| 100     | ~$0.10        | ~$0.50         | ~$0.60 |

*Method 2 uses 5x API calls (generates 5 candidates)

## 🔧 Troubleshooting

### Import errors (datasets, codebleu, etc.)
```bash
pip install -r requirements.txt
```

### Dataset download fails
```bash
# Remove partial download
rm -rf Datasets/code_contests/
# Run again - will re-download
python3 calculateLogits.py
```

### OpenAI API errors
- Verify API key in `.env` file
- Check API credits/billing
- Reduce `num_samples` if hitting rate limits
- Increase `OPENAI_REQUEST_TIMEOUT` in `.env`

### Tests timing out
Some problems are complex. The script continues processing other samples (timeout: 5 seconds per test).

### Language detection issues
Currently only Python test execution is fully supported. C++ and Java require compilation (to be implemented).

## 📂 Project Structure

```
logits/
├── calculateLogits.py              # Main experiment code
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── install.sh                      # Installation script
├── test_setup.py                  # Setup verification
├── README.md                       # This file
├── Results/                        # Experiment results
│   └── accuracy_experiment_results_code_contests.json
├── Datasets/
│   └── code_contests/             # Auto-downloaded dataset (2GB)
└── Tools/                          # Utility scripts
```

## 📚 Additional Resources

- **Dataset**: [DeepMind Code Contests on HuggingFace](https://huggingface.co/datasets/deepmind/code_contests)
- **Paper**: CodeContests - A competitive programming dataset
- **OpenAI API**: [OpenAI Platform Documentation](https://platform.openai.com/docs)

## 🤝 Contributing

Potential improvements:
- [ ] Implement C++/Java test execution
- [ ] Add parallel test execution for speed
- [ ] Support multi-file solutions
- [ ] Cache LLM responses to reduce costs
- [ ] Add error categorization and analysis
- [ ] Experiment with different selection criteria
- [ ] Try different prompting strategies

## 📝 Notes

- The dataset downloads automatically on first run (~2GB, may take several minutes)
- Test execution currently only supports Python
- Lower perplexity typically indicates higher model confidence
- Start with small `num_samples` (5-10) to verify setup
- Method 2 uses 5x more API calls than Method 1

## 🎓 Research Context

This project explores whether **token-level surprisal** (perplexity) can be used as a proxy for **solution correctness** in automated program repair. By generating multiple repair candidates and selecting the one the model is most "confident" about, we hypothesize that repair accuracy can be improved over single-shot approaches.
