

import importlib
import json
import math
from difflib import SequenceMatcher
from functools import lru_cache
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import tempfile
import textwrap
import pathlib

with open("accuracy_experiment_results.json", "r", encoding="utf-8") as f:
	results = json.load(f)

method1_df = pd.DataFrame(results["method1_direct"]["results"])
method1_df["method"] = "method1_direct"

method2_df = pd.DataFrame(results["method2_selection"]["results"])
method2_df["method"] = "method2_selection"

per_sample_df = pd.concat([method1_df, method2_df], ignore_index=True)

method_summary_df = pd.DataFrame(
	[
		{
			"method": "method1_direct",
			"correct": results["method1_direct"]["correct"],
			"total": results["method1_direct"]["total"],
			"accuracy": results["method1_direct"]["accuracy"],
		},
		{
			"method": "method2_selection",
			"correct": results["method2_selection"]["correct"],
			"total": results["method2_selection"]["total"],
			"accuracy": results["method2_selection"]["accuracy"],
		},
	]
)

comparison_df = pd.json_normalize(results["comparison"], sep="_")

print(comparison_df.columns)
"""
Index(['dataset_index', 'buggy_function', 'fixed_function',
       'method1_direct_repaired_code', 'method1_direct_is_correct',
       'method2_selection_repaired_code', 'method2_selection_is_correct',
       'method2_selection_candidates',
       'method2_selection
"""

def java_snippet_compiles(snippet: str):
    wrapped = f"""
    import java.sql.*;
    public class Snippet {{
        {snippet}

        public static void main(String[] args) throws Exception {{
            // Call the method to prove it links; adapt to your needs.
            printException("demo", new SQLException("msg", "23500"));
        }}
    }}
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "Snippet.java"
        path.write_text(textwrap.dedent(wrapped))
        resp = subprocess.run(
            ["javac", str(path)],
            text=True,
            capture_output=True,
            check=False,
        )
        return resp.returncode == 0, resp.stderr


"""method1_direct_repaired_code_runs = []
method2_selection_repaired_code_runs = []
method2_selection_candidates_runs = []

for idx, comparison in comparison_df.iterrows():
     method1_direct_repaired_code_runs.append(java_snippet_compiles(comparison['method1_direct_repaired_code'])[0])
     method2_selection_repaired_code_runs.append(java_snippet_compiles(comparison['method2_selection_repaired_code'])[0])
     method2_selection_candidates_runs.append([java_snippet_compiles(x)[0] for x in comparison['method2_selection_candidates']])

comparison_df["method1_direct_repaired_code_runs"] = method1_direct_repaired_code_runs
comparison_df["method2_selection_repaired_code_runs"] = method2_selection_repaired_code_runs
comparison_df["method2_selection_candidates_runs"] = method2_selection_candidates_runs

comparison_df.to_pickle("comparison_df.pkl")"""

#print(comparison_df.head)
"""
<bound method NDFrame.head of     dataset_index                                     buggy_function  ...                       method2_selection_candidates method2_selection_selected_perplexity
0               0  \tpublic static synchronized void printExcepti...  ...  [{'candidate_id': 1, 'code': 'public static sy...                              1.009654
1               1    public MonotonicAppendingLongBuffer(int init...  ...  [{'candidate_id': 1, 'code': 'public Monotonic...                              1.003142
2               2    public void testBuild() throws IOException {...  ...  [{'candidate_id': 1, 'code': 'public void test...                              1.003214
3               3    public void testExtendedResultsCount() throw...  ...  [{'candidate_id': 1, 'code': 'public void test...                              1.007116
4               4    public String[] listAll() {\n    ensureOpen(...  ...  [{'candidate_id': 1, 'code': 'public String[] ...                              1.001405
..            ...                                                ...  ...                                                ...                                   ...
95             95    public void setContext( TransformContext con...  ...  [{'candidate_id': 1, 'code': 'public void setC...                              1.001525
96             96    public static SimpleOrderedMap<Object> getIn...  ...  [{'candidate_id': 1, 'code': 'public static Si...                              1.006516
97             97      public static void enumeratekeys(String ss...  ...  [{'candidate_id': 1, 'code': '```java
public s...                              1.022839
98             98    private static Path prepareInput(FileSystem ...  ...  [{'candidate_id': 1, 'code': 'private static P...                              1.018837
99             99    public static Path prepareOutput(FileSystem ...  ...  [{'candidate_id': 1, 'code': 'public static Pa...                              1.022880

[100 rows x 9 columns]>
"""