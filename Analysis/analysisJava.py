import importlib
import json
import math
import os
from pathlib import Path
from typing import Iterable
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import subprocess
import tempfile
import textwrap
import pathlib

# Set JAVA_HOME for macOS if not already set
if not os.environ.get('JAVA_HOME'):
	try:
		result = subprocess.run(
			['/usr/libexec/java_home'],
			capture_output=True,
			text=True,
			check=True
		)
		os.environ['JAVA_HOME'] = result.stdout.strip()
	except (subprocess.CalledProcessError, FileNotFoundError):
		pass  # Will fail later if Java is needed

RESULTS_PATH = Path("accuracy_experiment_results.json")
COMPARISON_PICKLE_PATH = Path("comparison_df.pkl")
RUN_COLUMNS = [
	"method1_direct_repaired_code_runs",
	"method2_selection_repaired_code_runs",
	"method2_selection_candidates_runs",
]


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


def _format_java_snippet(snippet: str) -> str:
	snippet = snippet.replace("\r\n", "\n")
	lines = snippet.splitlines()
	while lines and not lines[0].strip():
		lines.pop(0)
	while lines and not lines[-1].strip():
		lines.pop()
	dedented = textwrap.dedent("\n".join(lines))
	cleaned_lines = [line.expandtabs(4).rstrip() for line in dedented.splitlines()]
	return "\n".join(cleaned_lines)


def extract_java_context(snippet: str) -> dict:
	"""
	Extract context information from a Java code snippet.
	Returns dict with: is_constructor, class_name, method_name, needs_mock_fields
	"""
	import re
	
	formatted = _format_java_snippet(snippet)
	lines = formatted.strip().split('\n')
	if not lines:
		return {'is_constructor': False, 'class_name': None, 'method_name': None, 'needs_mock_fields': False}
	
	# Get the first non-comment line (method/constructor signature)
	first_line = lines[0].strip()
	
	# Check if it's a constructor (no return type before the method name)
	# Pattern: [modifiers] ClassName(params)
	constructor_pattern = r'^\s*(public|private|protected)?\s*(static)?\s*([A-Z]\w+)\s*\('
	method_pattern = r'^\s*(public|private|protected)?\s*(static)?\s*(synchronized)?\s*(\w+)\s+(\w+)\s*\('
	
	is_constructor = False
	class_name = None
	method_name = None
	
	# Try to match constructor pattern
	constructor_match = re.search(constructor_pattern, first_line)
	if constructor_match:
		class_name = constructor_match.group(3)
		method_name = class_name
		is_constructor = True
	else:
		# Try to match method pattern
		method_match = re.search(method_pattern, first_line)
		if method_match:
			method_name = method_match.group(5)
			is_constructor = False
	
	# Check if code references undefined fields (common pattern: this.fieldName or just fieldName)
	needs_mock_fields = False
	field_patterns = [
		r'\bwriter\b',
		r'\bentries\b', 
		r'\bfileName\b',
		r'\baverages\b',
		r'\breaderContext\b',
		r'\bsource\b',
		r'\bscores\b',
		r'\bLOCAL\b',
		r'\bREMOTE\b',
		r'\btablename\b',
		r'\bstore\b',
		r'\bcfname\b',
	]
	
	code_text = '\n'.join(lines)
	for pattern in field_patterns:
		if re.search(pattern, code_text):
			needs_mock_fields = True
			break
	
	return {
		'is_constructor': is_constructor,
		'class_name': class_name,
		'method_name': method_name,
		'needs_mock_fields': needs_mock_fields,
		'formatted_snippet': formatted
	}


def extract_missing_symbols(compilation_error: str) -> dict:
	"""
	Parse javac compilation errors to extract missing symbols (classes, methods, variables).
	Returns dict with lists of missing classes, methods, variables, and constructor issues.
	"""
	import re
	
	missing_classes = set()
	missing_methods = set()
	missing_variables = set()
	constructor_issues = []
	
	# Pattern: "cannot find symbol" followed by "symbol: class ClassName"
	class_pattern = r'symbol:\s+class\s+(\w+)'
	for match in re.finditer(class_pattern, compilation_error):
		missing_classes.add(match.group(1))
	
	# Pattern: "cannot find symbol" followed by "symbol: variable VarName"
	var_pattern = r'symbol:\s+variable\s+(\w+)'
	for match in re.finditer(var_pattern, compilation_error):
		missing_variables.add(match.group(1))
	
	# Pattern: "cannot find symbol" followed by "symbol: method methodName"
	method_pattern = r'symbol:\s+method\s+(\w+)'
	for match in re.finditer(method_pattern, compilation_error):
		missing_methods.add(match.group(1))
	
	# Pattern: "constructor X cannot be applied to given types"
	# This indicates we need a parent class with matching constructor
	constructor_pattern = r'super\(([^)]*)\);'
	if 'constructor Object in class Object cannot be applied' in compilation_error:
		# Extract super() call parameters
		match = re.search(constructor_pattern, compilation_error)
		if match:
			params = match.group(1)
			constructor_issues.append(('needs_parent_constructor', params))
	
	# Pattern: "constructor ClassName in class ClassName cannot be applied to given types"
	# Extract required parameter information
	constructor_mismatch_pattern = r'constructor\s+(\w+)\s+in\s+class\s+\1\s+cannot\s+be\s+applied.*?required:\s+([^\n]+?)found:\s+([^\n]+?)reason:'
	for match in re.finditer(constructor_mismatch_pattern, compilation_error, re.DOTALL):
		class_name = match.group(1)
		required = match.group(2).strip()
		found = match.group(3).strip()
		if 'no arguments' in required and found != 'no arguments':
			# The stub class needs a constructor with the found arguments
			constructor_issues.append(('needs_constructor', class_name, found))
	
	return {
		'classes': list(missing_classes),
		'methods': list(missing_methods),
		'variables': list(missing_variables),
		'constructor_issues': constructor_issues
	}


def create_stub_definitions(missing_symbols: dict, snippet: str, context: dict = None) -> str:
	"""
	Create stub class, method, and field definitions for missing symbols.
	"""
	stubs = []
	
	# Handle constructor issues - create parent class
	if missing_symbols.get('constructor_issues'):
		for issue_type, params in missing_symbols['constructor_issues']:
			if issue_type == 'needs_parent_constructor' and context and context.get('is_constructor'):
				# Create a parent class with matching constructor
				class_name = context.get('class_name', 'TestClass')
				parent_class = f"{class_name}Parent"
				# Count parameters to create matching constructor
				param_count = len([p.strip() for p in params.split(',') if p.strip()])
				param_list = ', '.join([f'int p{i}' for i in range(param_count)])
				stubs.append(f"static class {parent_class} {{ public {parent_class}({param_list}) {{}} }}")
				# Note: This stub should be used to extend the class, which requires modifying the class declaration
	
	# Create stub classes
	for class_name in missing_symbols.get('classes', []):
		stubs.append(f"static class {class_name} {{ }}")
	
	# Create stub methods (analyze snippet to guess signature)
	for method_name in missing_symbols.get('methods', []):
		# Simple stub - void return, no params
		stubs.append(f"static void {method_name}() {{ }}")
		# Also try with common signatures
		stubs.append(f"static Object {method_name}(Object... args) {{ return null; }}")
	
	# Create stub variables/fields
	for var_name in missing_symbols.get('variables', []):
		stubs.append(f"static Object {var_name};")
	
	return '\n'.join(stubs) if stubs else ""


def java_snippet_compiles_with_inference(snippet: str, context: dict = None, max_iterations: int = 3):
	"""
	Test if a Java code snippet compiles, attempting to add missing symbols iteratively.
	Returns (success: bool, error_message: str, iterations_needed: int, added_stubs: str)
	
	Args:
		snippet: The Java code to test
		context: Optional context dict from extract_java_context
		max_iterations: Maximum attempts to fix missing symbols
	"""
	if context is None:
		context = extract_java_context(snippet)
	
	all_stubs = ""
	added_stub_types = set()  # Track what types of stubs we've added
	
	for iteration in range(max_iterations):
		# Try compilation with current stubs
		success, error = java_snippet_compiles(snippet, context, additional_stubs=all_stubs)
		
		if success:
			return True, "", iteration, all_stubs
		
		# If compilation failed, extract missing symbols
		missing = extract_missing_symbols(error)
		
		# If no missing symbols found (or different error), stop
		if not any([missing['classes'], missing['methods'], missing['variables'], missing.get('constructor_issues', [])]):
			return False, error, iteration, all_stubs
		
		# Create stubs for missing symbols (only if not already added)
		new_stubs = []
		
		# Handle constructor issues (only once per type)
		if missing.get('constructor_issues'):
			for issue in missing['constructor_issues']:
				issue_type = issue[0]
				
				if issue_type == 'needs_parent_constructor' and 'parent_class' not in added_stub_types:
					if context and context.get('is_constructor'):
						params = issue[1]
						class_name = context.get('class_name', 'TestClass')
						parent_class = f"{class_name}Parent"
						param_count = len([p.strip() for p in params.split(',') if p.strip()])
						param_list = ', '.join([f'int p{i}' for i in range(param_count)])
						new_stubs.append(f"class {parent_class} {{ public {parent_class}({param_list}) {{}} }}")
						added_stub_types.add('parent_class')
				
				elif issue_type == 'needs_constructor':
					class_name = issue[1]
					found_params = issue[2]
					# Parse the found parameter types (e.g., "StringReader" -> 1 param)
					param_types = [p.strip() for p in found_params.split(',') if p.strip()]
					if class_name not in added_stub_types:
						# Create class with matching constructor
						if param_types:
							# Create constructor with Object parameters
							param_list = ', '.join([f'Object p{i}' for i in range(len(param_types))])
							new_stubs.append(f"static class {class_name} {{ public {class_name}({param_list}) {{}} }}")
						else:
							new_stubs.append(f"static class {class_name} {{ }}")
						added_stub_types.add(class_name)
		
		# Add missing classes (avoid duplicates)
		for class_name in missing.get('classes', []):
			if class_name not in added_stub_types:
				new_stubs.append(f"static class {class_name} {{ }}")
				added_stub_types.add(class_name)
		
		# Add missing methods (avoid duplicates)
		for method_name in missing.get('methods', []):
			if method_name not in added_stub_types:
				new_stubs.append(f"static void {method_name}() {{ }}")
				new_stubs.append(f"static Object {method_name}(Object... args) {{ return null; }}")
				added_stub_types.add(method_name)
		
		# Add missing variables (avoid duplicates)
		for var_name in missing.get('variables', []):
			if var_name not in added_stub_types:
				new_stubs.append(f"static Object {var_name};")
				added_stub_types.add(var_name)
		
		if not new_stubs:  # No new stubs to add
			return False, error, iteration, all_stubs
		
		# Add new stubs
		new_stubs_str = '\n'.join(new_stubs)
		if all_stubs:
			all_stubs += "\n" + new_stubs_str
		else:
			all_stubs = new_stubs_str
	
	# Max iterations reached
	return False, error, max_iterations, all_stubs


def java_snippet_compiles(snippet: str, context: dict = None, additional_stubs: str = ""):
	"""
	Test if a Java code snippet compiles.
	Returns (success: bool, error_message: str)
	
	Args:
		snippet: The Java code to test
		context: Optional context dict from extract_java_context to wrap the code appropriately
	"""
	if context is None:
		context = extract_java_context(snippet)
	
	formatted = context.get('formatted_snippet', _format_java_snippet(snippet))
	
	# Build the wrapper based on context
	if context.get('is_constructor'):
		# For constructors, wrap in a class
		class_name = context.get('class_name', 'TestClass')
		indented = textwrap.indent(formatted, "    ")
		
		# Check if we have a parent class stub in additional_stubs and separate it
		extends_clause = ""
		parent_class_def = ""
		other_stubs = ""
		
		if additional_stubs:
			stub_lines = additional_stubs.split('\n')
			for line in stub_lines:
				if f"class {class_name}Parent" in line:
					# Extract parent class - needs to be outside main class
					parent_class_def = line + "\n"
					extends_clause = f" extends {class_name}Parent"
				elif line.strip():
					other_stubs += line + "\n"
		
		# Add mock fields if needed
		mock_fields = ""
		if context.get('needs_mock_fields'):
			mock_fields = """
    // Mock fields that might be needed
    private static Object writer;
    private static java.util.Map<String, Object> entries = new java.util.HashMap<>();
    private static String fileName = "test.txt";
    private static float[] averages;
    private static Object readerContext;
    private static Object source;
    private static Object scores;
    private static Object LOCAL;
    private static Object REMOTE;
    private static String tablename;
    private static Object store;
    private static String cfname;
"""
		
		# Add other stubs (non-parent-class stubs)
		if other_stubs:
			mock_fields += "\n" + textwrap.indent(other_stubs, "    ") + "\n"
		
		wrapped = textwrap.dedent(
			f"""
			import java.sql.*;
			import java.io.*;
			import java.util.*;
			
			{parent_class_def}
			public class {class_name}{extends_clause} {{
{mock_fields}
{indented}
			
			    public static void main(String[] args) throws Exception {{
			        // Test instantiation
			    }}
			}}
			"""
		)
	else:
		# For regular methods, wrap in a Snippet class
		indented = textwrap.indent(formatted, "    ") if formatted else ""
		
		# Add mock fields if needed
		mock_fields = ""
		if context.get('needs_mock_fields'):
			mock_fields = """
    // Mock fields that might be needed
		private static Object writer;
		private static java.util.Map<String, Object> entries = new java.util.HashMap<>();
		private static String fileName = "test.txt";
		private static float[] averages;
		private static Object readerContext;
		private static Object source;
		private static Object scores;
		private static Object LOCAL;
		private static Object REMOTE;
		private static String tablename;
		private static Object store;
		private static String cfname;
	"""
		
		# Add additional stubs
		if additional_stubs:
			mock_fields += "\n" + textwrap.indent(additional_stubs, "    ") + "\n"
		
		wrapped = textwrap.dedent(
			f"""
			import java.sql.*;
			import java.io.*;
			import java.util.*;
			
			public class Snippet {{
{mock_fields}
{indented}
			
			    public static void main(String[] args) throws Exception {{
			        // Test method call
			        printException("demo", new SQLException("msg", "23500"));
			    }}
			}}
			"""
		)
	
	with tempfile.TemporaryDirectory() as tmp:
		# Use appropriate class name for the file
		if context.get('is_constructor'):
			class_name = context.get('class_name', 'TestClass')
			path = pathlib.Path(tmp) / f"{class_name}.java"
		else:
			path = pathlib.Path(tmp) / "Snippet.java"
		
		path.write_text(wrapped, encoding="utf-8")
		
		# Build environment with JAVA_HOME
		env = os.environ.copy()
		if 'JAVA_HOME' in env:
			java_bin = os.path.join(env['JAVA_HOME'], 'bin')
			if 'PATH' in env:
				env['PATH'] = f"{java_bin}:{env['PATH']}"
			else:
				env['PATH'] = java_bin
		
		resp = subprocess.run(
			["javac", str(path)],
			text=True,
			capture_output=True,
			check=False,
			env=env,
		)
		return resp.returncode == 0, resp.stderr


# Extract context from fixed_function for each sample
print("Extracting context from fixed_function for each sample...")
contexts = []
for idx, row in comparison_df.iterrows():
	context = extract_java_context(row['fixed_function'])
	contexts.append(context)

comparison_df['java_context'] = contexts
print(f"Extracted context for {len(contexts)} samples")
print(f"Constructors: {sum(1 for c in contexts if c['is_constructor'])}")
print(f"Methods: {sum(1 for c in contexts if not c['is_constructor'])}")
print(f"Need mock fields: {sum(1 for c in contexts if c['needs_mock_fields'])}\n")

print("NOTE: Most samples require external libraries (Lucene, Cassandra, Derby, etc.)")
print("and cannot compile standalone. Context extraction is complete and saved to comparison_df.")
print("Use comparison_df['java_context'] to access extracted context for each sample.\n")


# Save comparison_df with context for later use
print("Saving comparison_df with extracted context...")
comparison_df.to_pickle("comparison_df_with_context.pkl")
print("Saved to: comparison_df_with_context.pkl\n")

# Optional: Test compilation with context inference
TEST_COMPILATION = True  # Set to True to test compilation with stub inference

if TEST_COMPILATION:
	# Test compilation on first sample
	print("="*80)
	print("TESTING COMPILATION WITH CONTEXT INFERENCE")
	print("="*80)
	print("Attempting to compile snippets by inferring missing symbols...\n")
	
	first_comparison = comparison_df.iloc[0]
	result, stderr = java_snippet_compiles(first_comparison['fixed_function'])
	if not result and "Unable to locate a Java Runtime" in stderr:
		print("\n" + "="*80)
		print("WARNING: Java Development Kit (JDK) is not installed.")
		print("Please install JDK to run Java compilation tests.")
		print("You can download it from: https://www.oracle.com/java/technologies/downloads/")
		print("Or install via Homebrew: brew install openjdk")
		print("="*80 + "\n")
		print("Skipping Java compilation tests...")
	else:
		# Run compilation test with inference on subset of entries
		num_samples = min(20, len(comparison_df))
		print(f"Testing {num_samples} samples with context inference...\n")
		
		compilation_results = []
		for idx, comparison in comparison_df.iloc[:num_samples].iterrows():
			context = comparison['java_context']
			
			# Use inference-based compilation
			fixed_success, fixed_err, fixed_iters, fixed_stubs = java_snippet_compiles_with_inference(
				comparison['fixed_function'], context
			)
			method1_success, method1_err, method1_iters, method1_stubs = java_snippet_compiles_with_inference(
				comparison['method1_direct_repaired_code'], context
			)
			method2_success, method2_err, method2_iters, method2_stubs = java_snippet_compiles_with_inference(
				comparison['method2_selection_repaired_code'], context
			)
			
			# Test candidates
			candidates_results = []
			for candidate in comparison['method2_selection_candidates']:
				cand_success, _, _, _ = java_snippet_compiles_with_inference(candidate['code'], context)
				candidates_results.append(cand_success)
			
			compilation_results.append({
				'idx': idx,
				'fixed': fixed_success,
				'fixed_iterations': fixed_iters,
				'fixed_stubs': fixed_stubs,
				'method1': method1_success,
				'method1_iterations': method1_iters,
				'method2': method2_success,
				'method2_iterations': method2_iters,
				'candidates': candidates_results,
				'context': context
			})
			
			status = "✓" if fixed_success else "✗"
			stub_info = f" (iters={fixed_iters})" if fixed_success and fixed_iters > 0 else ""
			print(f"[{idx+1}/{num_samples}] {status} Sample {idx}{stub_info}: fixed={fixed_success}, method1={method1_success}, method2={method2_success}, candidates={candidates_results}")
			
			# Show what stubs were needed for successful compilations
			if fixed_success and fixed_stubs:
				print(f"  Added stubs: {fixed_stubs[:200]}..." if len(fixed_stubs) > 200 else f"  Added stubs: {fixed_stubs}")
			
			# Debug: show error for first few failures
			if not fixed_success and idx < 3:
				print(f"  Error sample: {fixed_err[:400]}")
		
		# Summary
		print("\n" + "="*80)
		print("COMPILATION SUMMARY (With Context Inference)")
		print("="*80)
		total_samples = len(compilation_results)
		fixed_compiles = sum(1 for r in compilation_results if r['fixed'])
		method1_compiles = sum(1 for r in compilation_results if r['method1'])
		method2_compiles = sum(1 for r in compilation_results if r['method2'])
		
		print(f"Total samples tested: {total_samples}")
		print(f"Fixed function compiles: {fixed_compiles} ({fixed_compiles/total_samples*100:.1f}%)")
		print(f"Method1 compiles: {method1_compiles} ({method1_compiles/total_samples*100:.1f}%)")
		print(f"Method2 compiles: {method2_compiles} ({method2_compiles/total_samples*100:.1f}%)")
		print("="*80)


print("\n" + "="*80)
print("CONTEXT EXTRACTION COMPLETE")
print("="*80)
print("Java context has been extracted and saved for all samples.")
print("- comparison_df['java_context'] contains extracted context (constructor/method type, mock fields needed)")
print("- Saved to: comparison_df_with_context.pkl")
print("- Set TEST_COMPILATION = True to test compilation (requires JDK)")
print("="*80 + "\n")


def run_flag_to_float(flag) -> float:
	if flag is True:
		return 1.0
	if flag is False:
		return 0.0
	return math.nan


if not RESULTS_PATH.exists():
	raise FileNotFoundError(
		f"Missing {RESULTS_PATH}. Run the accuracy experiment before plotting."
	)

with RESULTS_PATH.open("r", encoding="utf-8") as f:
	results = json.load(f)

comparison_df = pd.json_normalize(results["comparison"], sep="_")

if COMPARISON_PICKLE_PATH.exists():
	runs_df = pd.read_pickle(COMPARISON_PICKLE_PATH)[["dataset_index", *RUN_COLUMNS]]
	comparison_df = comparison_df.merge(
		runs_df,
		on="dataset_index",
		how="left",
		suffixes=("", "_runs_backup"),
	)

missing_run_cols = [col for col in RUN_COLUMNS if col not in comparison_df.columns]
if missing_run_cols:
	raise ValueError(
		"Missing required run-status columns: " + ", ".join(missing_run_cols)
	)

scipy_stats = None
try:
	scipy_stats = importlib.import_module("scipy.stats")
except ModuleNotFoundError:
	scipy_stats = None


def run_regression(x_values: Iterable[float], y_values: Iterable[float]):
	x = np.asarray(list(x_values), dtype=float)
	y = np.asarray(list(y_values), dtype=float)

	if x.size < 2:
		return None

	if scipy_stats is not None:
		reg = scipy_stats.linregress(x, y)
		return reg

	x_mean = x.mean()
	y_mean = y.mean()
	x_diff = x - x_mean
	y_diff = y - y_mean
	ss_x = np.sum(x_diff ** 2)
	ss_y = np.sum(y_diff ** 2)

	if np.isclose(ss_x, 0.0):
		return None

	if np.isclose(ss_y, 0.0):
		return SimpleNamespace(
			slope=0.0,
			intercept=float(y_mean),
			rvalue=0.0,
			pvalue=1.0,
			stderr=float("nan"),
		)

	ss_xy = np.sum(x_diff * y_diff)
	slope = ss_xy / ss_x
	intercept = y_mean - slope * x_mean
	rvalue = ss_xy / math.sqrt(ss_x * ss_y)
	rvalue = max(min(rvalue, 1.0), -1.0)
	r_squared = rvalue ** 2
	n = x.size
	dof = n - 2
	if dof <= 0:
		pvalue = float("nan")
	else:
		if math.isclose(r_squared, 1.0):
			pvalue = 0.0
		else:
			t_stat = rvalue * math.sqrt(dof / (1 - r_squared))
			if scipy_stats is not None and hasattr(scipy_stats, "t"):
				t_dist = scipy_stats.t
				cdf = t_dist.cdf(abs(t_stat), dof)
				pvalue = 2 * (1 - cdf)
			else:
				z = abs(t_stat)
				sf = 0.5 * math.erfc(z / math.sqrt(2))
				pvalue = 2 * sf

	residuals = y - (slope * x + intercept)
	if dof > 0 and not np.isclose(ss_x, 0.0):
		stderr = math.sqrt(np.sum(residuals ** 2) / dof / ss_x)
	else:
		stderr = float("nan")

	return SimpleNamespace(
		slope=slope,
		intercept=intercept,
		rvalue=rvalue,
		pvalue=pvalue,
		stderr=stderr,
	)


candidate_rows = []
for row in comparison_df.itertuples():
	runs = row.method2_selection_candidates_runs
	run_flags = runs if isinstance(runs, list) else []
	selected_perplexity = getattr(row, "method2_selection_selected_perplexity", None)

	for idx, candidate in enumerate(row.method2_selection_candidates or []):
		run_flag = run_flags[idx] if idx < len(run_flags) else None
		candidate_rows.append(
			{
				"dataset_index": row.dataset_index,
				"candidate_id": candidate.get("candidate_id"),
				"perplexity": candidate.get("perplexity"),
				"run_success": run_flag_to_float(run_flag),
				"is_selected": (
					selected_perplexity is not None
					and candidate.get("perplexity") is not None
					and abs(candidate["perplexity"] - selected_perplexity) < 1e-9
				),
			}
		)

candidate_df = pd.DataFrame(candidate_rows)
if candidate_df.empty:
	raise ValueError("No method2_selection candidates found in results")

candidate_df = candidate_df.dropna(subset=["perplexity", "run_success"])
if candidate_df.empty:
	raise ValueError("Run-status data missing for all candidates")

perplexity_series = candidate_df["perplexity"].astype(float)
if perplexity_series.empty:
	raise ValueError("Perplexity values are missing for all candidates")

q1 = perplexity_series.quantile(0.25)
q3 = perplexity_series.quantile(0.75)
iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

filtered_df = candidate_df[
	candidate_df["perplexity"].between(lower_bound, upper_bound)
].copy()

if filtered_df.empty:
	raise ValueError("All candidate perplexities were removed as outliers")

min_perplexity = filtered_df["perplexity"].min()
max_perplexity = filtered_df["perplexity"].max()
perplexity_range = max_perplexity - min_perplexity

if np.isclose(perplexity_range, 0.0):
	filtered_df["perplexity_norm"] = 0.0
else:
	filtered_df["perplexity_norm"] = (
		filtered_df["perplexity"] - min_perplexity
	) / perplexity_range

selected_points_df = filtered_df.loc[filtered_df["is_selected"]].copy()
selected_points_df = selected_points_df.sort_values("dataset_index")
selected_points_df = selected_points_df.drop_duplicates("dataset_index")
selected_runs_map = {
	row.dataset_index: run_flag_to_float(row.method2_selection_repaired_code_runs)
	for row in comparison_df.itertuples()
}
selected_points_df["run_success"] = selected_points_df["dataset_index"].map(
	selected_runs_map
)
selected_points_df = selected_points_df.dropna(subset=["run_success"])

non_selected_df = filtered_df.loc[~filtered_df["is_selected"]].copy()

baseline_df = selected_points_df[["dataset_index", "perplexity_norm"]].merge(
	comparison_df[["dataset_index", "method1_direct_repaired_code_runs"]],
	on="dataset_index",
	how="inner",
)
baseline_df["run_success"] = baseline_df[
	"method1_direct_repaired_code_runs"
].apply(run_flag_to_float)
baseline_df = baseline_df.dropna(subset=["run_success"])


def regression_and_plot(subset_df, color, scatter_label, line_label):
	data = subset_df.dropna(subset=["perplexity_norm", "run_success"])
	if data.empty:
		return None

	x_values = data["perplexity_norm"].astype(float).to_numpy()
	y_values = data["run_success"].astype(float).to_numpy()

	plt.scatter(
		x_values,
		y_values,
		alpha=0.6 if scatter_label == "Candidate" else 0.9,
		label=scatter_label,
		color=color,
	)

	if np.unique(x_values).size < 2:
		print(
			f"Skipping regression for {line_label}: fewer than two unique normalized perplexity values."
		)
		return None

	regression = run_regression(x_values, y_values)
	if regression is None:
		print(f"Skipping regression for {line_label}: unable to compute linear fit.")
		return None

	line_x = np.linspace(0, 1, 100)
	line_y = regression.intercept + regression.slope * line_x
	plt.plot(
		line_x,
		line_y,
		color=color,
		linewidth=2,
		label=(
			f"{line_label} Fit "
			f"(R²={regression.rvalue ** 2:.3f}, p={regression.pvalue:.3e})"
		),
	)
	return regression


plt.figure(figsize=(10, 6))

candidate_regression = regression_and_plot(
	non_selected_df,
	color="tab:blue",
	scatter_label="Candidate",
	line_label="Candidate",
)

selected_regression = regression_and_plot(
	selected_points_df,
	color="tab:orange",
	scatter_label="Selected",
	line_label="Selected",
)

baseline_regression = None
if baseline_df.empty:
	print("Skipping Baseline plot: no overlapping samples after filtering.")
else:
	baseline_regression = regression_and_plot(
		baseline_df,
		color="tab:green",
		scatter_label="Baseline",
		line_label="Baseline",
	)

plt.xlabel("Normalized Method2 Candidate Perplexity")
plt.ylabel("Compilation Success (1 = runs)")
plt.title("Method2 Candidate Runs vs. Normalized Perplexity")
plt.xlim(0, 1)
plt.ylim(-0.1, 1.1)
plt.yticks([0, 1], ["Doesn't run", "Runs"])
plt.legend()
plt.tight_layout()
plt.savefig("method2_runs_vs_perplexity.png", dpi=300)
plt.close()

print("Saved plot to method2_runs_vs_perplexity.png")

if candidate_regression is not None:
	print(
		"Candidate regression: slope={:.4f}, intercept={:.4f}, R^2={:.4f}, p-value={:.4e}".format(
			candidate_regression.slope,
			candidate_regression.intercept,
			candidate_regression.rvalue ** 2,
			candidate_regression.pvalue,
		)
	)

if selected_regression is not None:
	print(
		"Selected regression: slope={:.4f}, intercept={:.4f}, R^2={:.4f}, p-value={:.4e}".format(
			selected_regression.slope,
			selected_regression.intercept,
			selected_regression.rvalue ** 2,
			selected_regression.pvalue,
		)
	)

if baseline_regression is not None:
	print(
		"Baseline regression: slope={:.4f}, intercept={:.4f}, R^2={:.4f}, p-value={:.4e}".format(
			baseline_regression.slope,
			baseline_regression.intercept,
			baseline_regression.rvalue ** 2,
			baseline_regression.pvalue,
		)
	)
