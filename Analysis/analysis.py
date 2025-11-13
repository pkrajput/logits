import importlib
import json
import math
from difflib import SequenceMatcher
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

candidate_rows = []
for sample in results["comparison"]:
	fixed_code = sample["fixed_function"]
	method2_data = sample.get("method2_selection", {})
	selected_perplexity = method2_data.get("selected_perplexity")

	for candidate in method2_data.get("candidates", []):
		similarity = SequenceMatcher(None, candidate["code"], fixed_code).ratio()
		candidate_rows.append(
			{
				"dataset_index": sample["dataset_index"],
				"candidate_id": candidate.get("candidate_id"),
				"perplexity": candidate.get("perplexity"),
				"mean_logprob": candidate.get("mean_logprob"),
				"similarity": similarity,
				"is_selected": (
					selected_perplexity is not None
					and candidate.get("perplexity") is not None
					and abs(candidate["perplexity"] - selected_perplexity) < 1e-12
				),
			}
		)

candidate_similarity_df = pd.DataFrame(candidate_rows)

baseline_rows = []
for sample in results["comparison"]:
	method1_data = sample.get("method1_direct", {})
	repaired_code = method1_data.get("repaired_code")
	if not repaired_code:
		continue
	similarity = SequenceMatcher(None, repaired_code, sample["fixed_function"]).ratio()
	baseline_rows.append(
		{
			"dataset_index": sample["dataset_index"],
			"similarity": similarity,
		}
	)

method1_similarity_df = pd.DataFrame(baseline_rows)

scipy_stats = None
try:
	scipy_stats = importlib.import_module("scipy.stats")
except ModuleNotFoundError:
	scipy_stats = None


def run_regression(x_values, y_values):
	"""Compute linear regression and return a namespace mimicking scipy's API."""
	x = np.asarray(x_values, dtype=float)
	y = np.asarray(y_values, dtype=float)

	if x.size < 2:
		return None

	if scipy_stats is not None:
		reg = scipy_stats.linregress(x, y)
		return reg

	# Manual fallback mirroring scipy.stats.linregress
	x_mean = x.mean()
	y_mean = y.mean()
	x_diff = x - x_mean
	y_diff = y - y_mean
	ss_x = np.sum(x_diff ** 2)
	ss_y = np.sum(y_diff ** 2)

	if np.isclose(ss_x, 0.0) or np.isclose(ss_y, 0.0):
		return None

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
				# Normal approximation fallback for environments without SciPy.
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


if candidate_similarity_df.empty:
	raise ValueError("No method2_selection candidates found in results")

perplexity_series = candidate_similarity_df["perplexity"].dropna()
if perplexity_series.empty:
	raise ValueError("Perplexity values are missing for all candidates")

q1 = perplexity_series.quantile(0.25)
q3 = perplexity_series.quantile(0.75)
iqr = q3 - q1

# Remove outliers using the IQR rule to avoid skewing normalization and regression.
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

filtered_df = candidate_similarity_df[
	candidate_similarity_df["perplexity"].between(lower_bound, upper_bound)
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

# Align method1 similarity with the normalized perplexity of the selected candidate.
baseline_df = pd.merge(
	selected_points_df[["dataset_index", "perplexity_norm"]],
	method1_similarity_df,
	on="dataset_index",
	how="inner",
)


def regression_and_plot(subset_df, color, scatter_label, line_label):
	if subset_df.empty:
		return None

	unique_x = subset_df["perplexity_norm"].nunique()
	if unique_x < 2:
		print(
			f"Skipping regression for {line_label}: normalized perplexity has fewer than two unique values."
		)
		return None

	regression = run_regression(
		subset_df["perplexity_norm"], subset_df["similarity"]
	)
	if regression is None:
		print(
			f"Skipping regression for {line_label}: unable to compute linear fit."
		)
		return None
	line_x = np.linspace(0, 1, 100)
	line_y = regression.intercept + regression.slope * line_x
	plt.scatter(
		subset_df["perplexity_norm"],
		subset_df["similarity"],
		alpha=0.6 if scatter_label == "Candidate" else 0.9,
		label=scatter_label,
		color=color,
	)
	plt.plot(
		line_x,
		line_y,
		color=color,
		linewidth=2,
		label=f"{line_label} Fit (R²={regression.rvalue ** 2:.3f}, p={regression.pvalue:.3e})",
	)
	return regression


plt.figure(figsize=(10, 6))

candidate_regression = regression_and_plot(
	filtered_df.loc[~filtered_df["is_selected"]],
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
plt.ylabel("Similarity to Fixed Function (SequenceMatcher ratio)")
plt.title("Method2 Candidate Similarity vs. Normalized Perplexity")
plt.xlim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("method2_similarity_vs_perplexity.png", dpi=300)
plt.close()

print("Saved plot to method2_similarity_vs_perplexity.png")

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
