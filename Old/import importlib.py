import importlib
import json
import math
from pathlib import Path
from typing import Iterable
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_PATH = Path("accuracy_experiment_results.json")
COMPARISON_PICKLE_PATH = Path("comparison_df.pkl")
RUN_COLUMNS = [
	"method1_direct_repaired_code_runs",
	"method2_selection_repaired_code_runs",
	"method2_selection_candidates_runs",
]


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

	plt.scatter(
		x_values,
		y_values,
		alpha=0.6 if scatter_label == "Candidate" else 0.9,
		label=scatter_label,
		color=color,
	)
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
