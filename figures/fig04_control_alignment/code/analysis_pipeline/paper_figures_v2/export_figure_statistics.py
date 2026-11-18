"""Export the 16-sheet figure workbook from stored trajectories.

Point estimates are recomputed from the cached runs. Bootstrap percentile
intervals use generator seed 20261030. Two-sided p-values are Monte Carlo
permutation tests of the mean difference (seed 20261031, 20,000 shuffles)
and are not read off the percentile interval. Holm adjustment is applied
inside each family listed in statistics_metadata.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

from analysis_pipeline.paper_figures.data import _trapz, _unique_rate
from analysis_pipeline.paper_figures_v2.data_bundle import OUT, load_or_build
from analysis_pipeline.paper_figures_v2.entropy_matched import analyse as entropy_matched_analyse
from analysis_pipeline.paper_figures_v2.entropy_matched import xy_table as entropy_matched_xy
from analysis_pipeline.paper_figures_v2.plot_style import LABELS, POLICIES

N_RUNS = 20
N_TRIALS = 200
DRAWS = 20000
THRESHOLD = 0.030
N_BINS = 24
MIN_FIGURE_BIN = 8
THETA = {"wx": 1.3, "wy": -0.9, "b": 0.4}
BOOT_SEED = 20261030
PERM_SEED = 20261031
XLSX = OUT / "DNE_Figure_Statistics.xlsx"
CSV_DIR = OUT / "figure_statistics_tables"
HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(color="FFFFFF", bold=True, name="Calibri", size=10)
SECTION_FONT = Font(bold=True, color="1F4E79", name="Calibri", size=12)
TITLE_FONT = Font(bold=True, name="Calibri", size=14)


def holm(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adjusted = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * float(pvals[idx])))
        adjusted[idx] = running
    return adjusted


def boot_mean_ci(values: np.ndarray, rng: np.random.Generator):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else np.nan
    sem = float(sd / np.sqrt(n)) if n > 1 else np.nan
    if n == 0:
        return mean, sd, sem, np.nan, np.nan, 0
    idx = rng.integers(0, n, size=(DRAWS, n))
    lo, hi = np.quantile(values[idx].mean(axis=1), [0.025, 0.975])
    return mean, sd, sem, float(lo), float(hi), n


def boot_curve_ci(samples: np.ndarray, rng: np.random.Generator):
    """samples: (run, trial). One resample of runs is applied to every trial."""
    n, n_trial = samples.shape
    idx = rng.integers(0, n, size=(DRAWS, n))
    means = samples[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975], axis=0)
    mean = samples.mean(axis=0)
    sd = samples.std(axis=0, ddof=1)
    return mean, sd, sd / np.sqrt(n), lo, hi


def boot_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ia = rng.integers(0, len(a), size=(DRAWS, len(a)))
    ib = rng.integers(0, len(b), size=(DRAWS, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def permutation_p(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = abs(float(a.mean() - b.mean()))
    pooled = np.concatenate([a, b])
    keys = rng.random((DRAWS, pooled.size))
    chosen = np.argpartition(keys, len(a), axis=1)[:, : len(a)]
    chosen_sum = pooled[chosen].sum(axis=1)
    other_mean = (pooled.sum() - chosen_sum) / len(b)
    diffs = np.abs(chosen_sum / len(a) - other_mean)
    return float((np.sum(diffs >= observed - 1e-12) + 1) / (DRAWS + 1))


def contrast_table(series: dict, family: str, environment: str, endpoint: str, rng_boot, rng_perm) -> pd.DataFrame:
    rows = []
    dne = np.asarray(series["dne"], dtype=float)
    for pol in POLICIES:
        if pol == "dne":
            continue
        other = np.asarray(series[pol], dtype=float)
        diff, lo, hi = boot_diff(dne, other, rng_boot)
        rows.append({
            "family": family,
            "environment": environment,
            "endpoint": endpoint,
            "comparator": pol,
            "n_dne": int(np.isfinite(dne).sum()),
            "n_comparator": int(np.isfinite(other).sum()),
            "mean_dne": float(np.nanmean(dne)),
            "mean_comparator": float(np.nanmean(other)),
            "diff_dne_minus_comparator": diff,
            "boot_ci_lo": lo,
            "boot_ci_hi": hi,
            "ci_excludes_zero": bool(hi < 0 or lo > 0),
            "p_permutation": permutation_p(dne[np.isfinite(dne)], other[np.isfinite(other)], rng_perm),
        })
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm(frame["p_permutation"].to_numpy())
    frame["holm_family_size"] = len(frame)
    return frame


def trajectory_tables(raw, environment: str, metric: str, value_name: str, rng):
    raw_rows = []
    summary_rows = []
    for pol in POLICIES:
        values = raw[environment][pol][metric]
        mean, sd, sem, lo, hi = boot_curve_ci(values, rng)
        for run in range(N_RUNS):
            for trial in range(N_TRIALS):
                raw_rows.append({
                    "environment": environment,
                    "policy": pol,
                    "run_id": run,
                    "trial": trial,
                    value_name: float(values[run, trial]),
                })
        for trial in range(N_TRIALS):
            summary_rows.append({
                "environment": environment,
                "policy": pol,
                "trial": trial,
                "n_runs": N_RUNS,
                "mean": float(mean[trial]),
                "sd": float(sd[trial]),
                "sem": float(sem[trial]),
                "boot_ci_lo": float(lo[trial]),
                "boot_ci_hi": float(hi[trial]),
            })
    return pd.DataFrame(raw_rows), pd.DataFrame(summary_rows)


def xy_graph_table(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per trial. Column A is X. Each model then has mean, SEM, and n."""
    wide = {"trial": np.arange(N_TRIALS)}
    for pol in POLICIES:
        part = summary.loc[summary.policy == pol].sort_values("trial")
        label = LABELS[pol]
        wide[f"{label} mean"] = part["mean"].to_numpy()
        wide[f"{label} sem"] = part["sem"].to_numpy()
        wide[f"{label} n"] = part["n_runs"].to_numpy()
    return pd.DataFrame(wide)


def model_summary_row(summary: pd.DataFrame) -> pd.DataFrame:
    """One row. Each model has mean, SEM, and n in adjacent columns."""
    wide = {}
    for pol in POLICIES:
        row = summary.loc[summary.policy == pol].iloc[0]
        label = LABELS[pol]
        wide[f"{label} mean"] = [float(row["mean"])]
        wide[f"{label} sem"] = [float(row["sem"])]
        wide[f"{label} n"] = [int(row["n_runs"])]
    return pd.DataFrame(wide)


def attainment_xy(raw_time: pd.DataFrame) -> pd.DataFrame:
    """Panel A: shared trial axis and one proportion column per model."""
    wide = {"trial": np.arange(N_TRIALS)}
    trials = np.arange(N_TRIALS)
    for pol in POLICIES:
        part = raw_time.loc[raw_time.policy == pol].sort_values("run_id")
        event = np.where(part["attained"].to_numpy() == 1, part["first_hit_trial"].to_numpy(), np.inf)
        wide[LABELS[pol]] = (event[:, None] <= trials[None, :]).mean(axis=0)
    return pd.DataFrame(wide)


def bin_xy(weighted: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    """X is the bin number. Each model then has functional-MAE mean, SEM, and n."""
    bins = np.sort(weighted["bin_id"].unique())
    wide = {"bin": bins}
    for pol in POLICIES:
        part = weighted.loc[weighted.policy == pol].sort_values("bin_id")
        pool = pooled.loc[pooled.policy == pol].sort_values("bin_id")
        show = pool["n_trials_pooled"].to_numpy() >= MIN_FIGURE_BIN
        label = LABELS[pol]
        mean = part["equal_run_mean_functional_mae"].to_numpy(float)
        sem = part["sem_functional_mae"].to_numpy(float)
        entered = part["n_runs_entered"].to_numpy(float)
        wide[f"{label} mean"] = np.where(show, mean, np.nan)
        wide[f"{label} sem"] = np.where(show, sem, np.nan)
        wide[f"{label} n"] = np.where(show, entered, np.nan)
    return pd.DataFrame(wide)


def grouped_summary(summary: pd.DataFrame, row_column: str, row_order: list, value_column: str, groups) -> pd.DataFrame:
    """groups is a list of (column_name, match_dict, header_prefix)."""
    records = []
    for row_value in row_order:
        record = {row_column: row_value}
        for prefix, match in groups:
            picked = summary
            for key, expected in match.items():
                picked = picked.loc[picked[key] == expected]
            picked = picked.loc[picked[row_column] == row_value]
            row = picked.iloc[0]
            record[f"{prefix} mean"] = float(row[value_column])
            record[f"{prefix} sem"] = float(row["sem"])
            record[f"{prefix} n"] = int(row["n_runs"])
        records.append(record)
    return pd.DataFrame(records)


def auc_tables(raw, environment: str, metric: str, value_name: str, family: str, rng_boot, rng_perm):
    per_run = {}
    raw_rows = []
    for pol in POLICIES:
        scores = _trapz(raw[environment][pol][metric])
        per_run[pol] = scores
        for run, score in enumerate(scores):
            raw_rows.append({
                "environment": environment,
                "policy": pol,
                "run_id": run,
                value_name: float(score),
            })
    summary = []
    for pol in POLICIES:
        mean, sd, sem, lo, hi, n = boot_mean_ci(per_run[pol], rng_boot)
        summary.append({
            "environment": environment, "policy": pol, "n_runs": n,
            "mean": mean, "sd": sd, "sem": sem, "boot_ci_lo": lo, "boot_ci_hi": hi,
        })
    comparisons = contrast_table(per_run, family, environment, value_name, rng_boot, rng_perm)
    return pd.DataFrame(raw_rows), pd.DataFrame(summary), comparisons, per_run


def first_hit(func: np.ndarray, threshold: float):
    hit = func <= threshold
    attained = hit.any(axis=1)
    first = np.where(attained, hit.argmax(axis=1), np.nan)
    return first, attained


def time_tables(raw, environment: str, family: str, rng_boot, rng_perm):
    raw_rows = []
    restricted = {}
    attained_matrix = {}
    for pol in POLICIES:
        first, attained = first_hit(raw[environment][pol]["func"], THRESHOLD)
        restricted[pol] = np.where(attained, first, N_TRIALS).astype(float)
        attained_matrix[pol] = attained
        for run in range(N_RUNS):
            raw_rows.append({
                "environment": environment,
                "policy": pol,
                "run_id": run,
                "threshold": THRESHOLD,
                "first_hit_trial": float(first[run]) if attained[run] else np.nan,
                "attained": int(attained[run]),
                "attained_on_trial_199": int(attained[run] and first[run] == 199),
                "censored_at_200": int(not attained[run]),
                "restricted_time": float(restricted[pol][run]),
            })
    summary = []
    for pol in POLICIES:
        mean, sd, sem, lo, hi, n = boot_mean_ci(restricted[pol], rng_boot)
        summary.append({
            "environment": environment, "policy": pol, "threshold": THRESHOLD, "n_runs": n,
            "n_attained": int(attained_matrix[pol].sum()),
            "n_censored": int((~attained_matrix[pol]).sum()),
            "rmst": mean, "sd": sd, "sem": sem, "boot_ci_lo": lo, "boot_ci_hi": hi,
        })
    cumulative = []
    for pol in POLICIES:
        times = restricted[pol]
        attained = attained_matrix[pol]
        event_time = np.where(attained, times, np.inf)
        for trial in range(N_TRIALS):
            count = int(np.sum(event_time <= trial))
            cumulative.append({
                "environment": environment, "policy": pol, "trial": trial, "threshold": THRESHOLD,
                "n_runs": N_RUNS, "n_attained_by_trial": count, "fraction_attained": count / N_RUNS,
            })
    comparisons = contrast_table(restricted, family, environment, "rmst_threshold_0.030", rng_boot, rng_perm)
    return pd.DataFrame(raw_rows), pd.DataFrame(summary), pd.DataFrame(cumulative), comparisons


def bin_edges(raw, environment: str):
    pooled = np.concatenate([raw[environment][pol]["func"].ravel() for pol in POLICIES])
    edges = np.linspace(float(pooled.min()), float(pooled.max()), N_BINS + 1)
    return edges


def bin_tables(raw, environment: str, edges: np.ndarray, rng):
    raw_rows = []
    run_means = {pol: [] for pol in POLICIES}
    for pol in POLICIES:
        func = raw[environment][pol]["func"]
        entropy = raw[environment][pol]["entropy"]
        for run in range(N_RUNS):
            for bin_id in range(N_BINS):
                if bin_id == N_BINS - 1:
                    mask = (func[run] >= edges[bin_id]) & (func[run] <= edges[bin_id + 1])
                else:
                    mask = (func[run] >= edges[bin_id]) & (func[run] < edges[bin_id + 1])
                n_trials = int(mask.sum())
                raw_rows.append({
                    "environment": environment,
                    "policy": pol,
                    "run_id": run,
                    "bin_id": bin_id + 1,
                    "bin_lo": float(edges[bin_id]),
                    "bin_hi": float(edges[bin_id + 1]),
                    "n_trials_in_bin": n_trials,
                    "entered_bin": int(n_trials > 0),
                    "mean_functional_mae": float(func[run, mask].mean()) if n_trials else np.nan,
                    "mean_posterior_entropy": float(entropy[run, mask].mean()) if n_trials else np.nan,
                })
    raw_frame = pd.DataFrame(raw_rows)
    pooled_rows = []
    weighted_rows = []
    for pol in POLICIES:
        func = raw[environment][pol]["func"].ravel()
        entropy = raw[environment][pol]["entropy"].ravel()
        entered = raw_frame[raw_frame.policy == pol]
        for bin_id in range(N_BINS):
            if bin_id == N_BINS - 1:
                mask = (func >= edges[bin_id]) & (func <= edges[bin_id + 1])
            else:
                mask = (func >= edges[bin_id]) & (func < edges[bin_id + 1])
            n_trials = int(mask.sum())
            pooled_rows.append({
                "environment": environment, "policy": pol, "bin_id": bin_id + 1,
                "n_trials_pooled": n_trials,
                "pooled_mean_functional_mae": float(func[mask].mean()) if n_trials else np.nan,
                "pooled_mean_posterior_entropy": float(entropy[mask].mean()) if n_trials else np.nan,
                "displayed_in_current_figure": int(n_trials >= MIN_FIGURE_BIN),
                "inference": "pooled trials are descriptive only; they are not an independent sample",
            })
            run_entropy = entered.loc[entered.bin_id == bin_id + 1, "mean_posterior_entropy"].to_numpy(float)
            run_mae = entered.loc[entered.bin_id == bin_id + 1, "mean_functional_mae"].to_numpy(float)
            present = np.isfinite(run_entropy)
            if present.any():
                mean_e, sd_e, sem_e, lo_e, hi_e, n_e = boot_mean_ci(run_entropy[present], rng)
                mean_m, sd_m, sem_m, lo_m, hi_m, _n = boot_mean_ci(run_mae[present], rng)
            else:
                mean_e = sd_e = sem_e = lo_e = hi_e = mean_m = sd_m = sem_m = lo_m = hi_m = np.nan
                n_e = 0
            weighted_rows.append({
                "environment": environment, "policy": pol, "bin_id": bin_id + 1,
                "n_runs_entered": n_e,
                "equal_run_mean_functional_mae": mean_m,
                "sd_functional_mae": sd_m, "sem_functional_mae": sem_m,
                "boot_ci_lo_functional_mae": lo_m, "boot_ci_hi_functional_mae": hi_m,
                "equal_run_mean_posterior_entropy": mean_e,
                "sd_entropy": sd_e, "sem_entropy": sem_e,
                "boot_ci_lo_entropy": lo_e, "boot_ci_hi_entropy": hi_e,
            })
    return raw_frame, pd.DataFrame(pooled_rows), pd.DataFrame(weighted_rows)


def parameter_tables(raw, rng_boot, rng_perm):
    raw_rows = []
    by_key = {}
    for environment in ("full", "noisy"):
        for pol in POLICIES:
            for run in range(N_RUNS):
                record = {"environment": environment, "policy": pol, "run_id": run, "trial": 199}
                for key, truth in THETA.items():
                    estimate = float(raw[environment][pol][key][run, -1])
                    signed = estimate - truth
                    record[f"posterior_mean_{key}"] = estimate
                    record[f"signed_error_{key}"] = signed
                    record[f"absolute_error_{key}"] = abs(signed)
                    by_key.setdefault((environment, key), {}).setdefault(pol, []).append(abs(signed))
                raw_rows.append(record)
    for key in by_key:
        for pol in by_key[key]:
            by_key[key][pol] = np.asarray(by_key[key][pol], dtype=float)
    summary = []
    comparisons = []
    for environment in ("full", "noisy"):
        for key in ("wx", "wy", "b"):
            for pol in POLICIES:
                values = by_key[(environment, key)][pol]
                mean, sd, sem, lo, hi, n = boot_mean_ci(values, rng_boot)
                summary.append({
                    "environment": environment, "parameter": key, "policy": pol, "n_runs": n,
                    "theta_star": THETA[key], "mean_absolute_error": mean, "sd": sd, "sem": sem,
                    "boot_ci_lo": lo, "boot_ci_hi": hi,
                })
            family = f"absolute_error_{key}_{environment}"
            comparisons.append(contrast_table(by_key[(environment, key)], family, environment, f"absolute_error_{key}", rng_boot, rng_perm))
    return pd.DataFrame(raw_rows), pd.DataFrame(summary), pd.concat(comparisons, ignore_index=True)


def spatial_tables(raw, rng_boot, rng_perm):
    circle = lambda x, y: x ** 2 + y ** 2 <= 0.25
    raw_rows = []
    trial_rates = {}
    unique_rates = {}
    for pol in POLICIES:
        sx = raw["full_trap"][pol]["sx"]
        sy = raw["full_trap"][pol]["sy"]
        trial_rates[pol] = []
        unique_rates[pol] = []
        reference_unique = _unique_rate(sx, sy, circle)
        for run in range(N_RUNS):
            inside = circle(sx[run], sy[run])
            coords = np.round(np.column_stack([sx[run], sy[run]]), 6)
            _values, inverse = np.unique(coords, axis=0, return_inverse=True)
            first = {}
            for position, code in enumerate(inverse.tolist()):
                first.setdefault(code, position)
            idx = np.fromiter(first.values(), dtype=int)
            n_unique = int(idx.size)
            n_unique_inside = int(np.sum(circle(sx[run, idx], sy[run, idx])))
            n_inside = int(inside.sum())
            trial_rates[pol].append(n_inside / N_TRIALS)
            unique_rates[pol].append(n_unique_inside / n_unique)
            if abs(unique_rates[pol][-1] - float(reference_unique[run])) > 1e-12:
                raise RuntimeError(f"unique-circle fraction does not match the stored definition for {pol} run {run}")
            raw_rows.append({
                "environment": "full_with_spatial_logging",
                "policy": pol,
                "run_id": run,
                "n_trials": N_TRIALS,
                "n_circle_trials": n_inside,
                "circle_trial_fraction": n_inside / N_TRIALS,
                "n_unique_points": n_unique,
                "n_unique_in_circle": n_unique_inside,
                "unique_circle_fraction": float(unique_rates[pol][run]),
                "note": "Spatial logging does not change the policy. The circle is a record, not a barrier.",
            })
        trial_rates[pol] = np.asarray(trial_rates[pol], dtype=float)
    summary = []
    for pol in POLICIES:
        for name, values in (("circle_trial_fraction", trial_rates[pol]), ("unique_circle_fraction", unique_rates[pol])):
            mean, sd, sem, lo, hi, n = boot_mean_ci(values, rng_boot)
            summary.append({
                "environment": "full_with_spatial_logging", "policy": pol, "metric": name,
                "n_runs": n, "mean": mean, "sd": sd, "sem": sem, "boot_ci_lo": lo, "boot_ci_hi": hi,
            })
    comparisons = pd.concat([
        contrast_table(trial_rates, "circle_trial_fraction", "full_with_spatial_logging", "circle_trial_fraction", rng_boot, rng_perm),
        contrast_table(unique_rates, "circle_unique_fraction", "full_with_spatial_logging", "unique_circle_fraction", rng_boot, rng_perm),
    ], ignore_index=True)
    return pd.DataFrame(raw_rows), pd.DataFrame(summary), comparisons


def write_sheet(writer, sheet: str, blocks: list[tuple[str, pd.DataFrame]]):
    row = 0
    ranges = []
    for title, frame in blocks:
        if title not in {"XY_GRAPH", "MODEL_SUMMARY"}:
            pd.DataFrame({title: [""]}).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=True)
            row += 2
        header_row = row
        frame.to_excel(writer, sheet_name=sheet, startrow=row, index=False)
        ranges.append((title, header_row + 1, header_row + 1 + len(frame), frame.shape[1]))
        row += len(frame) + 3
    return ranges


COUNT_HEADERS = {
    "trial", "run_id", "n_runs", "n_attained", "n_censored", "n_attained_by_trial",
    "n_trials", "n_trials_in_bin", "n_trials_pooled", "n_runs_entered", "n_circle_trials",
    "n_unique_points", "n_unique_in_circle", "bin_id", "attained", "attained_on_trial_199",
    "censored_at_200", "entered_bin", "displayed_in_current_figure", "holm_family_size",
    "n_dne", "n_comparator",
}
TEXT_HEADERS = {"environment", "policy", "note", "inference", "family", "endpoint", "comparator", "parameter", "metric"}


def _number_format(header: str):
    if header.startswith("p_") or header in {"ci_excludes_zero"}:
        return "0.000000" if header.startswith("p_") else "0"
    if header in COUNT_HEADERS or header in {"trial", "bin"} or header.endswith(" n"):
        return "0"
    if header in TEXT_HEADERS:
        return None
    return "0.000000"


def format_workbook(path: Path, layouts: dict):
    from openpyxl import load_workbook
    book = load_workbook(path)
    used_names = set()
    for sheet, ranges in layouts.items():
        ws = book[sheet]
        ws.freeze_panes = f"A{ranges[0][1] + 1}"
        ws.page_setup.orientation = "landscape"
        ws.page_setup.fitToPage = True
        ws.page_setup.fitToWidth = 1
        ws.page_setup.fitToHeight = 0
        ws.page_setup.paperSize = ws.PAPERSIZE_A4
        ws.sheet_properties.tabColor = "1F4E79"
        ws.sheet_view.showGridLines = False
        ws.page_setup.horizontalCentered = True
        ws.sheet_view.zoomScale = 110
        ws.oddHeader.left.text = sheet
        for index, (title, header_excel_row, last_row, ncols) in enumerate(ranges):
            if header_excel_row > 2:
                section_cell = ws.cell(header_excel_row - 2, 1)
                section_cell.font = SECTION_FONT
            formats = []
            for col in range(1, ncols + 1):
                cell = ws.cell(header_excel_row, col)
                cell.fill = HEADER_FILL
                cell.font = HEADER_FONT
                cell.alignment = Alignment(wrap_text=True, vertical="center")
                letter = get_column_letter(col)
                ws.column_dimensions[letter].width = max(ws.column_dimensions[letter].width or 0, min(42, len(str(cell.value)) + 3))
                formats.append(_number_format(str(cell.value)))
            for excel_row in ws.iter_rows(min_row=header_excel_row + 1, max_row=last_row, max_col=ncols):
                for cell, number_format in zip(excel_row, formats):
                    if number_format is not None and isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                        cell.number_format = number_format
            table_name = f"{sheet[:18]}_{index}"
            while table_name in used_names:
                table_name += "x"
            used_names.add(table_name)
            ref = f"A{header_excel_row}:{get_column_letter(ncols)}{last_row}"
            table = Table(displayName=table_name, ref=ref)
            table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False)
            ws.add_table(table)
        ws.row_dimensions[1].height = 18
        ws.sheet_view.view = "normal"
        ws.print_title_rows = "1:1" if ranges[0][1] == 1 else "1:3"
    book.save(path)


def export_csv(sheet: str, blocks: list[tuple[str, pd.DataFrame]]):
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    for title, frame in blocks:
        slug = title.lower().replace(" ", "_")
        frame.to_csv(CSV_DIR / f"{sheet}__{slug}.csv", index=False)


def verify(bundle, auc_values, rmst_values, abs_values, entropy_means, spatial_summary, rmst_comparisons, pooled_full) -> list[str]:
    notes = []
    for environment in ("full", "noisy"):
        for metric in ("func", "struct"):
            for pol in POLICIES:
                fresh = float(np.mean(auc_values[(environment, metric)][pol]))
                stored = float(np.mean(bundle["auc"][environment][pol][metric]))
                if abs(fresh - stored) > 1e-8:
                    notes.append(f"DISCREPANCY AUC {environment} {metric} {pol}: {fresh} vs {stored}")
        for pol in POLICIES:
            for key, truth in THETA.items():
                fresh = float(np.mean(abs_values[(environment, key)][pol]))
                stored = bundle["levels"][(environment, pol, f"abs_{key}")][0]
                if abs(fresh - stored) > 1e-8:
                    notes.append(f"DISCREPANCY absolute {key} {environment} {pol}: {fresh} vs {stored}")
            fresh_curve = entropy_means[(environment, pol)]
            stored_curve = bundle["series"][(environment, pol, "entropy")][0]
            gap = float(np.max(np.abs(fresh_curve - stored_curve)))
            if gap > 1e-8:
                notes.append(f"DISCREPANCY entropy mean {environment} {pol}: max abs {gap}")
    stored_rmst = pd.read_csv(OUT / "time_to_alignment_rmst.csv")
    hit = stored_rmst[(stored_rmst.kind == "first_hit") & np.isclose(stored_rmst.threshold, THRESHOLD)]
    for pol in POLICIES:
        row = hit[(hit.policy == pol)]
        if row.empty:
            notes.append(f"MISSING stored RMST for {pol}")
            continue
        fresh = float(np.mean(rmst_values[("full", pol)]))
        stored = float(row.iloc[0].rmst)
        if abs(fresh - stored) > 1e-8:
            notes.append(f"DISCREPANCY RMST full {pol}: {fresh} vs {stored}")
        fresh_reached = int(np.sum(rmst_values[("full", pol)] < N_TRIALS))
        if fresh_reached != int(row.iloc[0].n_reached):
            notes.append(f"DISCREPANCY attainment count full {pol}: {fresh_reached} vs {int(row.iloc[0].n_reached)}")
    stored_contrasts = pd.read_csv(OUT / "time_to_alignment_contrasts.csv")
    stored_hit = stored_contrasts[(stored_contrasts.kind == "first_hit") & np.isclose(stored_contrasts.threshold, THRESHOLD)]
    endpoint_gaps = []
    for _, stored_row in stored_hit.iterrows():
        match = rmst_comparisons[rmst_comparisons.comparator == stored_row.policy]
        if match.empty:
            notes.append(f"MISSING recomputed RMST contrast for {stored_row.policy}")
            continue
        fresh_row = match.iloc[0]
        if abs(float(fresh_row.diff_dne_minus_comparator) - float(stored_row.diff_dne_minus_other)) > 1e-8:
            notes.append(f"DISCREPANCY RMST difference {stored_row.policy}: {fresh_row.diff_dne_minus_comparator} vs {stored_row.diff_dne_minus_other}")
        endpoint_gaps.append(abs(float(fresh_row.boot_ci_lo) - float(stored_row.ci_lo)))
        endpoint_gaps.append(abs(float(fresh_row.boot_ci_hi) - float(stored_row.ci_hi)))
    if endpoint_gaps:
        notes.append(
            "Full RMST confidence-interval endpoints were recomputed with bootstrap seed "
            f"{BOOT_SEED}; the largest absolute difference from the figure file (seed 20261027) is {max(endpoint_gaps):.4f}."
        )
    circle = bundle["spatial"]
    circle = circle[circle.region == "circle"]
    for pol in POLICIES:
        stored_row = circle[circle.policy == pol].iloc[0]
        for metric, column in (("circle_trial_fraction", "trial_rate"), ("unique_circle_fraction", "unique_rate")):
            fresh = float(spatial_summary[(spatial_summary.policy == pol) & (spatial_summary.metric == metric)].iloc[0]["mean"])
            stored = float(stored_row[column])
            if abs(fresh - stored) > 1e-8:
                notes.append(f"DISCREPANCY spatial {metric} {pol}: {fresh} vs {stored}")
    notes.append("Noisy time-to-alignment and Noisy entropy bins have no earlier saved comparison file. Their values were calculated from the stored Noisy trajectories.")
    from analysis_pipeline.paper_figures_v2.figure3 import _bin_means
    plotted = _bin_means(bundle)
    displayed = pooled_full[pooled_full.displayed_in_current_figure == 1]
    for pol in POLICIES:
        mae, entropy = plotted[pol]
        sub = displayed[displayed.policy == pol].sort_values("bin_id")
        if len(sub) != len(mae):
            notes.append(f"DISCREPANCY Full entropy bins {pol}: {len(sub)} displayed bins vs {len(mae)} in the figure")
            continue
        if len(sub) and (
            np.max(np.abs(sub.pooled_mean_functional_mae.to_numpy() - mae)) > 1e-8
            or np.max(np.abs(sub.pooled_mean_posterior_entropy.to_numpy() - entropy)) > 1e-8
        ):
            notes.append(f"DISCREPANCY Full pooled entropy-bin means for {pol}")
    contrasts = pd.read_csv(OUT.parent / "essential" / "bootstrap_contrasts.csv")
    for environment, metric, endpoint in (("full", "func", "func_auc"), ("full", "struct", "struct_auc"), ("full", "entropy", "entropy_auc"), ("noisy", "func", "func_auc"), ("noisy", "struct", "struct_auc")):
        for pol in POLICIES:
            if pol == "dne":
                continue
            fresh = float(np.mean(auc_values[(environment, metric)]["dne"]) - np.mean(auc_values[(environment, metric)][pol]))
            stored = float(contrasts[(contrasts.env == environment) & (contrasts.baseline == pol) & (contrasts.endpoint == endpoint)].iloc[0].diff_dne_minus_base)
            if abs(fresh - stored) > 1e-8:
                notes.append(f"DISCREPANCY contrast {environment} {endpoint} {pol}: {fresh} vs {stored}")
    discrepancies = [note for note in notes if note.startswith("DISCREPANCY") or note.startswith("MISSING")]
    information = [note for note in notes if note not in discrepancies]
    if not discrepancies:
        information.insert(0, "AUC means, absolute-error means, entropy trajectory means, Full RMST, attainment counts, spatial-circle rates, and stored DNE-minus-comparator AUC means match the existing analyses.")
    return discrepancies + information


def main():
    bundle = load_or_build()
    raw = bundle["raw"]
    rng_boot = np.random.default_rng(BOOT_SEED)
    rng_perm = np.random.default_rng(PERM_SEED)
    sheets = {}
    auc_values = {}
    abs_values = {}
    entropy_means = {}
    rmst_values = {}

    raw_f, summary_f = trajectory_tables(raw, "full", "func", "functional_mae", rng_boot)
    sheets["F2A_Func_Traj"] = [("XY_GRAPH", xy_graph_table(summary_f)), ("RAW_RUNS", raw_f), ("SUMMARY_BY_TRIAL", summary_f)]
    raw_a, summary_a, comp_a, auc_values[("full", "func")] = auc_tables(raw, "full", "func", "functional_auc", "full_functional_auc", rng_boot, rng_perm)
    sheets["F2C_Func_AUC"] = [("MODEL_SUMMARY", model_summary_row(summary_a)), ("RAW_RUNS", raw_a), ("SUMMARY", summary_a), ("COMPARISONS", comp_a)]
    raw_s, summary_s = trajectory_tables(raw, "full", "struct", "structural_distance", rng_boot)
    sheets["F2B_Struct_Traj"] = [("XY_GRAPH", xy_graph_table(summary_s)), ("RAW_RUNS", raw_s), ("SUMMARY_BY_TRIAL", summary_s)]
    raw_sa, summary_sa, comp_sa, auc_values[("full", "struct")] = auc_tables(raw, "full", "struct", "structural_auc", "full_structural_auc", rng_boot, rng_perm)
    sheets["F2C_Struct_AUC"] = [("MODEL_SUMMARY", model_summary_row(summary_sa)), ("RAW_RUNS", raw_sa), ("SUMMARY", summary_sa), ("COMPARISONS", comp_sa)]

    raw_t, summary_t, cumulative_t, comp_t = time_tables(raw, "full", "full_rmst_0.030", rng_boot, rng_perm)
    sheets["Full_Time_Align"] = [("XY_GRAPH", attainment_xy(raw_t)), ("RAW_RUNS", raw_t), ("RMST_SUMMARY", summary_t), ("CUMULATIVE_ATTAINMENT", cumulative_t), ("COMPARISONS", comp_t)]
    for pol in POLICIES:
        rmst_values[("full", pol)] = raw_t.loc[raw_t.policy == pol, "restricted_time"].to_numpy()

    raw_e, summary_e = trajectory_tables(raw, "full", "entropy", "posterior_entropy", rng_boot)
    sheets["F3A_Entropy_Traj"] = [("XY_GRAPH", xy_graph_table(summary_e)), ("RAW_RUNS", raw_e), ("SUMMARY_BY_TRIAL", summary_e)]
    for pol in POLICIES:
        entropy_means[("full", pol)] = summary_e.loc[summary_e.policy == pol, "mean"].to_numpy()
    full_edges = bin_edges(raw, "full")
    raw_b, pooled_b, weighted_b = bin_tables(raw, "full", full_edges, rng_boot)
    sheets["F3B_Entropy_Bins"] = [("XY_GRAPH", bin_xy(weighted_b, pooled_b)), ("RAW_RUN_BINS", raw_b), ("POOLED_TRIAL_MEANS", pooled_b), ("EQUAL_RUN_WEIGHTED_MEANS", weighted_b)]
    sheets["F3B_Entropy_Matched"] = [("XY_GRAPH", entropy_matched_xy(entropy_matched_analyse(bundle)["summary"]))]

    raw_nf, summary_nf = trajectory_tables(raw, "noisy", "func", "functional_mae", rng_boot)
    sheets["F4A_Noisy_Func"] = [("XY_GRAPH", xy_graph_table(summary_nf)), ("RAW_RUNS", raw_nf), ("SUMMARY_BY_TRIAL", summary_nf)]
    raw_na, summary_na, comp_na, auc_values[("noisy", "func")] = auc_tables(raw, "noisy", "func", "functional_auc", "noisy_functional_auc", rng_boot, rng_perm)
    sheets["Noisy_Func_AUC"] = [("MODEL_SUMMARY", model_summary_row(summary_na)), ("RAW_RUNS", raw_na), ("SUMMARY", summary_na), ("COMPARISONS", comp_na)]
    raw_ns, summary_ns = trajectory_tables(raw, "noisy", "struct", "structural_distance", rng_boot)
    sheets["F4B_Noisy_Struct"] = [("XY_GRAPH", xy_graph_table(summary_ns)), ("RAW_RUNS", raw_ns), ("SUMMARY_BY_TRIAL", summary_ns)]
    raw_nsa, summary_nsa, comp_nsa, auc_values[("noisy", "struct")] = auc_tables(raw, "noisy", "struct", "structural_auc", "noisy_structural_auc", rng_boot, rng_perm)
    sheets["Noisy_Struct_AUC"] = [("MODEL_SUMMARY", model_summary_row(summary_nsa)), ("RAW_RUNS", raw_nsa), ("SUMMARY", summary_nsa), ("COMPARISONS", comp_nsa)]

    raw_p, summary_p, comp_p = parameter_tables(raw, rng_boot, rng_perm)
    parameter_groups = [
        (f"{LABELS[pol]} {env_label}", {"policy": pol, "environment": env})
        for pol in POLICIES
        for env, env_label in (("full", "Full"), ("noisy", "Noisy"))
    ]
    sheets["S6b_Parameter_Error"] = [
        ("MODEL_SUMMARY", grouped_summary(summary_p, "parameter", ["wx", "wy", "b"], "mean_absolute_error", parameter_groups)),
        ("RAW_RUNS", raw_p), ("SUMMARY_ABSOLUTE_ERROR", summary_p), ("COMPARISONS", comp_p),
    ]
    for environment in ("full", "noisy"):
        for key in ("wx", "wy", "b"):
            abs_values[(environment, key)] = {
                pol: np.abs(raw[environment][pol][key][:, -1] - THETA[key]) for pol in POLICIES
            }

    raw_sp, summary_sp, comp_sp = spatial_tables(raw, rng_boot, rng_perm)
    spatial_groups = [
        (f"{LABELS[pol]} Full with spatial logging", {"policy": pol, "environment": "full_with_spatial_logging"})
        for pol in POLICIES
    ]
    sheets["S3_Spatial_Circle"] = [
        ("MODEL_SUMMARY", grouped_summary(summary_sp, "metric", ["circle_trial_fraction", "unique_circle_fraction"], "mean", spatial_groups)),
        ("RAW_RUNS", raw_sp), ("SUMMARY", summary_sp), ("COMPARISONS", comp_sp),
    ]

    raw_ne, summary_ne = trajectory_tables(raw, "noisy", "entropy", "posterior_entropy", rng_boot)
    sheets["Noisy_Entropy_Traj"] = [("XY_GRAPH", xy_graph_table(summary_ne)), ("RAW_RUNS", raw_ne), ("SUMMARY_BY_TRIAL", summary_ne)]
    for pol in POLICIES:
        entropy_means[("noisy", pol)] = summary_ne.loc[summary_ne.policy == pol, "mean"].to_numpy()
    noisy_edges = bin_edges(raw, "noisy")
    raw_nb, pooled_nb, weighted_nb = bin_tables(raw, "noisy", noisy_edges, rng_boot)
    sheets["Noisy_Entropy_Bins"] = [("XY_GRAPH", bin_xy(weighted_nb, pooled_nb)), ("RAW_RUN_BINS", raw_nb), ("POOLED_TRIAL_MEANS", pooled_nb), ("EQUAL_RUN_WEIGHTED_MEANS", weighted_nb)]

    raw_nt, summary_nt, cumulative_nt, comp_nt = time_tables(raw, "noisy", "noisy_rmst_0.030", rng_boot, rng_perm)
    sheets["Noisy_Time_Align"] = [("XY_GRAPH", attainment_xy(raw_nt)), ("RAW_RUNS", raw_nt), ("RMST_SUMMARY", summary_nt), ("CUMULATIVE_ATTAINMENT", cumulative_nt), ("COMPARISONS", comp_nt)]

    raw_ea, summary_ea, comp_ea, auc_values[("full", "entropy")] = auc_tables(raw, "full", "entropy", "entropy_auc", "full_entropy_auc", rng_boot, rng_perm)
    sheets["F3A_Entropy_AUC"] = [("MODEL_SUMMARY", model_summary_row(summary_ea)), ("RAW_RUNS", raw_ea), ("SUMMARY", summary_ea), ("COMPARISONS", comp_ea)]

    order = [
        "F2A_Func_Traj", "F2C_Func_AUC", "F2B_Struct_Traj", "F2C_Struct_AUC", "Full_Time_Align",
        "F3A_Entropy_Traj", "F3A_Entropy_AUC", "F3B_Entropy_Bins", "F3B_Entropy_Matched", "F4A_Noisy_Func", "Noisy_Func_AUC", "F4B_Noisy_Struct",
        "Noisy_Struct_AUC", "S6b_Parameter_Error", "S3_Spatial_Circle", "Noisy_Entropy_Traj",
        "Noisy_Entropy_Bins", "Noisy_Time_Align",
    ]
    layouts = {}
    with pd.ExcelWriter(XLSX, engine="openpyxl") as writer:
        for sheet in order:
            layouts[sheet] = write_sheet(writer, sheet, sheets[sheet])
            export_csv(sheet, sheets[sheet])
    format_workbook(XLSX, layouts)

    notes = verify(bundle, auc_values, rmst_values, abs_values, entropy_means, summary_sp, comp_t, pooled_b)
    audit = []
    for sheet in order:
        blocks = {title: frame for title, frame in sheets[sheet]}
        raw_name = next((title for title in blocks if title.startswith("RAW")), None)
        raw_frame = blocks[raw_name] if raw_name else None
        graph = blocks.get("XY_GRAPH")
        comparisons = blocks.get("COMPARISONS")
        audit.append({
            "worksheet": sheet,
            "raw_rows": int(len(raw_frame)) if raw_frame is not None else int(len(graph)) if graph is not None else 0,
            "independent_runs_per_policy": int(raw_frame.groupby("policy").run_id.nunique().min()) if raw_frame is not None and "run_id" in raw_frame else None,
            "n_policies": int(raw_frame.policy.nunique()) if raw_frame is not None and "policy" in raw_frame else 10,
            "comparison_rows": int(len(comparisons)) if comparisons is not None else 0,
            "comparison_families": sorted(comparisons.family.unique().tolist()) if comparisons is not None else [],
            "tables": {title: int(len(frame)) for title, frame in sheets[sheet]},
        })
    metadata = {
        "workbook": str(XLSX),
        "sources": {
            "full": "results/trap2d/notrap",
            "noisy": "results_noisy_obs/trap2d/noisy_obs",
            "full_with_spatial_logging": "results/trap2d/trap",
            "cache": "analysis_outputs/paper_figures_v2/_bundle.pkl",
        },
        "policies": POLICIES,
        "runs": "run_id 0-19 within each policy. Matching run_id values are not paired across policies or environments.",
        "metrics": {
            "functional_mae": "alignment_func_abs after the update, mean absolute predictive error on 200 fixed probes",
            "structural_distance": "Euclidean distance of the posterior mean from (1.3, -0.9, 0.4)",
            "posterior_entropy": "discrete posterior entropy in nats, -sum w ln w",
            "auc": "numpy trapezoidal integral over trial indices 0-199",
            "absolute_error": "absolute deviation of the trial-199 posterior mean from theta_star; not the raw parameter magnitude",
            "signed_error": "posterior mean minus theta_star",
        },
        "bootstrap": {
            "replicates": DRAWS,
            "generator_seed": BOOT_SEED,
            "interval": "percentile 2.5 and 97.5 of the resampled mean, or of the resampled DNE-minus-comparator mean",
            "resampling": "independent runs, with replacement, separately for each policy",
        },
        "p_values": {
            "test": "two-sided Monte Carlo permutation test of the difference in means",
            "generator_seed": PERM_SEED,
            "shuffles": DRAWS,
            "not_derived_from_confidence_intervals": True,
        },
        "holm_families": [
            "full_functional_auc", "full_structural_auc", "full_entropy_auc", "full_rmst_0.030",
            "noisy_functional_auc", "noisy_structural_auc", "noisy_rmst_0.030",
            "absolute_error_wx_full", "absolute_error_wy_full", "absolute_error_b_full",
            "absolute_error_wx_noisy", "absolute_error_wy_noisy", "absolute_error_b_noisy",
            "circle_trial_fraction", "circle_unique_fraction",
        ],
        "holm_rule": "Within each family of nine DNE-minus-comparator tests, adjusted p equals the stepwise Holm value. Families are not pooled.",
        "trajectory_tests": "No trial-wise hypothesis tests. The first table on each trajectory sheet is an XY-graph block: column A is trial 0-199, then each of the ten policies has mean, SEM, and n. The long summary below it also stores SD and the pointwise bootstrap CI.",
        "auc_summary_layout": "The first row of each AUC sheet, including F3A_Entropy_AUC, places the ten policies side by side. Each policy has mean, SEM, and n in adjacent columns. Entropy AUC is the trapezoidal integral of posterior entropy over trials 0-199 in Full.",
        "time_graph_layout": "Full_Time_Align and Noisy_Time_Align start with the panel A curve. Column trial is X. Each following column is one model, and the value is the proportion of 20 runs that have reached functional MAE 0.030 at or before that trial. The same value is drawn as a step in the figure.",
        "bin_graph_layout": "F3B_Entropy_Bins uses 24 equal-width functional-MAE bins. The first table has bin as X. Each policy then has mean, SEM, and n for the equal-run functional MAE. A cell is blank when that policy has fewer than 8 pooled trials in the bin. F3B_Entropy_Matched is the separate entropy-binned Figure 3B: column entropy is the bin centre in nats, and each policy has mean functional MAE, SEM, and n whenever at least one run entered. A cell is blank only when no run entered.",
        "grouped_summary_layout": "S6b rows are wx, wy, and b. Each policy has Full and Noisy blocks of absolute-error mean, SEM, and n. S3 rows are the circle trial fraction and the unique-circle fraction. Each policy has one spatial-logging block of mean, SEM, and n.",
        "binning": {
            "n_bins": N_BINS,
            "rule": "equal-width bins from the pooled minimum to maximum functional MAE of all ten policies in that environment",
            "full_edges": full_edges.tolist(),
            "noisy_edges": noisy_edges.tolist(),
            "last_bin": "closed on the right",
            "figure_display_rule": "the current Full figure omits a bin with fewer than 8 pooled trials; that flag is descriptive and does not delete rows",
            "inference": "bootstrap CIs resample runs that entered the bin. Pooled trial means are not treated as independent samples and have no p-values.",
        },
        "censoring": {
            "threshold": THRESHOLD,
            "event": "first trial index 0-199 with functional MAE at or below 0.030",
            "non_attainment": "restricted_time is 200. first_hit_trial is blank. attained_on_trial_199 is 0. This is not attainment at trial 199.",
            "attainment_at_199": "first_hit_trial is 199, attained_on_trial_199 is 1, restricted_time is 199",
            "rmst": "mean of restricted_time across the 20 runs",
        },
        "spatial": "Circle ||s||<=0.5 in the spatially logged Full run. Logging does not change selection. Unique points are rounded to 6 decimals. Random's unique-circle fraction estimates pool composition because pools were not stored.",
        "verification": notes,
    }
    (OUT / "statistics_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = ["DNE figure statistics audit", ""]
    lines.extend(notes)
    lines.append("")
    for item in audit:
        lines.append(
            f"{item['worksheet']}: raw_rows={item['raw_rows']}, policies={item['n_policies']}, "
            f"runs_per_policy={item['independent_runs_per_policy']}, comparison_rows={item['comparison_rows']}, "
            f"families={item['comparison_families'] or 'none'}"
        )
    (OUT / "DNE_Figure_Statistics_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(XLSX)


if __name__ == "__main__":
    main()
