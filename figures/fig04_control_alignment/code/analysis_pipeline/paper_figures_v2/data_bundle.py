"""Load trajectories once and attach verified difference intervals.

Difference intervals are read from analysis_outputs/essential after the mean
difference is recomputed from the trajectories. Trajectory bands use a
separate generator (seed 20261023, 4,000 draws) so they do not reuse that
stream. Seed-level mean intervals use generator seed 20261024 and 20,000 draws.
Neither is adjusted for multiple comparisons, and seeds are not paired.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_pipeline.paper_figures.data import (
    BAND,
    LOW_R,
    first_reach,
    load_policy,
    replay_discrepancy,
)
from analysis_pipeline.paper_figures.data import _trapz, _unique_rate
from analysis_pipeline.paper_figures_v2.plot_style import POLICIES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis_outputs" / "paper_figures_v2"
CACHE = OUT / "_bundle.pkl"
CACHE_VERSION = 2
ENVS = ("full", "full_trap", "noisy")
METRICS = ("func", "struct", "entropy", "wx", "wy", "b")
THETA = {"wx": 1.3, "wy": -0.9, "b": 0.4}
B = 20000
BAND_DRAWS = 4000


def _band(samples, rng):
    n = samples.shape[0]
    mean = samples.mean(axis=0)
    parts = []
    for start in range(0, BAND_DRAWS, 1000):
        m = min(1000, BAND_DRAWS - start)
        idx = rng.integers(0, n, size=(m, n))
        parts.append(samples[idx].mean(axis=1))
    boot = np.concatenate(parts, axis=0)
    lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
    return mean, lo, hi


def _mean_ci(values, rng):
    values = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(values), size=(B, len(values)))
    draws = values[idx].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def _lookup(table, env, baseline, endpoint, estimate):
    hit = table[(table.env == env) & (table.baseline == baseline) & (table.endpoint == endpoint)]
    if hit.empty:
        raise KeyError((env, baseline, endpoint))
    row = hit.iloc[0]
    stored = float(row.diff_dne_minus_base)
    if abs(stored - estimate) > 1e-8:
        raise RuntimeError(f"Mean mismatch {env} {baseline} {endpoint}: {estimate} vs {stored}")
    return stored, float(row.ci_lo), float(row.ci_hi)


def build(progress=print):
    raw = {}
    for env in ENVS:
        raw[env] = {}
        for pol in POLICIES:
            progress(f"load {env} {pol}")
            raw[env][pol] = load_policy(env, pol)

    band_rng = np.random.default_rng(20261023)
    series = {}
    for env in ENVS:
        for metric in METRICS:
            for pol in POLICIES:
                series[(env, pol, metric)] = _band(raw[env][pol][metric], band_rng)

    disc = {}
    for env in ENVS:
        disc[env] = {}
        for pol in POLICIES:
            progress(f"replay discrepancy {env} {pol}")
            pack = raw[env][pol]
            disc[env][pol] = replay_discrepancy(pack["sx"], pack["sy"], pack["y"])
            series[(env, pol, "disc")] = _band(disc[env][pol], band_rng)
    replay_err = float(np.max(np.abs(disc["full"]["dne"][0] - raw["full"]["dne"]["score"][0])))

    auc = {env: {pol: {} for pol in POLICIES} for env in ENVS}
    for env in ENVS:
        for pol in POLICIES:
            for metric in ("func", "struct", "entropy"):
                auc[env][pol][metric] = _trapz(raw[env][pol][metric])

    reach = {}
    for env in ENVS:
        reach[env] = {}
        for pol in POLICIES:
            reach[env][pol] = {thr: first_reach(raw[env][pol]["func"], thr) for thr in (0.05, 0.03)}

    contrasts = pd.read_csv(ROOT / "analysis_outputs" / "essential" / "bootstrap_contrasts.csv")
    degradation = pd.read_csv(ROOT / "analysis_outputs" / "essential" / "degradation.csv")
    diffs = {}
    for env in ENVS:
        for endpoint in ("func_auc", "entropy_auc", "struct_auc", "func_final", "struct_final"):
            for pol in POLICIES:
                if pol == "dne":
                    continue
                if endpoint.endswith("auc"):
                    metric = endpoint.split("_")[0]
                    estimate = float(auc[env]["dne"][metric].mean() - auc[env][pol][metric].mean())
                else:
                    metric = "func" if endpoint.startswith("func") else "struct"
                    estimate = float(raw[env]["dne"][metric][:, -1].mean() - raw[env][pol][metric][:, -1].mean())
                diffs[(env, endpoint, pol)] = _lookup(contrasts, env, pol, endpoint, estimate)

    changes = {}
    for pol in POLICIES:
        for endpoint, metric in (("func_final", "func"), ("func_auc", "func"), ("struct_final", "struct")):
            if endpoint.endswith("auc"):
                estimate = float(auc["noisy"][pol][metric].mean() - auc["full"][pol][metric].mean())
                col = "diff_noisy_minus_full"
            else:
                estimate = float(raw["noisy"][pol][metric][:, -1].mean() - raw["full"][pol][metric][:, -1].mean())
                col = "diff_noisy_minus_full"
            hit = degradation[(degradation.policy == pol) & (degradation.endpoint == endpoint)].iloc[0]
            if abs(float(hit[col]) - estimate) > 1e-8:
                raise RuntimeError(f"Degradation mismatch {pol} {endpoint}")
            changes[(pol, endpoint)] = (estimate, float(hit.ci_lo), float(hit.ci_hi))

    level_rng = np.random.default_rng(20261024)
    levels = {}
    for env in ENVS:
        for pol in POLICIES:
            for metric in ("func", "struct", "entropy"):
                levels[(env, pol, f"auc_{metric}")] = _mean_ci(auc[env][pol][metric], level_rng)
            for key, truth in THETA.items():
                err = np.abs(raw[env][pol][key][:, -1] - truth)
                signed = raw[env][pol][key][:, -1] - truth
                levels[(env, pol, f"abs_{key}")] = _mean_ci(err, level_rng)
                levels[(env, pol, f"signed_{key}")] = _mean_ci(signed, level_rng)

    spatial = []
    trap = raw["full_trap"]
    for pol in POLICIES:
        for region, pred in (
            ("circle", lambda x, y: x ** 2 + y ** 2 <= LOW_R ** 2),
            ("band", lambda x, y: np.abs(y) <= BAND),
        ):
            mask = pred(trap[pol]["sx"], trap[pol]["sy"])
            spatial.append({
                "policy": pol,
                "region": region,
                "trial_rate": float(mask.mean()),
                "unique_rate": float(_unique_rate(trap[pol]["sx"], trap[pol]["sy"], pred).mean()),
            })

    return {
        "version": CACHE_VERSION,
        "raw": raw,
        "series": series,
        "auc": auc,
        "reach": reach,
        "diffs": diffs,
        "changes": changes,
        "levels": levels,
        "spatial": pd.DataFrame(spatial),
        "replay_err": replay_err,
        "disc": disc,
    }


def load_or_build(rebuild=False):
    OUT.mkdir(parents=True, exist_ok=True)
    if CACHE.exists() and not rebuild:
        with CACHE.open("rb") as handle:
            bundle = pickle.load(handle)
        if bundle.get("version") == CACHE_VERSION:
            return bundle
    bundle = build()
    with CACHE.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=4)
    return bundle
