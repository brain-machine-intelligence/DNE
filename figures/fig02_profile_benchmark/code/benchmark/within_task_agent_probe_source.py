#!/usr/bin/env python3
"""
Eval 5: geometry / probe / stability from sliding eval z JSONL files.

Inputs (per checkpoint stem under results_dir):
  {stem}_z_prefix_curve.jsonl   — prefix 0..L-1, L in {5,10,20,40,80}
  {stem}_z_profile_windows.jsonl — sliding L=80 windows

Outputs:
  {stem}_sliding_geometry.csv  — per-L probes + stability scalars
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from profile_benchmark.eval.analyse_profile_geometry import linear_probe, separability_ratio

EVAL_L = [5, 10, 20, 40, 80]


def _parse_stem(stem: str) -> Tuple[str, int]:
    m = re.match(r"^(.+)_seed(\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse stem: {stem}")
    return m.group(1), int(m.group(2))


def load_jsonl(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "z" in df.columns:
        df["z_arr"] = df["z"].apply(lambda x: np.asarray(x, dtype=np.float32))
    return df


def within_task_agent_probe(z: np.ndarray, agents: np.ndarray, tasks: np.ndarray, seed: int) -> float:
    accs: List[float] = []
    for task in np.unique(tasks):
        m = tasks == task
        if m.sum() < 8:
            continue
        p = linear_probe(z[m], agents[m], seed=seed)
        if not np.isnan(p["acc"]):
            accs.append(p["acc"])
    return float(np.mean(accs)) if accs else np.nan


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def profile_stability_metrics(df_slide: pd.DataFrame) -> Dict[str, float]:
    """Within-episode z consistency across sliding L=80 windows."""
    if df_slide.empty or "z_arr" not in df_slide.columns:
        return {
            "z_stability_mean_cos": np.nan,
            "z_stability_early_late_cos": np.nan,
            "z_stability_agent_centroid_r": np.nan,
        }

    mean_sims: List[float] = []
    early_late: List[float] = []
    agent_r: List[float] = []

    for _, g in df_slide.groupby("episode_id"):
        if len(g) < 2:
            continue
        zs = [z for z in g["z_arr"] if np.linalg.norm(z) > 1e-8]
        if len(zs) < 2:
            continue
        ps = g["prefix_start"].to_numpy()
        agents = g["agent"].astype(str).to_numpy()

        for i in range(len(zs)):
            for j in range(i + 1, len(zs)):
                s = cosine_sim(zs[i], zs[j])
                if not np.isnan(s):
                    mean_sims.append(s)

        g0 = g[g["prefix_start"] == ps.min()]
        g1 = g[g["prefix_start"] == ps.max()]
        if len(g0) and len(g1):
            z0, z1 = g0["z_arr"].iloc[0], g1["z_arr"].iloc[0]
            s = cosine_sim(z0, z1)
            if not np.isnan(s):
                early_late.append(s)

        # same-agent episodes: corr between mean z and one-hot agent (rough separability)
        if len(np.unique(agents)) == 1 and len(zs) >= 3:
            Z = np.stack(zs, axis=0)
            cen = Z.mean(axis=0, keepdims=True)
            dists = np.linalg.norm(Z - cen, axis=1)
            agent_r.append(float(1.0 / (1.0 + dists.std())))

    return {
        "z_stability_mean_cos": float(np.mean(mean_sims)) if mean_sims else np.nan,
        "z_stability_early_late_cos": float(np.mean(early_late)) if early_late else np.nan,
        "z_stability_within_ep_std": float(np.mean(agent_r)) if agent_r else np.nan,
    }


def analyse_prefix_curve(df: pd.DataFrame, seed: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for L in EVAL_L:
        sub = df[df["prefix_len"] == L]
        if sub.empty:
            continue
        z = np.stack(sub["z_arr"].tolist(), axis=0)
        agents = sub["agent"].astype(str).to_numpy()
        tasks = sub["task"].astype(str).to_numpy()
        ap = linear_probe(z, agents, seed=seed)
        tp = linear_probe(z, tasks, seed=seed + 1)
        w_in, w_bt, sep = separability_ratio(z, agents)
        rows.append(
            {
                "prefix_L": L,
                "agent_probe_acc": ap["acc"],
                "agent_probe_balanced_acc": ap["balanced_acc"],
                "task_probe_acc": tp["acc"],
                "agent_separability_ratio": sep,
                "within_task_agent_probe_acc": within_task_agent_probe(z, agents, tasks, seed),
                "n_samples": len(sub),
            }
        )
    return rows


def analyse_stem(results_dir: Path, stem: str, out_dir: Path) -> Optional[Path]:
    model, seed = _parse_stem(stem)
    path_prefix = results_dir / f"{stem}_z_prefix_curve.jsonl"
    path_slide = results_dir / f"{stem}_z_profile_windows.jsonl"

    if not path_prefix.is_file() and not path_slide.is_file():
        print(f"[WARN] no z jsonl for {stem}")
        return None

    rows: List[Dict] = []
    probe_by_L: Dict[int, float] = {}

    if path_prefix.is_file():
        df_p = load_jsonl(path_prefix)
        for r in analyse_prefix_curve(df_p, seed):
            r.update({"model": model, "seed": seed, "window_type": "prefix_curve"})
            rows.append(r)
            probe_by_L[int(r["prefix_L"])] = r["agent_probe_acc"]

    stab = profile_stability_metrics(load_jsonl(path_slide) if path_slide.is_file() else pd.DataFrame())
    stab_row = {
        "model": model,
        "seed": seed,
        "prefix_L": np.nan,
        "window_type": "stability_summary",
        **stab,
        "profile_gain_probe_5_80": (
            probe_by_L.get(80, np.nan) - probe_by_L.get(5, np.nan)
            if 80 in probe_by_L and 5 in probe_by_L
            else np.nan
        ),
        "profile_gain_probe_auc_L": _probe_auc(probe_by_L),
    }
    if probe_by_L:
        stab_row["agent_probe_L80"] = probe_by_L.get(80, np.nan)
        stab_row["within_task_agent_probe_L80"] = next(
            (r["within_task_agent_probe_acc"] for r in rows if r.get("prefix_L") == 80),
            np.nan,
        )
        stab_row["agent_separability_L80"] = next(
            (r["agent_separability_ratio"] for r in rows if r.get("prefix_L") == 80),
            np.nan,
        )
    rows.append(stab_row)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_sliding_geometry.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[OK] {out_path}")
    return out_path


def _probe_auc(probe_by_L: Dict[int, float]) -> float:
    Ls = sorted(probe_by_L.keys())
    if len(Ls) < 2:
        return np.nan
    ys = [probe_by_L[L] for L in Ls]
    return float(np.trapz(ys, Ls))


def flatten_geometry_for_summary(geo_path: Path) -> Dict[str, float]:
    df = pd.read_csv(geo_path)
    out: Dict[str, float] = {}
    pc = df[df["window_type"] == "prefix_curve"]
    for _, r in pc.iterrows():
        L = int(r["prefix_L"])
        out[f"agent_probe_L{L}"] = float(r["agent_probe_acc"])
        out[f"agent_sep_L{L}"] = float(r["agent_separability_ratio"])
        out[f"within_task_agent_probe_L{L}"] = float(r["within_task_agent_probe_acc"])
    stab = df[df["window_type"] == "stability_summary"]
    if not stab.empty:
        s = stab.iloc[0]
        for col in stab.columns:
            if col in ("model", "seed", "window_type", "prefix_L"):
                continue
            if pd.notna(s[col]):
                out[str(col)] = float(s[col])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--out-dir", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--ckpt", help="Checkpoint path (infers stem)")
    g.add_argument("--stem", help="Checkpoint stem e.g. rnn_seed0")
    g.add_argument("--all", action="store_true", help="Process all jsonl stems in results-dir")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)

    if args.all:
        stems = set()
        for f in results_dir.glob("*_z_profile_windows.jsonl"):
            stems.add(f.name.replace("_z_profile_windows.jsonl", ""))
        for f in results_dir.glob("*_z_prefix_curve.jsonl"):
            stems.add(f.name.replace("_z_prefix_curve.jsonl", ""))
        for stem in sorted(stems):
            analyse_stem(results_dir, stem, out_dir)
        return

    stem = args.stem if args.stem else Path(args.ckpt).stem
    analyse_stem(results_dir, stem, out_dir)


if __name__ == "__main__":
    main()
