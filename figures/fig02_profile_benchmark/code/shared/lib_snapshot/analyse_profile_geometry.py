#!/usr/bin/env python3
"""Agent/task linear probes and z_profile separability (per checkpoint, per prefix L)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DEFAULT_PREFIX_LENGTHS = (5, 10, 20, 40, 80)


def _parse_stem(stem: str) -> Tuple[str, int]:
    m = re.match(r"^(.+)_seed(\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse checkpoint stem: {stem}")
    return m.group(1), int(m.group(2))


def linear_probe(
    z: np.ndarray,
    labels: np.ndarray,
    seed: int,
    test_size: float = 0.25,
) -> Dict[str, float]:
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if len(np.unique(y)) < 2 or len(y) < 8:
        return {"acc": np.nan, "balanced_acc": np.nan, "macro_f1": np.nan}

    stratify = y if min(np.bincount(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        z, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    return {
        "acc": float(accuracy_score(y_te, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_te, pred)),
        "macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
    }


def separability_ratio(z: np.ndarray, agents: np.ndarray) -> Tuple[float, float, float]:
    """between_centroid_dist / within_agent_std (higher = more agent-separable)."""
    uniq = np.unique(agents)
    if len(uniq) < 2:
        return np.nan, np.nan, np.nan

    centroids = {a: z[agents == a].mean(axis=0) for a in uniq}
    within = []
    for a in uniq:
        za = z[agents == a]
        if len(za) < 2:
            continue
        c = centroids[a]
        within.append(float(np.linalg.norm(za - c, axis=1).mean()))
    within_mean = float(np.mean(within)) if within else np.nan

    between = []
    for i, a in enumerate(uniq):
        for b in list(uniq)[i + 1 :]:
            between.append(float(np.linalg.norm(centroids[a] - centroids[b])))
    between_mean = float(np.mean(between)) if between else np.nan

    ratio = between_mean / within_mean if within_mean and within_mean > 0 else np.nan
    return within_mean, between_mean, ratio


def analyse_one_L(
    z_path: Path,
    meta_path: Path,
    prefix_L: int,
    seed: int,
) -> Dict[str, float]:
    z = np.load(z_path)
    meta = pd.read_csv(meta_path)
    agents = meta["agent"].astype(str).to_numpy()
    tasks = meta["task"].astype(str).to_numpy()

    agent_probe = linear_probe(z, agents, seed=seed)
    task_probe = linear_probe(z, tasks, seed=seed + 1)
    w_in, w_bt, sep_ratio = separability_ratio(z, agents)

    return {
        "prefix_L": prefix_L,
        "agent_probe_acc": agent_probe["acc"],
        "agent_probe_balanced_acc": agent_probe["balanced_acc"],
        "agent_probe_macro_f1": agent_probe["macro_f1"],
        "task_probe_acc": task_probe["acc"],
        "task_probe_balanced_acc": task_probe["balanced_acc"],
        "task_probe_macro_f1": task_probe["macro_f1"],
        "within_agent_distance": w_in,
        "between_agent_distance": w_bt,
        "agent_separability_ratio": sep_ratio,
        "n_episodes": len(meta),
        "z_dim": z.shape[1],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Used only to infer model/seed from stem")
    p.add_argument("--rep-dir", default="profiled4/representations")
    p.add_argument("--out-dir", default="profiled4/analysis/geometry")
    p.add_argument(
        "--prefix-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_PREFIX_LENGTHS),
    )
    args = p.parse_args()

    stem = Path(args.ckpt).stem
    model, seed = _parse_stem(stem)
    rep_dir = Path(args.rep_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for L in args.prefix_lengths:
        z_path = rep_dir / f"{stem}_zprofile_L{L}.npy"
        meta_path = rep_dir / f"{stem}_zprofile_L{L}_meta.csv"
        if not z_path.exists() or not meta_path.exists():
            print(f"[WARN] missing L={L} for {stem}")
            continue
        row = analyse_one_L(z_path, meta_path, L, seed)
        row.update({"model": model, "seed": seed})
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No representation files for {stem} in {rep_dir}")

    df = pd.DataFrame(rows)
    out_path = out_dir / f"{stem}_geometry.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] {out_path}")


if __name__ == "__main__":
    main()
