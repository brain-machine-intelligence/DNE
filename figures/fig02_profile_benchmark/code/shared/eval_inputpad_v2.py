#!/usr/bin/env python3
"""Eval inputpad_v2 checkpoints into results/geometry dirs that do not touch mdl13."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SLIDING = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
for p in (str(REPO), str(SLIDING)):
    if p not in sys.path:
        sys.path.insert(0, p)
from config_profiled6_sliding import (  # noqa: E402
    CKPT_DIR_INPUTPAD_V2,
    DATA_CSV,
    GEOMETRY_INPUTPAD_V2,
    MODELS_INPUTPAD_V2,
    RESULTS_INPUTPAD_V2,
    TRAIN_SEEDS,
)


def _env() -> dict:
    env = os.environ.copy()
    prefix = os.pathsep.join([str(REPO), str(SLIDING)])
    env["PYTHONPATH"] = prefix + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def eval_one(ckpt: Path) -> None:
    stem = ckpt.stem
    ep = RESULTS_INPUTPAD_V2 / f"{stem}_sliding_episode.csv"
    zpref = RESULTS_INPUTPAD_V2 / f"{stem}_z_prefix_curve.jsonl"
    geom = GEOMETRY_INPUTPAD_V2 / f"{stem}_sliding_geometry.csv"
    if ep.is_file() and zpref.is_file() and geom.is_file():
        print(f"[skip] {stem}", flush=True)
        return
    cmd = [
        sys.executable, "-u", "-m", "profile_benchmark.eval.eval_sliding_suite",
        "--ckpt", str(ckpt),
        "--data", str(DATA_CSV),
        "--out-dir", str(RESULTS_INPUTPAD_V2),
        "--mode", "all",
        "--use-prefix-mask",
    ]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO), env=_env())
    cmd_g = [
        sys.executable, "-u", "-m", "profile_benchmark.eval.analyse_sliding_geometry",
        "--results-dir", str(RESULTS_INPUTPAD_V2),
        "--out-dir", str(GEOMETRY_INPUTPAD_V2),
        "--stem", stem,
    ]
    print("[RUN]", " ".join(cmd_g), flush=True)
    subprocess.run(cmd_g, check=True, cwd=str(REPO), env=_env())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default=",".join(MODELS_INPUTPAD_V2))
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in TRAIN_SEEDS))
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    RESULTS_INPUTPAD_V2.mkdir(parents=True, exist_ok=True)
    GEOMETRY_INPUTPAD_V2.mkdir(parents=True, exist_ok=True)
    jobs = []
    for model in models:
        for seed in seeds:
            ckpt = CKPT_DIR_INPUTPAD_V2 / f"{model}_seed{seed}.pt"
            if not ckpt.is_file():
                print(f"[missing] {ckpt.name}")
                continue
            jobs.append(ckpt)

    def _run(ckpt: Path) -> None:
        eval_one(ckpt)

    if args.workers <= 1:
        for j in jobs:
            _run(j)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run, j) for j in jobs]
            for f in as_completed(futs):
                f.result()
    print(f"[OK] inputpad_v2 eval done n_jobs={len(jobs)}")


if __name__ == "__main__":
    main()
