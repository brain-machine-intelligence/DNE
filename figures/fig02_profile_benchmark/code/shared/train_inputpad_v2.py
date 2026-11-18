#!/usr/bin/env python3
"""Train MODELS_INPUTPAD_V2 with one-hot actions + padding masks. Does not touch mdl13."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SLIDING = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[4]
for p in (str(REPO), str(SLIDING)):
    if p not in sys.path:
        sys.path.insert(0, p)
from config_profiled6_sliding import (  # noqa: E402
    CKPT_DIR,
    CKPT_DIR_INPUTPAD_V2,
    CKPT_DIR_INPUTPAD_V2_EPOCHS,
    LOG_DIR_INPUTPAD_V2,
    MODELS_INPUTPAD_V2,
    TRAIN_SEEDS,
)

COPY_UNCHANGED = {
    "mlp": CKPT_DIR,
    "single_layer": CKPT_DIR,
}
RETRAIN = [m for m in MODELS_INPUTPAD_V2 if m not in COPY_UNCHANGED]


def _copy_unchanged(seeds):
    dest = CKPT_DIR_INPUTPAD_V2
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for model, src_dir in COPY_UNCHANGED.items():
        for seed in seeds:
            src = src_dir / f"{model}_seed{seed}.pt"
            dst = dest / f"{model}_seed{seed}.pt"
            if dst.is_file():
                continue
            if not src.is_file():
                print(f"[missing copy] {src}")
                continue
            shutil.copy2(src, dst)
            src_m = src_dir / f"{model}_seed{seed}_metrics.json"
            if src_m.is_file():
                shutil.copy2(src_m, dest / src_m.name)
            print(f"[copy] {src.name}")
            n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default=",".join(RETRAIN))
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in TRAIN_SEEDS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--skip-copy", action="store_true")
    args = p.parse_args()
    print(f"[python] {sys.executable}", flush=True)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[cuda] {torch.cuda.get_device_name(0)}  torch={torch.__version__}", flush=True)
        else:
            print(f"[cuda] NOT AVAILABLE in this interpreter (torch={torch.__version__})", flush=True)
    except Exception as e:
        print(f"[cuda] import failed: {e}", flush=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    CKPT_DIR_INPUTPAD_V2.mkdir(parents=True, exist_ok=True)
    LOG_DIR_INPUTPAD_V2.mkdir(parents=True, exist_ok=True)
    if not args.skip_copy:
        _copy_unchanged(seeds)

    train_py = SLIDING / "run_train_profiled6_sliding.py"
    jobs = []
    for model in models:
        for seed in seeds:
            dest = CKPT_DIR_INPUTPAD_V2 / f"{model}_seed{seed}.pt"
            if dest.is_file():
                print(f"[skip] {dest.name}")
                continue
            jobs.append((model, seed))

    def run_one(job):
        model, seed = job
        cmd = [
            sys.executable, "-u", str(train_py),
            "--model", model,
            "--seed", str(seed),
            "--epochs", str(args.epochs),
            "--inputpad-v2",
            "--ckpt-dir", str(CKPT_DIR_INPUTPAD_V2),
            "--log-dir", str(LOG_DIR_INPUTPAD_V2),
            "--save-epoch-dir", str(CKPT_DIR_INPUTPAD_V2_EPOCHS),
        ]
        print("[RUN]", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=str(REPO))
        if r.returncode != 0:
            raise SystemExit(f"train failed {model} seed{seed} code={r.returncode}")

    if args.workers <= 1:
        for j in jobs:
            run_one(j)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, j) for j in jobs]
            for f in as_completed(futs):
                f.result()
    print(f"[OK] inputpad_v2 training done n_jobs={len(jobs)}")


if __name__ == "__main__":
    main()
