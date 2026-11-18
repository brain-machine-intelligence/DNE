from __future__ import annotations
import argparse
from pathlib import Path
import json
import pandas as pd
from tqdm import tqdm

from src.utils.config import load_yaml, ensure_dir
from src.utils.seeding import set_seed
from src.factory import make_env, make_target, make_posterior, make_policy
from src.experiment import run_episode, save_episode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, required=True)
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--exp", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, overrides the experiment file")
    ap.add_argument("--policies", type=str, default=None, help="Comma-separated policy names to run")
    args = ap.parse_args()

    env_cfg = load_yaml(args.env)
    target_cfg = load_yaml(args.target)
    exp_cfg = load_yaml(args.exp)

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    else:
        seeds = list(exp_cfg.get("seeds", [0]))
    trials = int(exp_cfg["trials"])
    out_root = Path(args.out_dir or exp_cfg.get("out_dir", "results"))
    compute_trap = bool((exp_cfg.get("metrics", {}) or {}).get("compute_trap_occupancy", True))

    # build once per sweep (seed-specific env is set below)
    target, theta_star, target_mode = make_target(target_cfg)

    summaries = []
    for seed in tqdm(seeds, desc="Seeds"):
        set_seed(int(seed))
        env_cfg["seed"] = int(seed)
        env = make_env(env_cfg)

        policies = exp_cfg["policies"]
        if args.policies:
            wanted = {p.strip() for p in args.policies.split(",") if p.strip()}
            policies = [p for p in policies if str(p.get("name", p["kind"])) in wanted]
            missing = wanted - {str(p.get("name", p["kind"])) for p in policies}
            if missing:
                raise ValueError(f"Unknown policies: {sorted(missing)}")
        for pol_cfg in policies:
            posterior = make_posterior(exp_cfg)
            policy = make_policy(pol_cfg, target)

            df, metrics = run_episode(
                env, target, posterior, policy,
                trials=trials, theta_star=theta_star, compute_trap=compute_trap, seed=int(seed),
            )

            out_dir = out_root / env.name / target_mode / policy.name / f"seed_{seed}"
            ensure_dir(out_dir)
            save_episode(out_dir, df, metrics)

            summaries.append({"seed": int(seed), "policy": policy.name, **metrics})

    # save summary table at env/target level
    out_summary_dir = out_root / env_cfg.get("name", "trap2d") / target_mode
    ensure_dir(out_summary_dir)
    df_sum = pd.DataFrame(summaries)
    summary_name = "summary.csv" if not args.policies else "summary_" + args.policies.replace(",", "_") + ".csv"
    df_sum.to_csv(out_summary_dir / summary_name, index=False)
    print(f"Saved summary: {out_summary_dir / summary_name}")

if __name__ == "__main__":
    main()
