from __future__ import annotations
import argparse
from pathlib import Path

from src.utils.config import load_yaml, ensure_dir
from src.utils.seeding import set_seed
from src.factory import make_env, make_target, make_posterior, make_policy
from src.experiment import run_episode, save_episode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=str, required=True)
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--exp", type=str, required=True)
    ap.add_argument("--policy", type=str, default=None, help="policy name override (run only one)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env_cfg = load_yaml(args.env)
    target_cfg = load_yaml(args.target)
    exp_cfg = load_yaml(args.exp)

    set_seed(args.seed)
    env_cfg["seed"] = args.seed

    env = make_env(env_cfg)
    target, theta_star, target_mode = make_target(target_cfg)

    trials = int(exp_cfg["trials"])
    trap_tag = "trap" if env.trap.enabled else "notrap"
    out_root = Path(exp_cfg.get("out_dir", "results"))
    compute_trap = bool((exp_cfg.get("metrics", {}) or {}).get("compute_trap_occupancy", True))

    for pol_cfg in exp_cfg["policies"]:
        name = str(pol_cfg.get("name", pol_cfg["kind"]))
        if args.policy is not None and name != args.policy:
            continue

        posterior = make_posterior(exp_cfg)
        policy = make_policy(pol_cfg, target)

        df, metrics = run_episode(
            env, target, posterior, policy,
            trials=trials, theta_star=theta_star, compute_trap=compute_trap, seed=int(args.seed),
        )

        out_dir = out_root / env.name / target_mode /  trap_tag /  policy.name / f"seed_{args.seed}"
        ensure_dir(out_dir)
        save_episode(out_dir, df, metrics)
        print(f"Saved episode: {out_dir}")

if __name__ == "__main__":
    main()
