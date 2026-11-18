#!/usr/bin/env python3
"""
Train 13 profile models with variable-L sliding-window samples (300-trial episodes).

Training windows: L in {10,20,40,80}, K=20, prefix_start stride 20, next-window targets.
Episode-level split (same seed=123 as profiled6 baseline); no cross-split leakage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_P6S = Path(__file__).resolve().parent
_P6 = _P6S.parents[1]  # profiled6/
_REPO = _P6.parent
for p in (_REPO, _P6, _P6S):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pandas as pd
import torch
from torch.optim import AdamW

from config_profiled6_sliding import (
    CKPT_DIR,
    CKPT_DIR_INPUTPAD_V2,
    CKPT_DIR_INPUTPAD_V2_EPOCHS,
    DATA_CSV,
    LOG_DIR,
    LOG_DIR_INPUTPAD_V2,
    MAX_PREFIX_L,
    PROFILED6_AGENTS,
    PROFILED6_SLIDING_ROOT,
    PROFILED6_TASKS,
    SLIDING_STRIDE,
    SPLIT_MANIFEST,
    TARGET_LEN,
    TRAIN_L_VALUES,
)
from profile_benchmark.data_profile import save_split_manifest
from profile_benchmark.data_profile_sliding import make_sliding_loaders
from profile_benchmark.engine.forward import forward_model, loss_on_batch
from profile_benchmark.engine.model_build import build_model_kwargs
from profile_benchmark.models import MODEL_REGISTRY
from profile_benchmark.utils import set_seed

try:
    from profile_benchmark.engine.model_build import count_parameters
except ImportError:
    import torch.nn as nn

    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def train_one_epoch(model, loader, optimiser, device, use_prefix_mask: bool = False) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        pmask = batch.get("prefix_mask")
        if pmask is not None:
            pmask = pmask.to(device)
        out = forward_model(
            model,
            batch["prefix_stim"].to(device),
            batch["prefix_action"].to(device),
            batch["prefix_reward"].to(device),
            batch["future_stim"].to(device),
            return_outputs=False,
            prefix_mask=pmask if use_prefix_mask else None,
        )
        loss = loss_on_batch(out, batch["future_action"].to(device), batch.get("future_mask"))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total += loss.item()
        n += 1
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device, use_prefix_mask: bool = False) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in loader:
        pmask = batch.get("prefix_mask")
        if pmask is not None:
            pmask = pmask.to(device)
        out = forward_model(
            model,
            batch["prefix_stim"].to(device),
            batch["prefix_action"].to(device),
            batch["prefix_reward"].to(device),
            batch["future_stim"].to(device),
            return_outputs=False,
            prefix_mask=pmask if use_prefix_mask else None,
        )
        logits = out if not isinstance(out, dict) else out["logits"]
        fa = batch["future_action"].to(device)
        fmask = batch.get("future_mask")
        B, K, A = logits.shape
        loss = torch.nn.functional.cross_entropy(
            logits.view(B * K, A), fa.view(-1), reduction="none"
        ).view(B, K)
        if fmask is not None:
            loss = loss[fmask.to(device)]
            preds = logits.argmax(dim=-1)
            correct = preds == fa
            correct = correct[fmask.to(device)]
            total_count += int(fmask.sum().item())
        else:
            loss = loss.view(-1)
            preds = logits.argmax(dim=-1)
            correct = (preds == fa).view(-1)
            total_count += fa.numel()
        total_loss += loss.sum().item()
        total_correct += correct.sum().item()
    return {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data", type=str, default=str(DATA_CSV))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    parser.add_argument("--log-dir", type=str, default=str(LOG_DIR))
    parser.add_argument("--split-manifest", type=str, default=str(SPLIT_MANIFEST))
    parser.add_argument("--stride", type=int, default=SLIDING_STRIDE)
    parser.add_argument("--max-prefix-len", type=int, default=MAX_PREFIX_L)
    parser.add_argument("--weak-transformer", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--padding-corrected", action="store_true")
    parser.add_argument("--inputpad-v2", action="store_true")
    parser.add_argument(
        "--save-epoch-dir",
        type=str,
        default="",
        help="If set, save weights after every epoch under {dir}/{model}/seed_{seed}/checkpoint_epoch_XXX.pt",
    )
    args = parser.parse_args()

    if args.inputpad_v2:
        if args.ckpt_dir == str(CKPT_DIR):
            args.ckpt_dir = str(CKPT_DIR_INPUTPAD_V2)
        if args.log_dir == str(LOG_DIR):
            args.log_dir = str(LOG_DIR_INPUTPAD_V2)
        if not args.save_epoch_dir:
            args.save_epoch_dir = str(CKPT_DIR_INPUTPAD_V2_EPOCHS)

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing {data_path}. Run env_v2.simulate first.")

    PROFILED6_SLIDING_ROOT.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(
            f"[device] cuda:{torch.cuda.current_device()} {torch.cuda.get_device_name(0)} "
            f"(torch {torch.__version__})",
            flush=True,
        )
    else:
        print(f"[device] cpu (torch {torch.__version__}; CUDA not available in this build)", flush=True)

    train_loader, val_loader, test_loader, stim_dim, n_actions, splits, train_specs = make_sliding_loaders(
        str(data_path),
        batch_size=args.batch_size,
        train_L_values=TRAIN_L_VALUES,
        target_len=TARGET_LEN,
        stride=args.stride,
        include_tasks=PROFILED6_TASKS,
    )

    split_path = Path(args.split_manifest)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    save_split_manifest(
        splits["train"],
        splits["val"],
        splits["test"],
        split_path,
        split_seed=123,
        extra={
            "benchmark": "profiled6_sliding",
            "training": "variable_L_sliding",
            "train_L": list(TRAIN_L_VALUES),
            "target_len": TARGET_LEN,
            "stride": args.stride,
            "n_train_windows_per_episode": len(train_specs),
            "tasks": PROFILED6_TASKS,
            "agents": PROFILED6_AGENTS,
            "data": str(data_path.resolve()),
        },
    )

    weak_tf = args.weak_transformer and args.model == "transformer"
    one_hot = bool(args.inputpad_v2)
    model_kw = build_model_kwargs(
        args.model,
        stim_dim,
        n_actions,
        prefix_len=args.max_prefix_len,
        weak_transformer=weak_tf,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        one_hot_action=one_hot,
    )
    model = MODEL_REGISTRY[args.model](**model_kw).to(device)
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    num_params = count_parameters(model)

    log_rows: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None

    use_prefix_mask = bool(args.padding_corrected or args.inputpad_v2)
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, train_loader, optimiser, device, use_prefix_mask=use_prefix_mask
        )
        val_stats = evaluate(model, val_loader, device, use_prefix_mask=use_prefix_mask)
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["acc"],
                "lr": args.lr,
            }
        )
        print(
            f"[sliding][{args.model}] ep {epoch}/{args.epochs} "
            f"train={tr_loss:.4f} val_loss={val_stats['loss']:.4f} val_acc={val_stats['acc']:.4f}"
        )
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if args.save_epoch_dir:
            ep_dir = Path(args.save_epoch_dir) / args.model / f"seed_{args.seed}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            ep_cfg = {
                "benchmark": "profiled6_sliding",
                "model": args.model,
                "seed": args.seed,
                "epoch": epoch,
                "one_hot_action": one_hot,
                "padding_corrected": use_prefix_mask,
                "implementation": "inputpad_v2" if args.inputpad_v2 else (
                    "padding_corrected" if args.padding_corrected else "legacy"
                ),
                "hidden_dim": model_kw["hidden_dim"],
                "num_layers": model_kw.get("num_layers", model_kw.get("n_layers")),
                "dropout": model_kw["dropout"],
                "max_prefix_len": args.max_prefix_len,
                "weak_transformer": weak_tf,
            }
            ep_obj: Dict[str, Any] = {
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "best_val_loss": best_val,
                "stim_dim": stim_dim,
                "n_actions": n_actions,
                "hidden_dim": model_kw["hidden_dim"],
                "num_layers": model_kw.get("num_layers", model_kw.get("n_layers")),
                "dropout": model_kw["dropout"],
                "num_params": num_params,
                "max_prefix_len": args.max_prefix_len,
                "config": ep_cfg,
            }
            if args.model == "transformer":
                ep_obj.update(
                    use_pos_embed=model_kw.get("use_pos_embed", True),
                    pool=model_kw.get("pool", "mean"),
                    n_heads=model_kw.get("n_heads", 1),
                )
            torch.save(ep_obj, ep_dir / f"checkpoint_epoch_{epoch:03d}.pt")

    if best_state is None:
        print("[WARN] No improvement; skipping checkpoint save")
        return

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.model}_seed{args.seed}.pt"

    config = {
        "benchmark": "profiled6_sliding",
        "model": args.model,
        "seed": args.seed,
        "training_scheme": "variable_L_sliding",
        "train_L_values": list(TRAIN_L_VALUES),
        "eval_L_values": [5, 10, 20, 40, 80],
        "max_prefix_len": args.max_prefix_len,
        "future_K": TARGET_LEN,
        "sliding_stride": args.stride,
        "operational_prefix": "0-79",
        "operational_target": "80-99",
        "tasks": PROFILED6_TASKS,
        "agents": PROFILED6_AGENTS,
        "split_seed": 123,
        "split_manifest": str(split_path.resolve()),
        "data": str(data_path.resolve()),
        "weak_transformer": weak_tf,
        "hidden_dim": model_kw["hidden_dim"],
        "num_layers": model_kw.get("num_layers", model_kw.get("n_layers")),
        "dropout": model_kw["dropout"],
        "lr": args.lr,
        "epochs": args.epochs,
        "num_params": num_params,
        "padding_corrected": bool(args.padding_corrected or args.inputpad_v2),
        "one_hot_action": one_hot,
        "implementation": (
            "inputpad_v2"
            if args.inputpad_v2
            else ("padding_corrected" if args.padding_corrected else "legacy")
        ),
    }

    model.load_state_dict(best_state)
    test_stats = evaluate(model, test_loader, device, use_prefix_mask=use_prefix_mask)

    save_obj: Dict[str, Any] = {
        "model_state_dict": best_state,
        "optimizer_state_dict": optimiser.state_dict(),
        "epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss": test_stats["loss"],
        "test_acc": test_stats["acc"],
        "stim_dim": stim_dim,
        "n_actions": n_actions,
        "hidden_dim": model_kw["hidden_dim"],
        "num_layers": model_kw.get("num_layers", model_kw.get("n_layers")),
        "dropout": model_kw["dropout"],
        "num_params": num_params,
        "max_prefix_len": args.max_prefix_len,
        "config": config,
    }
    if args.model == "transformer":
        save_obj.update(
            use_pos_embed=model_kw.get("use_pos_embed", True),
            pool=model_kw.get("pool", "mean"),
            n_heads=model_kw.get("n_heads", 1),
            max_prefix_len=model_kw.get("max_prefix_len", args.max_prefix_len),
        )

    torch.save(save_obj, ckpt_path)
    metrics_path = ckpt_dir / f"{args.model}_seed{args.seed}_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "test_loss": test_stats["loss"],
                "test_acc": test_stats["acc"],
                "num_params": num_params,
                "config": config,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log_rows).to_csv(log_dir / f"{args.model}_seed{args.seed}_trainlog.csv", index=False)
    print(f"[OK] checkpoint {ckpt_path}")
    print(f"[OK] test_acc={test_stats['acc']:.4f} (sliding val/test windows)")


if __name__ == "__main__":
    main()
