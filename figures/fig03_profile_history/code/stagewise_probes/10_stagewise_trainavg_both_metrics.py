#!/usr/bin/env python3
"""Stage-wise action prediction + agent decoding, averaged over training epochs 1–20.

Collects all representation stages at each epoch checkpoint (no retrain),
then per-seed mean over epochs, then mean±SEM across seeds.
Plots a two-panel figure matching the locked stagewise layout.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.gridspec import GridSpec
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parent
EXT = V3 / "fig3_representation_formation_extension"
PRED_SCRIPTS = V3 / "stagewise_prediction_all_models" / "scripts"
HANDOFF = V3.parent
SLIDING = HANDOFF.parent
REPO = SLIDING.parents[2]
EPOCH_CKPT = SLIDING / "checkpoints_inputpad_v2_epochs"
OUT = ROOT / "data" / "stagewise_trainavg"
FIG = ROOT / "figures" / "stagewise_trainavg"

sys.path[:0] = [str(EXT), str(SLIDING), str(REPO)]

from config import LABEL, L_PRIMARY, SEEDS  # noqa: E402
from lib.extract_stages import extract_formation_stages  # noqa: E402
from lib.io_util import DEVICE, load_test_episodes, make_loader, summarise, use_mask  # noqa: E402
from lib.stage_definitions import STAGE_SPEC  # noqa: E402
from profile_benchmark.engine.checkpoint import load_checkpoint  # noqa: E402
from profile_benchmark.eval.analyse_sliding_geometry import within_task_agent_probe  # noqa: E402

_spec_p = importlib.util.spec_from_file_location(
    "stagewise_pred", PRED_SCRIPTS / "01_stagewise_prediction.py"
)
_p = importlib.util.module_from_spec(_spec_p)
_spec_p.loader.exec_module(_p)
action_prediction_probe = _p.action_prediction_probe

DISPLAY = LABEL
COL = {
    "lstm_attn": "#0B3D91",
    "post_bilstm": "#6B4C9A",
    "lstm": "#3A6EA5",
    "bilstm": "#C0392B",
    "pre_lstm": "#5DADE2",
    "gru": "#1E8449",
    "pre_bilstm": "#D98880",
    "cnn": "#E67E22",
    "transformer": "#B7950B",
    "tinyrnn_paper": "#48C9B0",
    "mlp_pool": "#6E2C00",
}
ARCH_ORDER = [
    "lstm_attn",
    "post_bilstm",
    "lstm",
    "bilstm",
    "pre_lstm",
    "gru",
    "pre_bilstm",
    "cnn",
    "transformer",
    "tinyrnn_paper",
    "mlp_pool",
]
EPOCHS = list(range(1, 21))
STAGE_DISPLAY = {
    "trial_input": "Trial input",
    "preattn_transformed": "Pre-attention",
    "encoded_history": "History encoding",
    "pooled_history": "History encoding",
    "integrated_readout": "Final profile",
    "attention_readout": "Final profile",
    "pooled_readout": "Final profile",
    "final_z_profile": "Final profile",
}
STAGE_X_ORDER = ["Trial input", "Pre-attention", "History encoding", "Final profile"]


def write_wb(path: Path, sheets: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    wb.remove(wb.active)
    for name, df in sheets.items():
        ws = wb.create_sheet(str(name)[:31])
        for r in dataframe_to_rows(df if df is not None else pd.DataFrame(), index=False, header=True):
            ws.append(list(r))
        ws.auto_filter.ref = ws.dimensions
    wb.save(path)


@torch.no_grad()
def collect_all_stages(model, loader, use_pmask: bool, arch: str, probe_seed: int) -> list[dict]:
    buckets: dict[int, list] = {}
    fa_list, agents, tasks = [], [], []
    meta_same, meta_dim, meta_rep = {}, {}, {}
    enc_idx = fin_idx = None
    identical = False

    for batch in loader:
        ps = batch["prefix_stim"].to(DEVICE)
        pa = batch["prefix_action"].to(DEVICE)
        pr = batch["prefix_reward"].to(DEVICE)
        fs = batch["future_stim"].to(DEVICE)
        fa = batch["future_action"].cpu().numpy()
        pmask = batch.get("prefix_mask")
        pmask = pmask.to(DEVICE) if (use_pmask and pmask is not None) else None
        st = extract_formation_stages(model, ps, pa, pr, fs, pmask)
        enc_idx = st["encoder_idx"]
        fin_idx = st["final_idx"]
        identical = bool(st["identical_enc_final"])
        for i, Z in st["stages"].items():
            buckets.setdefault(i, []).append(Z.detach().cpu().numpy())
            meta_same[i] = bool(st["same_flags"][i])
            meta_dim[i] = int(st["dims"][i])
            meta_rep[i] = st["rep_names"][i]
        fa_list.append(fa)
        agents.extend(batch["agent"])
        tasks.extend(batch["task"])

    FA = np.concatenate(fa_list, 0)
    ag = np.asarray(agents, str)
    tk = np.asarray(tasks, str)
    agent_acc, pred_acc, canonical = {}, {}, {}
    sorted_idx = sorted(buckets.keys())
    for i in sorted_idx:
        Zi = np.concatenate(buckets[i], 0)
        reused = None
        for j in sorted_idx:
            if j >= i:
                break
            if meta_same.get(i):
                Zj = np.concatenate(buckets[j], 0)
                if Zj.shape == Zi.shape and np.allclose(Zi[:8], Zj[:8], atol=1e-4):
                    reused = j
                    break
        if reused is not None:
            agent_acc[i] = agent_acc[reused]
            pred_acc[i] = pred_acc[reused]
            canonical[i] = reused
        else:
            agent_acc[i] = float(within_task_agent_probe(Zi, ag, tk, seed=probe_seed))
            pred_acc[i] = float(action_prediction_probe(Zi, FA, seed=probe_seed)["prediction_accuracy"])
            canonical[i] = i

    label_map = {a: b for a, b, *_ in STAGE_SPEC[arch]}
    rows = []
    for i in sorted_idx:
        rows.append(
            {
                "architecture": arch,
                "architecture_display": DISPLAY.get(arch, arch),
                "stage_index": int(i),
                "stage_label": label_map.get(i, f"stage_{i}"),
                "stage_display": STAGE_DISPLAY.get(label_map.get(i, ""), label_map.get(i, f"stage_{i}")),
                "representation_name": meta_rep[i],
                "representation_dimension": meta_dim[i],
                "same_representation_flag": bool(meta_same[i] or canonical[i] != i),
                "same_as_probed_stage": int(canonical[i]),
                "structural_zero_flag": bool(identical and i in (enc_idx, fin_idx)),
                "is_native_final_profile": bool(i == fin_idx),
                "encoder_stage_index": int(enc_idx),
                "final_stage_index": int(fin_idx),
                "Agent Decoding Accuracy": float(agent_acc[i]),
                "Behaviour Prediction Accuracy": float(pred_acc[i]),
                "n_windows": int(np.concatenate(buckets[i], 0).shape[0]),
                "probe_seed": int(probe_seed),
                "probe_policy": "model_seed",
            }
        )
    return rows


def run_collect() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / "raw_stagewise_epoch_both.csv"
    rows, done = [], set()
    if raw_path.is_file():
        prev = pd.read_csv(raw_path)
        rows = prev.to_dict("records")
        done = {(r["architecture"], int(r["seed"]), int(r["epoch"])) for r in rows}

    test_eps = load_test_episodes()
    loader = make_loader(test_eps, L=L_PRIMARY, batch_size=64)
    jobs = [(a, s, e) for a in ARCH_ORDER for s in SEEDS for e in EPOCHS]
    n_todo = sum(1 for j in jobs if j not in done)
    print(f"[stagewise-trainavg] device={DEVICE} remaining={n_todo}", flush=True)

    for arch, seed, ep in jobs:
        if (arch, seed, ep) in done:
            continue
        path = EPOCH_CKPT / arch / f"seed_{seed}" / f"checkpoint_epoch_{ep:03d}.pt"
        if not path.is_file():
            print(f"[skip] missing {path}", flush=True)
            continue
        try:
            model, meta = load_checkpoint(path, DEVICE)
        except Exception as e:
            print(f"[skip] {arch} s{seed} e{ep}: {e}", flush=True)
            continue
        recs = collect_all_stages(model, loader, use_mask(meta), arch, probe_seed=seed)
        for r in recs:
            r.update(
                {
                    "seed": int(seed),
                    "epoch": int(ep),
                    "checkpoint": str(path),
                }
            )
            rows.append(r)
        done.add((arch, seed, ep))
        pd.DataFrame(rows).to_csv(raw_path, index=False)
        fin = [r for r in recs if r["is_native_final_profile"]][0]
        print(
            f"[OK] {arch} s{seed} e{ep:02d} "
            f"A={fin['Behaviour Prediction Accuracy']:.3f} "
            f"P={fin['Agent Decoding Accuracy']:.3f}",
            flush=True,
        )
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def unique_stage_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per canonical probed stage (drop structural duplicates)."""
    out = []
    for _, g in df.groupby(["architecture", "seed", "epoch"]):
        g = g[g.stage_index == g.same_as_probed_stage].copy()
        for disp, gg in g.groupby("stage_display", sort=False):
            if disp == "Final profile" and gg.is_native_final_profile.any():
                out.append(gg[gg.is_native_final_profile].iloc[0])
            else:
                out.append(gg.sort_values("stage_index").iloc[0])
    return pd.DataFrame(out)


def summarize_trainavg(raw_u: pd.DataFrame):
    # per seed: mean over epochs
    seed_lvl = (
        raw_u.groupby(["architecture", "architecture_display", "seed", "stage_display", "stage_label"], as_index=False)
        .agg(
            agent=("Agent Decoding Accuracy", "mean"),
            pred=("Behaviour Prediction Accuracy", "mean"),
            n_epochs=("epoch", "nunique"),
        )
    )
    rows = []
    for (arch, disp, stage), g in seed_lvl.groupby(["architecture", "architecture_display", "stage_display"]):
        sa = summarise(g.agent.values)
        sp = summarise(g.pred.values)
        rows.append(
            {
                "architecture": arch,
                "architecture_display": disp,
                "stage_display": stage,
                "x": STAGE_X_ORDER.index(stage) if stage in STAGE_X_ORDER else 99,
                "agent_mean": sa["mean"],
                "agent_SEM": sa["SEM"],
                "agent_N": sa["N"],
                "pred_mean": sp["mean"],
                "pred_SEM": sp["SEM"],
                "pred_N": sp["N"],
                "n_epochs_per_seed": int(g.n_epochs.iloc[0]),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["architecture", "x"])
    return seed_lvl, summary


def plot_figure(summary: pd.DataFrame, seed_lvl: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 240,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    metrics = [
        (axes[0], "pred_mean", "pred_SEM", "pred", "a. Action prediction across representation stages", "Action prediction probe accuracy", (0.40, 0.58)),
        (axes[1], "agent_mean", "agent_SEM", "agent", "b. Agent identification across representation stages", "Agent identification accuracy", (0.10, 0.50)),
    ]
    for ax, mean_c, sem_c, seed_c, title, ylab, ylim in metrics:
        for arch in ARCH_ORDER:
            g = summary[summary.architecture.eq(arch)].sort_values("x")
            if g.empty:
                continue
            ax.errorbar(
                g["x"],
                g[mean_c],
                yerr=g[sem_c],
                fmt="-o",
                color=COL.get(arch, "#666"),
                ms=5,
                lw=1.6 if arch in ("lstm_attn", "post_bilstm") else 1.2,
                capsize=2,
                label=DISPLAY.get(arch, arch),
                zorder=5 if arch in ("lstm_attn", "post_bilstm") else 3,
            )
            # light seed points
            ss = seed_lvl[seed_lvl.architecture.eq(arch)].copy()
            ss["x"] = ss["stage_display"].map(lambda s: STAGE_X_ORDER.index(s) if s in STAGE_X_ORDER else np.nan)
            ss = ss.dropna(subset=["x"])
            ax.scatter(
                ss["x"] + np.random.default_rng(abs(hash(arch)) % 10_000).normal(0, 0.04, len(ss)),
                ss[seed_c],
                s=7,
                color=COL.get(arch, "#666"),
                alpha=0.12,
                lw=0,
                zorder=1,
            )
        ax.set_xticks(range(len(STAGE_X_ORDER)))
        ax.set_xticklabels(STAGE_X_ORDER, rotation=15, ha="right")
        ax.set_xlabel("Representation stage")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(*ylim)
        if seed_c == "agent":
            ax.axhline(1 / 6, color="#999", ls="--", lw=0.8)
        ax.legend(fontsize=6.2, frameon=False, loc="best", ncol=2)

    fig.suptitle(
        "Action prediction and agent identification across representation stages\n"
        "(mean over training epochs 1–20, then mean±SEM across seeds)",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    stem = "Figure3_stagewise_trainavg_both_metrics"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    return FIG / f"{stem}.png"


def main():
    raw = run_collect()
    raw_u = unique_stage_rows(raw)
    seed_lvl, summary = summarize_trainavg(raw_u)
    fig_path = plot_figure(summary, seed_lvl)
    write_wb(
        OUT / "Figure3_stagewise_trainavg_both_metrics_DATA.xlsx",
        {
            "00_README": pd.DataFrame(
                [
                    ("definition", "Per seed: mean over epochs 1-20 of stagewise probes; then mean±SEM across seeds"),
                    ("probe_seed", "model seed (plane-unified)"),
                    ("episodes", "locked test split"),
                    ("created_utc", datetime.now(timezone.utc).isoformat()),
                ],
                columns=["item", "description"],
            ),
            "summary_trainavg": summary,
            "seed_level_trainavg": seed_lvl,
            "raw_unique_head": raw_u.head(200),
        },
    )
    print(f"Figure: {fig_path}")
    print(f"Excel:  {OUT / 'Figure3_stagewise_trainavg_both_metrics_DATA.xlsx'}")
    print(f"Raw:    {OUT / 'raw_stagewise_epoch_both.csv'}")


if __name__ == "__main__":
    main()
