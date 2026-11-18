#!/usr/bin/env python3
"""During-training history vs final action prediction (all 11 arches).

Mirrors 04_training_trajectory_enrichment.py but for Behaviour Prediction
Accuracy (linear probe → future action). Uses existing epoch checkpoints.
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parent
EXT = V3 / "fig3_representation_formation_extension"
PRED_SCRIPTS = V3 / "stagewise_prediction_all_models" / "scripts"
HANDOFF = V3.parent
SLIDING = HANDOFF.parent
REPO = SLIDING.parents[2]
EPOCH_CKPT = SLIDING / "checkpoints_inputpad_v2_epochs"
FINAL_CKPT = SLIDING / "checkpoints_inputpad_v2"
OUT = ROOT / "data" / "training_trajectory_action_pred"
FIG = ROOT / "figures" / "training_trajectory_action_pred"

sys.path[:0] = [str(EXT), str(SLIDING), str(REPO)]

from config import LABEL, L_PRIMARY, SEEDS  # noqa: E402
from lib.extract_stages import extract_formation_stages  # noqa: E402
from lib.io_util import DEVICE, load_test_episodes, make_loader, summarise, use_mask  # noqa: E402
from lib.stage_definitions import STAGE_SPEC  # noqa: E402
from profile_benchmark.engine.checkpoint import load_checkpoint  # noqa: E402

_spec_u = importlib.util.spec_from_file_location(
    "unified", Path(__file__).resolve().parent / "01_recompute_and_build.py"
)
_u = importlib.util.module_from_spec(_spec_u)
_spec_u.loader.exec_module(_u)

_spec_p = importlib.util.spec_from_file_location(
    "stagewise_pred", PRED_SCRIPTS / "01_stagewise_prediction.py"
)
_p = importlib.util.module_from_spec(_spec_p)
_spec_p.loader.exec_module(_p)
action_prediction_probe = _p.action_prediction_probe

DISPLAY = LABEL
COL = _u.COL
stars = _u.stars
holm = _u.holm
write_wb = _u.write_wb
STRUCTURAL_ZERO = _u.STRUCTURAL_ZERO

ARCHES = [
    "lstm_attn",
    "post_bilstm",
    "lstm",
    "bilstm",
    "gru",
    "pre_lstm",
    "pre_bilstm",
    "transformer",
    "cnn",
    "mlp_pool",
    "tinyrnn_paper",
]
EPOCHS = list(range(1, 21))
POST_ATTN = {"lstm_attn", "post_bilstm"}


@torch.no_grad()
def collect_pair(model, loader, use_pmask: bool, arch: str, probe_seed: int) -> dict:
    buckets: dict[int, list] = {}
    fa_list = []
    meta_same = {}
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
        identical = bool(st["identical_enc_final"]) or arch in STRUCTURAL_ZERO
        for i, Z in st["stages"].items():
            if i in (enc_idx, fin_idx):
                buckets.setdefault(i, []).append(Z.detach().cpu().numpy())
                meta_same[i] = bool(st["same_flags"][i])
        fa_list.append(fa)

    FA = np.concatenate(fa_list, 0)
    acc = {}
    for i in (enc_idx, fin_idx):
        Zi = np.concatenate(buckets[i], 0)
        if i == fin_idx and i != enc_idx and meta_same.get(i):
            Zj = np.concatenate(buckets[enc_idx], 0)
            if Zj.shape == Zi.shape and np.allclose(Zi[:8], Zj[:8], atol=1e-4):
                acc[i] = acc[enc_idx]
                continue
        if i not in acc:
            m = action_prediction_probe(Zi, FA, seed=probe_seed)
            acc[i] = float(m["prediction_accuracy"])
    return {
        "pred_history": float(acc[enc_idx]),
        "pred_final": float(acc[fin_idx]),
        "pred_change": float(acc[fin_idx] - acc[enc_idx]),
        "structural_zero_flag": bool(identical),
        "encoder_label": STAGE_SPEC[arch][enc_idx][1] if enc_idx < len(STAGE_SPEC[arch]) else f"s{enc_idx}",
        "final_label": STAGE_SPEC[arch][fin_idx][1] if fin_idx < len(STAGE_SPEC[arch]) else f"s{fin_idx}",
        "n_windows": int(np.concatenate(buckets[enc_idx], 0).shape[0]),
    }


def load_epoch_model(arch: str, seed: int, epoch: int):
    path = EPOCH_CKPT / arch / f"seed_{seed}" / f"checkpoint_epoch_{epoch:03d}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    model, meta = load_checkpoint(path, DEVICE)
    return model, meta, path


def load_final_model(arch: str, seed: int):
    path = FINAL_CKPT / f"{arch}_seed{seed}.pt"
    model, meta = load_checkpoint(path, DEVICE)
    return model, meta, path


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / "raw_epoch_pairs_action_pred.csv"

    rows, done = [], set()
    if raw_path.is_file():
        prev = pd.read_csv(raw_path)
        rows = prev.to_dict("records")
        done = {(r["architecture"], int(r["seed"]), str(r["epoch_label"])) for r in rows}

    test_eps = load_test_episodes()
    loader = make_loader(test_eps, L=L_PRIMARY, batch_size=64)
    print(f"[traj-pred] device={DEVICE} n_test={len(test_eps)} arches={ARCHES}", flush=True)

    jobs = []
    for arch in ARCHES:
        for seed in SEEDS:
            for ep in EPOCHS:
                jobs.append((arch, seed, ep, f"epoch_{ep:03d}", False))
            jobs.append((arch, seed, -1, "final_best", True))

    n_todo = sum(1 for a, s, e, lab, _ in jobs if (a, s, lab) not in done)
    print(f"[traj-pred] remaining jobs={n_todo}", flush=True)

    for arch, seed, ep, label, is_final in jobs:
        key = (arch, seed, label)
        if key in done:
            continue
        try:
            if is_final:
                model, meta, path = load_final_model(arch, seed)
                ep_val = int(meta.get("epoch") or -1)
            else:
                model, meta, path = load_epoch_model(arch, seed, ep)
                ep_val = ep
        except Exception as e:
            print(f"[skip] {arch} s{seed} {label}: {e}", flush=True)
            continue

        # plane-unified: probe_seed = model seed (same policy as agent traj)
        rec = collect_pair(model, loader, use_mask(meta), arch, probe_seed=seed)
        row = {
            "architecture": arch,
            "architecture_display": DISPLAY.get(arch, arch),
            "seed": int(seed),
            "epoch": int(ep_val) if ep_val is not None else -1,
            "epoch_label": label,
            "is_final_best": bool(is_final),
            "checkpoint": str(path),
            "probe_seed": int(seed),
            "probe_policy": "model_seed",
            **rec,
        }
        rows.append(row)
        done.add(key)
        pd.DataFrame(rows).to_csv(raw_path, index=False)
        print(
            f"[OK] {arch} s{seed} {label} Ph={rec['pred_history']:.3f} "
            f"Pf={rec['pred_final']:.3f} d={rec['pred_change']:+.3f}",
            flush=True,
        )
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def summarize(raw: pd.DataFrame):
    ep = raw[~raw.is_final_best].copy()
    sum_rows = []
    for (arch, epoch), g in ep.groupby(["architecture", "epoch"]):
        for col, name in [
            ("pred_history", "pred_history"),
            ("pred_final", "pred_final"),
            ("pred_change", "change"),
        ]:
            st = summarise(g[col].values)
            sum_rows.append(
                {
                    "architecture": arch,
                    "architecture_display": DISPLAY.get(arch, arch),
                    "epoch": int(epoch),
                    "metric": name,
                    "mean": st["mean"],
                    "SEM": st["SEM"],
                    "SD": st["SD"],
                    "N": st["N"],
                }
            )
    summary = pd.DataFrame(sum_rows)

    chg_rows = []
    for (arch, epoch), g in ep.groupby(["architecture", "epoch"]):
        d = g.pred_change.values
        st = summarise(d)
        if np.isfinite(d).sum() >= 2:
            tt = stats.ttest_1samp(d, 0.0)
            tp = float(tt.pvalue)
        else:
            tp = np.nan
        chg_rows.append(
            {
                "architecture": arch,
                "architecture_display": DISPLAY.get(arch, arch),
                "epoch": int(epoch),
                "change_mean": st["mean"],
                "change_SEM": st["SEM"],
                "frac_positive": float(np.mean(d > 0)),
                "p_vs_0": tp,
                "stars": stars(tp),
            }
        )
    chg = pd.DataFrame(chg_rows)
    outs = []
    for arch, g in chg.groupby("architecture"):
        g = g.copy()
        g["p_Holm"] = holm(g["p_vs_0"].values)
        g["stars_Holm"] = g["p_Holm"].map(stars)
        outs.append(g)
    chg = pd.concat(outs, ignore_index=True)

    peak_rows = []
    for arch, g in chg.groupby("architecture"):
        i = g["change_mean"].idxmax()
        r = g.loc[i]
        fin = g[g.epoch == g.epoch.max()].iloc[0]
        peak_rows.append(
            {
                "architecture": arch,
                "architecture_display": DISPLAY.get(arch, arch),
                "peak_epoch": int(r.epoch),
                "peak_change": float(r.change_mean),
                "peak_stars": r.stars_Holm,
                "final_epoch": int(fin.epoch),
                "final_change": float(fin.change_mean),
                "final_stars": fin.stars_Holm,
                "peak_minus_final": float(r.change_mean - fin.change_mean),
                "ever_positive_sig": bool(
                    ((g.change_mean > 0) & (g.stars_Holm.isin(["*", "**", "***"]))).any()
                ),
            }
        )
    peak = pd.DataFrame(peak_rows)
    return summary, chg, peak


def train_avg_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ep = raw[~raw.is_final_best].copy()
    seed_lvl = (
        ep.groupby(["architecture", "seed"], as_index=False)
        .agg(
            pred_history=("pred_history", "mean"),
            pred_final=("pred_final", "mean"),
            pred_change=("pred_change", "mean"),
            n_epochs=("epoch", "nunique"),
        )
    )
    rows = []
    for arch, g in seed_lvl.groupby("architecture"):
        a, b, d = g.pred_history.values, g.pred_final.values, g.pred_change.values
        if len(d) >= 2:
            tp = float(stats.ttest_rel(b, a).pvalue)
        else:
            tp = np.nan
        rows.append(
            {
                "architecture": arch,
                "architecture_display": DISPLAY.get(arch, arch),
                "n_seeds": int(len(g)),
                "n_epochs_per_seed": int(g.n_epochs.iloc[0]),
                "pred_history_mean": float(np.mean(a)),
                "pred_history_SEM": float(np.std(a, ddof=1) / np.sqrt(len(a))),
                "pred_final_mean": float(np.mean(b)),
                "pred_final_SEM": float(np.std(b, ddof=1) / np.sqrt(len(b))),
                "change_mean": float(np.mean(d)),
                "change_SEM": float(np.std(d, ddof=1) / np.sqrt(len(d))),
                "paired_p": tp,
                "stars": stars(tp),
                "is_post_attn": arch in POST_ATTN,
            }
        )
    out = pd.DataFrame(rows)
    out["paired_p_Holm"] = holm(out["paired_p"].values)
    out["stars_Holm"] = out["paired_p_Holm"].map(stars)
    return out, seed_lvl


def plot(summary: pd.DataFrame, chg: pd.DataFrame, peak: pd.DataFrame, train_stats: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.set_title("A. Training-trajectory question (action pred)", loc="left", fontweight="bold")
    ax0.text(
        0.02,
        0.95,
        "Does history→final action-prediction Δ\n"
        "change across training?\n\n"
        "Metric: pred_final − pred_history\n"
        "(Behaviour Prediction Accuracy)\n"
        "probe_seed = model seed\n"
        "Locked test episodes (n=120).\n\n"
        "Epoch ckpts: checkpoints_inputpad_v2_epochs\n"
        "No retrain. All 11 architectures.",
        va="top",
        fontsize=8,
        transform=ax0.transAxes,
        family="monospace",
    )

    axb = fig.add_subplot(gs[0, 1])
    for arch in ARCHES:
        g = chg[chg.architecture.eq(arch)].sort_values("epoch")
        if g.empty:
            continue
        lw = 2.4 if arch in POST_ATTN else 1.15
        axb.plot(
            g.epoch,
            g.change_mean,
            color=COL.get(arch, "#666"),
            lw=lw,
            marker="o",
            ms=3.5 if arch in POST_ATTN else 2.2,
            label=DISPLAY.get(arch, arch),
            alpha=0.95 if arch in POST_ATTN else 0.7,
        )
        axb.fill_between(
            g.epoch,
            g.change_mean - g.change_SEM,
            g.change_mean + g.change_SEM,
            color=COL.get(arch, "#666"),
            alpha=0.15 if arch in POST_ATTN else 0.05,
        )
    axb.axhline(0, color="#222", lw=0.9)
    axb.set_xlabel("Training epoch")
    axb.set_ylabel("Action prediction Δ\n(final − history)")
    axb.set_title("B. Δ across training", loc="left", fontweight="bold")
    axb.legend(fontsize=5.0, frameon=False, ncol=2, loc="best")

    # C: absolute pred_final over epochs for post vs others
    axc = fig.add_subplot(gs[1, 0])
    for arch in ARCHES:
        g = summary[(summary.architecture.eq(arch)) & (summary.metric.eq("pred_final"))].sort_values("epoch")
        if g.empty:
            continue
        axc.plot(
            g.epoch,
            g["mean"],
            color=COL.get(arch, "#666"),
            lw=2.2 if arch in POST_ATTN else 1.0,
            label=DISPLAY.get(arch, arch),
            alpha=0.95 if arch in POST_ATTN else 0.65,
        )
    axc.set_xlabel("Training epoch")
    axc.set_ylabel("Final-stage Behaviour Prediction Accuracy")
    axc.set_title("C. Final-stage action prediction over training", loc="left", fontweight="bold")
    axc.legend(fontsize=4.8, frameon=False, ncol=2, loc="lower right")

    axd = fig.add_subplot(gs[1, 1])
    show = peak.set_index("architecture").reindex([a for a in ARCHES if a in set(peak.architecture)])
    x = np.arange(len(show))
    w = 0.38
    axd.bar(x - w / 2, show.peak_change.values, w, color="#7A8F9A", label="Peak Δ during training")
    axd.bar(x + w / 2, show.final_change.values, w, color="#3D5A80", label="Epoch-20 Δ")
    axd.axhline(0, color="#222", lw=0.8)
    axd.set_xticks(x)
    axd.set_xticklabels([DISPLAY.get(a, a) for a in show.index], rotation=30, ha="right", fontsize=6)
    axd.set_ylabel("Action prediction Δ")
    axd.set_title("D. Peak mid-training vs end", loc="left", fontweight="bold")
    axd.legend(fontsize=6.5, frameon=False)

    fig.suptitle(
        "Training trajectory: action prediction (history → final)",
        fontsize=12,
        fontweight="bold",
        y=0.995,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"Figure3_training_trajectory_action_pred.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    raw = run()
    summary, chg, peak = summarize(raw)
    train_stats, seed_lvl = train_avg_table(raw)
    plot(summary, chg, peak, train_stats)

    methods = pd.DataFrame(
        [
            ("metric", "Behaviour Prediction Accuracy = action prediction (logistic probe)"),
            ("delta", "pred_final − pred_history"),
            ("probe_seed", "model seed (plane-unified policy)"),
            ("episodes", "locked test episodes"),
            ("arches", "all 11"),
            ("created_utc", datetime.now(timezone.utc).isoformat()),
        ],
        columns=["item", "description"],
    )
    xlsx = OUT / "Figure3_training_trajectory_action_pred_DATA.xlsx"
    write_wb(
        xlsx,
        {
            "00_README": methods,
            "epoch_metric_summary": summary,
            "epoch_change_stats": chg,
            "peak_vs_final": peak,
            "trainavg_arch": train_stats,
            "trainavg_seed_level": seed_lvl,
            "raw_epoch_pairs": raw,
        },
    )
    print("=== PEAK vs FINAL (action pred Δ) ===")
    print(
        peak.set_index("architecture")
        .reindex(ARCHES)[
            ["architecture_display", "peak_epoch", "peak_change", "final_change", "final_stars"]
        ]
        .round(4)
        .to_string(index=False)
    )
    print(f"Excel: {xlsx}")
    print(f"Figure: {FIG / 'Figure3_training_trajectory_action_pred.png'}")


if __name__ == "__main__":
    main()
