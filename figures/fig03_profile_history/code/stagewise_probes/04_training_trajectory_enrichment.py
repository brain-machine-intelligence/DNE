#!/usr/bin/env python3
"""Track history vs final agent decoding across training epochs.

Uses existing checkpoints_inputpad_v2_epochs (no retrain).
Question: does post-attn enrichment appear mid-training then vanish?
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
HANDOFF = V3.parent
SLIDING = HANDOFF.parent
REPO = SLIDING.parents[2]
EPOCH_CKPT = SLIDING / "checkpoints_inputpad_v2_epochs"
FINAL_CKPT = SLIDING / "checkpoints_inputpad_v2"
OUT = ROOT / "data" / "training_trajectory"
FIG = ROOT / "figures" / "training_trajectory"

sys.path[:0] = [str(EXT), str(SLIDING), str(REPO)]

from config import LABEL, L_PRIMARY, SEEDS  # noqa: E402
from lib.extract_stages import extract_formation_stages  # noqa: E402
from lib.io_util import DEVICE, load_test_episodes, make_loader, summarise, use_mask  # noqa: E402
from lib.stage_definitions import STAGE_SPEC  # noqa: E402
from profile_benchmark.engine.checkpoint import load_checkpoint  # noqa: E402
from profile_benchmark.eval.analyse_sliding_geometry import within_task_agent_probe  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "unified", Path(__file__).resolve().parent / "01_recompute_and_build.py"
)
_u = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_u)
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
    agents, tasks = [], []
    meta_same = {}
    enc_idx = fin_idx = None
    identical = False
    for batch in loader:
        ps = batch["prefix_stim"].to(DEVICE)
        pa = batch["prefix_action"].to(DEVICE)
        pr = batch["prefix_reward"].to(DEVICE)
        fs = batch["future_stim"].to(DEVICE)
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
        agents.extend(batch["agent"])
        tasks.extend(batch["task"])

    ag = np.asarray(agents, str)
    tk = np.asarray(tasks, str)
    acc = {}
    for i in (enc_idx, fin_idx):
        Zi = np.concatenate(buckets[i], 0)
        if i == fin_idx and i != enc_idx and meta_same.get(i):
            Zj = np.concatenate(buckets[enc_idx], 0)
            if Zj.shape == Zi.shape and np.allclose(Zi[:8], Zj[:8], atol=1e-4):
                acc[i] = acc[enc_idx]
                continue
        if i not in acc:
            acc[i] = float(within_task_agent_probe(Zi, ag, tk, seed=probe_seed))
    return {
        "P_encoder": float(acc[enc_idx]),
        "P_final": float(acc[fin_idx]),
        "Final_readout_change": float(acc[fin_idx] - acc[enc_idx]),
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
    raw_path = OUT / "raw_epoch_pairs.csv"

    rows, done = [], set()
    if raw_path.is_file():
        prev = pd.read_csv(raw_path)
        rows = prev.to_dict("records")
        done = {(r["architecture"], int(r["seed"]), str(r["epoch_label"])) for r in rows}

    test_eps = load_test_episodes()
    loader = make_loader(test_eps, L=L_PRIMARY, batch_size=64)
    print(f"[traj] device={DEVICE} n_test={len(test_eps)} arches={ARCHES}", flush=True)

    jobs = []
    for arch in ARCHES:
        for seed in SEEDS:
            for ep in EPOCHS:
                jobs.append((arch, seed, ep, f"epoch_{ep:03d}", False))
            jobs.append((arch, seed, int(meta_epoch_or_best(arch, seed)), "final_best", True))

    for arch, seed, ep, label, is_final in jobs:
        key = (arch, seed, label)
        if key in done:
            continue
        try:
            if is_final:
                model, meta, path = load_final_model(arch, seed)
                ep_val = int(meta.get("epoch") or ep or -1)
            else:
                model, meta, path = load_epoch_model(arch, seed, ep)
                ep_val = ep
        except Exception as e:
            print(f"[skip] {arch} s{seed} {label}: {e}", flush=True)
            continue

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
            f"[OK] {arch} s{seed} {label} Penc={rec['P_encoder']:.3f} "
            f"Pfin={rec['P_final']:.3f} d={rec['Final_readout_change']:+.3f}",
            flush=True,
        )
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def meta_epoch_or_best(arch: str, seed: int) -> int:
    # placeholder; actual epoch read from final ckpt meta
    return -1


def summarize(raw: pd.DataFrame):
    # epoch curves (exclude final_best from epoch axis; keep separately)
    ep = raw[~raw.is_final_best].copy()
    sum_rows = []
    for (arch, epoch), g in ep.groupby(["architecture", "epoch"]):
        for col, name in [
            ("P_encoder", "P_encoder"),
            ("P_final", "P_final"),
            ("Final_readout_change", "change"),
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

    # per-epoch paired stats for change
    chg_rows = []
    for (arch, epoch), g in ep.groupby(["architecture", "epoch"]):
        d = g.Final_readout_change.values
        st = summarise(d)
        if np.isfinite(d).sum() >= 2:
            # test change vs 0
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
    # Holm within each architecture across epochs
    outs = []
    for arch, g in chg.groupby("architecture"):
        g = g.copy()
        g["p_Holm"] = holm(g["p_vs_0"].values)
        g["stars_Holm"] = g["p_Holm"].map(stars)
        outs.append(g)
    chg = pd.concat(outs, ignore_index=True)

    # peak enrichment epoch per arch
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
                "ever_positive_sig": bool(((g.change_mean > 0) & (g.stars_Holm.isin(["*", "**", "***"]))).any()),
            }
        )
    peak = pd.DataFrame(peak_rows)
    return summary, chg, peak


def plot(summary: pd.DataFrame, chg: pd.DataFrame, peak: pd.DataFrame, raw: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 140, "savefig.dpi": 220, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(13.5, 8.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.set_title("A. Training-trajectory question", loc="left", fontweight="bold")
    ax0.text(
        0.02,
        0.95,
        "Does post-attn enrichment appear mid-training\n"
        "then disappear by convergence?\n\n"
        "Metric: P_final − P_history (agent decoding)\n"
        "probe_seed = model seed (plane-unified)\n"
        "Locked test episodes (n=120).\n\n"
        "Epoch ckpts: checkpoints_inputpad_v2_epochs\n"
        "No retrain.",
        va="top",
        fontsize=8,
        transform=ax0.transAxes,
    )

    # B: change trajectories all arches
    axb = fig.add_subplot(gs[0, 1])
    for arch in ARCHES:
        g = chg[chg.architecture.eq(arch)].sort_values("epoch")
        if g.empty:
            continue
        lw = 2.2 if arch in POST_ATTN else 1.1
        axb.plot(
            g.epoch,
            g.change_mean,
            color=COL.get(arch, "#666"),
            lw=lw,
            marker="o",
            ms=3 if arch in POST_ATTN else 2,
            label=DISPLAY.get(arch, arch),
            alpha=0.95 if arch in POST_ATTN else 0.75,
        )
        axb.fill_between(
            g.epoch,
            g.change_mean - g.change_SEM,
            g.change_mean + g.change_SEM,
            color=COL.get(arch, "#666"),
            alpha=0.12 if arch in POST_ATTN else 0.05,
        )
    axb.axhline(0, color="#333", lw=0.8)
    axb.set_xlabel("Epoch")
    axb.set_ylabel(r"$P_{final}-P_{history}$")
    axb.set_title("B. Enrichment Δ across training", loc="left", fontweight="bold")
    axb.legend(fontsize=5.5, frameon=False, ncol=2, loc="best")

    # C: post-attn P_encoder vs P_final
    axc = fig.add_subplot(gs[1, 0])
    for arch, ls in [("lstm_attn", "-"), ("post_bilstm", "--")]:
        for metric, color, lab in [
            ("P_encoder", COL.get(arch, "#666"), f"{DISPLAY.get(arch)} history"),
            ("P_final", COL.get(arch, "#666"), f"{DISPLAY.get(arch)} final"),
        ]:
            g = summary[(summary.architecture.eq(arch)) & (summary.metric.eq(metric))].sort_values("epoch")
            axc.plot(
                g.epoch,
                g["mean"],
                ls=ls if metric == "P_final" else ":",
                color=color,
                lw=1.8 if metric == "P_final" else 1.4,
                marker="o",
                ms=3,
                label=lab,
                alpha=0.9 if metric == "P_final" else 0.7,
            )
            axc.fill_between(g.epoch, g["mean"] - g.SEM, g["mean"] + g.SEM, color=color, alpha=0.08)
    axc.axhline(1 / 6, color="#999", ls="--", lw=0.8)
    axc.set_xlabel("Epoch")
    axc.set_ylabel("Agent Decoding Accuracy")
    axc.set_title("C. Post-attn: history vs final over epochs", loc="left", fontweight="bold")
    axc.legend(fontsize=6, frameon=False, loc="lower right")
    axc.set_ylim(0.05, 0.55)

    # D: peak vs final change for post-attn + refs
    axd = fig.add_subplot(gs[1, 1])
    show = peak.set_index("architecture").reindex([a for a in ARCHES if a in set(peak.architecture)])
    x = np.arange(len(show))
    w = 0.35
    axd.bar(x - w / 2, show.peak_change.values, w, color="#7A8F9A", label="Peak Δ during training")
    axd.bar(x + w / 2, show.final_change.values, w, color="#2F4A6A", label="Δ at epoch 20")
    axd.axhline(0, color="#333", lw=0.8)
    axd.set_xticks(x)
    axd.set_xticklabels([DISPLAY.get(a, a) for a in show.index], rotation=30, ha="right", fontsize=7)
    axd.set_ylabel(r"$P_{final}-P_{history}$")
    axd.set_title("D. Peak mid-training vs end", loc="left", fontweight="bold")
    axd.legend(fontsize=7, frameon=False)
    for i, a in enumerate(show.index):
        axd.text(i - w / 2, show.loc[a, "peak_change"], f"e{int(show.loc[a,'peak_epoch'])}", ha="center", va="bottom", fontsize=5.5, color="#444")

    fig.suptitle(
        "Training trajectory — does post-attn enrichment appear then vanish?",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"Figure3_training_trajectory_enrichment.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] figures -> {FIG}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "NOTE.txt").write_text(
        f"Built: {datetime.now(timezone.utc).isoformat()}\nDevice: {DEVICE}\n"
        f"Epoch ckpts: {EPOCH_CKPT}\nProbe: model_seed plane-unified\n",
        encoding="utf-8",
    )
    raw = run()
    summary, chg, peak = summarize(raw)

    methods = pd.DataFrame(
        [
            ("question", "Does post-attn enrichment appear mid-training then vanish?"),
            ("epochs", "1..20 from checkpoints_inputpad_v2_epochs"),
            ("final_best", "also evaluated locked final best checkpoint"),
            ("probe", "within-task agent; probe_seed=model seed"),
            ("episodes", "locked test split n=120"),
        ],
        columns=["item", "description"],
    )

    def sort_a(df):
        order = {a: i for i, a in enumerate(ARCHES)}
        out = df.copy()
        out["_o"] = out["architecture"].map(lambda a: order.get(a, 999))
        cols = [c for c in out.columns if c != "_o"]
        if "epoch" in out.columns:
            return out.sort_values(["_o", "epoch"]).drop(columns="_o")
        return out.sort_values("_o").drop(columns="_o")

    xlsx = OUT / "Figure3_training_trajectory_enrichment_DATA.xlsx"
    write_wb(
        xlsx,
        {
            "00_README": methods,
            "raw_pairs": sort_a(raw),
            "epoch_metric_summary": sort_a(summary),
            "epoch_change_stats": sort_a(chg),
            "peak_vs_final": sort_a(peak),
        },
    )
    plot(summary, chg, peak, raw)

    print("\n=== PEAK vs FINAL ===")
    print(sort_a(peak)[
        ["architecture_display", "peak_epoch", "peak_change", "peak_stars", "final_change", "final_stars", "ever_positive_sig"]
    ].round(4).to_string(index=False))
    print(f"Excel: {xlsx}")
    print(f"Figure: {FIG / 'Figure3_training_trajectory_enrichment.png'}")


if __name__ == "__main__":
    main()
