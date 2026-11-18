#!/usr/bin/env python3
"""Recompute stagewise agent decoding with probe_seed=model seed (plane-unified).

Does NOT overwrite original fig3_representation_formation_extension outputs.
Also merges stagewise Behaviour Prediction Accuracy and exports figure + Excel.
"""
from __future__ import annotations

import shutil
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
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT.parent
EXT = V3 / "fig3_representation_formation_extension"
PRED = V3 / "stagewise_prediction_all_models" / "data"
HANDOFF = V3.parent
SLIDING = HANDOFF.parent
REPO = SLIDING.parents[2]
FIG1 = HANDOFF / "profile_module_storyline_final" / "data" / "inputpad_v2" / "csv" / "Figure1_plot_data"
DATA = ROOT / "data"
FIG = ROOT / "figures"
QC = ROOT / "qc"

sys.path[:0] = [str(EXT), str(SLIDING), str(REPO)]

from config import ARCHES, LABEL, L_PRIMARY, SEEDS  # noqa: E402
from lib.extract_stages import extract_formation_stages  # noqa: E402
from lib.io_util import DEVICE, load_model, load_test_episodes, make_loader, summarise, use_mask  # noqa: E402
from lib.stage_definitions import STAGE_SPEC  # noqa: E402
from profile_benchmark.eval.analyse_sliding_geometry import within_task_agent_probe  # noqa: E402

DISPLAY = LABEL
COL = {
    "lstm_attn": "#0B3D91",
    "post_bilstm": "#5E7A6C",
    "lstm": "#424242",
    "bilstm": "#7A8F6A",
    "gru": "#3D5A80",
    "pre_lstm": "#C45C26",
    "pre_bilstm": "#B07848",
    "transformer": "#6E5F7A",
    "cnn": "#A14A3A",
    "mlp_pool": "#8A6A3A",
    "tinyrnn_paper": "#787878",
}
ARCH_ORDER = [
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
STAGE_DISPLAY = {
    "trial_input": "Trial representation",
    "preattn_transformed": "Pre-attn transform",
    "encoded_history": "History encoding",
    "integrated_readout": "Readout",
    "attention_readout": "Readout",
    "pooled_readout": "Readout",
    "pooled_history": "History encoding",
    "final_z_profile": "Final profile",
}
STAGE_X_ORDER = [
    "Trial representation",
    "Pre-attn transform",
    "History encoding",
    "Readout",
    "Final profile",
]
STRUCTURAL_ZERO = {"cnn", "transformer", "mlp_pool"}
PROBE_NOTE = "within_task agent probe; probe_seed = model seed (unified with Fig1 plane)"


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def holm(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    m = len(pv)
    order = np.argsort(pv)
    adjusted = np.empty(m, float)
    prev = 0.0
    for i, idx in enumerate(order):
        adj = min(1.0, pv[idx] * (m - i))
        adj = max(adj, prev)
        adjusted[idx] = adj
        prev = adj
    out[np.where(valid)[0]] = adjusted
    return out


@torch.no_grad()
def collect(model, loader, use_pmask: bool, arch: str, seed: int) -> list[dict]:
    buckets: dict[int, list] = {}
    agents, tasks = [], []
    meta_same, meta_dim, meta_rep = {}, {}, {}
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
            buckets.setdefault(i, []).append(Z.detach().cpu().numpy())
            meta_rep[i] = st["rep_names"][i]
            meta_dim[i] = int(st["dims"][i])
            meta_same[i] = bool(st["same_flags"][i])
        agents.extend(batch["agent"])
        tasks.extend(batch["task"])

    ag = np.asarray(agents, str)
    tk = np.asarray(tasks, str)
    stage_acc = {}
    canonical = {}
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
            stage_acc[i] = stage_acc[reused]
            canonical[i] = reused
        else:
            # UNIFIED with plane: probe_seed = model seed
            stage_acc[i] = float(within_task_agent_probe(Zi, ag, tk, seed=seed))
            canonical[i] = i

    label_map = {a: b for a, b, *_ in STAGE_SPEC[arch]}
    rows = []
    for i in sorted_idx:
        rows.append(
            {
                "architecture": arch,
                "architecture_display": DISPLAY.get(arch, arch),
                "seed": seed,
                "checkpoint": f"{arch}_seed{seed}.pt",
                "history_length": L_PRIMARY,
                "stage_index": i,
                "stage_label": label_map.get(i, f"stage_{i}"),
                "representation_name": meta_rep[i],
                "representation_dimension": meta_dim[i],
                "same_representation_flag": bool(meta_same[i] or canonical[i] != i),
                "same_as_probed_stage": canonical[i],
                "structural_zero_flag": bool(identical and i in (enc_idx, fin_idx)),
                "Agent Decoding Accuracy": stage_acc[i],
                "encoder_stage_index": enc_idx,
                "final_stage_index": fin_idx,
                "n_windows": int(np.concatenate(buckets[i], 0).shape[0]),
                "probe_seed": seed,
                "probe_seed_policy": "model_seed_unified_with_plane",
            }
        )
    return rows


def recompute_agent() -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    raw_path = DATA / "raw_stagewise_agent_unified.csv"
    rows = []
    done = set()
    if raw_path.is_file():
        prev = pd.read_csv(raw_path)
        rows = prev.to_dict("records")
        done = {(r["architecture"], int(r["seed"])) for r in rows}

    test_eps = load_test_episodes()
    loader = make_loader(test_eps, L=L_PRIMARY, batch_size=64)
    print(f"[recompute] device={DEVICE} n_test={len(test_eps)} probe_seed=model_seed", flush=True)

    for arch in ARCHES:
        for seed in SEEDS:
            if (arch, seed) in done:
                continue
            try:
                model, meta, _ = load_model(arch, seed)
            except Exception as e:
                print(f"[skip] {arch} s{seed}: {e}", flush=True)
                continue
            recs = collect(model, loader, use_mask(meta), arch, seed)
            rows.extend(recs)
            done.add((arch, seed))
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            enc = [r for r in recs if r["stage_index"] == r["encoder_stage_index"]][0]
            fin = [r for r in recs if r["stage_index"] == r["final_stage_index"]][0]
            print(
                f"[OK] {arch} s{seed} Penc={enc['Agent Decoding Accuracy']:.4f} "
                f"Pfin={fin['Agent Decoding Accuracy']:.4f}",
                flush=True,
            )
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    raw.to_csv(raw_path, index=False)
    return raw


def build_summaries(raw: pd.DataFrame):
    # stage summary
    sum_rows = []
    for (arch, si), g in raw.groupby(["architecture", "stage_index"]):
        st = summarise(g["Agent Decoding Accuracy"].values)
        sum_rows.append(
            {
                "architecture": arch,
                "architecture_display": DISPLAY.get(arch, arch),
                "stage_index": int(si),
                "stage_label": g.stage_label.iloc[0],
                "stage_display": STAGE_DISPLAY.get(str(g.stage_label.iloc[0]), str(g.stage_label.iloc[0])),
                "representation_name": g.representation_name.iloc[0],
                "mean": st["mean"],
                "SD": st["SD"],
                "SEM": st["SEM"],
                "N": st["N"],
                "CI95_low": st["CI95_low"],
                "CI95_high": st["CI95_high"],
            }
        )
    summary = pd.DataFrame(sum_rows)

    # encoder vs final paired
    pair_rows = []
    for (arch, seed), g in raw.groupby(["architecture", "seed"]):
        enc = g[g.stage_index == g.encoder_stage_index.iloc[0]].iloc[0]
        fin = g[g.stage_index == g.final_stage_index.iloc[0]].iloc[0]
        pair_rows.append(
            {
                "architecture": arch,
                "architecture_display": DISPLAY.get(arch, arch),
                "seed": int(seed),
                "P_encoder": float(enc["Agent Decoding Accuracy"]),
                "P_final": float(fin["Agent Decoding Accuracy"]),
                "Final_readout_change": float(fin["Agent Decoding Accuracy"] - enc["Agent Decoding Accuracy"]),
                "structural_zero_flag": bool(enc.structural_zero_flag or fin.structural_zero_flag),
                "encoder_label": enc.stage_label,
                "final_label": fin.stage_label,
            }
        )
    pairs = pd.DataFrame(pair_rows)

    stats_rows = []
    for arch, g in pairs.groupby("architecture"):
        a = g.P_encoder.values
        b = g.P_final.values
        d = b - a
        st = summarise(d)
        if np.isfinite(d).sum() >= 2:
            tt = stats.ttest_rel(b, a)
            tp = float(tt.pvalue)
            tstat = float(tt.statistic)
        else:
            tp = tstat = np.nan
        stats_rows.append(
            {
                "architecture": arch,
                "architecture_display": DISPLAY.get(arch, arch),
                "n": int(len(g)),
                "P_encoder_mean": float(np.mean(a)),
                "P_encoder_SEM": float(np.std(a, ddof=1) / np.sqrt(len(a))),
                "P_final_mean": float(np.mean(b)),
                "P_final_SEM": float(np.std(b, ddof=1) / np.sqrt(len(b))),
                "change_mean": st["mean"],
                "change_SEM": st["SEM"],
                "change_SD": st["SD"],
                "paired_t": tstat,
                "paired_p_two_sided": tp,
                "stars": stars(tp),
                "structural_zero_flag": bool(g.structural_zero_flag.any()),
            }
        )
    st_df = pd.DataFrame(stats_rows)
    st_df["paired_p_Holm"] = holm(st_df["paired_p_two_sided"].values)
    st_df["stars_Holm"] = st_df["paired_p_Holm"].map(stars)
    return summary, pairs, st_df


def merge_prediction(pairs: pd.DataFrame):
    """Add stagewise prediction + Fig1 plane prediction/agent."""
    pred_raw = pd.read_csv(PRED / "prediction_profile_aligned_raw.csv")
    rows = []
    for (arch, seed), g in pred_raw.groupby(["architecture", "seed"]):
        enc = g[g.stage_label.isin(["encoded_history", "pooled_history"])]
        if enc.empty and "stage_display" in g.columns:
            enc = g[g.stage_display.eq("History encoding")]
        fin = g[g["is_native_final_profile"] == True]  # noqa: E712
        if fin.empty:
            fin = g[g.stage_label.eq("final_z_profile")]
        if fin.empty:
            fin = g[g.stage_label.isin(["attention_readout", "integrated_readout", "final_z_profile"])]
            if not fin.empty:
                fin = fin.sort_values("stage_index").tail(1)
        if enc.empty or fin.empty:
            continue
        rows.append(
            {
                "architecture": arch,
                "seed": int(seed),
                "pred_history": float(enc.iloc[0]["Behaviour Prediction Accuracy"]),
                "pred_final": float(fin.iloc[0]["Behaviour Prediction Accuracy"]),
                "pred_source": "stagewise_prediction_all_models",
            }
        )
    pred_hf = pd.DataFrame(rows)

    fig1 = pd.read_csv(FIG1 / "Fig1B_raw.csv").rename(
        columns={
            "architecture_internal": "architecture",
            "operational_acc_L80": "plane_pred",
            "probe_L80": "plane_agent",
        }
    )

    out = pairs.merge(pred_hf, on=["architecture", "seed"], how="left")
    out = out.merge(fig1[["architecture", "seed", "plane_pred", "plane_agent"]], on=["architecture", "seed"], how="left")
    # Arches without stagewise prediction (gru/cnn/transformer/mlp_pool/tinyrnn): use plane pred as final
    miss = out["pred_final"].isna()
    out.loc[miss, "pred_final"] = out.loc[miss, "plane_pred"]
    out.loc[miss, "pred_source"] = "plane_Fig1B_operational_acc_L80"
    out.loc[out["pred_source"].isna() & out["pred_final"].notna(), "pred_source"] = "stagewise_prediction_all_models"
    out["abs_final_minus_plane_agent"] = (out["P_final"] - out["plane_agent"]).abs()
    out["pred_final_minus_history"] = out["pred_final"] - out["pred_history"]
    return out, pred_raw


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


def _paired_means(merged: pd.DataFrame, hist_col: str, fin_col: str):
    arches = [a for a in ARCH_ORDER if a in set(merged.architecture)]
    hist_m, hist_e, fin_m, fin_e = [], [], [], []
    for arch in arches:
        g = merged[merged.architecture.eq(arch)]
        h = g[hist_col].dropna()
        f = g[fin_col].dropna()
        hist_m.append(float(h.mean()) if len(h) else np.nan)
        hist_e.append(float(h.std(ddof=1) / np.sqrt(len(h))) if len(h) > 1 else 0.0)
        fin_m.append(float(f.mean()) if len(f) else np.nan)
        fin_e.append(float(f.std(ddof=1) / np.sqrt(len(f))) if len(f) > 1 else 0.0)
    return arches, hist_m, hist_e, fin_m, fin_e


def plot_figure(summary: pd.DataFrame, pairs: pd.DataFrame, stats_df: pd.DataFrame, merged: pd.DataFrame):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 140, "savefig.dpi": 220, "pdf.fonttype": 42})

    fig = plt.figure(figsize=(14.5, 8.4))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.05, 1.05], hspace=0.38, wspace=0.32)

    # A: schematic text
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.set_title("A. Representation stages", loc="left", fontweight="bold")
    ax0.text(
        0.02,
        0.95,
        "Trial → (Pre-attn) → History → Final\n\n"
        "Same within-task linear agent probe.\n"
        "probe_seed = model seed (= Fig1 plane).\n\n"
        "D: agent decoding (history vs final)\n"
        "E: action prediction (history vs final)\n"
        "  — prediction Δ is small (~0.005–0.014).",
        va="top",
        fontsize=8,
        transform=ax0.transAxes,
    )

    # B: stagewise agent trajectories
    axb = fig.add_subplot(gs[0, 1:])
    for arch in ARCH_ORDER:
        g = summary[summary.architecture.eq(arch)].copy()
        if g.empty:
            continue
        g = g.sort_values("stage_index")
        xs, ys, yerr = [], [], []
        for _, r in g.iterrows():
            disp = r.stage_display
            if disp not in STAGE_X_ORDER:
                continue
            xs.append(STAGE_X_ORDER.index(disp))
            ys.append(r["mean"])
            yerr.append(r["SEM"])
        if not xs:
            continue
        axb.errorbar(
            xs,
            ys,
            yerr=yerr,
            color=COL.get(arch, "#666"),
            marker="o",
            ms=3.5,
            lw=1.2,
            alpha=0.9,
            label=DISPLAY.get(arch, arch),
        )
    axb.set_xticks(range(len(STAGE_X_ORDER)))
    axb.set_xticklabels(STAGE_X_ORDER, rotation=15, ha="right", fontsize=7)
    axb.set_ylabel("Agent Decoding Accuracy")
    axb.set_title("B. Stage-wise agent decoding (plane-unified)", loc="left", fontweight="bold")
    axb.axhline(1 / 6, color="#999", ls="--", lw=0.8)
    axb.set_ylim(0.05, 0.55)
    axb.legend(fontsize=5.5, ncol=3, frameon=False, loc="upper left")

    # C: agent delta bars
    axc = fig.add_subplot(gs[1, 0])
    st = stats_df.set_index("architecture").reindex([a for a in ARCH_ORDER if a in set(stats_df.architecture)])
    y = np.arange(len(st))
    axc.barh(
        y,
        st.change_mean.values,
        xerr=st.change_SEM.values,
        color=[COL.get(a, "#666") for a in st.index],
        alpha=0.9,
        height=0.7,
    )
    axc.axvline(0, color="#333", lw=0.8)
    axc.set_yticks(y)
    axc.set_yticklabels([DISPLAY.get(a, a) for a in st.index], fontsize=7)
    axc.invert_yaxis()
    axc.set_xlabel(r"$P_{final} - P_{history}$")
    axc.set_title("C. Agent Δ (final − history)", loc="left", fontweight="bold")
    for yi, a in enumerate(st.index):
        axc.text(
            st.loc[a, "change_mean"] + (0.002 if st.loc[a, "change_mean"] >= 0 else -0.002),
            yi,
            st.loc[a, "stars_Holm"],
            va="center",
            ha="left" if st.loc[a, "change_mean"] >= 0 else "right",
            fontsize=7,
            color="#333",
        )

    width = 0.35
    # D: agent decoding history vs final (same style as previous D)
    axd = fig.add_subplot(gs[1, 1])
    arches_d, ah, ae, af, afe = _paired_means(merged, "P_encoder", "P_final")
    x = np.arange(len(arches_d))
    axd.bar(x - width / 2, ah, width, yerr=ae, color="#7A8F9A", alpha=0.9, label="History", error_kw={"lw": 0.8})
    axd.bar(x + width / 2, af, width, yerr=afe, color="#2F4A6A", alpha=0.9, label="Final", error_kw={"lw": 0.8})
    axd.axhline(1 / 6, color="#999", ls="--", lw=0.8)
    axd.set_xticks(x)
    axd.set_xticklabels([DISPLAY.get(a, a) for a in arches_d], rotation=35, ha="right", fontsize=6)
    axd.set_ylabel("Agent Decoding Accuracy")
    axd.set_title("D. Agent decoding (history vs final)", loc="left", fontweight="bold")
    axd.set_ylim(0.05, 0.55)
    axd.legend(fontsize=6.5, frameon=False, loc="upper right")
    # annotate Holm stars from stats_df for visible agent drops
    star_map = stats_df.set_index("architecture")["stars_Holm"].to_dict()
    for xi, arch in enumerate(arches_d):
        s = star_map.get(arch, "")
        if s and s != "ns":
            axd.text(xi, max(ah[xi], af[xi]) + 0.02, s, ha="center", va="bottom", fontsize=6.5, color="#333")

    # E: action prediction — zoomed y so small Δ is visible
    axe = fig.add_subplot(gs[1, 2])
    arches_e, ph, pe, pf, pfe = _paired_means(merged, "pred_history", "pred_final")
    x = np.arange(len(arches_e))
    axe.bar(x - width / 2, ph, width, yerr=pe, color="#A8B8A0", alpha=0.95, label="History", error_kw={"lw": 0.8})
    axe.bar(x + width / 2, pf, width, yerr=pfe, color="#3D5A40", alpha=0.95, label="Final / plane", error_kw={"lw": 0.8})
    axe.set_xticks(x)
    axe.set_xticklabels([DISPLAY.get(a, a) for a in arches_e], rotation=35, ha="right", fontsize=6)
    axe.set_ylabel("Behaviour Prediction Accuracy")
    axe.set_title("E. Action prediction (zoomed; Δ small)", loc="left", fontweight="bold")
    # zoom around data
    vals = [v for v in ph + pf if np.isfinite(v)]
    if vals:
        lo, hi = min(vals) - 0.01, max(vals) + 0.015
        axe.set_ylim(lo, hi)
    axe.legend(fontsize=6.5, frameon=False, loc="upper left")
    for xi, arch in enumerate(arches_e):
        if np.isfinite(ph[xi]) and np.isfinite(pf[xi]):
            d = pf[xi] - ph[xi]
            axe.text(xi, max(ph[xi], pf[xi]) + 0.002, f"{d:+.3f}", ha="center", va="bottom", fontsize=5.5, color="#444")

    fig.suptitle(
        "Stage-wise profile formation — plane-unified agent probe (seed = model seed)",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIG / f"Figure3_stagewise_plane_unified.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] figures -> {FIG}")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)

    # backup note of old policy
    (DATA / "UNIFICATION_NOTE.txt").write_text(
        "Agent decoding recomputed with probe_seed = model seed to match Fig1 plane.\n"
        "Original extension used probe_seed = model seed + 1.\n"
        f"Built: {datetime.now(timezone.utc).isoformat()}\n"
        f"Device: {DEVICE}\n",
        encoding="utf-8",
    )

    raw = recompute_agent()
    summary, pairs, stats_df = build_summaries(raw)
    merged, pred_raw = merge_prediction(pairs)

    # prediction summaries
    pred_sum_rows = []
    for arch, g in merged.groupby("architecture"):
        for col, name in [
            ("pred_history", "action_pred_history"),
            ("pred_final", "action_pred_final"),
            ("plane_pred", "plane_pred"),
            ("P_encoder", "agent_history_unified"),
            ("P_final", "agent_final_unified"),
            ("plane_agent", "plane_agent"),
        ]:
            st = summarise(g[col].values)
            pred_sum_rows.append(
                {
                    "architecture": arch,
                    "architecture_display": DISPLAY.get(arch, arch),
                    "metric": name,
                    "mean": st["mean"],
                    "SD": st["SD"],
                    "SEM": st["SEM"],
                    "N": st["N"],
                    "pred_source": g["pred_source"].dropna().iloc[0] if g["pred_source"].notna().any() else "",
                }
            )
    metric_sum = pd.DataFrame(pred_sum_rows)

    # paired prediction change where history exists
    pred_stats_rows = []
    for arch, g in merged.groupby("architecture"):
        h = g["pred_history"].values
        f = g["pred_final"].values
        ok = np.isfinite(h) & np.isfinite(f)
        if ok.sum() < 2:
            pred_stats_rows.append(
                {
                    "architecture": arch,
                    "architecture_display": DISPLAY.get(arch, arch),
                    "n": int(ok.sum()),
                    "pred_history_mean": float(np.nanmean(h)),
                    "pred_final_mean": float(np.nanmean(f)),
                    "pred_change_mean": np.nan,
                    "pred_change_SEM": np.nan,
                    "paired_p": np.nan,
                    "stars": "",
                    "note": "no stagewise history prediction; final=plane",
                }
            )
            continue
        d = f[ok] - h[ok]
        tt = stats.ttest_rel(f[ok], h[ok])
        pred_stats_rows.append(
            {
                "architecture": arch,
                "architecture_display": DISPLAY.get(arch, arch),
                "n": int(ok.sum()),
                "pred_history_mean": float(np.mean(h[ok])),
                "pred_final_mean": float(np.mean(f[ok])),
                "pred_change_mean": float(np.mean(d)),
                "pred_change_SEM": float(np.std(d, ddof=1) / np.sqrt(len(d))),
                "paired_p": float(tt.pvalue),
                "stars": stars(float(tt.pvalue)),
                "note": "stagewise Behaviour Prediction Accuracy",
            }
        )
    pred_stats = pd.DataFrame(pred_stats_rows)
    if pred_stats["paired_p"].notna().any():
        pred_stats["paired_p_Holm"] = holm(pred_stats["paired_p"].values)
        pred_stats["stars_Holm"] = pred_stats["paired_p_Holm"].map(stars)
    else:
        pred_stats["paired_p_Holm"] = np.nan
        pred_stats["stars_Holm"] = ""

    # QC vs plane
    qc = pd.DataFrame(
        [
            ("n_agent_rows", len(raw), True),
            ("n_arch", raw.architecture.nunique(), True),
            (
                "mean_abs_Pfinal_minus_plane_agent",
                float(merged["abs_final_minus_plane_agent"].mean()),
                float(merged["abs_final_minus_plane_agent"].mean()) < 1e-9,
            ),
            (
                "max_abs_Pfinal_minus_plane_agent",
                float(merged["abs_final_minus_plane_agent"].max()),
                float(merged["abs_final_minus_plane_agent"].max()) < 1e-9,
            ),
            ("lstm_attn_P_final", float(merged[merged.architecture.eq("lstm_attn")].P_final.mean()), True),
            ("lstm_attn_plane_agent", float(merged[merged.architecture.eq("lstm_attn")].plane_agent.mean()), True),
            ("probe_policy", PROBE_NOTE, True),
        ],
        columns=["check", "value", "pass"],
    )
    qc.to_csv(QC / "UNIFY_QC.csv", index=False)

    methods = pd.DataFrame(
        [
            ("agent_probe", "within-task linear agent decoding"),
            ("probe_seed", "model seed (unified with Fig1 plane; NOT seed+1)"),
            ("window", "operational prefix L=80, K=20; n_test episodes=120"),
            ("prediction_stagewise", "Behaviour Prediction Accuracy (= action prediction) from stagewise_prediction_all_models"),
            ("prediction_plane", "Fig1B operational_acc_L80 (used as pred_final when stagewise unavailable)"),
            ("agent_plane", "Fig1B probe_L80 (= unified P_final)"),
            ("stats_H", "paired t-test P_final vs P_encoder; Holm across architectures"),
            ("stats_pred", "paired t-test pred_final vs pred_history where stagewise available"),
        ],
        columns=["item", "description"],
    )

    # order helpers
    def sort_a(df, col="architecture"):
        out = df.copy()
        out["_o"] = out[col].map(lambda a: ARCH_ORDER.index(a) if a in ARCH_ORDER else 999)
        return out.sort_values("_o").drop(columns="_o")

    xlsx = DATA / "Figure3_stagewise_plane_unified_DATA.xlsx"
    write_wb(
        xlsx,
        {
            "00_README": methods,
            "agent_stage_summary": sort_a(summary),
            "agent_stage_raw": sort_a(raw),
            "H_history_vs_final_raw": sort_a(merged),
            "H_history_vs_final_stats": sort_a(stats_df),
            "pred_history_vs_final_stats": sort_a(pred_stats),
            "pred_stage_raw": sort_a(pred_raw),
            "metric_means_sem_n": sort_a(metric_sum),
            "ZZ_QC": qc,
        },
    )

    # also copy into confirmed redraw package update for H
    redraw = V3 / "supplementary_profile_module" / "confirmed" / "plot_data" / "Figure_DEFGH_redraw_DATA.xlsx"
    if redraw.exists():
        from openpyxl import load_workbook

        wb = load_workbook(redraw)
        for name, df in [
            ("H_stage_scatter_summary", sort_a(stats_df.rename(columns={
                "P_encoder_mean": "P_encoder_mean",
                "P_encoder_SEM": "P_encoder_SEM",
                "P_final_mean": "P_final_mean",
                "P_final_SEM": "P_final_SEM",
                "change_mean": "change_mean",
                "change_SEM": "change_SEM",
            }))),
            ("H_stage_raw", sort_a(merged)),
            ("H_stage_stats_paired", sort_a(stats_df)),
            ("H_UNIFIED_NOTE", pd.DataFrame(
                [
                    ("note", "H sheets updated to plane-unified probe_seed=model seed"),
                    ("prediction", "H_stage_raw: pred_history/pred_final/plane_pred (= action prediction accuracy)"),
                    ("pred_fill", "gru/cnn/transformer/mlp_pool/tinyrnn: pred_final=plane_pred (no stagewise history)"),
                ],
                columns=["item", "description"],
            )),
            ("H_pred_stats", sort_a(pred_stats)),
        ]:
            if name in wb.sheetnames:
                del wb[name]
            ws = wb.create_sheet(name)
            for r in dataframe_to_rows(df, index=False, header=True):
                ws.append(list(r))
        wb.save(redraw)
        print(f"[OK] updated redraw H sheets: {redraw}")

    plot_figure(summary, pairs, stats_df, merged)

    print("\n=== KEY MEANS ===")
    show = sort_a(stats_df)[
        ["architecture_display", "P_encoder_mean", "P_final_mean", "change_mean", "stars_Holm"]
    ]
    print(show.round(4).to_string(index=False))
    print("QC mean |P_final - plane|:", merged.abs_final_minus_plane_agent.mean())
    print(f"Excel: {xlsx}")
    print(f"Figure: {FIG / 'Figure3_stagewise_plane_unified.png'}")


if __name__ == "__main__":
    main()
