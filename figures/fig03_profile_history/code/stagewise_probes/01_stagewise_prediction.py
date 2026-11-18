#!/usr/bin/env python3
"""Stage-wise behavioural prediction probes (full architecture set).

Reuses fig3_representation_formation_extension stage extraction + same
LogisticRegression probe family as agent decoding (native dims, no PCA).

Isolated workspace — does not modify trained models or agent-decoding outputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

WS = Path(__file__).resolve().parents[1]
FIG3_EXT = WS.parent / "fig3_representation_formation_extension"
SLIDING = WS.parents[2]
REPO = SLIDING.parents[2]
sys.path[:0] = [str(FIG3_EXT), str(WS), str(SLIDING), str(REPO)]

from config import ARCHES, CKPT_DIR, LABEL, L_PRIMARY, SEEDS  # noqa: E402
from lib.extract_stages import extract_formation_stages  # noqa: E402
from lib.io_util import (  # noqa: E402
    DEVICE,
    episode_grid,
    load_model,
    make_loader,
    summarise,
    use_mask,
)
from lib.stage_definitions import STAGE_SPEC  # noqa: E402

DATA = WS / "data"
OUT = WS / "outputs"
QC = WS / "qc"
for d in (DATA, OUT, QC):
    d.mkdir(parents=True, exist_ok=True)

FAMILY = {
    "lstm": "Plain recurrent",
    "bilstm": "Plain recurrent",
    "gru": "Plain recurrent",
    "tinyrnn_paper": "Plain recurrent",
    "pre_lstm": "Pre-attention recurrent",
    "pre_bilstm": "Pre-attention recurrent",
    "lstm_attn": "Post-attention recurrent",
    "post_bilstm": "Post-attention recurrent",
    "cnn": "CNN / Transformer / pooled",
    "transformer": "CNN / Transformer / pooled",
    "mlp_pool": "CNN / Transformer / pooled",
}

AGG_RULE = "masked_mean_valid_prefix (same as stage-wise Agent Decoding)"


def write_stage_definitions_prediction() -> pd.DataFrame:
    rows = []
    for arch in ARCHES:
        specs = STAGE_SPEC[arch]
        for stage_index, stage_label, rep_name, same_as, is_final in specs:
            # shapes documented conceptually; runtime dims filled in Raw sheet
            rows.append(
                {
                    "model": arch,
                    "architecture_family": FAMILY.get(arch, ""),
                    "architecture_display": LABEL.get(arch, arch),
                    "stage_index": stage_index,
                    "stage_display_name": {
                        "trial_input": "Trial representation",
                        "preattn_transformed": "Pre-attn transform",
                        "encoded_history": "History encoding",
                        "integrated_readout": "Readout",
                        "attention_readout": "Readout",
                        "pooled_readout": "Readout",
                        "pooled_history": "History encoding",
                        "final_z_profile": "Final profile",
                    }.get(stage_label, stage_label),
                    "stage_label": stage_label,
                    "source_tensor_code_path": rep_name,
                    "tensor_shape_before_aggregation": "sequence [B,L,D] if sequence-valued; else vector [B,D]",
                    "aggregation_rule": AGG_RULE
                    if "mean" in rep_name or "masked" in rep_name or stage_label
                    in (
                        "trial_input",
                        "encoded_history",
                        "preattn_transformed",
                        "pooled_readout",
                        "pooled_history",
                        "attention_readout",
                    )
                    else "native vector (last hidden / pooled profile)",
                    "tensor_shape_after_aggregation": "[B, D_stage]",
                    "identical_to_stage_index": same_as if same_as is not None else "",
                    "whether_identical_to_another_stage": same_as is not None,
                    "whether_native_final_profile": bool(is_final),
                    "notes": (
                        "Structural identity with stage "
                        f"{same_as}; do not interpret zero Δ as a measured effect."
                        if same_as is not None
                        else (
                            "Post-attention: attention_context_mean ≡ native z_profile "
                            "when same_as links stages 2 and 3."
                            if arch in ("lstm_attn", "post_bilstm") and stage_index in (2, 3)
                            else ""
                        )
                    ),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "stage_definitions_prediction.csv", index=False)
    return df


def action_prediction_probe(
    Z: np.ndarray,
    future_actions: np.ndarray,
    seed: int,
    test_size: float = 0.25,
) -> dict:
    """Same probe family as agent decoding: LogisticRegression, test_size=0.25.

    Expands each window across K future actions (tile z), matching operational
    next-action evaluation that scores all future steps.
    """
    Z = np.asarray(Z, dtype=np.float64)
    fa = np.asarray(future_actions)
    if fa.ndim != 2:
        raise ValueError(f"future_actions expected [N,K], got {fa.shape}")
    n, k = fa.shape
    if n < 8 or Z.shape[0] != n:
        return {
            "prediction_accuracy": np.nan,
            "prediction_nll": np.nan,
            "n_samples_expanded": 0,
            "n_windows": int(n),
            "n_future_steps": int(k) if fa.ndim == 2 else 0,
        }

    X = np.repeat(Z, k, axis=0)
    y_raw = fa.reshape(-1)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    if len(np.unique(y)) < 2 or len(y) < 8:
        return {
            "prediction_accuracy": np.nan,
            "prediction_nll": np.nan,
            "n_samples_expanded": int(len(y)),
            "n_windows": int(n),
            "n_future_steps": int(k),
        }

    stratify = y if min(np.bincount(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    # NLL on test
    try:
        proba = clf.predict_proba(X_te)
        nll = float(log_loss(y_te, proba, labels=list(range(len(le.classes_)))))
    except Exception:
        nll = float("nan")
    return {
        "prediction_accuracy": acc,
        "prediction_nll": nll,
        "n_samples_expanded": int(len(y)),
        "n_windows": int(n),
        "n_future_steps": int(k),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
    }


@torch.no_grad()
def collect(model, loader, use_pmask: bool, arch: str, seed: int) -> list[dict]:
    buckets: dict[int, list] = {}
    fa_list = []
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
        identical = bool(st["identical_enc_final"]) or arch in ("cnn", "mlp_pool")
        for i, Z in st["stages"].items():
            buckets.setdefault(i, []).append(Z.detach().cpu().numpy())
            meta_same[i] = bool(st["same_flags"][i])
            meta_dim[i] = int(st["dims"][i])
            meta_rep[i] = st["rep_names"][i]
        fa_list.append(fa)

    FA = np.concatenate(fa_list, 0)
    stage_metrics = {}
    canonical = {}
    sorted_idx = sorted(buckets.keys())
    for i in sorted_idx:
        Zi = np.concatenate(buckets[i], 0)
        reused = None
        for j in sorted_idx:
            if j >= i:
                break
            if meta_same.get(i) and buckets[j][0].shape[-1] == Zi.shape[-1]:
                Zj = np.concatenate(buckets[j], 0)
                if Zj.shape == Zi.shape and np.allclose(Zi[:8], Zj[:8], atol=1e-4):
                    reused = j
                    break
        if reused is not None:
            stage_metrics[i] = stage_metrics[reused]
            canonical[i] = reused
        else:
            stage_metrics[i] = action_prediction_probe(Zi, FA, seed=seed + 1)
            canonical[i] = i

    label_map = {a: b for a, b, c, _, _ in STAGE_SPEC[arch]}
    rows = []
    for i in sorted_idx:
        m = stage_metrics[i]
        rows.append(
            {
                "architecture": arch,
                "architecture_display": LABEL.get(arch, arch),
                "architecture_family": FAMILY.get(arch, ""),
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
                "is_native_final_profile": bool(i == fin_idx),
                "Behaviour Prediction Accuracy": m["prediction_accuracy"],
                "Behaviour Prediction NLL": m.get("prediction_nll", np.nan),
                "encoder_stage_index": enc_idx,
                "final_stage_index": fin_idx,
                "n_windows": m.get("n_windows", int(np.concatenate(buckets[i], 0).shape[0])),
                "n_samples_expanded": m.get("n_samples_expanded", np.nan),
                "n_future_steps": m.get("n_future_steps", np.nan),
                "probe_seed": seed + 1,
            }
        )
    return rows


def main():
    write_stage_definitions_prediction()
    raw_path = DATA / "raw_stagewise_prediction.parquet"
    sample = episode_grid()
    (OUT / "episode_grid_manifest.json").write_text(
        json.dumps({"n": len(sample), "L": L_PRIMARY}, indent=2), encoding="utf-8"
    )
    loader = make_loader(sample, L=L_PRIMARY, batch_size=64)
    print(f"[pred] device={DEVICE} episodes={len(sample)} L={L_PRIMARY}", flush=True)

    rows = []
    done = set()
    if raw_path.is_file():
        prev = pd.read_parquet(raw_path)
        rows = prev.to_dict("records")
        done = {(r["architecture"], int(r["seed"])) for r in rows}
        print(f"[resume] {len(done)} arch×seed done", flush=True)

    for arch in ARCHES:
        for seed in SEEDS:
            if (arch, seed) in done:
                continue
            ckpt = CKPT_DIR / f"{arch}_seed{seed}.pt"
            if not ckpt.is_file():
                print(f"[skip] {arch} s{seed}", flush=True)
                continue
            model, meta, _ = load_model(arch, seed)
            recs = collect(model, loader, use_mask(meta), arch, seed)
            rows.extend(recs)
            done.add((arch, seed))
            pd.DataFrame(rows).to_parquet(raw_path, index=False)
            fin = [r for r in recs if r["stage_index"] == r["final_stage_index"]][0]
            print(
                f"[OK] {arch} s{seed} stages={len(recs)} "
                f"A_final={fin['Behaviour Prediction Accuracy']:.3f}",
                flush=True,
            )
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    raw.to_parquet(raw_path, index=False)
    raw.to_csv(DATA / "raw_stagewise_prediction.csv", index=False)

    sum_rows = []
    for (arch, si), g in raw.groupby(["architecture", "stage_index"]):
        st = summarise(g["Behaviour Prediction Accuracy"].values)
        st_nll = summarise(g["Behaviour Prediction NLL"].values)
        sum_rows.append(
            {
                "architecture": arch,
                "architecture_display": LABEL.get(arch, arch),
                "architecture_family": FAMILY.get(arch, ""),
                "stage_index": int(si),
                "stage_label": g.stage_label.iloc[0],
                "representation_name": g.representation_name.iloc[0],
                "same_representation_flag": bool(g.same_representation_flag.iloc[0]),
                "is_native_final_profile": bool(g.is_native_final_profile.iloc[0]),
                "mean": st["mean"],
                "SD": st["SD"],
                "SEM": st["SEM"],
                "N": st["N"],
                "CI95_low": st["CI95_low"],
                "CI95_high": st["CI95_high"],
                "median": st["median"],
                "min": st["min"],
                "max": st["max"],
                "nll_mean": st_nll["mean"],
                "nll_SEM": st_nll["SEM"],
            }
        )
    summary = pd.DataFrame(sum_rows)
    summary.to_csv(DATA / "summary_stagewise_prediction.csv", index=False)
    print(f"[done] raw={len(raw)} summary={len(summary)} -> {DATA}", flush=True)


if __name__ == "__main__":
    main()
