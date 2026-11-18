"""Prespecified representation-formation stages (LOCKED before probing).

Conceptual framework (architecture-general):
  Stage 0 — trial-level / input behavioural representation
  Stage 1 — encoded behavioural-history representation
  Stage 2 — integrated / readout-stage representation (if distinct)
  Stage 3 — native final z_profile

Aggregation for sequence-valued stages (prespecified, architecture-neutral):
  masked mean over valid history positions.

same_representation=TRUE means two conceptual stages share the exact same tensor;
a zero difference must NOT be interpreted as a mechanistic effect.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import ARCHES, DATA, LABEL

# stage_index, stage_label, representation_name, notes
# same_as: stage_index of the stage this equals (or None)
# is_native_final: True if this is the architecture's stock z_profile

STAGE_SPEC = {
    # Uni-LSTM / GRU / TinyRNN: embed → h_seq → final hidden (= z)
    "lstm": [
        (0, "trial_input", "masked_mean_trial_embed", None, False),
        (1, "encoded_history", "masked_mean_h_seq", None, False),
        (2, "integrated_readout", "native_final_hidden", None, True),
        (3, "final_z_profile", "native_z_profile", 2, True),  # same as 2
    ],
    "gru": [
        (0, "trial_input", "masked_mean_trial_embed", None, False),
        (1, "encoded_history", "masked_mean_h_seq", None, False),
        (2, "integrated_readout", "native_final_hidden", None, True),
        (3, "final_z_profile", "native_z_profile", 2, True),
    ],
    "tinyrnn_paper": [
        (0, "trial_input", "masked_mean_trial_raw_concat", None, False),
        (1, "encoded_history", "masked_mean_h_seq", None, False),
        (2, "integrated_readout", "native_final_hidden", None, True),
        (3, "final_z_profile", "native_z_profile", 2, True),
    ],
    "bilstm": [
        (0, "trial_input", "masked_mean_trial_embed", None, False),
        (1, "encoded_history", "masked_mean_h_seq", None, False),
        (2, "integrated_readout", "native_bidirectional_final", None, True),
        (3, "final_z_profile", "native_z_profile", 2, True),
    ],
    # Pre-attention: embed → pre-attn seq → h_seq → final hidden (= z)
    "pre_lstm": [
        (0, "trial_input", "masked_mean_trial_embed", None, False),
        (1, "preattn_transformed", "masked_mean_preattn_seq", None, False),
        (2, "encoded_history", "masked_mean_h_seq", None, False),
        (3, "final_z_profile", "native_z_profile", None, True),
    ],
    "pre_bilstm": [
        (0, "trial_input", "masked_mean_trial_embed", None, False),
        (1, "preattn_transformed", "masked_mean_preattn_seq", None, False),
        (2, "encoded_history", "masked_mean_h_seq", None, False),
        (3, "final_z_profile", "native_z_profile", None, True),
    ],
    # Post-attention: embed → h_seq → attention context mean (= z)
    "lstm_attn": [
        (0, "trial_input", "masked_mean_trial_embed", None, False),
        (1, "encoded_history", "masked_mean_h_seq", None, False),
        (2, "attention_readout", "attention_context_mean", None, True),
        (3, "final_z_profile", "native_z_profile", 2, True),
    ],
    "post_bilstm": [
        (0, "trial_input", "masked_mean_trial_embed", None, False),
        (1, "encoded_history", "masked_mean_h_seq", None, False),
        (2, "attention_readout", "attention_context_mean", None, True),
        (3, "final_z_profile", "native_z_profile", 2, True),
    ],
    # CNN: embed → conv h_seq → masked_mean pool (= z)  [corrected CNN]
    "cnn": [
        (0, "trial_input", "masked_mean_trial_embed", None, False),
        (1, "encoded_history", "masked_mean_conv_h_seq", None, True),
        (2, "pooled_readout", "masked_mean_conv_pool", 1, True),
        (3, "final_z_profile", "native_z_profile", 1, True),
    ],
    # Transformer: embed → encoder h_seq → native pool (= z when pool≡masked mean)
    "transformer": [
        (0, "trial_input", "masked_mean_trial_embed", None, False),
        (1, "encoded_history", "masked_mean_encoder_h_seq", None, False),
        (2, "pooled_readout", "native_pooled_encoder", None, True),
        (3, "final_z_profile", "native_z_profile", 2, True),
    ],
    # MLP-pool: embed sequence → masked pool (= z); no distinct deeper encoder
    "mlp_pool": [
        (0, "trial_input", "masked_mean_trial_embed", None, True),
        (1, "pooled_history", "masked_mean_trial_embed", 0, True),
        (2, "final_z_profile", "native_z_profile", 0, True),
        # only stages 0/1/2 defined; stage 3 omitted (would duplicate)
    ],
}


def build_stage_definitions_df() -> pd.DataFrame:
    rows = []
    for arch in ARCHES:
        specs = STAGE_SPEC[arch]
        by_idx = {s[0]: s for s in specs}
        for stage_index, stage_label, rep_name, same_as, is_final in specs:
            same = same_as is not None
            structural_zero_pair = False
            if same and same_as is not None:
                # structural zero for enrichment-style pairs when final == encoder-like
                pass
            rows.append(
                {
                    "architecture": arch,
                    "architecture_display": LABEL.get(arch, arch),
                    "stage_index": stage_index,
                    "stage_label": stage_label,
                    "representation_name": rep_name,
                    "same_representation": bool(same),
                    "same_as_stage_index": same_as if same else "",
                    "is_native_final": bool(is_final),
                    "sequence_aggregation": "masked_mean_valid_prefix"
                    if "masked_mean" in rep_name or rep_name.endswith("_seq")
                    else ("native_vector" if is_final else "architecture_native"),
                    "prespecified_before_results": True,
                    "notes": _note(arch, stage_index, stage_label, same_as),
                }
            )
        # Mark structural-zero architectures: encoder summary == final by construction
        if arch in ("cnn", "mlp_pool"):
            for r in rows:
                if r["architecture"] == arch and r["is_native_final"]:
                    r["structural_zero_vs_encoded_history"] = True
        elif arch == "transformer":
            for r in rows:
                if r["architecture"] == arch:
                    r["structural_zero_vs_encoded_history"] = "runtime_check"
    # fill missing column
    for r in rows:
        r.setdefault("structural_zero_vs_encoded_history", False)
    return pd.DataFrame(rows)


def _note(arch, stage_index, stage_label, same_as):
    if same_as is not None:
        return f"Identical tensor to stage {same_as}; zero delta is not a measured effect"
    if arch == "cnn" and stage_index >= 1:
        return "Corrected CNN: masked mean of conv features; no future_stim fallback"
    if arch == "mlp_pool":
        return "Masked mean over valid prefix trial embeddings"
    return ""


def write_stage_definitions(path: Path | None = None) -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    df = build_stage_definitions_df()
    path = path or (DATA / "stage_definitions.csv")
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df = write_stage_definitions()
    print(df.to_string(index=False))
    print(f"Wrote {DATA / 'stage_definitions.csv'}")
