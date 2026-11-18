"""Multi-stage representation extraction (prespecified; architecture-general)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from profile_benchmark.engine.forward import _prefix_x
from profile_benchmark.engine.prefix_features import prefix_history_input
from profile_benchmark.engine.prefix_packing import (
    key_padding_mask,
    last_hidden,
    masked_dot_attention,
    masked_mean,
    run_packed_rnn,
)

from lib.stage_definitions import STAGE_SPEC


def _allclose(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.allclose(a, b, atol=1e-5, rtol=1e-4))


@torch.no_grad()
def extract_formation_stages(
    model,
    prefix_stim,
    prefix_action,
    prefix_reward,
    future_stim,
    prefix_mask=None,
) -> Dict[str, object]:
    """
    Return dict:
      stages: {stage_index: tensor [B,D]}
      meta: representation names, dims, same flags, model_class
    """
    name = model.__class__.__name__
    stages: Dict[int, torch.Tensor] = {}
    rep_names: Dict[int, str] = {}

    # ---- Post-attention LSTM / BiLSTM ----
    if hasattr(model, "query_proj") and hasattr(model, "lstm"):
        x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
        if prefix_mask is None:
            h_seq, _ = model.lstm(x)
        else:
            h_seq, _ = run_packed_rnn(model.lstm, x, prefix_mask)
        queries = model.query_proj(future_stim)
        _, context = masked_dot_attention(queries, h_seq, prefix_mask)
        z = context.mean(dim=1)
        stages[0] = masked_mean(x, prefix_mask)
        stages[1] = masked_mean(h_seq, prefix_mask)
        stages[2] = z
        stages[3] = z
        rep_names = {0: "masked_mean_trial_embed", 1: "masked_mean_h_seq", 2: "attention_context_mean", 3: "native_z_profile"}

    # ---- Pre-attention ----
    elif hasattr(model, "pre_attn") and hasattr(model, "lstm"):
        x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
        pad_k = key_padding_mask(prefix_mask)
        x_attn, _ = model.pre_attn(
            x, x, x, need_weights=True, average_attn_weights=True, key_padding_mask=pad_k
        )
        if prefix_mask is None:
            h_seq, (h_last, _) = model.lstm(x_attn)
            if getattr(model.lstm, "bidirectional", False):
                h = torch.cat([h_last[-2], h_last[-1]], dim=-1)
            else:
                h = h_last[-1]
        else:
            h_seq, hidden = run_packed_rnn(model.lstm, x_attn, prefix_mask)
            h = last_hidden(hidden, bidirectional=bool(getattr(model.lstm, "bidirectional", False)))
        stages[0] = masked_mean(x, prefix_mask)
        stages[1] = masked_mean(x_attn, prefix_mask)
        stages[2] = masked_mean(h_seq, prefix_mask)
        stages[3] = h
        rep_names = {
            0: "masked_mean_trial_embed",
            1: "masked_mean_preattn_seq",
            2: "masked_mean_h_seq",
            3: "native_z_profile",
        }

    # ---- Uni-LSTM ----
    elif (
        hasattr(model, "lstm")
        and not getattr(model.lstm, "bidirectional", False)
        and not hasattr(model, "query_proj")
        and not hasattr(model, "pre_attn")
    ):
        x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
        if prefix_mask is None:
            h_seq, (h_last, _) = model.lstm(x)
            h = h_last[-1]
        else:
            h_seq, hidden = run_packed_rnn(model.lstm, x, prefix_mask)
            h = last_hidden(hidden, bidirectional=False)
        stages[0] = masked_mean(x, prefix_mask)
        stages[1] = masked_mean(h_seq, prefix_mask)
        stages[2] = h
        stages[3] = h
        rep_names = {0: "masked_mean_trial_embed", 1: "masked_mean_h_seq", 2: "native_final_hidden", 3: "native_z_profile"}

    # ---- BiLSTM (no post-attn) ----
    elif hasattr(model, "lstm") and getattr(model.lstm, "bidirectional", False) and not hasattr(model, "query_proj"):
        x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
        if prefix_mask is None:
            h_seq, (h_last, _) = model.lstm(x)
            h = torch.cat([h_last[-2], h_last[-1]], dim=-1)
        else:
            h_seq, hidden = run_packed_rnn(model.lstm, x, prefix_mask)
            h = last_hidden(hidden, bidirectional=True)
        stages[0] = masked_mean(x, prefix_mask)
        stages[1] = masked_mean(h_seq, prefix_mask)
        stages[2] = h
        stages[3] = h
        rep_names = {
            0: "masked_mean_trial_embed",
            1: "masked_mean_h_seq",
            2: "native_bidirectional_final",
            3: "native_z_profile",
        }

    # ---- GRU / RNN ----
    elif hasattr(model, "rnn"):
        x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
        if prefix_mask is None:
            if getattr(model, "rnn_type", "gru") == "lstm":
                h_seq, (h_last, _) = model.rnn(x)
                h = h_last[-1]
            else:
                h_seq, h_last = model.rnn(x)
                h = h_last[-1]
        else:
            h_seq, hidden = run_packed_rnn(model.rnn, x, prefix_mask)
            h = last_hidden(hidden, bidirectional=bool(getattr(model.rnn, "bidirectional", False)))
        stages[0] = masked_mean(x, prefix_mask)
        stages[1] = masked_mean(h_seq, prefix_mask)
        stages[2] = h
        stages[3] = h
        rep_names = {0: "masked_mean_trial_embed", 1: "masked_mean_h_seq", 2: "native_final_hidden", 3: "native_z_profile"}

    # ---- Paper TinyRNN ----
    elif name == "PaperTinyRNNProfile" or getattr(model, "paper_tiny_rnn", False):
        # raw concat trial features (no shared embed layer in encode_prefix path)
        n_actions = int(getattr(model, "n_actions", 4))
        x_raw = prefix_history_input(
            prefix_stim,
            prefix_action,
            prefix_reward,
            n_actions,
            bool(getattr(model, "one_hot_action", False)),
        )
        h, h_seq = model.encode_prefix(prefix_stim, prefix_action, prefix_reward, prefix_mask=prefix_mask)
        stages[0] = masked_mean(x_raw, prefix_mask)
        stages[1] = masked_mean(h_seq, prefix_mask)
        stages[2] = h
        stages[3] = h
        rep_names = {
            0: "masked_mean_trial_raw_concat",
            1: "masked_mean_h_seq",
            2: "native_final_hidden",
            3: "native_z_profile",
        }

    # ---- Transformer ----
    elif hasattr(model, "encoder"):
        B, L, _ = prefix_stim.shape
        n_actions = int(getattr(model, "n_actions", 4))
        x = prefix_history_input(
            prefix_stim,
            prefix_action,
            prefix_reward,
            n_actions,
            bool(getattr(model, "one_hot_action", False)),
        )
        if hasattr(model, "embed"):
            x = model.embed(x)
        x0 = x
        if getattr(model, "pos_embedding", None) is not None:
            x = x + model.pos_embedding[:L].unsqueeze(0)
        pad_k = key_padding_mask(prefix_mask)
        h_seq = model.encoder(x, src_key_padding_mask=pad_k)
        if hasattr(model, "_pool_prefix"):
            h = model._pool_prefix(h_seq, prefix_mask)
        elif prefix_mask is not None:
            h = masked_mean(h_seq, prefix_mask)
        else:
            h = h_seq.mean(dim=1)
        stages[0] = masked_mean(x0, prefix_mask)
        stages[1] = masked_mean(h_seq, prefix_mask)
        stages[2] = h
        stages[3] = h
        rep_names = {
            0: "masked_mean_trial_embed",
            1: "masked_mean_encoder_h_seq",
            2: "native_pooled_encoder",
            3: "native_z_profile",
        }

    # ---- MLP-pool ----
    elif hasattr(model, "mlp") and hasattr(model, "embed") and name == "MLPPoolingProfile":
        x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
        pooled = masked_mean(x, prefix_mask)
        stages[0] = pooled
        stages[1] = pooled
        stages[2] = pooled
        rep_names = {
            0: "masked_mean_trial_embed",
            1: "masked_mean_trial_embed",
            2: "native_z_profile",
        }

    # ---- CNN (corrected) ----
    elif name == "CNNProfile" or hasattr(model, "convs"):
        x = _prefix_x(model, prefix_stim, prefix_action, prefix_reward)
        h_seq = x.transpose(1, 2)
        for conv in model.convs:
            h_seq = conv(h_seq)
            h_seq = torch.relu(h_seq)
            h_seq = model.dropout(h_seq)
        h_seq = h_seq.transpose(1, 2)
        pooled = masked_mean(h_seq, prefix_mask)
        stages[0] = masked_mean(x, prefix_mask)
        stages[1] = pooled
        stages[2] = pooled
        stages[3] = pooled
        rep_names = {
            0: "masked_mean_trial_embed",
            1: "masked_mean_conv_h_seq",
            2: "masked_mean_conv_pool",
            3: "native_z_profile",
        }

    else:
        raise ValueError(f"Unsupported model class for formation stages: {name}")

    # dims + same_representation flags vs previous unique tensors
    dims = {i: int(t.shape[-1]) for i, t in stages.items()}
    same_flags = {}
    for i, t in stages.items():
        same_flags[i] = False
    # Mark duplicates relative to earliest identical stage
    keys = sorted(stages.keys())
    for i in keys:
        for j in keys:
            if j >= i:
                break
            if stages[i].shape == stages[j].shape and _allclose(stages[i], stages[j]):
                same_flags[i] = True
                break

    # Main-Fig3-compatible encoder reference:
    # masked-mean of the architecture's encoded history sequence (h_seq / conv / encoder),
    # NOT the trial-input embed and NOT the pre-attention input transform alone.
    encoder_idx = None
    for i, rn in rep_names.items():
        if rn in (
            "masked_mean_h_seq",
            "masked_mean_conv_h_seq",
            "masked_mean_encoder_h_seq",
            "masked_mean_trial_embed",  # mlp_pool only encoded form
        ):
            # Prefer true encoded-history names over trial embed when both exist
            if rn != "masked_mean_trial_embed":
                encoder_idx = i
                break
            if encoder_idx is None:
                encoder_idx = i
    if encoder_idx is None:
        encoder_idx = 1 if 1 in stages else min(stages.keys())
    # For mlp_pool, encoder == final pooled embed
    if name == "MLPPoolingProfile":
        encoder_idx = 0

    final_idx = max(stages.keys())
    for i, rn in rep_names.items():
        if rn == "native_z_profile":
            final_idx = i
            break
    identical_enc_final = (
        _allclose(stages[encoder_idx], stages[final_idx])
        if stages[encoder_idx].shape == stages[final_idx].shape
        else False
    )

    return {
        "stages": stages,
        "rep_names": rep_names,
        "dims": dims,
        "same_flags": same_flags,
        "encoder_idx": encoder_idx,
        "final_idx": final_idx,
        "identical_enc_final": identical_enc_final,
        "model_class": name,
    }


def ordered_stage_items(arch: str) -> List[Tuple[int, str, str]]:
    """Return (stage_index, stage_label, representation_name) from STAGE_SPEC."""
    return [(a, b, c) for a, b, c, _, _ in STAGE_SPEC[arch]]
