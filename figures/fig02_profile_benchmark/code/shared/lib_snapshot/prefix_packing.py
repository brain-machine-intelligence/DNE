"""Packed-sequence and attention masking for variable-length prefixes.

Used only when ``prefix_mask`` is provided. Unmasked call sites keep the
legacy tensor path. ``prefix_mask`` is True on valid timesteps.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

RNNHidden = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def prefix_lengths(prefix_mask: torch.Tensor) -> torch.Tensor:
    return prefix_mask.to(dtype=torch.long).sum(dim=1).clamp(min=1)


def run_packed_rnn(
    rnn: nn.Module,
    x: torch.Tensor,
    prefix_mask: torch.Tensor,
) -> Tuple[torch.Tensor, RNNHidden]:
    """x: [B, L, H] → padded h_seq [B, L, H*] and native RNN hidden."""
    lengths = prefix_lengths(prefix_mask)
    packed = pack_padded_sequence(
        x, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    packed_out, hidden = rnn(packed)
    h_seq, _ = pad_packed_sequence(
        packed_out, batch_first=True, total_length=x.size(1)
    )
    return h_seq, hidden


def last_hidden(hidden: RNNHidden, bidirectional: bool) -> torch.Tensor:
    h_n = hidden[0] if isinstance(hidden, tuple) else hidden
    if bidirectional:
        return torch.cat([h_n[-2], h_n[-1]], dim=-1)
    return h_n[-1]


def masked_dot_attention(
    queries: torch.Tensor,
    h_seq: torch.Tensor,
    prefix_mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product; pad positions get -inf before softmax.

    queries: [B, K, H], h_seq: [B, L, H]
    """
    scale = h_seq.size(-1) ** 0.5
    attn_logits = torch.bmm(queries, h_seq.transpose(1, 2)) / scale
    if prefix_mask is not None:
        valid = prefix_mask.to(device=attn_logits.device, dtype=torch.bool)
        attn_logits = attn_logits.masked_fill(~valid.unsqueeze(1), torch.finfo(attn_logits.dtype).min)
    attn_weights = torch.softmax(attn_logits, dim=-1)
    context = torch.bmm(attn_weights, h_seq)
    return attn_weights, context


def key_padding_mask(prefix_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """PyTorch convention: True = ignore."""
    if prefix_mask is None:
        return None
    return ~prefix_mask.to(dtype=torch.bool)


def masked_mean(h_seq: torch.Tensor, prefix_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if prefix_mask is None:
        return h_seq.mean(dim=1)
    m = prefix_mask.to(device=h_seq.device, dtype=h_seq.dtype).unsqueeze(-1)
    return (h_seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
