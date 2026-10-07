"""Training-time guard against runaway attention scores.

A head whose attention becomes all-or-nothing can keep growing its query and
key weights without limit (attention logit growth): its scores explode, the
layer starts passing garbage forward and training collapses. AraSpellX's first
full pretraining run collapsed this way: one head in layer 6 reached scores of
2,750 by step 35k and 10 million by step 45k, while every other head stayed
below ~30.

`cap_attention` measures every head's largest attention score on a fixed probe
batch and scales the query of any head above the cap so that its largest score
equals the cap. Scores are linear in the query for absolute and relative_key
positions, so the cap is exact and what each head attends to is unchanged; the
saved model stays a standard BERT.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor


@torch.no_grad()
def max_attention_scores(model, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
    """Largest absolute attention score of every head on input_ids, as a (layers, heads) tensor."""
    layers = model.bert.encoder.layer
    inputs = {}
    hooks = [layer.attention.self.register_forward_hook(
        lambda module, args, output, i=i: inputs.__setitem__(i, args[0])) for i, layer in enumerate(layers)]
    training = model.training
    model.eval()
    try:
        with torch.autocast(input_ids.device.type, enabled=False):
            model.bert(input_ids, attention_mask)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(training)
    real = (torch.ones_like(input_ids) if attention_mask is None else attention_mask).bool()
    pairs = real[:, None, :, None] & real[:, None, None, :]  # (batch, 1, query, key): both real characters
    result = []
    for i, layer in enumerate(layers):
        attention = layer.attention.self
        hidden = inputs[i].float()
        query = attention._split_heads(attention.query(hidden))
        key = attention._split_heads(attention.key(hidden))
        scores = query @ key.transpose(-1, -2) / math.sqrt(attention.head_size)
        if attention.position_type != "absolute":
            scores = scores + attention._relative_scores(query, key)
        result.append(scores.abs().masked_fill(~pairs, 0).amax(dim=(0, 2, 3)))
    return torch.stack(result)


@torch.no_grad()
def cap_attention(model, input_ids: Tensor, cap: float,
                  attention_mask: Optional[Tensor] = None) -> List[Tuple[int, int, float]]:
    """Scale the query of every head whose largest score exceeds `cap` down to the cap.

    Returns (layer, head, largest score before) for each head that was scaled.
    """
    scores = max_attention_scores(model, input_ids, attention_mask)
    capped = []
    for i, layer in enumerate(model.bert.encoder.layer):
        attention = layer.attention.self
        for head in range(attention.heads):
            score = scores[i, head].item()
            if score > cap:
                rows = slice(head * attention.head_size, (head + 1) * attention.head_size)
                attention.query.weight[rows] *= cap / score
                attention.query.bias[rows] *= cap / score
                capped.append((i, head, score))
    return capped
