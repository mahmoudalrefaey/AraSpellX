"""AraSpellX's own BERT encoder, weight-compatible with Hugging Face BERT.

Parameter names, shapes and computations follow Hugging Face's
`BertForTokenClassification` and `BertForMaskedLM` exactly (post-norm
layers, erf GELU, learned absolute or relative positions, a single segment type), so
checkpoints saved here load into those classes without custom code, and
their checkpoints load here (verified: outputs agree within 1e-5 for absolute,
relative_key and relative_key_query positions).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import BertConfig

PathLike = Union[str, Path]
RELATIVE = ("relative_key", "relative_key_query")


class Embeddings(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).unsqueeze(0), persistent=False)

        # With relative positions the absolute table is kept (Hugging Face keeps it too) but unused.
        self.absolute = getattr(config, "position_embedding_type", "absolute") == "absolute"

    def forward(self, input_ids: Tensor) -> Tensor:
        embeddings = self.word_embeddings(input_ids) + self.token_type_embeddings(torch.zeros_like(input_ids))
        if self.absolute:
            embeddings = embeddings + self.position_embeddings(self.position_ids[:, :input_ids.shape[1]])
        return self.dropout(self.LayerNorm(embeddings))


class SelfAttention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout_p = config.attention_probs_dropout_prob
        self.position_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_type in RELATIVE:
            self.max_positions = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.head_size)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.heads, self.head_size).transpose(1, 2)

    def _relative_scores(self, query: Tensor, key: Tensor) -> Tensor:
        """Scores from the distance between positions, scaled like the attention scores.

        relative_key: query . d(i - j); relative_key_query adds key . d(i - j),
        where d is a learned embedding of the distance (Hugging Face's formulation).
        """
        length = query.shape[2]
        # Embeddings of the distances -(length - 1) .. length - 1 (row k is distance k - length + 1).
        start = self.max_positions - length
        table = self.distance_embedding.weight[start:start + 2 * length - 1].to(query.dtype)
        # Score every vector against every distance once, then read distance i - j for each pair.
        scores = _diagonals(query @ table.flip(0).T)  # query_i . d(i - j)
        if self.position_type == "relative_key_query":
            scores = scores + _diagonals(key @ table.T).transpose(-1, -2)  # key_j . d(i - j)
        return scores / math.sqrt(self.head_size)

    def forward(self, hidden: Tensor, additive_mask: Tensor) -> Tensor:
        query, key, value = (self._split_heads(f(hidden)) for f in (self.query, self.key, self.value))
        if self.position_type in RELATIVE:
            additive_mask = additive_mask.to(query.dtype) + self._relative_scores(query, key)
        context = F.scaled_dot_product_attention(
            query, key, value, attn_mask=additive_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        batch, _, length, _ = context.shape
        return context.transpose(1, 2).reshape(batch, length, self.heads * self.head_size)


def _diagonals(scores: Tensor) -> Tensor:
    """(..., L, 2L - 1) -> (..., L, L) with out[i, j] = scores[i, L - 1 - i + j].

    Element (i, j) sits at flat offset (L - 1) + i * (2L - 2) + j of each
    (L, 2L - 1) matrix, so a slice and two reshapes pick all of them out
    (a view in PyTorch, and plain Slice/Reshape operators in ONNX).
    """
    *lead, length, width = scores.shape
    flat = scores.reshape(*lead, length * width)[..., length - 1:length - 1 + length * (width - 1)]
    return flat.reshape(*lead, length, width - 1)[..., :length]


class AddNorm(nn.Module):
    """Dense projection, dropout, then LayerNorm over the residual sum."""

    def __init__(self, in_size: int, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(in_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden: Tensor, residual: Tensor) -> Tensor:
        return self.LayerNorm(self.dropout(self.dense(hidden)) + residual)


class Attention(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.self = SelfAttention(config)
        self.output = AddNorm(config.hidden_size, config)

    def forward(self, hidden: Tensor, additive_mask: Tensor) -> Tensor:
        return self.output(self.self(hidden, additive_mask), hidden)


class Intermediate(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def forward(self, hidden: Tensor) -> Tensor:
        return F.gelu(self.dense(hidden))


class Layer(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = AddNorm(config.intermediate_size, config)

    def forward(self, hidden: Tensor, additive_mask: Tensor) -> Tensor:
        attended = self.attention(hidden, additive_mask)
        return self.output(self.intermediate(attended), attended)


class Encoder(nn.Module):
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden: Tensor, additive_mask: Tensor) -> Tensor:
        for layer in self.layer:
            hidden = layer(hidden, additive_mask)
        return hidden


class Bert(nn.Module):
    """The encoder shared by both heads (Hugging Face `BertModel` without pooler)."""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        hidden = self.embeddings(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        # 0 where attention is allowed, a large negative number on padding keys.
        additive = (1.0 - attention_mask[:, None, None, :].to(hidden.dtype)) * torch.finfo(hidden.dtype).min
        return self.encoder(hidden, additive)


def init_weights(module: nn.Module, std: float) -> None:
    """Hugging Face BERT initialization: N(0, std) weights, zero biases, unit LayerNorm."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(0.0, std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(0.0, std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.weight.data.fill_(1.0)
        module.bias.data.zero_()


class _Pretrained(nn.Module):
    """Saving and loading in Hugging Face format (config.json + model.safetensors)."""

    architecture = ""

    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config

    def save_pretrained(self, directory: PathLike) -> None:
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.architectures = [self.architecture]
        self.config.save_pretrained(directory)
        state = {name: tensor.contiguous() for name, tensor in self.state_dict().items()
                 if name not in self._tied_names()}
        save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})

    @classmethod
    def from_pretrained(cls, directory: PathLike) -> "_Pretrained":
        from safetensors.torch import load_file

        directory = Path(directory)
        model = cls(BertConfig.from_pretrained(directory))
        model.load_hf_state_dict(load_file(str(directory / "model.safetensors")))
        return model

    def _tied_names(self) -> set:
        return set()

    def load_hf_state_dict(self, state: Dict[str, Tensor]) -> None:
        """Load a Hugging Face state dict (tied and buffer entries may be absent)."""
        state = {name: tensor for name, tensor in state.items()
                 if not name.endswith(("position_ids", "token_type_ids"))}
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [name for name in missing if name not in self._tied_names()]
        if missing or unexpected:
            raise ValueError(f"Incompatible checkpoint: missing {missing}, unexpected {unexpected}")


class BertForTokenClassification(_Pretrained):
    """Encoder plus one label per character (the correction model)."""

    architecture = "BertForTokenClassification"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = Bert(config)
        dropout = config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.apply(lambda m: init_weights(m, config.initializer_range))

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        return self.classifier(self.dropout(self.bert(input_ids, attention_mask)))


class PredictionHead(nn.Module):
    def __init__(self, config: BertConfig, embeddings: nn.Embedding) -> None:
        super().__init__()
        self.transform = nn.Module()
        self.transform.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.decoder.weight = embeddings.weight  # tied input and output embeddings
        self.bias = self.decoder.bias

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = self.transform.LayerNorm(F.gelu(self.transform.dense(hidden)))
        return self.decoder(hidden)


class BertForMaskedLM(_Pretrained):
    """Encoder plus masked-character prediction (pretraining)."""

    architecture = "BertForMaskedLM"

    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.bert = Bert(config)
        self.cls = nn.Module()
        self.apply(lambda m: init_weights(m, config.initializer_range))
        self.cls.predictions = PredictionHead(config, self.bert.embeddings.word_embeddings)
        init_weights(self.cls.predictions.transform.dense, config.initializer_range)
        self.cls.predictions.decoder.bias.data.zero_()

    def _tied_names(self) -> set:
        return {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}

    def forward(self, input_ids: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        return self.cls.predictions(self.bert(input_ids, attention_mask))


def make_config(vocab_size: int, num_labels: int = 2, layers: int = 8, hidden: int = 384,
                max_positions: int = 512, pad_token_id: int = 0,
                position_embedding_type: str = "relative_key", **kwargs) -> BertConfig:
    """BERT configuration for AraSpellX (default: the size fixed by the CPU speed test).

    Positions are relative (attention sees the distance between characters):
    with absolute positions a character-level masked LM can stay stuck
    predicting character frequencies, never learning to look at neighbours.
    """
    return BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=hidden // 64,
        intermediate_size=4 * hidden,
        max_position_embeddings=max_positions,
        type_vocab_size=1,
        pad_token_id=pad_token_id,
        num_labels=num_labels,
        position_embedding_type=position_embedding_type,
        **kwargs,
    )
