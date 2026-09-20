import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional
from torch import Tensor
from core.utils import get_positionals
from torch.nn.utils.rnn import (
    pad_packed_sequence, pack_padded_sequence
    )


class MultiHeadAtt(nn.Module):
    """Optimized multi-head attention using PyTorch's scaled_dot_product_attention.

    Preserves mathematical equivalence with the original MultiHeadAtt:
    - Same Q/K/V projections (with bias)
    - Same concat(query, attn_output) + proj_fc pattern
    - Same dropout
    - Compatible mask semantics
    """

    def __init__(
            self,
            d_model: int,
            h: int,
            p_dropout: float,
            device: str
            ) -> None:
        super().__init__()
        assert d_model % h == 0, 'd_model is not divisible by h'
        self.fc_key = nn.Linear(d_model, d_model, bias=True)
        self.fc_query = nn.Linear(d_model, d_model, bias=True)
        self.fc_value = nn.Linear(d_model, d_model, bias=True)
        self.proj_fc = nn.Linear(2 * d_model, d_model, bias=True)
        self.dropout = nn.Dropout(p_dropout)
        self.d_model = d_model
        self.h = h
        self.dk = d_model // h
        self.device = device

    def _prepare_mask_for_sdpa(
            self,
            mask: Optional[Tensor],
            query_len: int,
            key_len: int,
            batch_size: int,
            num_heads: int,
            is_query_mask: bool = True,
            is_causal: bool = False
            ) -> Optional[Tensor]:
        """Convert padding mask to SDPA format.
        
        Current mask: True = padding (masked out)
        SDPA boolean mask: True = participate (keep)
        So we need to invert.
        
        For query_mask (is_query_mask=True): [B, query_len] -> [B, h, query_len, key_len]
        For key_mask (is_query_mask=False): [B, key_len] -> [B, h, query_len, key_len]
        
        CRITICAL: Ensure no row is fully masked (prevents NaN in SDPA).
        For self-attention, allow padded queries to attend to themselves.
        """
        if mask is None:
            return None
        mask = ~mask  # invert: True (padding) -> False (don't participate)
        if is_query_mask:
            # Query mask: [B, query_len] -> [B, 1, query_len, 1] -> [B, h, query_len, key_len]
            mask = mask.unsqueeze(1).unsqueeze(3)
        else:
            # Key mask: [B, key_len] -> [B, 1, 1, key_len] -> [B, h, query_len, key_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.expand(batch_size, num_heads, query_len, key_len)
        
        # Prevent fully-masked rows (causes NaN in SDPA softmax)
        # For self-attention (query_len == key_len), ensure diagonal is not masked
        if query_len == key_len:
            # Create identity mask for diagonal
            diag_mask = torch.eye(query_len, dtype=torch.bool, device=mask.device)
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, query_len, key_len)
            # Allow attending to self even if padded
            mask = mask | diag_mask
        
        return mask

    def forward(
            self,
            key: Tensor,
            query: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None,
            query_mask: Optional[Tensor] = None,
            key_mask: Optional[Tensor] = None,
            need_weights: bool = False
            ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            key: [B, Tk, d_model]
            query: [B, Tq, d_model]
            value: [B, Tk, d_model]
            mask: Padding mask for self-attention [B, M] (True = padding)
            query_mask: Padding mask for query [B, Tq] (True = padding)
            key_mask: Padding mask for key [B, Tk] (True = padding)
            need_weights: If True, compute and return attention weights
        Returns:
            att: [h, B, Tq, Tk] attention weights (or empty tensor if need_weights=False)
            out: [B, Tq, d_model] output
        """
        b, tq, _ = query.shape
        tk = key.shape[1]

        Q = self.fc_query(query)
        K = self.fc_key(key)
        V = self.fc_value(value)

        Q = Q.view(b, tq, self.h, self.dk).transpose(1, 2)
        K = K.view(b, tk, self.h, self.dk).transpose(1, 2)
        V = V.view(b, tk, self.h, self.dk).transpose(1, 2)

        attn_mask = None
        is_causal = False

        if mask is not None:
            attn_mask = self._prepare_mask_for_sdpa(mask, tq, tk, b, self.h)
        elif query_mask is not None and key_mask is not None:
            q_mask = self._prepare_mask_for_sdpa(query_mask, tq, tk, b, self.h, is_query_mask=True)
            k_mask = self._prepare_mask_for_sdpa(key_mask, tq, tk, b, self.h, is_query_mask=False)
            attn_mask = q_mask & k_mask

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        if need_weights:
            scale = 1.0 / math.sqrt(self.dk)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(~attn_mask, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            att = attn_weights.permute(1, 0, 2, 3)
        else:
            att = torch.empty(self.h, b, tq, tk, device=query.device)

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, tq, -1)
        result = torch.cat([query, attn_output], dim=-1)
        result = self.proj_fc(result)
        out = self.dropout(result)

        return att, out


class MultiHeadSelfAtt(MultiHeadAtt):
    """Optimized multi-head self-attention with causal masking.
    
    For decoder self-attention, only causal mask is applied.
    Padding is handled by the loss function.
    """

    def __init__(
            self,
            d_model: int,
            h: int,
            p_dropout: float,
            device: str
            ) -> None:
        super().__init__(d_model, h, p_dropout, device)

    def forward(
            self,
            key: Tensor,
            query: Tensor,
            value: Tensor,
            mask: Optional[Tensor] = None,
            query_mask: Optional[Tensor] = None,
            key_mask: Optional[Tensor] = None,
            need_weights: bool = False
            ) -> Tuple[Tensor, Tensor]:
        b, tq, _ = query.shape
        tk = key.shape[1]
        assert tq == tk, "Self-attention requires query_len == key_len"

        Q = self.fc_query(query)
        K = self.fc_key(key)
        V = self.fc_value(value)

        Q = Q.view(b, tq, self.h, self.dk).transpose(1, 2)
        K = K.view(b, tk, self.h, self.dk).transpose(1, 2)
        V = V.view(b, tk, self.h, self.dk).transpose(1, 2)

        # Causal mask only (standard decoder self-attention)
        # Padding is handled by the loss function
        causal_mask = torch.triu(
            torch.ones(tq, tk, dtype=torch.bool, device=self.device),
            diagonal=1
        )
        causal_mask = ~causal_mask  # True = keep (j <= i)
        attn_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(b, self.h, tq, tk)

        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            is_causal=False,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        if need_weights:
            scale = 1.0 / math.sqrt(self.dk)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn_scores = attn_scores.masked_fill(~attn_mask, -1e9)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            att = attn_weights.permute(1, 0, 2, 3)
        else:
            att = torch.empty(self.h, b, tq, tk, device=query.device)

        attn_output = attn_output.transpose(1, 2).contiguous().view(b, tq, -1)
        result = torch.cat([query, attn_output], dim=-1)
        result = self.proj_fc(result)
        out = self.dropout(result)

        return att, out


class FeedForward(nn.Module):
    def __init__(
            self,
            d_model: int,
            hidden_size: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class AddAndNorm(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.lnrom = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, out: Tensor):
        return self.lnrom(x + out)


class EncoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            h: int,
            hidden_size: int,
            p_dropout: float,
            device: str
            ) -> None:
        super().__init__()
        self.mhsa = MultiHeadAtt(
            d_model=d_model,
            h=h,
            p_dropout=p_dropout,
            device=device
            )
        self.mhsa_add_and_norm = AddAndNorm(d_model=d_model)
        self.ff = FeedForward(
            d_model=d_model,
            hidden_size=hidden_size,
            p_dropout=p_dropout
        )
        self.ff_add_and_norm = AddAndNorm(d_model=d_model)

    def forward(self, x: Tensor, mask: Union[Tensor, None], need_weights: bool = False) -> Tensor:
        _, out = self.mhsa(x, x, x, key_mask=mask, need_weights=need_weights)
        out = self.mhsa_add_and_norm(x, out)
        ff_out = self.ff(out)
        out = self.ff_add_and_norm(out, ff_out)
        return out


class DecoderLayer(nn.Module):
    def __init__(
            self,
            d_model: int,
            h: int,
            p_dropout: float,
            hidden_size: int,
            device: str
            ) -> None:
        super().__init__()
        self.mhsa = MultiHeadSelfAtt(
            d_model=d_model,
            h=h,
            p_dropout=p_dropout,
            device=device
        )
        self.add_and_norm_1 = AddAndNorm(d_model=d_model)
        self.mha = MultiHeadAtt(
            d_model=d_model,
            h=h,
            p_dropout=p_dropout,
            device=device
        )
        self.add_and_norm_2 = AddAndNorm(d_model=d_model)
        self.ff = FeedForward(
            d_model=d_model,
            hidden_size=hidden_size,
            p_dropout=p_dropout
        )
        self.add_and_norm_3 = AddAndNorm(d_model=d_model)

    def forward(
            self,
            x: Tensor,
            encoder_values: Tensor,
            mask: Union[Tensor, None] = None,
            query_mask: Union[Tensor, None] = None,
            key_mask: Union[Tensor, None] = None,
            need_weights: bool = False
            ) -> Tuple[Tensor, Tensor]:
        _, out = self.mhsa(x, x, x, mask=mask, need_weights=need_weights)
        out_1 = self.add_and_norm_1(x, out)
        att, out = self.mha(
            query=out_1,
            key=encoder_values,
            value=encoder_values,
            key_mask=key_mask,
            need_weights=need_weights
            )
        out = self.add_and_norm_2(out_1, out)
        out_1 = self.ff(out)
        out = self.add_and_norm_3(out_1, out)
        return out, att


class PositionalEmb(nn.Module):
    def __init__(
            self,
            voc_size: int,
            d_model: int,
            pad_idx: int,
            device: str
            ) -> None:
        super().__init__()
        self.emb = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=d_model,
            padding_idx=pad_idx
        )
        self.device = device
        self.d_model = d_model
        max_len = 2048
        pe = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                denom = 10000 ** (2 * i / d_model)
                pe[pos, i] = math.sin(pos / denom)
                pe[pos, i + 1] = math.cos(pos / denom)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[-1]
        out = self.emb(x)
        return out + self.pe[:seq_len].to(self.device)


class EncoderLayers(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layers: int,
            voc_size: int,
            hidden_size: int,
            h: int,
            p_dropout: float,
            pad_idx: int,
            device: str
            ) -> None:
        super().__init__()
        self.emb = PositionalEmb(
            voc_size=voc_size,
            d_model=d_model,
            pad_idx=pad_idx,
            device=device
        )
        self.layers = nn.ModuleList([
            EncoderLayer(
                d_model=d_model,
                h=h,
                hidden_size=hidden_size,
                p_dropout=p_dropout,
                device=device
            )
            for _ in range(n_layers)
        ])

    def forward(
            self, x: Tensor, mask: Union[Tensor, None], need_weights: bool = False
            ) -> Tensor:
        out = self.emb(x)
        for layer in self.layers:
            out = layer(out, mask=mask, need_weights=need_weights)
        return out


class DecoderLayers(nn.Module):
    def __init__(
            self,
            voc_size: int,
            d_model: int,
            n_layers: int,
            h: int,
            p_dropout: float,
            hidden_size: int,
            pad_idx: int,
            device: str
            ) -> None:
        super().__init__()
        self.emb = PositionalEmb(
            voc_size=voc_size,
            d_model=d_model,
            pad_idx=pad_idx,
            device=device
        )
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model,
                h=h,
                p_dropout=p_dropout,
                hidden_size=hidden_size,
                device=device
            )
            for _ in range(n_layers)
        ])

    def forward(
            self,
            x: Tensor,
            mask: Tensor,
            enc_values: Tensor,
            key_mask: Union[Tensor, None] = None,
            need_weights: bool = False
            ):
        out = self.emb(x)
        att = None
        for layer in self.layers:
            out, att = layer(
                x=out,
                encoder_values=enc_values,
                mask=mask,
                key_mask=key_mask,
                need_weights=need_weights
                )
        return out, att


class PackedGRU(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bidirectional: bool,
            padding_value: Union[float, int],
            num_layers=1
            ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            num_layers=num_layers,
            batch_first=True
        )
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.padding_value = padding_value
        self.num_layers = num_layers

    def forward(self, x: Tensor, lengths: List[int], hn=None) -> Tensor:
        packed_seq = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
            )
        if hn is None:
            hn = torch.zeros(
                self.num_layers,
                x.shape[0],
                self.hidden_size
                ).to(x.device)
        output, hn = self.gru(packed_seq, hn)
        output, lengths = pad_packed_sequence(output, batch_first=True)
        return output, hn


class GRUBlock(nn.Module):
    def __init__(
            self,
            inp_size: int,
            hidden_size: int,
            p_dropout: float,
            bidirectional: bool,
            padding_value: Union[float, int]
            ) -> None:
        super().__init__()
        self.gru = PackedGRU(
            input_size=inp_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            padding_value=padding_value
        )

        self.ff = FeedForward(
            d_model=hidden_size if bidirectional is False else 2 * hidden_size,
            hidden_size=2 * hidden_size if bidirectional is False else 4 * hidden_size,
            p_dropout=p_dropout
        )
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(p_dropout)
        self.lnorm = nn.LayerNorm(normalized_shape=hidden_size)

    def forward(
            self, x: Tensor, lengths: List[int], hn=None
            ) -> Tuple[Tensor, Tensor]:
        out, h = self.gru(x, lengths, hn=hn)
        out = self.dropout(out)
        out = self.ff(out)
        out = self.lnorm(out)
        return out, h


class GRUStack(nn.Module):
    def __init__(
            self,
            n_layers: int,
            inp_size: int,
            hidden_size: int,
            p_dropout: float,
            bidirectional: bool,
            padding_value: Union[float, int]
            ) -> None:
        super().__init__()
        self.grus = nn.ModuleList([
            GRUBlock(
                inp_size=inp_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                p_dropout=p_dropout,
                bidirectional=bidirectional,
                padding_value=padding_value
            )
            for i in range(n_layers)
        ])
        self.hidden_size = hidden_size

    def forward(self, x: Tensor, lengths: List[int], hn=None) -> Tensor:
        out = x
        hns = []
        for i, layer in enumerate(self.grus):
            if hn is not None:
                out, h = layer(
                    out,
                    lengths,
                    hn=hn if hn.shape[0] != len(self.grus) else hn[i:i+1, ...]
                    )
            else:
                out, h = layer(out, lengths, hn=hn)
            hns.append(h)
        hns = torch.vstack(hns)
        return out, hns


class RNNEncoder(nn.Module):
    def __init__(
            self,
            voc_size: int,
            emb_size: int,
            n_layers: int,
            hidden_size: int,
            p_dropout: float,
            bidirectional: bool,
            padding_idx: int,
            padding_value: Union[float, int],
            ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=emb_size,
            padding_idx=padding_idx
        )
        self.gru_stack = GRUStack(
            n_layers=n_layers,
            inp_size=emb_size,
            hidden_size=hidden_size,
            p_dropout=p_dropout,
            bidirectional=bidirectional,
            padding_value=padding_value
        )

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        out = self.embedding(x)
        out, hn = self.gru_stack(out, lengths)
        return out, hn


class Attention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(
            in_features=2 * hidden_size,
            out_features=hidden_size
        )

    def forward(self, query, key, value):
        query = query.permute(1, 0, 2)
        key = key.permute(0, 2, 1)
        e = torch.softmax(torch.matmul(query, key), dim=-1)
        result = torch.matmul(e, value)
        if result.shape[0] != query.shape[0]:
            query = query.repeat(result.shape[0], 1, 1)
        result = torch.cat([result, query], dim=-1)
        result = self.fc(result)
        result = result.permute(1, 0, 2)
        return result, e


class RNNDecoder(nn.Module):
    def __init__(
            self,
            max_len: int,
            voc_size: int,
            emb_size: int,
            n_layers: int,
            hidden_size: int,
            p_dropout: float,
            bidirectional: bool,
            padding_idx: int,
            padding_value: Union[float, int]
            ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=voc_size,
            embedding_dim=emb_size,
            padding_idx=padding_idx
        )
        self.gru_stack = GRUStack(
            n_layers=n_layers,
            inp_size=emb_size,
            hidden_size=hidden_size,
            p_dropout=p_dropout,
            bidirectional=False,
            padding_value=padding_value
        )
        self.pred_fc = nn.Linear(
            in_features=hidden_size,
            out_features=voc_size
        )
        self.max_len = max_len
        self.attention = Attention(
            hidden_size=hidden_size
            )
        self.key_fc = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size
        )
        self.value_fc = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size
        )
        self.query_fc = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size
        )

    def _process_query(self, h: Tensor):
        h = h.permute(1, 0, 2)
        h = h.contiguous().view(h.shape[0], 1, -1)
        h = self.query_fc(h)
        h = h.permute(1, 0, 2)
        return h

    def forward(
            self,
            enc_values: Tensor,
            hn: Tensor,
            x: Tensor,
            lengths: Tensor
            ) -> Tensor:
        max_len = lengths.max().item()
        out = self.embedding(x)
        key = self.key_fc(enc_values)
        value = self.value_fc(enc_values)
        attention = []
        result = []
        for i in range(max_len):
            step_lens = torch.ones(x.shape[0], dtype=torch.long)
            hn = self.query_fc(hn)
            hn, att = self.attention(key=key, value=value, query=hn)
            output, hn = self.gru_stack(
                out[..., i:i+1, :], lengths=step_lens, hn=hn
                )
            result.append(output)
            attention.append(att[:, -1:, :])
        result = torch.hstack(result)
        attention = torch.hstack(attention)
        result = self.pred_fc(result)
        return result, attention

    def predict(self, hn, x, enc_values, key=None, value=None):
        out = self.embedding(x)
        step_lens = torch.ones(x.shape[0], dtype=torch.long)
        if enc_values is not None:
            key = self.key_fc(enc_values)
            value = self.value_fc(enc_values)
        hn = self.query_fc(hn)
        hn, att = self.attention(key=key, value=value, query=hn)
        output, hn = self.gru_stack(out, lengths=step_lens, hn=hn)
        result = self.pred_fc(output)
        return hn, att, result, key, value