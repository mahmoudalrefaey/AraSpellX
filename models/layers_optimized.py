import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Union, Optional
from torch import Tensor


class OptimizedMultiHeadAtt(nn.Module):
    """Optimized multi-head attention using PyTorch's scaled_dot_product_attention.

    Preserves mathematical equivalence with the original MultiHeadAtt:
    - Same Q/K/V projections
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
        self.fc_key = nn.Linear(d_model, d_model, bias=False)
        self.fc_query = nn.Linear(d_model, d_model, bias=False)
        self.fc_value = nn.Linear(d_model, d_model, bias=False)
        self.proj_fc = nn.Linear(2 * d_model, d_model)
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
            is_causal: bool = False
            ) -> Optional[Tensor]:
        """Convert padding mask to SDPA format.
        
        Current mask: True = padding (masked out)
        SDPA boolean mask: True = participate (keep)
        So we need to invert.
        """
        if mask is None:
            return None
        # mask shape: [B, M] where True = padding
        # Expand to [B, 1, 1, M] for broadcasting
        mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, M]
        # Invert: True -> False (padding positions should not participate)
        mask = ~mask
        # Expand for heads: [B, h, 1, M] -> will broadcast to [B, h, query_len, key_len]
        mask = mask.expand(batch_size, num_heads, query_len, key_len)
        return mask

    def _prepare_causal_mask(
            self,
            query_len: int,
            key_len: int,
            batch_size: int,
            num_heads: int,
            device: torch.device
            ) -> Tensor:
        """Create causal mask for SDPA (is_causal=True is preferred but we need custom for combined)."""
        # SDPA's is_causal=True handles this automatically
        # But for combined causal + padding, we need explicit mask
        causal_mask = torch.triu(
            torch.ones(query_len, key_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        # causal_mask[i,j] = True means j > i (future positions)
        # SDPA expects True = participate, so invert
        causal_mask = ~causal_mask  # True = keep (j <= i)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, query_len, key_len]
        causal_mask = causal_mask.expand(batch_size, num_heads, query_len, key_len)
        return causal_mask

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

        # Project Q, K, V
        Q = self.fc_query(query)  # [B, Tq, d_model]
        K = self.fc_key(key)      # [B, Tk, d_model]
        V = self.fc_value(value)  # [B, Tk, d_model]

        # Reshape for multi-head: [B, T, h, dk] -> [B, h, T, dk]
        Q = Q.view(b, tq, self.h, self.dk).transpose(1, 2)  # [B, h, Tq, dk]
        K = K.view(b, tk, self.h, self.dk).transpose(1, 2)  # [B, h, Tk, dk]
        V = V.view(b, tk, self.h, self.dk).transpose(1, 2)  # [B, h, Tk, dk]

        # Prepare attention mask for SDPA
        attn_mask = None
        is_causal = False

        if mask is not None:
            # Self-attention case: mask is padding mask [B, M]
            attn_mask = self._prepare_mask_for_sdpa(mask, tq, tk, b, self.h)
        elif query_mask is not None and key_mask is not None:
            # Cross-attention case: separate query and key masks
            q_mask = self._prepare_mask_for_sdpa(query_mask, tq, tk, b, self.h)
            k_mask = self._prepare_mask_for_sdpa(key_mask, tq, tk, b, self.h)
            attn_mask = q_mask & k_mask

        # Compute attention using SDPA
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # Compute attention weights only if needed (for visualization)
        if need_weights:
            scale = 1.0 / math.sqrt(self.dk)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=-1)  # [B, h, Tq, Tk]
            att = attn_weights.permute(1, 0, 2, 3)  # [h, B, Tq, Tk]
        else:
            # Return empty tensor with correct shape for compatibility
            att = torch.empty(self.h, b, tq, tk, device=query.device)

        # Reshape output: [B, h, Tq, dk] -> [B, Tq, h, dk] -> [B, Tq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, tq, -1)

        # Concatenate with original query (preserves original behavior)
        result = torch.cat([query, attn_output], dim=-1)  # [B, Tq, 2*d_model]
        result = self.proj_fc(result)  # [B, Tq, d_model]
        out = self.dropout(result)

        return att, out


class OptimizedMultiHeadSelfAtt(OptimizedMultiHeadAtt):
    """Optimized multi-head self-attention with causal masking."""

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
        """
        For self-attention, mask combines causal + padding.
        The original implementation creates a combined mask in get_mask.
        We'll replicate that logic here.
        """
        b, tq, _ = query.shape
        tk = key.shape[1]
        assert tq == tk, "Self-attention requires query_len == key_len"

        # Project Q, K, V
        Q = self.fc_query(query)
        K = self.fc_key(key)
        V = self.fc_value(value)

        # Reshape: [B, T, h, dk] -> [B, h, T, dk]
        Q = Q.view(b, tq, self.h, self.dk).transpose(1, 2)
        K = K.view(b, tk, self.h, self.dk).transpose(1, 2)
        V = V.view(b, tk, self.h, self.dk).transpose(1, 2)

        # Build combined causal + padding mask for SDPA
        # SDPA boolean mask: True = participate, False = mask
        attn_mask = None
        if mask is not None:
            # mask: [B, M] where True = padding
            # Create padding mask: [B, h, Tq, Tk] where True = keep
            pad_mask = (~mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, M]
            pad_mask = pad_mask.expand(b, self.h, tq, tk)
            
            # Create causal mask: [B, h, Tq, Tk] where True = keep (j <= i)
            causal_mask = torch.triu(
                torch.ones(tq, tk, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            causal_mask = ~causal_mask  # True = keep (j <= i)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(b, self.h, tq, tk)
            
            # Combine: both padding and causal must allow
            attn_mask = pad_mask & causal_mask
        else:
            # Pure causal mask
            causal_mask = torch.triu(
                torch.ones(tq, tk, dtype=torch.bool, device=self.device),
                diagonal=1
            )
            causal_mask = ~causal_mask
            attn_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(b, self.h, tq, tk)

        # SDPA with combined mask
        attn_output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            is_causal=False,  # We provide explicit mask
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # Compute attention weights only if needed (for visualization)
        if need_weights:
            scale = 1.0 / math.sqrt(self.dk)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=-1)
            att = attn_weights.permute(1, 0, 2, 3)
        else:
            att = torch.empty(self.h, b, tq, tk, device=query.device)

        # Reshape output
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, tq, -1)

        # Concat + project (original behavior)
        result = torch.cat([query, attn_output], dim=-1)
        result = self.proj_fc(result)
        out = self.dropout(result)

        return att, out


# Backward compatibility aliases
MultiHeadAtt = OptimizedMultiHeadAtt
MultiHeadSelfAtt = OptimizedMultiHeadSelfAtt