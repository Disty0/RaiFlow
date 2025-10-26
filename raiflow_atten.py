from typing import Optional, Union

import torch
from torch import nn


def dispatch_attention_fn(
    query: torch.FloatTensor,
    key: torch.FloatTensor,
    value: torch.FloatTensor,
    attn_mask: Optional[torch.FloatTensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    **kwargs,
) -> torch.FloatTensor:
    if enable_gqa:
        kwargs["enable_gqa"] = enable_gqa
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = torch.nn.functional.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        **kwargs,
    ).permute(0, 2, 1, 3)
    return out


class RaiFlowAttention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        out_dim: Optional[int] = None,
        out_context_dim: Optional[int] = None,
        heads: int = 12,
        head_dim: int = 128,
        dropout: float = 0.1,
        is_joint_attention: bool = False,
        is_cross_attention: bool = False,
        processor: Optional[Union["RaiFlowAttnProcessor", "RaiFlowCrossAttnProcessor"]] = None,
    ):
        super().__init__()

        self.query_dim = query_dim
        self.is_joint_attention = is_joint_attention
        self.is_cross_attention = is_cross_attention

        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * heads

        self.out_dim = out_dim if out_dim is not None else self.query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else self.query_dim
        self.processor = processor if processor is not None else RaiFlowAttnProcessor()

        self.gate = nn.Sequential(
            nn.Linear(self.query_dim, self.inner_dim, bias=True),
            nn.GELU(approximate="none"),
            nn.Dropout(dropout),
        )

        self.to_out = nn.Linear(self.inner_dim, self.out_dim, bias=True)
        self.norm_q = nn.RMSNorm(self.head_dim)
        self.norm_k = nn.RMSNorm(self.head_dim)

        if self.is_cross_attention:
            self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=True)
            self.to_kv = nn.Linear(self.query_dim, self.inner_dim*2, bias=True)
        else:
            self.to_qkv = nn.Linear(self.query_dim, self.inner_dim*3, bias=True)

        if self.is_joint_attention:
            self.to_added_qkv = nn.Linear(self.query_dim, self.inner_dim*3, bias=True)
            self.norm_added_q = nn.RMSNorm(self.head_dim)
            self.norm_added_k = nn.RMSNorm(self.head_dim)
            self.to_add_out = nn.Linear(self.inner_dim, self.out_context_dim, bias=True)

    def forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )


class RaiFlowAttnProcessor:
    def __call__(self, attn: RaiFlowAttention, hidden_states: torch.FloatTensor, encoder_hidden_states: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        heads = attn.heads
        head_dim = attn.head_dim

        query, key, value = attn.to_qkv(hidden_states).unflatten(-1, (3*heads, head_dim)).split(heads, dim=-2)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            _, encoder_seq_len, _ = encoder_hidden_states.shape
            (
                encoder_hidden_states_query_proj,
                encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj,
            ) = attn.to_added_qkv(encoder_hidden_states).unflatten(-1, (3*heads, head_dim)).split(heads, dim=-2)

            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=1)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)

        attn_output = dispatch_attention_fn(query, key, value).flatten(-2, -1).to(query.dtype)
        attn_output = attn_output * attn.gate(
            torch.cat([encoder_hidden_states, hidden_states], dim=1) if encoder_hidden_states is not None else hidden_states
        )

        if encoder_hidden_states is not None:
            attn_output, context_attn_output = attn_output[:, encoder_seq_len :], attn_output[:, : encoder_seq_len]
            context_attn_output = attn.to_add_out(context_attn_output)
        attn_output = attn.to_out(attn_output)

        if encoder_hidden_states is not None:
            return attn_output, context_attn_output
        else:
            return attn_output


class RaiFlowCrossAttnProcessor:
    def __call__(self, attn: RaiFlowAttention, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        heads = attn.heads
        head_dim = attn.head_dim

        query = attn.to_q(hidden_states).unflatten(-1, (heads, head_dim))
        key, value = attn.to_kv(encoder_hidden_states).unflatten(-1, (2*heads, head_dim)).split(heads, dim=-2)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        attn_output = dispatch_attention_fn(query, key, value).flatten(-2, -1).to(query.dtype)
        attn_output = attn.to_out(attn_output * attn.gate(hidden_states))

        return attn_output
