from typing import Tuple, Optional

import torch
from torch import nn

fp16_max = 65504


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


def _get_projections(attn: "RaiFlowAttention", hidden_states: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    heads = attn.heads
    head_dim = attn.head_dim

    query = attn.to_q(hidden_states).unflatten(-1, (heads, head_dim))
    key = attn.to_k(hidden_states).unflatten(-1, (heads, head_dim))
    value = attn.to_v(hidden_states).unflatten(-1, (heads, head_dim))

    query = attn.norm_q(query.clamp(-fp16_max, fp16_max))
    key = attn.norm_k(key.clamp(-fp16_max, fp16_max))
    return query, key, value


def _get_fused_projections(attn: "RaiFlowAttention", hidden_states: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    heads = attn.heads
    head_dim = attn.head_dim

    query, key, value = attn.to_qkv(hidden_states).unflatten(-1, (3*heads, head_dim)).split(heads, dim=-2)
    query = attn.norm_q(query.clamp(-fp16_max, fp16_max))
    key = attn.norm_k(key.clamp(-fp16_max, fp16_max))
    return query, key, value


def _get_qkv_projections(attn: "RaiFlowAttention", hidden_states: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states)
    return _get_projections(attn, hidden_states)


class RaiFlowAttnProcessor:
    def __call__(self, attn: "RaiFlowAttention", hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return attn.to_out(
            torch.mul(
                attn.gate(hidden_states),
                dispatch_attention_fn(*_get_qkv_projections(attn, hidden_states)).flatten(-2, -1).to(dtype=hidden_states.dtype),
            )
        )


class RaiFlowAttention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        out_dim: Optional[int] = None,
        out_context_dim: Optional[int] = None,
        heads: int = 12,
        head_dim: int = 128,
        dropout: float = 0.1,
        eps: float = 1e-5,
        bias: bool = False,
        elementwise_affine: bool = False,
        processor: Optional[RaiFlowAttnProcessor] = None,
    ):
        super().__init__()
        self.query_dim = query_dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * heads

        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else self.query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else self.query_dim
        self.processor = processor if processor is not None else RaiFlowAttnProcessor()

        self.gate = nn.Sequential(
            nn.Linear(self.query_dim, self.inner_dim, bias=bias),
            nn.GELU(approximate="none"),
            nn.Dropout(dropout),
        )

        self.to_out = nn.Linear(self.inner_dim, self.out_dim, bias=bias)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=elementwise_affine)

        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.query_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(self.query_dim, self.inner_dim, bias=bias)

    @torch.no_grad()
    def fuse_projections(self):
        device = self.to_q.weight.data.device
        dtype = self.to_q.weight.data.dtype
        self.fused_projections = True

        output_channel_size, channel_size = self.to_q.weight.shape
        output_channel_size = output_channel_size * 3
        use_bias = bool(self.to_qkv.bias is not None)
        self.to_qkv = nn.Linear(channel_size, output_channel_size, bias=use_bias, device=device, dtype=dtype)
        self.to_qkv.weight.copy_(torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data]))
        if use_bias:
            self.to_qkv.bias.copy_(torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data]))
        del self.to_q, self.to_k, self.to_v

    @torch.no_grad()
    def unfuse_projections(self):
        self.fused_projections = False

        device = self.to_qkv.weight.data.device
        dtype = self.to_qkv.weight.data.dtype
        output_channel_size, channel_size = self.to_qkv.weight.shape
        output_channel_size = output_channel_size // 3
        use_bias = bool(self.to_qkv.bias is not None)
        self.to_q = nn.Linear(channel_size, output_channel_size, bias=use_bias, device=device, dtype=dtype)
        self.to_k = nn.Linear(channel_size, output_channel_size, bias=use_bias, device=device, dtype=dtype)
        self.to_v = nn.Linear(channel_size, output_channel_size, bias=use_bias, device=device, dtype=dtype)
        to_q, to_k, to_v = self.to_qkv.weight.data.chunk(3)
        self.to_q.weight.copy_(to_q)
        self.to_k.weight.copy_(to_k)
        self.to_v.weight.copy_(to_v)
        if use_bias:
            to_q_bias, to_k_bias, to_v_bias = self.to_qkv.bias.data.chunk(3)
            self.to_q.bias.copy_(to_q_bias)
            self.to_k.bias.copy_(to_k_bias)
            self.to_v.bias.copy_(to_v_bias)
            del to_q_bias, to_k_bias, to_v_bias
        del self.to_qkv, to_q, to_k, to_v

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.processor(self, hidden_states)
