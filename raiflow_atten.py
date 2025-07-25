from typing import Optional

import torch
from diffusers.models.attention_processor import Attention

def dispatch_attention_fn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    **kwargs,
) -> torch.Tensor:
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
    )
    out = out.permute(0, 2, 1, 3)
    return out


class RaiFlowAttnProcessor2_0:
    _attention_backend = None

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("RaiFlowAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        attn_heads = attn.heads
        head_dim = attn.inner_dim // attn_heads

        # `sample` projections.
        qkv = attn.to_qkv(hidden_states)
        qkv = qkv.unflatten(-1, (3, attn_heads, head_dim))
        query, key, value = qkv[:,:,0,:,:], qkv[:,:,1,:,:], qkv[:,:,2,:,:]

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            _, encoder_seq_len, _ = encoder_hidden_states.shape
            enc_qkv = attn.to_added_qkv(encoder_hidden_states)
            enc_qkv = enc_qkv.unflatten(-1, (3, attn_heads, head_dim))
            (
                encoder_hidden_states_query_proj,
                encoder_hidden_states_key_proj,
                encoder_hidden_states_value_proj,
            ) = (
                enc_qkv[:,:,0,:,:],
                enc_qkv[:,:,1,:,:],
                enc_qkv[:,:,2,:,:],
            )

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=1)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)

        attn_output = dispatch_attention_fn(query, key, value)
        attn_output = attn_output.flatten(2, 3)
        attn_output = attn_output.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            attn_output, context_attn_output = (
                attn_output[:, encoder_seq_len :],
                attn_output[:, : encoder_seq_len],
            )
            context_attn_output = attn.to_add_out(context_attn_output)

        # linear proj
        attn_output = attn.to_out[0](attn_output)
        # dropout
        attn_output = attn.to_out[1](attn_output)

        if encoder_hidden_states is not None:
            return attn_output, context_attn_output
        else:
            return attn_output


class RaiFlowCrossAttnProcessor2_0:
    _attention_backend = None

    def __init__(self):
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("RaiFlowCrossAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        attn_heads = attn.heads
        head_dim = attn.inner_dim // attn_heads

        query = attn.to_q(hidden_states)
        query = query.unflatten(-1, (attn_heads, head_dim))

        kv = attn.to_kv(encoder_hidden_states)
        kv = kv.unflatten(-1, (2, attn_heads, head_dim))
        key, value = kv[:,:,0,:,:], kv[:,:,1,:,:]

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        attn_output = dispatch_attention_fn(query, key, value)
        attn_output = attn_output.flatten(2, 3)
        attn_output = attn_output.to(query.dtype)

        # linear proj
        attn_output = attn.to_out[0](attn_output)
        # dropout
        attn_output = attn.to_out[1](attn_output)

        return attn_output
