import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention


class SoteDiffusionV3AttnProcessor2_0:
    """Attention processor used in processing the Sote Diffusion V3 self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SoteDiffusionV3AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, seq_len, _ = hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states).clamp(-32768,32768)

        attn_heads = attn.heads
        head_dim = attn.inner_dim // attn_heads

        query = query.view(batch_size, seq_len, attn_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, attn_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, attn_heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query.clamp(-32768,32768))
        else:
            query = query.clamp(-255,255)
        if attn.norm_k is not None:
            key = attn.norm_k(key.clamp(-32768,32768))
        else:
            key = key.clamp(-255,255)

        total_seq_len = seq_len
        # `context` projections.
        if encoder_hidden_states is not None:
            _, encoder_seq_len, _ = encoder_hidden_states.shape
            total_seq_len = seq_len + encoder_seq_len
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states).clamp(-32768,32768)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, encoder_seq_len, attn_heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, encoder_seq_len, attn_heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, encoder_seq_len, attn_heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj.clamp(-32768,32768))
            else:
                encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.clamp(-255,255)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj.clamp(-32768,32768))
            else:
                encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.clamp(-255,255)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        attn_output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, total_seq_len, attn_heads * head_dim)
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


class SoteDiffusionV3CrossAttnProcessor2_0:
    """Attention processor used in processing the Sote Diffusion V3 cross-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("SoteDiffusionV3CrossAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        secondary_hidden_states: torch.FloatTensor,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size, seq_len, _ = hidden_states.shape
        _, secondary_seq_len, _ = secondary_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(secondary_hidden_states)
        value = attn.to_v(secondary_hidden_states).clamp(-32768,32768)

        attn_heads = attn.heads
        head_dim = attn.inner_dim // attn_heads

        query = query.view(batch_size, seq_len, attn_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, secondary_seq_len, attn_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, secondary_seq_len, attn_heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query.clamp(-32768,32768))
        else:
            query = query.clamp(-255,255)
        if attn.norm_k is not None:
            key = attn.norm_k(key.clamp(-32768,32768))
        else:
            key = key.clamp(-255,255)

        attn_output = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, attn_heads * head_dim)
        attn_output = attn_output.to(query.dtype)

        # linear proj
        attn_output = attn.to_out[0](attn_output)
        # dropout
        attn_output = attn.to_out[1](attn_output)

        return attn_output