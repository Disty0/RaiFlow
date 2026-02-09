from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from .raiflow_embedder import RaiFlowLatentEmbedder, RaiFlowTextEmbedder, RaiFlowLatentUnembedder
from .raiflow_pipeline_output import RaiFlowTransformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
fp16_max = 65504


class RaiFlowTransformerBlock(nn.Module):
    r"""
    A Transformer block as part of the RaiFlow DiT architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        eps (`float`, *optional*, defaults to 1e-5): The eps used with nn modules.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.1,
        eps: float = 1e-5,
        bias: bool = False,
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim

        self.ff_in = nn.Linear(dim, dim*12, bias=bias)
        self.ff_out = nn.Linear(dim*5, dim, bias=bias)

        self.ff_gate = nn.Sequential(nn.GELU(approximate="none"), nn.Dropout(dropout))
        self.attn_gate = nn.Sequential(nn.Sigmoid(), nn.Dropout(dropout))

        self.norm = nn.RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = hidden_states.clamp(-fp16_max, fp16_max)
        attn_hidden_states, ff_hidden_states, ff_gate = self.ff_in(self.norm(hidden_states)).chunk(3, dim=-1)

        ff_gate = self.ff_gate(ff_gate)
        ff_hidden_states = torch.mul(ff_gate, ff_hidden_states).contiguous()
        del ff_gate

        query, key, value, attn_gate = attn_hidden_states.unflatten(-1, (-1, self.head_dim)).chunk(4, dim=-2)
        del attn_hidden_states

        query = self.norm_q(query.clamp(-fp16_max, fp16_max))
        key = self.norm_k(key.clamp(-fp16_max, fp16_max))
        attn_hidden_states = dispatch_attention_fn(query, key, value)
        del query, key, value

        attn_gate = self.attn_gate(attn_gate)
        attn_hidden_states = torch.mul(attn_gate, attn_hidden_states).flatten(-2, -1).contiguous()
        del attn_gate

        ff_hidden_states = torch.cat([attn_hidden_states, ff_hidden_states], dim=-1)
        del attn_hidden_states

        ff_hidden_states = self.ff_out(ff_hidden_states)
        hidden_states = hidden_states + ff_hidden_states
        del ff_hidden_states

        return hidden_states


class RaiFlowTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    The Multi Modal Convoluted Transformer model introduced in RaiFlow.

    Parameters:
        sample_size (`int`, *optional*, defaults to 64): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        in_channels (`int`, *optional*, defaults to 384): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 16): The number of Transformer blocks to use.
        num_refiner_layers (`int`, *optional*, defaults to 4): The number of unconditional refiner Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 24): The number of heads to use for multi-head attention.
        encoder_in_channels (`int`, *optional*, defaults to 1536): The number of `encoder_hidden_states` dimensions to use.
        encoder_max_sequence_length (`int`, *optional*, defaults to 1024): The sequence lenght of the text encoder embeds.
            This is fixed during training since it is used to learn a number of position embeddings.
        out_channels (`int`, defaults to 384): Number of output channels.
        patch_size (`int`, *optional*, (`int`, *optional*, defaults to 2):
            The size of each patch in the image. This parameter defines the resolution of patches fed into the model.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        eps (`float`, *optional*, defaults to 1e-5): The eps used with nn modules.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "latent_embedder", "unembedder", "text_embedder", "token_embedding",
        "norm_latent_embedder", "norm_text_embedder", "norm_unembed",
        "norm_ff", "norm_attn", "norm_q", "norm_k", "norm", "bias",
    ]
    _keep_in_fp32_modules = [
        "latent_embedder", "unembedder", "text_embedder_proj",
        "norm_latent_embedder", "norm_text_embedder", "norm_unembed",
    ]

    @register_to_config
    def __init__(
        self,
        max_freqs: int = 8,
        sample_size: int = 64,
        in_channels: int = 384,
        num_layers: int = 16,
        num_refiner_layers: int = 4,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        encoder_max_sequence_length: int = 1024,
        encoder_pad_to_multiple_of: int = 256,
        vocab_size: int = 151936,
        pad_token_id: int = 151643,
        embedding_dim: int = None,
        out_channels: int = None,
        patch_size: int = 1,
        dropout: float = 0.1,
        eps: float = 1e-5,
        bias: bool = False,
        embedder_bias: bool = True,
        elementwise_affine: bool = False,
        embedder_elementwise_affine: bool = True,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.out_channels = out_channels if out_channels is not None else self.config.in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.embedding_dim = embedding_dim or self.inner_dim
        self.base_seq_len = (self.config.sample_size // self.config.patch_size) * (self.config.sample_size // self.config.patch_size)

        self.latent_embedder = RaiFlowLatentEmbedder(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            base_seq_len=self.base_seq_len,
            max_freqs=self.config.max_freqs,
            inner_dim=self.inner_dim,
            eps=self.config.eps,
            bias=self.config.embedder_bias,
            elementwise_affine=self.config.embedder_elementwise_affine,
        )

        self.text_embedder = RaiFlowTextEmbedder(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.embedding_dim,
            pad_token_id=self.config.pad_token_id,
            base_seq_len=self.config.encoder_max_sequence_length,
            max_freqs=self.config.max_freqs,
            inner_dim=self.inner_dim,
            eps=self.config.eps,
            bias=self.config.embedder_bias,
            elementwise_affine=self.config.embedder_elementwise_affine,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                RaiFlowTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    eps=self.config.eps,
                    bias=self.config.bias,
                    elementwise_affine=self.config.elementwise_affine,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.refiner_transformer_blocks = nn.ModuleList(
            [
                RaiFlowTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    eps=self.config.eps,
                    bias=self.config.bias,
                    elementwise_affine=self.config.elementwise_affine,
                )
                for _ in range(self.config.num_refiner_layers)
            ]
        )

        self.unembedder = RaiFlowLatentUnembedder(
            patch_size=self.config.patch_size,
            inner_dim=self.inner_dim,
            out_channels=self.out_channels,
            eps=self.config.eps,
            bias=self.config.embedder_bias,
            elementwise_affine=self.config.embedder_elementwise_affine,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.Tensor,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[RaiFlowTransformer2DModelOutput, Tuple[torch.FloatTensor]]:
        """
        The [`RaiFlowTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, dim, height, width)`):
                The latent input.
            encoder_hidden_states (`torch.Tensor` of shape `(batch size, sequence_len)`):
                Input IDs from the tokenizer to use.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`RaiFlowTransformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`RaiFlowTransformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        dtype = self.text_embedder.token_embedding.weight.dtype # pipe can be quantized
        use_checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing

        batch_size, channels, height, width = hidden_states.shape
        _, encoder_seq_len = encoder_hidden_states.shape
        latents_seq_len = (height // self.config.patch_size) * (width // self.config.patch_size)

        if use_checkpointing:
            encoder_hidden_states = self._gradient_checkpointing_func(
                self.text_embedder,
                encoder_hidden_states,
                dtype,
                latents_seq_len,
                encoder_seq_len,
                batch_size,
            )
        else:
            encoder_hidden_states = self.text_embedder(
                encoder_hidden_states=encoder_hidden_states,
                dtype=dtype,
                latents_seq_len=latents_seq_len,
                encoder_seq_len=encoder_seq_len,
                batch_size=batch_size,
            )

        if use_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                self.latent_embedder,
                hidden_states,
                dtype,
                latents_seq_len,
                encoder_seq_len,
                batch_size,
                height,
                width,
            )
        else:
            hidden_states = self.latent_embedder(
                hidden_states=hidden_states,
                dtype=dtype,
                latents_seq_len=latents_seq_len,
                encoder_seq_len=encoder_seq_len,
                batch_size=batch_size,
                height=height,
                width=width,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=-2)
        if use_checkpointing:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = self._gradient_checkpointing_func(block, hidden_states)
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = block(hidden_states=hidden_states)
        hidden_states = hidden_states[:, encoder_seq_len :]

        if use_checkpointing:
            for index_block, block in enumerate(self.refiner_transformer_blocks):
                hidden_states = self._gradient_checkpointing_func(block, hidden_states)
        else:
            for index_block, block in enumerate(self.refiner_transformer_blocks):
                hidden_states = block(hidden_states=hidden_states)

        output = self.unembedder(hidden_states, height=height, width=width)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return RaiFlowTransformer2DModelOutput(sample=output)
