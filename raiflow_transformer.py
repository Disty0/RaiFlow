from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from .raiflow_layers import RaiFlowFeedForward
from .raiflow_atten import RaiFlowAttention, RaiFlowAttnProcessor, RaiFlowCrossAttnProcessor
from .raiflow_embedder import RaiFlowLatentEmbedder, RaiFlowTextEmbedder, RaiFlowLatentUnembedder
from .raiflow_pipeline_output import RaiFlowTransformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class RaiFlowSingleTransformerBlock(nn.Module):
    r"""
    A Single Transformer block as part of the RaiFlow MMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the conv feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        eps (`float`, *optional*, defaults to 1e-05): The eps used with nn modules.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-05,
    ):
        super().__init__()

        self.norm_attn = nn.RMSNorm(dim)
        self.attn = RaiFlowAttention(
            query_dim=dim,
            out_dim=dim,
            out_context_dim=dim,
            heads=num_attention_heads,
            head_dim=attention_head_dim,
            dropout=dropout,
            processor=RaiFlowAttnProcessor(),
            is_joint_attention=False,
            is_cross_attention=False,
        )

        self.norm_ff = nn.RMSNorm(dim)
        self.ff = RaiFlowFeedForward(dim=dim, dim_out=dim, ff_mult=ff_mult, dropout=dropout)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = hidden_states + self.attn(hidden_states=self.norm_attn(hidden_states), encoder_hidden_states=None)
        hidden_states = hidden_states + self.ff(hidden_states=self.norm_ff(hidden_states))
        return hidden_states


class RaiFlowJointTransformerBlock(nn.Module):
    r"""
    A Joint Transformer block as part of the RaiFlow MMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the conv feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        eps (`float`, *optional*, defaults to 1e-05): The eps used with nn modules.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ff_mult: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-05,
    ):
        super().__init__()

        self.norm_attn = nn.RMSNorm(dim)
        self.norm_attn_context = nn.RMSNorm(dim)
        self.attn = RaiFlowAttention(
            query_dim=dim,
            out_dim=dim,
            out_context_dim=dim,
            heads=num_attention_heads,
            head_dim=attention_head_dim,
            dropout=dropout,
            processor=RaiFlowAttnProcessor(),
            is_joint_attention=True,
            is_cross_attention=False,
        )

        self.encoder_transformer = RaiFlowSingleTransformerBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ff_mult=ff_mult,
            dropout=dropout,
            eps=eps,
        )

        self.latent_transformer = RaiFlowSingleTransformerBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ff_mult=ff_mult,
            dropout=dropout,
            eps=eps,
        )

    def forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        attn_output, context_attn_output = self.attn(hidden_states=self.norm_attn(hidden_states), encoder_hidden_states=self.norm_attn_context(encoder_hidden_states))
        hidden_states = hidden_states + attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        hidden_states = self.latent_transformer(hidden_states=hidden_states)
        encoder_hidden_states = self.encoder_transformer(hidden_states=encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class RaiFlowConditionalTransformer2DBlock(nn.Module):
    r"""
    A Conditional Transformer block as part of the RaiFlow MMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the conv feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        eps (`float`, *optional*, defaults to 1e-05): The eps used with nn modules.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ff_mult: int = 2,
        dropout: float = 0.1,
        eps: float = 1e-05,
    ):
        super().__init__()

        self.norm_cross_attn = nn.RMSNorm(dim)
        self.cross_attn = RaiFlowAttention(
            query_dim=dim,
            out_dim=dim,
            out_context_dim=dim,
            heads=num_attention_heads,
            head_dim=attention_head_dim,
            dropout=dropout,
            processor=RaiFlowCrossAttnProcessor(),
            is_joint_attention=False,
            is_cross_attention=True,
        )

        self.norm_attn = nn.RMSNorm(dim)
        self.attn = RaiFlowAttention(
            query_dim=dim,
            out_dim=dim,
            out_context_dim=dim,
            heads=num_attention_heads,
            head_dim=attention_head_dim,
            dropout=dropout,
            processor=RaiFlowAttnProcessor(),
            is_joint_attention=False,
            is_cross_attention=False,
        )

        self.norm_ff = nn.RMSNorm(dim)
        self.ff = RaiFlowFeedForward(dim=dim, dim_out=dim, ff_mult=ff_mult, dropout=dropout)

    def forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = hidden_states + self.cross_attn(hidden_states=self.norm_cross_attn(hidden_states), encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + self.attn(hidden_states=self.norm_attn(hidden_states), encoder_hidden_states=None)
        hidden_states = hidden_states + self.ff(hidden_states=self.norm_ff(hidden_states))
        return hidden_states


class RaiFlowTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    The Multi Modal Convoluted Transformer model introduced in RaiFlow.

    Parameters:
        sample_size (`int`, *optional*, defaults to 128): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        in_channels (`int`, *optional*, defaults to 384): The number of channels in the input.
        num_joint_layers (`int`, *optional*, defaults to 4): The number of joint layers of Transformer blocks to use.
        num_layers (`int`, *optional*, defaults to 16): The number of conditional layers of Transformer blocks to use.
        num_refiner_layers (`int`, *optional*, defaults to 4): The number of refiner layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 24): The number of heads to use for multi-head attention.
        encoder_in_channels (`int`, *optional*, defaults to 1536): The number of `encoder_hidden_states` dimensions to use.
        encoder_max_sequence_length (`int`, *optional*, defaults to 1024): The sequence lenght of the text encoder embeds.
            This is fixed during training since it is used to learn a number of position embeddings.
        num_train_timesteps (`int`, defaults to 1000): The number of diffusion steps to train the model.
        out_channels (`int`, defaults to 384): Number of output channels.
        patch_size (`int`, *optional*, (`int`, *optional*, defaults to 2):
            The size of each patch in the image. This parameter defines the resolution of patches fed into the model.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        eps (`float`, *optional*, defaults to 1e-05): The eps used with nn modules.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        "latent_embedder", "unembedder", "text_embedder", "token_embedding",
        "norm_unembed", "norm_ff", "norm_attn", "norm_attn_context", "norm",
        "norm_cross_attn","norm_q", "norm_k", "norm_added_q", "norm_added_k",
        "shift_latent", "shift_latent_out", "shift_in", "shift_out", "bias",
        "scale_latent", "scale_latent_out", "scale_in", "scale_out",
    ]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        in_channels: int = 384,
        num_joint_layers: int = 4,
        num_layers: int = 16,
        num_refiner_layers: int = 4,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        num_train_timesteps: int = 1000,
        encoder_max_sequence_length: int = 1024,
        encoder_pad_to_multiple_of: int = 256,
        vocab_size: int = 151936,
        pad_token_id: int = 151643,
        embedding_dim: int = None,
        out_channels: int = None,
        patch_size: int = 1,
        ff_mult: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.out_channels = out_channels if out_channels is not None else self.config.in_channels
        self.out_channels = self.out_channels * self.config.patch_size*self.config.patch_size
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.embedding_dim = embedding_dim or self.inner_dim
        self.base_seq_len = (self.config.sample_size // self.config.patch_size) * (self.config.sample_size // self.config.patch_size)
        self.patched_in_channels = 4 + ((self.config.in_channels + 4) * self.config.patch_size*self.config.patch_size) # patched + pos channels
        self.encoder_in_channels = 4 + self.embedding_dim # pos channels

        self.latent_embedder = RaiFlowLatentEmbedder(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            base_seq_len=self.base_seq_len,
            dim=self.patched_in_channels,
            dim_out=self.inner_dim,
            ff_mult=self.config.ff_mult,
            dropout=dropout,
        )

        self.text_embedder = RaiFlowTextEmbedder(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.embedding_dim,
            pad_token_id=self.config.pad_token_id,
            base_seq_len=self.config.encoder_max_sequence_length,
            dim=self.encoder_in_channels, # dim + pos
            dim_out=self.inner_dim,
            ff_mult=self.config.ff_mult,
            dropout=dropout,
        )

        self.joint_transformer_blocks = nn.ModuleList(
            [
                RaiFlowJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_mult=self.config.ff_mult,
                    dropout=dropout,
                    eps=eps,
                )
                for _ in range(self.config.num_joint_layers)
            ]
        )

        self.norm_context = nn.RMSNorm(self.inner_dim)
        self.cond_transformer_blocks = nn.ModuleList(
            [
                RaiFlowConditionalTransformer2DBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_mult=self.config.ff_mult,
                    dropout=dropout,
                    eps=eps,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.refiner_transformer_blocks = nn.ModuleList(
            [
                RaiFlowSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_mult=self.config.ff_mult,
                    dropout=dropout,
                    eps=eps,
                )
                for _ in range(self.config.num_refiner_layers)
            ]
        )

        self.unembedder = RaiFlowLatentUnembedder(
            patch_size=self.config.patch_size,
            dim=self.inner_dim,
            dim_out=self.out_channels,
            ff_mult=self.config.ff_mult,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.FloatTensor,
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
            timestep (`torch.LongTensor`):
                Used to indicate the current denoising step.
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

        with torch.no_grad():
            timestep = timestep.view(batch_size, 1, 1)

        if use_checkpointing:
            encoder_hidden_states = self._gradient_checkpointing_func(
                self.text_embedder,
                encoder_hidden_states,
                timestep,
                dtype,
                latents_seq_len,
                encoder_seq_len,
                batch_size,
            )
        else:
            encoder_hidden_states = self.text_embedder(
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                dtype=dtype,
                latents_seq_len=latents_seq_len,
                encoder_seq_len=encoder_seq_len,
                batch_size=batch_size,
            )

        if use_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                self.latent_embedder,
                hidden_states,
                timestep,
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
                timestep=timestep,
                dtype=dtype,
                latents_seq_len=latents_seq_len,
                encoder_seq_len=encoder_seq_len,
                batch_size=batch_size,
                height=height,
                width=width,
            )

        residual = hidden_states
        for index_block, block in enumerate(self.joint_transformer_blocks):
            if use_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states)
            else:
                hidden_states, encoder_hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + residual
        encoder_hidden_states = self.norm_context(encoder_hidden_states)

        residual = hidden_states
        for index_block, block in enumerate(self.cond_transformer_blocks):
            if use_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, encoder_hidden_states)
            else:
                hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        for index_block, block in enumerate(self.refiner_transformer_blocks):
            if use_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states)
            else:
                hidden_states = block(hidden_states=hidden_states)
        hidden_states = hidden_states + residual

        if use_checkpointing:
            output = self._gradient_checkpointing_func(self.unembedder, hidden_states, height, width)
        else:
            output = self.unembedder(hidden_states, height=height, width=width)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, hidden_states, encoder_hidden_states)

        return RaiFlowTransformer2DModelOutput(sample=output, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
