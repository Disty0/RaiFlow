from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward
from diffusers.models.normalization import RMSNorm
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

from .dynamic_tanh import DynamicTanh
from .raiflow_atten import RaiFlowAttnProcessor2_0, RaiFlowCrossAttnProcessor2_0
from .raiflow_embedder import RaiFlowPosEmbed1D, RaiFlowPosEmbed2D, pack_2d_latents_to_1d, unpack_1d_latents_to_2d

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class RaiFlowSingleTransformerBlock(nn.Module):
    r"""
    A Single Transformer block as part of the RaiFlow MMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        eps (`float`, *optional*, defaults to 0.1): The eps used with nn modules.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the linear feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to "rms_norm"): The qk normalization to use in attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-05,
        ff_mult: int = 4,
        dropout: float = 0.1,
        qk_norm: str = "rms_norm",
    ):
        super().__init__()

        if hasattr(F, "scaled_dot_product_attention"):
            processor = RaiFlowAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.norm_attn = DynamicTanh(dim=dim, init_alpha=0.2, elementwise_affine=True, bias=True)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            dropout=dropout,
            qk_norm=qk_norm if qk_norm != "dynamic_tanh" else None,
            elementwise_affine=True,
            eps=eps,
        )
        if qk_norm == "dynamic_tanh":
            self.attn.norm_q = DynamicTanh(dim=attention_head_dim, init_alpha=0.2, elementwise_affine=True, bias=True)
            self.attn.norm_k = DynamicTanh(dim=attention_head_dim, init_alpha=0.2, elementwise_affine=True, bias=True)

        self.norm_ff = DynamicTanh(dim=dim, init_alpha=0.2, elementwise_affine=True, bias=True)
        self.ff = FeedForward(
            dim=dim,
            dim_out=dim,
            mult=ff_mult,
            dropout=dropout,
            activation_fn="gelu-approximate",
            bias=True
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        norm_hidden_states = self.norm_attn(hidden_states)
        hidden_states = hidden_states + self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=None)

        norm_hidden_states = self.norm_ff(hidden_states)
        hidden_states = hidden_states + self.ff(norm_hidden_states)
        return hidden_states


@maybe_allow_in_graph
class RaiFlowJointTransformerBlock(nn.Module):
    r"""
    A Joint Transformer block as part of the RaiFlow MMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        eps (`float`, *optional*, defaults to 0.1): The eps used with nn modules.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the linear feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to "rms_norm"): The qk normalization to use in attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-05,
        ff_mult: int = 4,
        dropout: float = 0.1,
        qk_norm: str = "rms_norm",
    ):
        super().__init__()

        if hasattr(F, "scaled_dot_product_attention"):
            processor = RaiFlowAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.norm_attn = DynamicTanh(dim=dim, init_alpha=0.2, elementwise_affine=True, bias=True)
        self.norm_attn_context = DynamicTanh(dim=dim, init_alpha=0.2, elementwise_affine=True, bias=True)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=processor,
            dropout=dropout,
            qk_norm=qk_norm if qk_norm != "dynamic_tanh" else None,
            elementwise_affine=True,
            eps=eps,
        )
        if qk_norm == "dynamic_tanh":
            self.attn.norm_q = DynamicTanh(dim=attention_head_dim, init_alpha=0.2, elementwise_affine=True, bias=True)
            self.attn.norm_k = DynamicTanh(dim=attention_head_dim, init_alpha=0.2, elementwise_affine=True, bias=True)
            self.attn.norm_added_q = DynamicTanh(dim=attention_head_dim, init_alpha=0.2, elementwise_affine=True, bias=True)
            self.attn.norm_added_k = DynamicTanh(dim=attention_head_dim, init_alpha=0.2, elementwise_affine=True, bias=True)

        self.encoder_transformer = RaiFlowSingleTransformerBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            eps=eps,
            ff_mult=ff_mult,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.latent_transformer = RaiFlowSingleTransformerBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            eps=eps,
            ff_mult=ff_mult,
            dropout=dropout,
            qk_norm=qk_norm,
        )


    def forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        norm_hidden_states = self.norm_attn(hidden_states)
        norm_encoder_hidden_states = self.norm_attn_context(encoder_hidden_states)

        attn_output, context_attn_output = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states)
        hidden_states = hidden_states + attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        hidden_states = self.latent_transformer(hidden_states)
        encoder_hidden_states = self.encoder_transformer(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class RaiFlowConditionalTransformer2DBlock(nn.Module):
    r"""
    A Conditional Transformer block as part of the RaiFlow MMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        eps (`float`, *optional*, defaults to 0.1): The eps used with nn modules.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the linear feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to "rms_norm"): The qk normalization to use in attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-05,
        ff_mult: int = 4,
        dropout: float = 0.1,
        qk_norm: str = "rms_norm",
    ):
        super().__init__()

        if hasattr(F, "scaled_dot_product_attention"):
            cross_processor = RaiFlowCrossAttnProcessor2_0()
            processor = RaiFlowAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.norm_cross_attn = DynamicTanh(dim=dim, init_alpha=0.2, elementwise_affine=True, bias=True)
        self.cross_attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=cross_processor,
            dropout=dropout,
            qk_norm=qk_norm if qk_norm != "dynamic_tanh" else None,
            elementwise_affine=True,
            eps=eps,
        )
        if qk_norm == "dynamic_tanh":
            self.cross_attn.norm_q = DynamicTanh(dim=attention_head_dim, init_alpha=0.2, elementwise_affine=True, bias=True)
            self.cross_attn.norm_k = DynamicTanh(dim=attention_head_dim, init_alpha=0.2, elementwise_affine=True, bias=True)

        self.norm_attn = DynamicTanh(dim=dim, init_alpha=0.2, elementwise_affine=True, bias=True)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            dropout=dropout,
            qk_norm=qk_norm if qk_norm != "dynamic_tanh" else None,
            elementwise_affine=True,
            eps=eps,
        )
        if qk_norm == "dynamic_tanh":
            self.attn.norm_q = DynamicTanh(dim=attention_head_dim, init_alpha=0.2, elementwise_affine=True, bias=True)
            self.attn.norm_k = DynamicTanh(dim=attention_head_dim, init_alpha=0.2, elementwise_affine=True, bias=True)

        self.norm_ff = DynamicTanh(dim=dim, init_alpha=0.2, elementwise_affine=True, bias=True)
        self.ff = FeedForward(
            dim=dim*2,
            dim_out=dim,
            mult=ff_mult/2,
            dropout=dropout,
            activation_fn="gelu-approximate",
            bias=True
        )

    def forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor) -> torch.FloatTensor:
        norm_hidden_states = self.norm_cross_attn(hidden_states)
        hidden_states = hidden_states + self.cross_attn(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)

        norm_hidden_states = self.norm_attn(hidden_states)
        hidden_states = hidden_states + self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=None)

        norm_hidden_states = self.norm_ff(hidden_states)
        ff_hidden_states = torch.cat([norm_hidden_states, temb], dim=-1)
        hidden_states = hidden_states + self.ff(ff_hidden_states)
        return hidden_states


class RaiFlowTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    The Multi Modal Convoluted Transformer model introduced in RaiFlow.

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        in_channels (`int`, *optional*, defaults to 384): The number of channels in the input.
        num_joint_layers (`int`, *optional*, defaults to 4): The number of joint layers of Transformer blocks to use.
        num_layers (`int`, *optional*, defaults to 24): The number of single layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 32): The number of heads to use for multi-head attention.
        encoder_in_channels (`int`, *optional*, defaults to 1536): The number of `encoder_hidden_states` dimensions to use.
        out_channels (`int`, defaults to 384): Number of output channels.
        eps (`float`, *optional*, defaults to 0.1): The eps used with nn modules.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the linear feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to "rms_norm"): The qk normalization to use in attention.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        in_channels: int = 16,
        num_joint_layers: int = 4,
        num_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 32,
        encoder_in_channels: int = 1536,
        encoder_base_seq_len: int = 1024,
        num_train_timesteps: int = 1000,
        out_channels: int = None,
        patch_size: int = 2,
        eps: float = 1e-05,
        ff_mult: int = 4,
        dropout: float = 0.1,
        qk_norm: str = "rms_norm",
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.out_channels = self.out_channels * self.config.patch_size*self.config.patch_size
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.base_seq_len = (self.config.sample_size // self.config.patch_size) * (self.config.sample_size // self.config.patch_size)
        self.in_channels = 4 + ((in_channels + 4) * self.config.patch_size*self.config.patch_size) # patched + pos channels
        self.encoder_in_channels = encoder_in_channels + 4 # pos channels

        self.timestep_embedder = nn.Linear(4 + (4 * self.config.patch_size*self.config.patch_size), self.in_channels, bias=True)
        self.temb_embedder = FeedForward(
            dim=self.in_channels * 2,
            dim_out=self.inner_dim,
            mult=self.config.ff_mult,
            dropout=dropout,
            activation_fn="gelu-approximate",
            bias=True
        )

        self.embedder = FeedForward(
            dim=self.in_channels,
            dim_out=self.inner_dim,
            mult=self.config.ff_mult,
            dropout=dropout,
            activation_fn="gelu-approximate",
            bias=True
        )

        self.norm_context_in = RMSNorm(encoder_in_channels, eps=eps, elementwise_affine=True)
        self.context_embedder = FeedForward(
            dim=self.encoder_in_channels, # dim + pos
            dim_out=self.inner_dim,
            mult=self.config.ff_mult,
            dropout=dropout,
            activation_fn="gelu-approximate",
            bias=True
        )

        self.norm_temb = DynamicTanh(dim=self.inner_dim, init_alpha=0.2, elementwise_affine=True, bias=True)
        self.norm_context = DynamicTanh(dim=self.inner_dim, init_alpha=0.2, elementwise_affine=True, bias=True)

        self.joint_transformer_blocks = nn.ModuleList(
            [
                RaiFlowJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_mult=self.config.ff_mult,
                    eps=eps,
                    dropout=dropout,
                    qk_norm=qk_norm,
                )
                for _ in range(self.config.num_joint_layers)
            ]
        )

        self.transformer_blocks = nn.ModuleList(
            [
                RaiFlowConditionalTransformer2DBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_mult=self.config.ff_mult,
                    eps=eps,
                    dropout=dropout,
                    qk_norm=qk_norm,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        self.unembedder = FeedForward(
            dim=self.inner_dim * 2,
            dim_out=self.out_channels,
            mult=self.config.ff_mult,
            dropout=dropout,
            activation_fn="gelu-approximate",
            bias=True
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        flip_target: bool = True,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.FloatTensor]]:
        """
        The [`RaiFlowTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, dim, height, width)`):
                The latent input.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            flip_target (`bool`, *optional*, defaults to `True`):
                Whether or not to flip the outputs for inference.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
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

        batch_size, channels, height, width = hidden_states.shape
        latents_seq_len = (height // self.config.patch_size) * (width // self.config.patch_size)

        sigmas = timestep.to(dtype=hidden_states.dtype) / self.config.num_train_timesteps
        sigmas = sigmas.view(batch_size, 1, 1)

        posed_latents_2d = RaiFlowPosEmbed2D(
            shape=hidden_states.shape,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        posed_latents_1d = RaiFlowPosEmbed1D(
            shape=(batch_size, latents_seq_len, (channels*self.config.patch_size*self.config.patch_size)),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            secondary_seq_len=encoder_hidden_states.shape[1],
            base_seq_len=self.base_seq_len,
            sigmas=sigmas,
        )

        hidden_states = torch.cat([hidden_states, posed_latents_2d], dim=1)
        hidden_states = pack_2d_latents_to_1d(hidden_states, patch_size=self.config.patch_size)
        hidden_states = torch.cat([hidden_states, posed_latents_1d], dim=2)

        temb = pack_2d_latents_to_1d(posed_latents_2d, patch_size=self.config.patch_size)
        temb = torch.cat([temb, posed_latents_1d], dim=2)
        temb = self.timestep_embedder(temb)
        temb = torch.cat([hidden_states, temb], dim=2)

        temb = self.temb_embedder(temb)
        temb = self.norm_temb(temb)

        hidden_states = self.embedder(hidden_states)

        posed_encoder_1d = RaiFlowPosEmbed1D(
            shape=encoder_hidden_states.shape,
            device=encoder_hidden_states.device,
            dtype=hidden_states.dtype,
            secondary_seq_len=latents_seq_len,
            base_seq_len=self.config.encoder_base_seq_len,
            sigmas=sigmas,
        )

        encoder_hidden_states = self.norm_context_in(encoder_hidden_states)
        encoder_hidden_states = torch.cat([encoder_hidden_states, posed_encoder_1d], dim=2)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.joint_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                )

        encoder_hidden_states = self.norm_context(encoder_hidden_states)

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                )

        hidden_states = torch.cat([hidden_states, temb], dim=-1)
        output = self.unembedder(hidden_states)
        output = unpack_1d_latents_to_2d(output, patch_size=self.config.patch_size, original_height=height, original_widht=width)

        if flip_target: # latents - noise to noise - latents
            output = -output

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
