from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_utils import ModelMixin

from .dynamic_tanh import DynamicTanh
from .raiflow_atten import RaiFlowAttnProcessor2_0, RaiFlowCrossAttnProcessor2_0
from .raiflow_embedder import RaiFlowPosEmbed1D, RaiFlowPosEmbed2D, pack_2d_latents_to_1d, unpack_1d_latents_to_2d, prepare_latent_image_ids, prepare_text_embed_ids, FluxPosEmbed
from .raiflow_pipeline_output import RaiFlowTransformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class RaiFlowFeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int, num_attention_heads: int, attention_head_dim: int, heads_per_group: int = 2, router_mult: int = 2, ff_mult: int = 2, dropout: float = 0.1, is_2d: bool = False):
        super().__init__()
        self.is_2d = is_2d
        self.num_groups = num_attention_heads // heads_per_group
        self.inner_dim = int(self.num_groups * attention_head_dim * heads_per_group * router_mult)
        self.ff_dim = int(self.inner_dim * ff_mult)

        self.router = nn.Sequential(
            nn.Linear(dim, self.inner_dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
        )

        if self.is_2d:
            self.conv = nn.Sequential(
                nn.Conv2d(self.inner_dim, self.ff_dim, 3, padding=1, groups=self.num_groups),
                nn.GELU(approximate='tanh'),
                nn.Dropout(dropout),
                nn.Conv2d(self.ff_dim, self.inner_dim, 3, padding=1, groups=self.num_groups),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(self.inner_dim, self.ff_dim, 3, padding=1, groups=self.num_groups),
                nn.GELU(approximate='tanh'),
                nn.Dropout(dropout),
                nn.Conv1d(self.ff_dim, self.inner_dim, 3, padding=1, groups=self.num_groups),
            )

        self.proj_out = nn.Sequential(
            nn.GELU(approximate='tanh'),
            nn.Dropout(dropout),
            nn.Linear(self.inner_dim, dim_out, bias=True),
        )

    def forward(self, hidden_states: torch.FloatTensor, height: Optional[int] = None, width: Optional[int] = None) -> torch.FloatTensor:
        batch_size, seq_len, inner_dim = hidden_states.shape

        router_outputs = self.router(hidden_states)

        ff_hidden_states = router_outputs.transpose(1,2)
        if self.is_2d:
            ff_hidden_states = ff_hidden_states.view(batch_size, self.inner_dim, height, width)

        ff_hidden_states = self.conv(ff_hidden_states)

        if self.is_2d:
            ff_hidden_states = ff_hidden_states.view(batch_size, self.inner_dim, seq_len)
        ff_hidden_states = ff_hidden_states.transpose(1,2)

        ff_hidden_states = ff_hidden_states + router_outputs
        ff_hidden_states = self.proj_out(ff_hidden_states)

        return ff_hidden_states


@maybe_allow_in_graph
class RaiFlowSingleTransformerBlock(nn.Module):
    r"""
    A Single Transformer block as part of the RaiFlow MMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        heads_per_group (`int`): The number of heads to use per conv goup.
        router_mult (`int`, *optional*, defaults to 2): The multiplier to use for the router feed forward hidden dimension.
        ff_mult (`int`, *optional*, defaults to 2): The multiplier to use for the conv feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to "dynamic_tanh"): The qk normalization to use in attention.
        eps (`float`, *optional*, defaults to 1e-05): The eps used with nn modules.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        heads_per_group: int,
        router_mult: int = 2,
        ff_mult: int = 2,
        is_2d: bool = True,
        dropout: float = 0.1,
        qk_norm: str = "dynamic_tanh",
        eps: float = 1e-05,
    ):
        super().__init__()

        if hasattr(F, "scaled_dot_product_attention"):
            processor = RaiFlowAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.norm_attn = DynamicTanh(dim=dim, init_alpha=0.2, elementwise_affine=True, bias=True)
        self.scale_attn = nn.Parameter(torch.ones(dim))
        self.shift_attn = nn.Parameter(torch.zeros(dim))

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
        self.scale_ff = nn.Parameter(torch.ones(dim))
        self.shift_ff = nn.Parameter(torch.zeros(dim))

        self.ff = RaiFlowFeedForward(
            dim=dim,
            dim_out=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            heads_per_group=heads_per_group,
            router_mult=router_mult,
            ff_mult=ff_mult,
            dropout=dropout,
            is_2d=is_2d,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        rotary_emb: Optional[Tuple[torch.FloatTensor]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.FloatTensor:
        norm_hidden_states = self.norm_attn(hidden_states)
        attn_output = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=None, rotary_emb=rotary_emb)
        attn_output = torch.addcmul(self.shift_attn, attn_output, self.scale_attn)
        hidden_states = hidden_states + attn_output


        norm_hidden_states = self.norm_ff(hidden_states)
        ff_output = self.ff(norm_hidden_states, height=height, width=width)
        ff_output = torch.addcmul(self.shift_ff, ff_output, self.scale_ff)
        hidden_states = hidden_states + ff_output

        return hidden_states


@maybe_allow_in_graph
class RaiFlowJointTransformerBlock(nn.Module):
    r"""
    A Joint Transformer block as part of the RaiFlow MMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        heads_per_group (`int`): The number of heads to use per conv goup.
        router_mult (`int`, *optional*, defaults to 2): The multiplier to use for the router feed forward hidden dimension.
        ff_mult (`int`, *optional*, defaults to 2): The multiplier to use for the conv feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to "dynamic_tanh"): The qk normalization to use in attention.
        eps (`float`, *optional*, defaults to 1e-05): The eps used with nn modules.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        heads_per_group: int,
        router_mult: int = 2,
        ff_mult: int = 2,
        is_2d: bool = True,
        dropout: float = 0.1,
        qk_norm: str = "dynamic_tanh",
        eps: float = 1e-05,
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

        self.scale_attn = nn.Parameter(torch.ones(dim))
        self.shift_attn = nn.Parameter(torch.zeros(dim))

        self.scale_attn_context = nn.Parameter(torch.ones(dim))
        self.shift_attn_context = nn.Parameter(torch.zeros(dim))

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
            heads_per_group=heads_per_group,
            ff_mult=ff_mult,
            is_2d=False,
            dropout=dropout,
            qk_norm=qk_norm,
            eps=eps,
        )

        self.latent_transformer = RaiFlowSingleTransformerBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            heads_per_group=heads_per_group,
            ff_mult=ff_mult,
            is_2d=is_2d,
            dropout=dropout,
            qk_norm=qk_norm,
            eps=eps,
        )


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        combined_rotary_emb: Tuple[torch.FloatTensor],
        image_rotary_emb: Tuple[torch.FloatTensor],
        text_rotary_emb: Tuple[torch.FloatTensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.FloatTensor:
        norm_hidden_states = self.norm_attn(hidden_states)
        norm_encoder_hidden_states = self.norm_attn_context(encoder_hidden_states)

        attn_output, context_attn_output = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states, rotary_emb=combined_rotary_emb)

        attn_output = torch.addcmul(self.shift_attn, attn_output, self.scale_attn)
        hidden_states = hidden_states + attn_output

        context_attn_output = torch.addcmul(self.shift_attn_context, context_attn_output, self.scale_attn_context)
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        hidden_states = self.latent_transformer(hidden_states, rotary_emb=image_rotary_emb, height=height, width=width)
        encoder_hidden_states = self.encoder_transformer(encoder_hidden_states, rotary_emb=text_rotary_emb)
        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class RaiFlowConditionalTransformer2DBlock(nn.Module):
    r"""
    A Conditional Transformer block as part of the RaiFlow MMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        heads_per_group (`int`): The number of heads to use per conv goup.
        router_mult (`int`, *optional*, defaults to 2): The multiplier to use for the router feed forward hidden dimension.
        ff_mult (`int`, *optional*, defaults to 2): The multiplier to use for the conv feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to "dynamic_tanh"): The qk normalization to use in attention.
        eps (`float`, *optional*, defaults to 1e-05): The eps used with nn modules.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        heads_per_group: int,
        router_mult: int = 2,
        ff_mult: int = 2,
        is_2d: bool = True,
        dropout: float = 0.1,
        qk_norm: str = "dynamic_tanh",
        eps: float = 1e-05,
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
        self.scale_cross_attn = nn.Parameter(torch.ones(dim))
        self.shift_cross_attn = nn.Parameter(torch.zeros(dim))

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
        self.scale_attn = nn.Parameter(torch.ones(dim))
        self.shift_attn = nn.Parameter(torch.zeros(dim))

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
        self.scale_ff = nn.Parameter(torch.ones(dim))
        self.shift_ff = nn.Parameter(torch.zeros(dim))

        self.ff = RaiFlowFeedForward(
            dim=dim,
            dim_out=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            heads_per_group=heads_per_group,
            router_mult=router_mult,
            ff_mult=ff_mult,
            dropout=dropout,
            is_2d=is_2d,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        image_rotary_emb: Tuple[torch.FloatTensor],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.FloatTensor:
        norm_hidden_states = self.norm_cross_attn(hidden_states)
        cross_attn_output = self.cross_attn(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        cross_attn_output = torch.addcmul(self.shift_cross_attn, cross_attn_output, self.scale_cross_attn)
        hidden_states = hidden_states + cross_attn_output

        norm_hidden_states = self.norm_attn(hidden_states)
        attn_output = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=None, rotary_emb=image_rotary_emb)
        attn_output = torch.addcmul(self.shift_attn, attn_output, self.scale_attn)
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm_ff(hidden_states)
        ff_output = self.ff(norm_hidden_states, height=height, width=width)
        ff_output = torch.addcmul(self.shift_ff, ff_output, self.scale_ff)
        hidden_states = hidden_states + ff_output

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
        heads_per_group (`int`, *optional*, defaults to 1): The number of heads to use per conv goup.
        encoder_in_channels (`int`, *optional*, defaults to 1536): The number of `encoder_hidden_states` dimensions to use.
        encoder_max_sequence_length (`int`, *optional*, defaults to 1024): The sequence lenght of the text encoder embeds.
            This is fixed during training since it is used to learn a number of position embeddings.
        num_train_timesteps (`int`, defaults to 1000): The number of diffusion steps to train the model.
            This is used to convert timesteps into flow-match sigmas.
        out_channels (`int`, defaults to 384): Number of output channels.
        patch_size (`int`, *optional*, (`int`, *optional*, defaults to 2):
            The size of each patch in the image. This parameter defines the resolution of patches fed into the model.
        router_mult (`int`, *optional*, defaults to 2): The multiplier to use for the router feed forward hidden dimension.
        ff_mult (`int`, *optional*, defaults to 2): The multiplier to use for the conv feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to "dynamic_tanh"): The qk normalization to use in attention.
        axes_dims_rope (Tuple[int], *optional*, defaults to (16, 24, 24)): The dimensions for rotart positional embeds.
            First dimension defines how much of the attention_head_dim won't be used for positional embeds.
            Rest of them are used with positional embeds. Mainly for the height and the width of the latents.
        eps (`float`, *optional*, defaults to 1e-05): The eps used with nn modules.
    """

    _supports_gradient_checkpointing = True
    _keep_in_fp32_modules = ["embedder", "unembedder", "scale_in", "shift_in", "scale_out", "shift_out", "norm_unembed"]

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
        heads_per_group: int = 1,
        num_train_timesteps: int = 1000,
        encoder_max_sequence_length: int = 1024,
        encoder_pad_to_multiple_of: int = 256,
        vocab_size: int = 151936,
        pad_token_id: int = 151643,
        out_channels: int = None,
        patch_size: int = 1,
        router_mult: int = 2,
        ff_mult: int = 2,
        dropout: float = 0.1,
        qk_norm: str = "dynamic_tanh",
        axes_dims_rope = (16, 24, 24),
        eps: float = 1e-05,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        self.out_channels = out_channels if out_channels is not None else self.config.in_channels
        self.out_channels = self.out_channels * self.config.patch_size*self.config.patch_size
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.base_seq_len = (self.config.sample_size // self.config.patch_size) * (self.config.sample_size // self.config.patch_size)
        self.patched_in_channels = 4 + ((self.config.in_channels + 4) * self.config.patch_size*self.config.patch_size) # patched + pos channels
        self.encoder_in_channels = self.inner_dim + 4 # pos channels

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=self.config.axes_dims_rope)
        self.postemb_embedder = nn.Linear(4 + (4 * self.config.patch_size*self.config.patch_size), self.patched_in_channels, bias=True)

        self.scale_in = nn.Parameter(torch.ones(self.config.in_channels))
        self.shift_in = nn.Parameter(torch.zeros(self.config.in_channels))

        self.embedder = RaiFlowFeedForward(
            dim=self.patched_in_channels,
            dim_out=self.inner_dim,
            num_attention_heads=self.config.num_attention_heads,
            attention_head_dim=self.config.attention_head_dim,
            heads_per_group=self.config.heads_per_group,
            router_mult=self.config.router_mult,
            ff_mult=self.config.ff_mult,
            dropout=dropout,
            is_2d=True,
        )

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.inner_dim, self.config.pad_token_id)
        self.context_embedder = RaiFlowFeedForward(
            dim=self.encoder_in_channels, # dim + pos
            dim_out=self.inner_dim,
            num_attention_heads=self.config.num_attention_heads,
            attention_head_dim=self.config.attention_head_dim,
            heads_per_group=self.config.heads_per_group,
            router_mult=self.config.router_mult,
            ff_mult=self.config.ff_mult,
            dropout=dropout,
            is_2d=False,
        )
        self.norm_context = DynamicTanh(dim=self.inner_dim, init_alpha=0.2, elementwise_affine=True, bias=True)

        self.joint_transformer_blocks = nn.ModuleList(
            [
                RaiFlowJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    heads_per_group=self.config.heads_per_group,
                    router_mult=self.config.router_mult,
                    ff_mult=self.config.ff_mult,
                    is_2d=True,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    eps=eps,
                )
                for _ in range(self.config.num_joint_layers)
            ]
        )

        self.cond_transformer_blocks = nn.ModuleList(
            [
                RaiFlowConditionalTransformer2DBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    heads_per_group=self.config.heads_per_group,
                    router_mult=self.config.router_mult,
                    ff_mult=self.config.ff_mult,
                    is_2d=True,
                    dropout=dropout,
                    qk_norm=qk_norm,
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
                    heads_per_group=self.config.heads_per_group,
                    router_mult=self.config.router_mult,
                    ff_mult=self.config.ff_mult,
                    is_2d=True,
                    dropout=dropout,
                    qk_norm=qk_norm,
                    eps=eps,
                )
                for _ in range(self.config.num_refiner_layers)
            ]
        )

        self.unembedder = RaiFlowFeedForward(
            dim=self.inner_dim,
            dim_out=self.out_channels,
            num_attention_heads=self.config.num_attention_heads,
            attention_head_dim=self.config.attention_head_dim,
            heads_per_group=self.config.heads_per_group,
            router_mult=self.config.router_mult,
            ff_mult=self.config.ff_mult,
            dropout=dropout,
            is_2d=True,
        )

        self.norm_unembed = DynamicTanh(dim=self.inner_dim, init_alpha=0.2, elementwise_affine=True, bias=True)
        self.scale_out = nn.Parameter(torch.ones(self.out_channels))
        self.shift_out = nn.Parameter(torch.zeros(self.out_channels))

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        combined_rotary_emb: Optional[Tuple[torch.FloatTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.FloatTensor]] = None,
        text_rotary_emb: Optional[Tuple[torch.FloatTensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        flip_target: bool = True,
        use_timesteps_sigmas: bool = True,
    ) -> Union[RaiFlowTransformer2DModelOutput, Tuple[torch.FloatTensor]]:
        """
        The [`RaiFlowTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, dim, height, width)`):
                The latent input.
            encoder_hidden_states (`torch.Tensor` of shape `(batch size, sequence_len)`):
                Input IDs from the tokenizer to use.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            combined_rotary_emb (Tuple[`torch.FloatTensor`] of shape `(combined_sequence_len, 3)`, *optional*):
                Used for rotary positional embeddings. combined_sequence_len is encoder_seq_len + latents_seq_len.
            image_rotary_emb (Tuple[`torch.FloatTensor`] of shape `(latents_seq_len, 3)`, *optional*):
                Used for rotary positional embeddings for the latents.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`RaiFlowTransformer2DModelOutput`] instead of a plain
                tuple.
            flip_target (`bool`, *optional*, defaults to `True`):
                Whether or not to flip the outputs for inference.

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

        device = hidden_states.device
        dtype = hidden_states.dtype

        batch_size, channels, height, width = hidden_states.shape
        _, encoder_seq_len = encoder_hidden_states.shape
        encoder_seq_len = encoder_seq_len + 2
        
        padded_height = height + 2
        padded_width = width + 2

        patched_height = padded_height // self.config.patch_size
        patched_width = padded_width // self.config.patch_size
        latents_seq_len = patched_height * patched_width

        if combined_rotary_emb is None:
            txt_ids = prepare_text_embed_ids(encoder_seq_len, device, dtype)
            img_ids = prepare_latent_image_ids(patched_height, patched_width, device, dtype)
            combined_ids = torch.cat((txt_ids, img_ids), dim=0)
            combined_rotary_emb = self.pos_embed(combined_ids, freqs_dtype=torch.float32)

        if image_rotary_emb is None:
            image_rotary_emb = (combined_rotary_emb[0][encoder_seq_len :], combined_rotary_emb[1][encoder_seq_len :])
        if text_rotary_emb is None:
            text_rotary_emb = (combined_rotary_emb[0][: encoder_seq_len], combined_rotary_emb[1][: encoder_seq_len])

        sigmas = timestep.to(dtype=dtype)
        if use_timesteps_sigmas:
            sigmas = sigmas / self.config.num_train_timesteps
        sigmas = sigmas.view(batch_size, 1, 1)
        sigmas_enc = sigmas.expand(batch_size, 1, self.inner_dim)

        posed_latents_2d = RaiFlowPosEmbed2D(
            batch_size=batch_size,
            height=padded_height,
            width=padded_width,
            device=device,
            dtype=dtype,
        )

        posed_latents_1d = RaiFlowPosEmbed1D(
            batch_size=batch_size,
            seq_len=latents_seq_len,
            device=device,
            dtype=dtype,
            secondary_seq_len=encoder_seq_len,
            base_seq_len=self.base_seq_len,
            sigmas=sigmas,
        )

        posed_encoder_1d = RaiFlowPosEmbed1D(
            batch_size=batch_size,
            seq_len=encoder_seq_len,
            device=device,
            dtype=dtype,
            secondary_seq_len=latents_seq_len,
            base_seq_len=self.config.encoder_max_sequence_length,
            sigmas=sigmas,
        )

        sigmas = sigmas.view(batch_size, 1, 1, 1)
        sigmas_h = sigmas.expand(batch_size, self.config.in_channels, 1, width).float()
        sigmas_w = sigmas.expand(batch_size, self.config.in_channels, padded_height, 1).float()

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            hidden_states = hidden_states.float() # embedder wants precision
            hidden_states = torch.addcmul(self.shift_in.view(1,-1,1,1), hidden_states, self.scale_in.view(1,-1,1,1))
            hidden_states = torch.cat([sigmas_h, hidden_states, sigmas_h], dim=2)
            hidden_states = torch.cat([sigmas_w, hidden_states, sigmas_w], dim=3)

            hidden_states = torch.cat([hidden_states, posed_latents_2d], dim=1)
            hidden_states = pack_2d_latents_to_1d(hidden_states, patch_size=self.config.patch_size)
            hidden_states = torch.cat([hidden_states, posed_latents_1d], dim=2)
            hidden_states = self.embedder(hidden_states, height=patched_height, width=patched_width).to(dtype=dtype)

        encoder_hidden_states = self.embed_tokens(encoder_hidden_states)
        encoder_hidden_states = torch.cat([sigmas_enc, encoder_hidden_states, sigmas_enc], dim=1)
        encoder_hidden_states = torch.cat([encoder_hidden_states, posed_encoder_1d], dim=2)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.joint_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    combined_rotary_emb,
                    image_rotary_emb,
                    text_rotary_emb,
                    patched_height,
                    patched_width,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    combined_rotary_emb=combined_rotary_emb,
                    image_rotary_emb=image_rotary_emb,
                    text_rotary_emb=text_rotary_emb,
                    height=patched_height,
                    width=patched_width,
                )

        encoder_hidden_states = self.norm_context(encoder_hidden_states)

        for index_block, block in enumerate(self.cond_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    image_rotary_emb,
                    patched_height,
                    patched_width,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    height=patched_height,
                    width=patched_width,
                )

        for index_block, block in enumerate(self.refiner_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    image_rotary_emb,
                    patched_height,
                    patched_width,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    rotary_emb=image_rotary_emb,
                    height=patched_height,
                    width=patched_width,
                )

        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            hidden_states = hidden_states.float() # optional extra precision for dct
            hidden_states = self.norm_unembed(hidden_states)
            hidden_states = self.unembedder(hidden_states, height=patched_height, width=patched_width)
            hidden_states = torch.addcmul(self.shift_out, hidden_states, self.scale_out)
            hidden_states = unpack_1d_latents_to_2d(hidden_states, patch_size=self.config.patch_size, original_height=padded_height, original_width=padded_width)
            output = hidden_states[:, :, 1:-1, 1:-1] # remove attention sinks

        if flip_target: # latents - noise to noise - latents
            output = -output

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, hidden_states, encoder_hidden_states)

        return RaiFlowTransformer2DModelOutput(sample=output, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
