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

from .raiflow_feedforward import RaiFlowFeedForward, DynamicTanh
from .raiflow_atten import RaiFlowAttnProcessor2_0, RaiFlowCrossAttnProcessor2_0
from .raiflow_embedder import RaiFlowLatentEmbedder, RaiFlowTextEmbedder, RaiFlowLatentUnembedder, prepare_latent_image_ids, prepare_text_embed_ids, FluxPosEmbed
from .raiflow_pipeline_output import RaiFlowTransformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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
        embedding_dim: int = None,
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
        self.embedding_dim = embedding_dim or self.inner_dim
        self.base_seq_len = (self.config.sample_size // self.config.patch_size) * (self.config.sample_size // self.config.patch_size)
        self.patched_in_channels = 4 + ((self.config.in_channels + 4) * self.config.patch_size*self.config.patch_size) # patched + pos channels
        self.encoder_in_channels = 4 + self.embedding_dim # pos channels

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=self.config.axes_dims_rope)

        self.embedder = RaiFlowLatentEmbedder(
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            base_seq_len=self.base_seq_len,
            dim=self.patched_in_channels,
            dim_out=self.inner_dim,
            num_attention_heads=self.config.num_attention_heads,
            attention_head_dim=self.config.attention_head_dim,
            heads_per_group=self.config.heads_per_group,
            router_mult=self.config.router_mult * 2,
            ff_mult=self.config.ff_mult * 2,
            dropout=dropout,
        )

        self.text_embedder = RaiFlowTextEmbedder(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.embedding_dim,
            pad_token_id=self.config.pad_token_id,
            base_seq_len=self.config.encoder_max_sequence_length,
            dim=self.encoder_in_channels, # dim + pos
            dim_out=self.inner_dim,
            num_attention_heads=self.config.num_attention_heads,
            attention_head_dim=self.config.attention_head_dim,
            heads_per_group=self.config.heads_per_group,
            router_mult=self.config.router_mult,
            ff_mult=self.config.ff_mult,
            dropout=dropout,
        )

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

        self.norm_context = DynamicTanh(dim=self.inner_dim, init_alpha=0.2, elementwise_affine=True, bias=True)

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

        self.unembedder = RaiFlowLatentUnembedder(
            patch_size=self.config.patch_size,
            dim=self.inner_dim,
            dim_out=self.out_channels,
            num_attention_heads=self.config.num_attention_heads,
            attention_head_dim=self.config.attention_head_dim,
            heads_per_group=self.config.heads_per_group,
            router_mult=self.config.router_mult * 2,
            ff_mult=self.config.ff_mult * 2,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.FloatTensor,
        combined_rotary_emb: Optional[Tuple[torch.FloatTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.FloatTensor]] = None,
        text_rotary_emb: Optional[Tuple[torch.FloatTensor]] = None,
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

        use_checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing
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

        timestep = timestep.view(batch_size, 1, 1)

        if combined_rotary_emb is None:
            txt_ids = prepare_text_embed_ids(encoder_seq_len, device, dtype)
            img_ids = prepare_latent_image_ids(patched_height, patched_width, device, dtype)
            combined_ids = torch.cat((txt_ids, img_ids), dim=0)
            combined_rotary_emb = self.pos_embed(combined_ids, freqs_dtype=torch.float32)

        if image_rotary_emb is None:
            image_rotary_emb = (combined_rotary_emb[0][encoder_seq_len :], combined_rotary_emb[1][encoder_seq_len :])
        if text_rotary_emb is None:
            text_rotary_emb = (combined_rotary_emb[0][: encoder_seq_len], combined_rotary_emb[1][: encoder_seq_len])

        if use_checkpointing:
            encoder_hidden_states = self._gradient_checkpointing_func(self.text_embedder, encoder_hidden_states, timestep, latents_seq_len, encoder_seq_len)
        else:
            encoder_hidden_states = self.text_embedder(encoder_hidden_states=encoder_hidden_states, timestep=timestep, latents_seq_len=latents_seq_len, encoder_seq_len=encoder_seq_len)

        if use_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                self.embedder,
                hidden_states,
                timestep,
                latents_seq_len,
                encoder_seq_len,
                padded_height,
                padded_width,
                patched_height,
                patched_width,
            )
        else:
            hidden_states = self.embedder(
                hidden_states=hidden_states,
                timestep=timestep,
                latents_seq_len=latents_seq_len,
                encoder_seq_len=encoder_seq_len,
                padded_height=padded_height,
                padded_width=padded_width,
                patched_height=patched_height,
                patched_width=patched_width,
            ).to(dtype=dtype)

        for index_block, block in enumerate(self.joint_transformer_blocks):
            if use_checkpointing:
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

        self.norm_context(encoder_hidden_states)

        for index_block, block in enumerate(self.cond_transformer_blocks):
            if use_checkpointing:
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
            if use_checkpointing:
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

        if use_checkpointing:
            output = self._gradient_checkpointing_func(self.unembedder, hidden_states, padded_height, padded_width, patched_height, patched_width)
        else:
            output = self.unembedder(hidden_states, padded_height=padded_height, padded_width=padded_width, patched_height=patched_height, patched_width=patched_width)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, hidden_states, encoder_hidden_states)

        return RaiFlowTransformer2DModelOutput(sample=output, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
