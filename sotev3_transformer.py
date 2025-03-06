from typing import Any, Dict, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward, _chunked_feed_forward
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from .sotev3_atten import SoteDiffusionV3AttnProcessor2_0, SoteDiffusionV3CrossAttnProcessor2_0
from .sotev3_embedder import SoteDiffusionV3PosEmbed1D, SoteDiffusionV3PosEmbed2D
from .sotev3_pipeline_output import SoteDiffusionV3Transformer2DModelOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class SoteDiffusionV3SingleTransformer1DBlock(nn.Module):
    r"""
    A Single Transformer block as part of the Sote Diffusion V3 EMMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        eps (`float`, *optional*, defaults to 0.1): The eps used with nn modules.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the linear feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to None): The qk normalization to use in attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-05,
        ff_mult: int = 4,
        dropout: float = 0.1,
        qk_norm: str = None,
    ):
        super().__init__()

        if hasattr(F, "scaled_dot_product_attention"):
            processor = SoteDiffusionV3AttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
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
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm_ff = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = FeedForward(dim=dim, dim_out=dim, mult=ff_mult, dropout=dropout, activation_fn="gelu-approximate", bias=True)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        norm_hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states + self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=None)

        norm_hidden_states = self.norm_ff(hidden_states)
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + ff_output

        return hidden_states


@maybe_allow_in_graph
class SoteDiffusionV3EncoderTransformerBlock(nn.Module):
    r"""
    An Encoder Transformer block as part of the Sote Diffusion V3 EMMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        eps (`float`, *optional*, defaults to 0.1): The eps used with nn modules.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the linear feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to None): The qk normalization to use in attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-05,
        ff_mult: int = 4,
        dropout: float = 0.1,
        qk_norm: str = None,
    ):
        super().__init__()

        if hasattr(F, "scaled_dot_product_attention"):
            processor = SoteDiffusionV3AttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_context = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)

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
            qk_norm=qk_norm,
            eps=eps,
        )

        self.encoder_transformer = SoteDiffusionV3SingleTransformer1DBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            eps=eps,
            ff_mult=ff_mult,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        self.latent_transformer = SoteDiffusionV3SingleTransformer1DBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            eps=eps,
            ff_mult=ff_mult,
            dropout=dropout,
            qk_norm=qk_norm,
        )


    def forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        norm_hidden_states = self.norm(hidden_states)
        norm_encoder_hidden_states = self.norm_context(encoder_hidden_states)

        attn_output, context_attn_output = self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states)

        hidden_states = hidden_states + attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        hidden_states = self.latent_transformer(hidden_states)
        encoder_hidden_states = self.encoder_transformer(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class SoteDiffusionV3ConditionalTransformer2DBlock(nn.Module):
    r"""
    A Conditional Transformer block as part of the Sote Diffusion V3 EMMDit architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        eps (`float`, *optional*, defaults to 0.1): The eps used with nn modules.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the linear feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to None): The qk normalization to use in attention.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-05,
        ff_mult: int = 4,
        dropout: float = 0.1,
        qk_norm: str = None,
    ):
        super().__init__()

        if hasattr(F, "scaled_dot_product_attention"):
            cross_processor = SoteDiffusionV3CrossAttnProcessor2_0()
            processor = SoteDiffusionV3AttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.norm_cross_attn = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_cross_attn_context = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
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
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm_attn = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
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
            qk_norm=qk_norm,
            eps=eps,
        )

        self.norm_ff = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.norm_ff_secondary = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)

        self.ff = nn.Sequential(
            nn.Linear(dim*2, dim*ff_mult, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(dim*ff_mult, dim, bias=True),
        )

    def forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, secondary_hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        norm_hidden_states = self.norm_cross_attn(hidden_states)
        norm_encoder_hidden_states = self.norm_cross_attn_context(encoder_hidden_states)
        hidden_states = hidden_states + self.cross_attn(hidden_states=norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states)

        norm_hidden_states = self.norm_attn(hidden_states)
        hidden_states = hidden_states + self.attn(hidden_states=norm_hidden_states, encoder_hidden_states=None)

        norm_hidden_states = self.norm_ff(hidden_states)
        norm_secondary_hidden_states = self.norm_ff_secondary(secondary_hidden_states)

        norm_hidden_states = torch.cat([norm_hidden_states, norm_secondary_hidden_states], dim=-1)
        hidden_states = hidden_states + self.ff(norm_hidden_states)

        return hidden_states


class SoteDiffusionV3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    The Multi Modal Convoluted Transformer model introduced in Sote Diffusion 3.

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        in_channels (`int`, *optional*, defaults to 384): The number of channels in the input.
        num_eps_encoder_layers (`int`, *optional*, defaults to 4): The number of eps encoder layers of Transformer blocks to use.
        num_x0_encoder_layers (`int`, *optional*, defaults to 4): The number of x0 encoder layers of Transformer blocks to use.
        num_conditional_layers (`int`, *optional*, defaults to 24): The number of single layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 32): The number of heads to use for multi-head attention.
        encoder_in_channels (`int`, *optional*, defaults to 1536): The number of `encoder_hidden_states` dimensions to use.
        out_channels (`int`, defaults to 384): Number of output channels.
        eps (`float`, *optional*, defaults to 0.1): The eps used with nn modules.
        ff_mult (`int`, *optional*, defaults to 4): The multiplier to use for the linear feed forward hidden dimension.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        qk_norm (`str`, *optional*, defaults to None): The qk normalization to use in attention.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 64,
        in_channels: int = 384,
        num_eps_encoder_layers: int = 4,
        num_x0_encoder_layers: int = 4,
        num_eps_layers: int = 24,
        num_x0_layers: int = 24,
        attention_head_dim: int = 64,
        num_attention_heads: int = 32,
        encoder_in_channels: int = 1536,
        encoder_base_seq_len: int = 1024,
        num_train_timesteps: int = 1000,
        out_channels: int = None,
        eps: float = 1e-05,
        ff_mult: int = 4,
        dropout: float = 0.1,
        qk_norm: str = None,
    ):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.num_entagled_layers = min(self.config.num_eps_layers, self.config.num_x0_layers)

        if self.config.num_eps_layers > self.num_entagled_layers:
            self.last_layer_type = "eps"
            self.num_last_layers = self.config.num_eps_layers - self.num_entagled_layers
        elif self.config.num_x0_layers > self.num_entagled_layers:
            self.last_layer_type = "x0"
            self.num_last_layers = self.config.num_x0_layers - self.num_entagled_layers
        else:
            self.last_layer_type = "none"
            self.num_last_layers = 0

        self.eps_embedder = nn.Linear((in_channels + 8), self.inner_dim, bias=True)
        self.x0_embedder = nn.Linear((in_channels + 8), self.inner_dim, bias=True)

        self.eps_context_embedder = nn.Linear((encoder_in_channels + 4), self.inner_dim, bias=True)
        self.x0_context_embedder = nn.Linear((encoder_in_channels + 4), self.inner_dim, bias=True)

        self.eps_encoder_blocks = nn.ModuleList(
            [
                SoteDiffusionV3EncoderTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_mult=ff_mult,
                    eps=eps,
                    dropout=dropout,
                    qk_norm=qk_norm,
                )
                for _ in range(self.config.num_eps_encoder_layers)
            ]
        )

        self.x0_encoder_blocks = nn.ModuleList(
            [
                SoteDiffusionV3EncoderTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_mult=ff_mult,
                    eps=eps,
                    dropout=dropout,
                    qk_norm=qk_norm,
                )
                for _ in range(self.config.num_x0_encoder_layers)
            ]
        )

        self.eps_transformer_blocks = nn.ModuleList(
            [
                SoteDiffusionV3ConditionalTransformer2DBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_mult=ff_mult,
                    eps=eps,
                    dropout=dropout,
                    qk_norm=qk_norm,
                )
                for _ in range(self.config.num_eps_layers)
            ]
        )

        self.x0_transformer_blocks = nn.ModuleList(
            [
                SoteDiffusionV3ConditionalTransformer2DBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_mult=ff_mult,
                    eps=eps,
                    dropout=dropout,
                    qk_norm=qk_norm,
                )
                for _ in range(self.config.num_x0_layers)
            ]
        )

        self.eps_unembedder = nn.Linear(self.inner_dim, self.out_channels, bias=True)
        self.x0_unembedder = nn.Linear(self.inner_dim, self.out_channels, bias=True)

        self.gradient_checkpointing = False

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, SoteDiffusionV3Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, dim, height, width)`):
                JPEG latent input.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`SoteDiffusionV3Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`SoteDiffusionV3Transformer2DModelOutput`] is returned, otherwise a
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
        latents_seq_len = height * width

        sigmas = timestep.to(dtype=hidden_states.dtype) / self.config.num_train_timesteps

        hidden_states = SoteDiffusionV3PosEmbed2D(latents=hidden_states, sigmas=sigmas)
        hidden_states = hidden_states.view(batch_size, (channels + 4), latents_seq_len).transpose(1,2)

        hidden_states = SoteDiffusionV3PosEmbed1D(
            embeds=hidden_states,
            sigmas=sigmas,
            secondary_seq_len=encoder_hidden_states.shape[1],
            base_seq_len=(self.config.sample_size*self.config.sample_size)
        )

        eps_hidden_states = self.eps_embedder(hidden_states)
        x0_hidden_states = self.x0_embedder(hidden_states)

        encoder_hidden_states = SoteDiffusionV3PosEmbed1D(
            embeds=encoder_hidden_states,
            sigmas=sigmas,
            secondary_seq_len=latents_seq_len,
            base_seq_len=self.config.encoder_base_seq_len
        )

        eps_encoder_hidden_states = self.eps_context_embedder(encoder_hidden_states)
        x0_encoder_hidden_states = self.x0_context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.eps_encoder_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                eps_hidden_states, eps_encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    eps_hidden_states,
                    eps_encoder_hidden_states,
                )
            else:
                eps_hidden_states, eps_encoder_hidden_states = block(
                    hidden_states=eps_hidden_states,
                    encoder_hidden_states=eps_encoder_hidden_states,
                )

        for index_block, block in enumerate(self.x0_encoder_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x0_hidden_states, x0_encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    x0_hidden_states,
                    x0_encoder_hidden_states,
                )
            else:
                x0_hidden_states, x0_encoder_hidden_states = block(
                    hidden_states=x0_hidden_states,
                    encoder_hidden_states=x0_encoder_hidden_states,
                )

        for index_block in range(self.num_entagled_layers):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                eps_hidden_states = self._gradient_checkpointing_func(
                    self.eps_transformer_blocks[index_block],
                    eps_hidden_states,
                    eps_encoder_hidden_states,
                    x0_hidden_states,
                )
                x0_hidden_states = self._gradient_checkpointing_func(
                    self.x0_transformer_blocks[index_block],
                    x0_hidden_states,
                    x0_encoder_hidden_states,
                    eps_hidden_states,
                )
            else:
                eps_hidden_states = self.eps_transformer_blocks[index_block](
                    hidden_states=eps_hidden_states,
                    encoder_hidden_states=eps_encoder_hidden_states,
                    secondary_hidden_states=x0_hidden_states,
                )
                x0_hidden_states = self.x0_transformer_blocks[index_block](
                    hidden_states=x0_hidden_states,
                    encoder_hidden_states=x0_encoder_hidden_states,
                    secondary_hidden_states=eps_hidden_states,
                )

        if self.last_layer_type == "eps":
            for index_block in range(self.num_last_layers):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    eps_hidden_states = self._gradient_checkpointing_func(
                        self.eps_transformer_blocks[index_block + self.num_entagled_layers],
                        eps_hidden_states,
                        eps_encoder_hidden_states,
                        x0_hidden_states,
                    )
                else:
                    eps_hidden_states = self.eps_transformer_blocks[index_block + self.num_entagled_layers](
                        hidden_states=eps_hidden_states,
                        encoder_hidden_states=eps_encoder_hidden_states,
                        secondary_hidden_states=x0_hidden_states,
                    )
        elif self.last_layer_type == "x0":
            for index_block in range(self.num_last_layers):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    x0_hidden_states = self._gradient_checkpointing_func(
                        self.x0_transformer_blocks[index_block + self.num_entagled_layers],
                        x0_hidden_states,
                        x0_encoder_hidden_states,
                        eps_hidden_states,
                    )
                else:
                    x0_hidden_states = self.x0_transformer_blocks[index_block + self.num_entagled_layers](
                        hidden_states=x0_hidden_states,
                        encoder_hidden_states=x0_encoder_hidden_states,
                        secondary_hidden_states=eps_hidden_states,
                    )

        eps_pred = self.eps_unembedder(eps_hidden_states).transpose(1,2).view(batch_size, channels, height, width)
        x0_pred = self.x0_unembedder(x0_hidden_states).transpose(1,2).view(batch_size, channels, height, width)
        flow_pred = eps_pred - x0_pred

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (flow_pred, eps_pred, x0_pred)

        return SoteDiffusionV3Transformer2DModelOutput(sample=flow_pred, noise=eps_pred, latent=x0_pred)
