from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.attention import FeedForward, _chunked_feed_forward
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from sotev3.sotev3_atten import SoteDiffusionV3AttnProcessor2_0, SoteDiffusionV3CrossAttnProcessor2_0
from sotev3.sotev3_embedder import SoteDiffusionV3PosEmbed1D, SoteDiffusionV3PosEmbed2D, SoteDiffusionV3PatchEmbed2D

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class SoteDiffusionV3SingleTransformerBlock(nn.Module):
    r"""
    A Single Transformer block as part of the Sote Diffusion V3 MMxDiT architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()

        if hasattr(F, "scaled_dot_product_attention"):
            processor = SoteDiffusionV3AttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", bias=True)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(self, hidden_states: torch.FloatTensor):
        attn_output = self.attn(hidden_states=hidden_states, encoder_hidden_states=None)
        hidden_states = hidden_states + attn_output.clamp(-32768,32768)
        hidden_states = hidden_states.clamp(-16384,16384)

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(hidden_states)

        hidden_states = hidden_states + ff_output.clamp(-32768,32768)
        hidden_states = hidden_states.clamp(-16384,16384)
        return hidden_states


@maybe_allow_in_graph
class SoteDiffusionV3JointTransformerBlock(nn.Module):
    r"""
     A Joint Transformer block as part of the Sote Diffusion V3 MMxDiT architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: Optional[str] = None,
        norm: Optional[str] = None,
    ):
        super().__init__()

        if hasattr(F, "scaled_dot_product_attention"):
            processor = SoteDiffusionV3AttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )

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
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", bias=True)
        self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", bias=True)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor):
        # Attention.
        attn_output, context_attn_output = self.attn(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)

        hidden_states = hidden_states + attn_output.clamp(-32768,32768)
        hidden_states = hidden_states.clamp(-16384,16384)

        encoder_hidden_states = encoder_hidden_states + context_attn_output.clamp(-32768,32768)
        encoder_hidden_states = encoder_hidden_states.clamp(-16384,16384)

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(hidden_states)

        hidden_states = hidden_states - ff_output.clamp(-32768,32768)
        hidden_states = hidden_states.clamp(-16384,16384)

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_context_output = _chunked_feed_forward(self.ff_context, encoder_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_context_output = self.ff_context(encoder_hidden_states)

        encoder_hidden_states = encoder_hidden_states + ff_context_output.clamp(-32768,32768)
        encoder_hidden_states = encoder_hidden_states.clamp(-16384,16384)

        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class SoteDiffusionV3CrossTransformerBlock(nn.Module):
    r"""
    A Cross Transformer block as part of the Sote Diffusion V3 MMxDiT architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        qk_norm: Optional[str] = None,
    ):
        super().__init__()

        if hasattr(F, "scaled_dot_product_attention"):
            cross_processor = SoteDiffusionV3CrossAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )


        self.cross_attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            processor=cross_processor,
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.single_transformer = SoteDiffusionV3SingleTransformerBlock(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            qk_norm=qk_norm,
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(self, hidden_states: torch.FloatTensor, secondary_hidden_states: torch.FloatTensor):
        cross_attn_output = self.cross_attn(hidden_states=hidden_states, secondary_hidden_states=secondary_hidden_states)
        hidden_states = hidden_states + cross_attn_output.clamp(-32768,32768)
        hidden_states = hidden_states.clamp(-16384,16384)

        hidden_states = self.single_transformer(hidden_states)
        return hidden_states


class SoteDiffusionV3Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    The Transformer model introduced in Sote Diffusion 3.

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        secondary_patch_size (`int`): Patch size to turn the input data into secondary small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of nMMDiT layers of Transformer blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of Single DiT layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        encoder_in_channels (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        out_channels (`int`, defaults to 16): Number of output channels.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        secondary_patch_size: int = 8,
        in_channels: int = 16,
        num_layers: int = 8,
        num_single_layers: int = 4,
        num_cross_layers: int = 36,
        attention_head_dim: int = 64,
        num_attention_heads: int = 48,
        encoder_in_channels: int = 4096,
        num_timesteps: int = 1000,
        out_channels: int = None,
        qk_norm: Optional[str] = None,
        norm: Optional[str] = None,
    ):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.embedder = nn.Linear((((in_channels+4) * patch_size*patch_size) + 4), self.inner_dim, bias=True)
        self.secondary_embedder = nn.Linear((((in_channels+4) * secondary_patch_size*secondary_patch_size) + 4), self.inner_dim, bias=True)

        self.context_embedder = nn.Linear((encoder_in_channels + 4), self.inner_dim, bias=True)
        self.context_embedder_norm = nn.LayerNorm(encoder_in_channels, eps=1e-6, elementwise_affine=True)



        self.single_transformer_blocks = nn.ModuleList(
            [
                SoteDiffusionV3SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    qk_norm=qk_norm,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.transformer_blocks = nn.ModuleList(
            [
                SoteDiffusionV3JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    qk_norm=qk_norm,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.cross_transformer_blocks = nn.ModuleList(
            [
                SoteDiffusionV3CrossTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    qk_norm=qk_norm,
                )
                for i in range(self.config.num_cross_layers)
            ]
        )

        self.unembedder = nn.Linear(self.inner_dim, self.out_channels*patch_size*patch_size, bias=True)

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
        block_controlnet_hidden_states: List = None,
        single_block_controlnet_hidden_states: List = None,
        cross_block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, dim, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step. Should be the sigmas with a range between 0 and 1.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            single_block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of single transformer blocks.
            cross_block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of cross transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

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

        batch_size, _, height, width = hidden_states.shape

        sigmas = timestep.view(batch_size,1).to(dtype=hidden_states.dtype) / self.config.num_timesteps
    
        secondary_hidden_states = SoteDiffusionV3PosEmbed2D(hidden_states)
        hidden_states = SoteDiffusionV3PatchEmbed2D(latents=secondary_hidden_states, sigmas=sigmas, patch_size=self.config.patch_size, embeds_seq_len=encoder_hidden_states.shape[1])
        hidden_states = self.embedder(hidden_states).clamp(-16384,16384)

        secondary_hidden_states = SoteDiffusionV3PatchEmbed2D(latents=secondary_hidden_states, sigmas=sigmas, patch_size=self.config.secondary_patch_size, embeds_seq_len=(hidden_states.shape[1]+encoder_hidden_states.shape[1]))
        secondary_hidden_states = self.secondary_embedder(secondary_hidden_states).clamp(-16384,16384)

        encoder_hidden_states = self.context_embedder_norm(encoder_hidden_states)
        encoder_hidden_states = SoteDiffusionV3PosEmbed1D(embeds=encoder_hidden_states, sigmas=sigmas, latents_seq_len=hidden_states.shape[1])
        encoder_hidden_states = self.context_embedder(encoder_hidden_states).clamp(-16384,16384)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                secondary_hidden_states = self._gradient_checkpointing_func(
                    block,
                    secondary_hidden_states,
                )
            else:
                secondary_hidden_states = block(
                    hidden_states=secondary_hidden_states,
                )
            # controlnet residual
            if single_block_controlnet_hidden_states is not None:
                interval_control = len(self.single_transformer_blocks) / len(single_block_controlnet_hidden_states)
                secondary_hidden_states = secondary_hidden_states + single_block_controlnet_hidden_states[int(index_block / interval_control)]

        for index_block, block in enumerate(self.transformer_blocks):

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

            # controlnet residual
            if block_controlnet_hidden_states is not None:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                hidden_states = hidden_states + block_controlnet_hidden_states[int(index_block / interval_control)]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.cross_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    secondary_hidden_states,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    secondary_hidden_states=secondary_hidden_states,
                )
            # controlnet residual
            if cross_block_controlnet_hidden_states is not None:
                interval_control = len(self.cross_transformer_blocks) / len(cross_block_controlnet_hidden_states)
                hidden_states = hidden_states + cross_block_controlnet_hidden_states[int(index_block / interval_control)]

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :]

        hidden_states = self.unembedder(hidden_states)
        output = hidden_states.reshape((batch_size, self.out_channels, height, width))

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
