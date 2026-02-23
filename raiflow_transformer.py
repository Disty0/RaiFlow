from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.configuration_utils import ConfigMixin, register_to_config

from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.loaders import PeftAdapterMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
fp16_max = 65504


def get_raiflow_pos_embed_1d(batch_size: int, seq_len: int, secondary_seq_len: int, base_seq_len: int, max_freqs: int, is_latent: bool, device: torch.device, dtype: torch.dtype) -> torch.FloatTensor:
    global_pos = torch.linspace(start=0, end=1, steps=(seq_len + secondary_seq_len), device=device, dtype=dtype)
    if is_latent:
        global_pos = global_pos[secondary_seq_len:]
    else:
        global_pos = global_pos[:seq_len]

    pos_embeds = torch.stack([torch.linspace(start=0, end=1, steps=seq_len, device=device, dtype=dtype), global_pos], dim=-1)
    del global_pos

    # batch_size, seq_len, channels, dct
    pos_embeds = torch.cat([pos_embeds, torch.full((seq_len, 1), (seq_len / base_seq_len), device=device, dtype=dtype)], dim=-1)
    pos_embeds = torch.pi * pos_embeds.unsqueeze(-1) * (2 ** torch.linspace(0, max_freqs - 1, max_freqs, device=device, dtype=dtype))
    pos_embeds = pos_embeds.flatten(-2,-1)

    # batch_size, seq_len, (channels * max_freqs * 2)
    pos_embeds = torch.cat([torch.sin(pos_embeds), torch.cos(pos_embeds)], dim=-1)
    pos_embeds = pos_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
    return pos_embeds


def get_raiflow_pos_embed_2d(batch_size: int, height: int, width: int, max_freqs: int, device: torch.device, dtype: torch.dtype) -> torch.FloatTensor:
    pos_x, pos_y = torch.meshgrid(
        torch.linspace(0, 1, height, device=device, dtype=dtype),
        torch.linspace(0, 1, width, device=device, dtype=dtype),
        indexing="ij",
    )

    max_dim_linspace = torch.linspace(0, 1, max(width, height), device=device, dtype=dtype)
    relative_pos_x, relative_pos_y = torch.meshgrid(max_dim_linspace[:height], max_dim_linspace[:width], indexing="ij")
    del max_dim_linspace

    # channels, height, width -> channels, dct_x, dct_y, height, width
    pos_x = torch.stack([pos_x, relative_pos_x], dim=0).view(2, 1, 1, height, width)
    pos_y = torch.stack([pos_y, relative_pos_y], dim=0).view(2, 1, 1, height, width)
    del relative_pos_x, relative_pos_y

    # channels, dct_x, dct_y, height, width
    freqs = torch.pi * (2 ** torch.linspace(0, max_freqs - 1, max_freqs, device=device, dtype=dtype))

    pos_embeds = torch.mul(
        torch.cos(pos_x * freqs.view(1, max_freqs, 1, 1, 1)),
        torch.cos(pos_y * freqs.view(1, 1, max_freqs, 1, 1)),
    )
    del freqs

    # batch_size, (channels * max_freqs ** 2), height, width
    pos_embeds = pos_embeds.flatten(0,2).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return pos_embeds


class RaiFlowLatentEmbedder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        base_seq_len: int,
        max_freqs: int,
        inner_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.base_seq_len = base_seq_len
        self.max_freqs = max_freqs

        dim_in = (3 * self.max_freqs * 2) + ((in_channels + (2 * self.max_freqs ** 2)) * (self.patch_size ** 2))
        self.latent_embedder_proj = nn.Linear(dim_in, inner_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        dtype: torch.dtype,
        latents_seq_len: int,
        encoder_seq_len: int,
        batch_size: int,
        height: int,
        width: int,
    ) -> torch.FloatTensor:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False): # force fp32
            with torch.no_grad():
                pos_embed_2d = get_raiflow_pos_embed_2d(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    max_freqs=self.max_freqs,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                pos_embed_1d = get_raiflow_pos_embed_1d(
                    batch_size=batch_size,
                    seq_len=latents_seq_len,
                    secondary_seq_len=encoder_seq_len,
                    base_seq_len=self.base_seq_len,
                    max_freqs=self.max_freqs,
                    is_latent=True,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )

            hidden_states = torch.cat([hidden_states.to(dtype=torch.float32), pos_embed_2d], dim=1)
            hidden_states = torch.nn.functional.pixel_unshuffle(hidden_states, self.patch_size)
            hidden_states = hidden_states.flatten(-2,-1).transpose(-1,-2).contiguous()
            hidden_states = torch.cat([hidden_states, pos_embed_1d], dim=2)
            hidden_states = self.latent_embedder_proj(hidden_states).to(dtype=dtype)
            return hidden_states


class RaiFlowTextEmbedder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pad_token_id: int,
        base_seq_len: int,
        max_freqs: int,
        inner_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.base_seq_len = base_seq_len
        self.max_freqs = max_freqs

        dim_in = (3 * self.max_freqs * 2) + self.embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, pad_token_id)
        self.text_embedder_proj = nn.Conv1d(dim_in, inner_dim, 3, padding=1, bias=bias)

    def forward(
        self,
        encoder_hidden_states: torch.FloatTensor,
        dtype: torch.dtype,
        latents_seq_len: int,
        encoder_seq_len: int,
        batch_size: int,
    ) -> torch.FloatTensor:
        encoder_hidden_states = self.token_embedding(encoder_hidden_states)
        with torch.autocast(device_type=encoder_hidden_states.device.type, enabled=False): # force fp32
            with torch.no_grad():
                pos_embed_1d = get_raiflow_pos_embed_1d(
                    batch_size=batch_size,
                    seq_len=encoder_seq_len,
                    secondary_seq_len=latents_seq_len,
                    base_seq_len=self.base_seq_len,
                    max_freqs=self.max_freqs,
                    is_latent=False,
                    device=encoder_hidden_states.device,
                    dtype=torch.float32,
                )

            encoder_hidden_states = torch.cat([encoder_hidden_states.to(dtype=torch.float32), pos_embed_1d], dim=2)
            encoder_hidden_states = self.text_embedder_proj(encoder_hidden_states.transpose(-1,-2)).transpose(-1,-2)
            encoder_hidden_states = encoder_hidden_states.to(dtype=dtype, memory_format=torch.contiguous_format)
            return encoder_hidden_states


class RaiFlowLatentUnembedder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        inner_dim: int,
        out_channels: int,
        eps: float = 1e-5,
        bias: bool = True,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size

        dim_out = out_channels * (self.patch_size ** 2)
        self.norm_unembed = nn.RMSNorm(inner_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.unembedder_proj = nn.Linear(inner_dim, dim_out, bias=bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        height: int,
        width: int,
    ) -> torch.FloatTensor:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False): # force fp32
            hidden_states = self.norm_unembed(hidden_states.clamp(-fp16_max, fp16_max).to(dtype=torch.float32))
            hidden_states = self.unembedder_proj(hidden_states)
            hidden_states = hidden_states.transpose(-1,-2).unflatten(-1, (height//self.patch_size, width//self.patch_size))
            hidden_states = torch.nn.functional.pixel_shuffle(hidden_states, self.patch_size)
            return hidden_states


class RaiFlowAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: "RaiFlowAttention", hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        query, key, value, attn_gate = attn.attn_in(hidden_states).unflatten(-1, (-1, attn.head_dim)).chunk(4, dim=-2)
        query = attn.norm_q(query.clamp(-fp16_max, fp16_max))
        key = attn.norm_k(key.clamp(-fp16_max, fp16_max))

        hidden_states = dispatch_attention_fn(query, key, value)
        del query, key, value

        attn_gate = attn.attn_gate(attn_gate)
        hidden_states = torch.mul(attn_gate, hidden_states).flatten(-2, -1).contiguous()
        del attn_gate

        hidden_states = attn.attn_out(hidden_states)
        return hidden_states


class RaiFlowAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = RaiFlowAttnProcessor
    _available_processors = [RaiFlowAttnProcessor]
    _supports_qkv_fusion = False

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        input_dim: int = None,
        output_dim: int = None,
        dropout: float = 0.1,
        eps: float = 1e-5,
        bias: bool = False,
        elementwise_affine: bool = False,
        processor: RaiFlowAttnProcessor = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.inner_dim = self.head_dim * self.num_heads
        self.input_dim = input_dim if input_dim is not None else self.inner_dim
        self.output_dim = output_dim if output_dim is not None else self.input_dim

        self.attn_in = nn.Linear(self.input_dim, self.inner_dim*4, bias=bias)
        self.attn_out = nn.Linear(self.inner_dim, self.output_dim, bias=bias)
        self.attn_gate = nn.Sequential(nn.Sigmoid(), nn.Dropout(dropout))

        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=elementwise_affine)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(self, hidden_states: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        return self.processor(self, hidden_states, **kwargs)


class RaiFlowFeedForward(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = None, inner_dim: int = None, ff_mult: int = 4, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.ff_mult = ff_mult
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.inner_dim = (inner_dim if inner_dim is not None else self.input_dim) * self.ff_mult

        self.ff_in = nn.Linear(self.input_dim, self.inner_dim*2, bias=bias)
        self.ff_out = nn.Linear(self.inner_dim, self.output_dim, bias=bias)
        self.ff_gate = nn.Sequential(nn.GELU(approximate="none"), nn.Dropout(dropout))

    def forward(self, hidden_states):
        hidden_states, ff_gate = self.ff_in(hidden_states).chunk(2, dim=-1)
        ff_gate = self.ff_gate(ff_gate)
        hidden_states = torch.mul(ff_gate, hidden_states).contiguous()
        del ff_gate

        hidden_states = self.ff_out(hidden_states)
        return hidden_states


class RaiFlowTransformerBlock(nn.Module):
    r"""
    A Transformer block as part of the RaiFlow architecture.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.1): The dropout probability to use.
        eps (`float`, *optional*, defaults to 1e-5): The eps used with nn modules.
    """

    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        inner_dim: int = None,
        ff_mult: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-5,
        bias: bool = False,
        elementwise_affine: bool = False,
    ):
        super().__init__()
        self.inner_dim = inner_dim if inner_dim is not None else (num_attention_heads * attention_head_dim)

        self.ff = RaiFlowFeedForward(self.inner_dim, ff_mult=ff_mult, dropout=dropout, bias=bias)
        self.attn = RaiFlowAttention(num_attention_heads, attention_head_dim, input_dim=inner_dim, dropout=dropout, eps=eps, bias=bias, elementwise_affine=elementwise_affine)

        self.norm_ff = nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=elementwise_affine)
        self.norm_attn = nn.RMSNorm(self.inner_dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = hidden_states.clamp(-fp16_max, fp16_max)
        hidden_states = hidden_states + self.attn(self.norm_attn(hidden_states))
        hidden_states = hidden_states + self.ff(self.norm_ff(hidden_states))
        return hidden_states


class RaiFlowTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, CacheMixin, AttentionMixin):
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
        "norm_unembed", "norm_ff", "norm_attn", "norm_q", "norm_k", "norm", "bias",
    ]
    _keep_in_fp32_modules = ["latent_embedder", "unembedder", "text_embedder_proj", "norm_unembed"]

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
        ff_mult: int = 4,
        patch_size: int = 1,
        dropout: float = 0.1,
        eps: float = 1e-5,
        bias: bool = True,
        embedder_bias: bool = True,
        elementwise_affine: bool = True,
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
            bias=self.config.embedder_bias,
        )

        self.text_embedder = RaiFlowTextEmbedder(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.embedding_dim,
            pad_token_id=self.config.pad_token_id,
            base_seq_len=self.config.encoder_max_sequence_length,
            max_freqs=self.config.max_freqs,
            inner_dim=self.inner_dim,
            bias=self.config.embedder_bias,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                RaiFlowTransformerBlock(
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_mult=self.config.ff_mult,
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
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    ff_mult=self.config.ff_mult,
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
    ) -> Union[Transformer2DModelOutput, Tuple[torch.FloatTensor]]:
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
                Whether or not to return a [`Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`Transformer2DModelOutput`] is returned, otherwise a
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

        return Transformer2DModelOutput(sample=output)
