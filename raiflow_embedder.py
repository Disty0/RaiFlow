import torch
from torch import nn

from diffusers.utils import logging

from .raiflow_layers import RaiFlowFeedForward, DynamicTanh

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class RaiFlowLatentEmbedder(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, base_seq_len: int, dim: int, dim_out: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.base_seq_len = base_seq_len
        self.scale_latent = nn.Parameter(torch.ones((1, self.in_channels, 1, 1)))
        self.shift_latent = nn.Parameter(torch.zeros((1, self.in_channels, 1, 1)))
        self.latent_embedder = RaiFlowFeedForward(dim=dim, dim_out=dim_out, ff_mult=ff_mult, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        timestep: torch.FloatTensor,
        dtype: torch.dtype,
        latents_seq_len: int,
        encoder_seq_len: int,
        batch_size: int,
        height: int,
        width: int,
    ):
        with torch.no_grad():
            posed_latents_2d = RaiFlowPosEmbed2D(
                batch_size=batch_size,
                height=height,
                width=width,
                device=hidden_states.device,
                dtype=dtype,
            )
            posed_latents_1d = RaiFlowPosEmbed1D(
                batch_size=batch_size,
                seq_len=latents_seq_len,
                device=hidden_states.device,
                dtype=dtype,
                secondary_seq_len=encoder_seq_len,
                base_seq_len=self.base_seq_len,
                timestep=timestep,
                is_latent=True,
            )

        hidden_states = torch.addcmul(self.shift_latent, hidden_states, self.scale_latent)
        hidden_states = torch.cat([hidden_states, posed_latents_2d], dim=1)
        hidden_states = pack_2d_latents_to_1d(hidden_states, patch_size=self.patch_size)
        hidden_states = torch.cat([hidden_states, posed_latents_1d], dim=2)
        hidden_states = self.latent_embedder(hidden_states)
        return hidden_states


class RaiFlowTextEmbedder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pad_token_id: int,
        base_seq_len: int,
        dim: int,
        dim_out: int,
        ff_mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.base_seq_len = base_seq_len
        self.embed_tokens = nn.Embedding(vocab_size, embedding_dim, pad_token_id)
        self.text_embedder = RaiFlowFeedForward(dim=dim, dim_out=dim_out, ff_mult=ff_mult, dropout=dropout)

    def forward(
        self,
        encoder_hidden_states: torch.FloatTensor,
        timestep: torch.FloatTensor,
        dtype: torch.dtype,
        latents_seq_len: int,
        encoder_seq_len: int,
        batch_size: int,
    ):
        with torch.no_grad():
            posed_encoder_1d = RaiFlowPosEmbed1D(
                batch_size=batch_size,
                seq_len=encoder_seq_len,
                device=encoder_hidden_states.device,
                dtype=dtype,
                secondary_seq_len=latents_seq_len,
                base_seq_len=self.base_seq_len,
                timestep=timestep,
                is_latent=True,
            )

        encoder_hidden_states = self.embed_tokens(encoder_hidden_states)
        encoder_hidden_states = torch.cat([encoder_hidden_states, posed_encoder_1d], dim=2)
        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        return encoder_hidden_states


class RaiFlowLatentUnembedder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        dim: int,
        dim_out: int,
        ff_mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.patch_size = patch_size
        self.scale_latent_out = nn.Parameter(torch.ones(dim_out))
        self.shift_latent_out = nn.Parameter(torch.zeros(dim_out))
        self.norm_unembed = DynamicTanh(dim=dim, init_alpha=0.2, elementwise_affine=True, bias=True)
        self.unembedder = RaiFlowFeedForward(dim=dim, dim_out=dim_out, ff_mult=ff_mult, dropout=dropout)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        height: int,
        width: int
    ):
        hidden_states = self.unembedder(self.norm_unembed(hidden_states))
        hidden_states = torch.addcmul(self.shift_latent_out, hidden_states, self.scale_latent_out)
        hidden_states = unpack_1d_latents_to_2d(hidden_states, patch_size=self.patch_size, original_height=height, original_width=width)
        return hidden_states


def RaiFlowPosEmbed1D(batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype, secondary_seq_len: int, base_seq_len: int, timestep: torch.FloatTensor, is_latent: bool) -> torch.FloatTensor:
    ones = torch.ones((batch_size, seq_len, 1), device=device, dtype=dtype)

    # Create 1D linspace positions
    posed_embeds_ch0 = torch.linspace(start=0, end=1, steps=seq_len, device=device, dtype=dtype)
    posed_embeds_ch1 = torch.linspace(start=0, end=1, steps=(seq_len + secondary_seq_len), device=device, dtype=dtype)
    posed_embeds_ch1 = posed_embeds_ch1[secondary_seq_len if is_latent else seq_len :]
    posed_embeds_ch2 = ones * (seq_len / base_seq_len)
    posed_embeds_ch3 = ones * timestep

    # stack and repeat for batch_size
    posed_embeds = torch.stack([posed_embeds_ch0, posed_embeds_ch1], dim=1)
    posed_embeds = posed_embeds.unsqueeze(0).repeat(batch_size, 1, 1)

    posed_embeds = torch.cat([posed_embeds, posed_embeds_ch2, posed_embeds_ch3], dim=2)
    return posed_embeds


def RaiFlowPosEmbed2D(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.FloatTensor:
    max_dim = max(width, height)
    max_dim_linspace = torch.linspace(0, 1, max_dim, device=device, dtype=dtype)

    # create 2D linspace positions grid
    posed_latents_ch0 = torch.linspace(0, 1, height, device=device, dtype=dtype).reshape(height, 1).repeat(1, width)
    posed_latents_ch1 = max_dim_linspace[:height].reshape(height, 1).repeat(1, width)
    posed_latents_ch2 = torch.linspace(0, 1, width, device=device, dtype=dtype).reshape(1, width).repeat(height, 1)
    posed_latents_ch3 = max_dim_linspace[:width].reshape(1, width).repeat(height, 1)

    # stack and repeat for batch_size
    posed_latents = torch.stack([posed_latents_ch0, posed_latents_ch1, posed_latents_ch2, posed_latents_ch3], dim=0) # (4, height, width)
    posed_latents = posed_latents.unsqueeze(0).repeat(batch_size, 1, 1, 1) # (batch_size, 4, height, width)
    return posed_latents


def pack_2d_latents_to_1d(latents: torch.FloatTensor, patch_size: int) -> torch.FloatTensor:
    batch_size, channels, height, width  = latents.shape
    packed_latents = latents.view(batch_size, channels, (height // patch_size), patch_size, (width // patch_size), patch_size)
    packed_latents = packed_latents.permute(0, 2, 4, 1, 3, 5)
    packed_latents = packed_latents.reshape(batch_size, (height // patch_size) * (width // patch_size), (channels * patch_size * patch_size))
    return packed_latents


def unpack_1d_latents_to_2d(latents: torch.FloatTensor, patch_size: int, original_height: int, original_width: int) -> torch.FloatTensor:
        batch_size, _, channels = latents.shape

        latents = latents.view(
            batch_size,
            (original_height // patch_size),
            (original_width // patch_size),
            (channels // (patch_size * patch_size)),
            patch_size,
            patch_size
        )

        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, (channels // (patch_size * patch_size)), original_height, original_width)
        return latents
