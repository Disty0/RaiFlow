import torch
from torch import nn

from .raiflow_layers import RaiFlowRMSNorm


class RaiFlowLatentEmbedder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        base_seq_len: int,
        dim: int,
        dim_out: int,
        eps: float = 1e-5,
        bias: bool = True,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.base_seq_len = base_seq_len
        self.latent_embedder_proj = nn.Conv2d(dim, dim_out, 3, padding=1, bias=bias)
        self.norm_latent_embedder = RaiFlowRMSNorm(dim_out, eps=eps, elementwise_affine=elementwise_affine)

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
    ) -> torch.FloatTensor:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            with torch.no_grad():
                posed_latents_2d = RaiFlowPosEmbed2D(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                posed_latents_1d = RaiFlowPosEmbed1D(
                    batch_size=batch_size,
                    seq_len=latents_seq_len,
                    device=hidden_states.device,
                    dtype=torch.float32,
                    secondary_seq_len=encoder_seq_len,
                    base_seq_len=self.base_seq_len,
                    timestep=timestep,
                    is_latent=True,
                ).transpose(-1,-2).unflatten(-1, (height//self.patch_size, width//self.patch_size))

            hidden_states = torch.cat([hidden_states.to(dtype=torch.float32), posed_latents_2d], dim=1)
            hidden_states = torch.nn.functional.pixel_unshuffle(hidden_states, self.patch_size)
            hidden_states = torch.cat([hidden_states, posed_latents_1d], dim=-3)
            hidden_states = self.latent_embedder_proj(hidden_states).flatten(-2,-1).transpose(-1,-2)
            hidden_states = self.norm_latent_embedder(hidden_states).to(dtype=dtype, memory_format=torch.contiguous_format)
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
        eps: float = 1e-5,
        bias: bool = True,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.base_seq_len = base_seq_len
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, pad_token_id)
        self.text_embedder_proj = nn.Conv1d(dim, dim_out, 3, padding=1, bias=bias)
        self.norm_text_embedder = RaiFlowRMSNorm(dim_out, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self,
        encoder_hidden_states: torch.FloatTensor,
        timestep: torch.FloatTensor,
        dtype: torch.dtype,
        latents_seq_len: int,
        encoder_seq_len: int,
        batch_size: int,
    ) -> torch.FloatTensor:
        encoder_hidden_states = self.token_embedding(encoder_hidden_states).to(dtype=torch.float32)
        with torch.autocast(device_type=encoder_hidden_states.device.type, enabled=False):
            with torch.no_grad():
                posed_encoder_1d = RaiFlowPosEmbed1D(
                    batch_size=batch_size,
                    seq_len=encoder_seq_len,
                    device=encoder_hidden_states.device,
                    dtype=torch.float32,
                    secondary_seq_len=latents_seq_len,
                    base_seq_len=self.base_seq_len,
                    timestep=timestep,
                    is_latent=True,
                )

            encoder_hidden_states = torch.cat([encoder_hidden_states, posed_encoder_1d], dim=2)
            encoder_hidden_states = self.text_embedder_proj(encoder_hidden_states.transpose(-1,-2)).transpose(-1,-2)
            encoder_hidden_states = self.norm_text_embedder(encoder_hidden_states).to(dtype=dtype, memory_format=torch.contiguous_format)
            return encoder_hidden_states


class RaiFlowLatentUnembedder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        dim: int,
        dim_out: int,
        eps: float = 1e-5,
        bias: bool = True,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.norm_unembed = RaiFlowRMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.unembedder_proj = nn.Conv2d(dim, dim_out, 3, padding=1, bias=bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        height: int,
        width: int,
    ) -> torch.FloatTensor:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            hidden_states = self.norm_unembed(hidden_states.to(dtype=torch.float32))
            hidden_states = hidden_states.transpose(-1,-2).unflatten(-1, (height//self.patch_size, width//self.patch_size))
            hidden_states = torch.nn.functional.pixel_shuffle(self.unembedder_proj(hidden_states), self.patch_size)
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
    posed_latents_ch0 = torch.linspace(0, 1, height, device=device, dtype=dtype).unsqueeze(-1).repeat(1, width)
    posed_latents_ch1 = max_dim_linspace[:height].unsqueeze(-1).repeat(1, width)
    posed_latents_ch2 = torch.linspace(0, 1, width, device=device, dtype=dtype).unsqueeze(0).repeat(height, 1)
    posed_latents_ch3 = max_dim_linspace[:width].unsqueeze(0).repeat(height, 1)

    # stack and repeat for batch_size
    posed_latents = torch.stack([posed_latents_ch0, posed_latents_ch1, posed_latents_ch2, posed_latents_ch3], dim=0) # (4, height, width)
    posed_latents = posed_latents.unsqueeze(0).repeat(batch_size, 1, 1, 1) # (batch_size, 4, height, width)
    return posed_latents
