import torch

from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def SoteDiffusionV3PosEmbed1D(embeds: torch.Tensor, sigmas: torch.Tensor, secondary_seq_len: int, base_seq_len: int):
    batch_size, seq_len, _ = embeds.shape
    device = embeds.device
    dtype = embeds.dtype

    # Create 1D linspace tensors on the target device
    posed_embeds_ch0 = torch.linspace(start=0, end=1, steps=seq_len, device=device, dtype=dtype)
    posed_embeds_ch1 = torch.linspace(start=0, end=1, steps=(seq_len + secondary_seq_len), device=device, dtype=dtype)[:seq_len]

    ones = torch.ones((batch_size, seq_len, 1), device=device, dtype=dtype)
    posed_embeds_ch2 = ones * (seq_len / base_seq_len)
    posed_embeds_ch3 = ones * sigmas.view(batch_size, 1, 1)

    # stack and repeat for batch_size
    posed_embeds = torch.stack([posed_embeds_ch0, posed_embeds_ch1], dim=1)
    posed_embeds = posed_embeds.unsqueeze(0).repeat(batch_size, 1, 1)

    posed_embeds = torch.cat([posed_embeds, posed_embeds_ch2, posed_embeds_ch3], dim=2)
    posed_embeds = torch.cat([embeds, posed_embeds], dim=2)

    return posed_embeds


def SoteDiffusionV3PosEmbed2D(latents: torch.FloatTensor, sigmas: torch.FloatTensor):
    batch_size, _, height, width = latents.shape
    max_dim = max(width, height)
    device = latents.device
    dtype = latents.dtype

    # create 1D linspace tensors on the target device
    width_1d = torch.linspace(0, 1, width, device=device, dtype=dtype)
    height_1d = torch.linspace(0, 1, height, device=device, dtype=dtype)
    max_dim_width_1d = torch.linspace(0, 1, max_dim, device=device, dtype=dtype)[:width]
    max_dim_height_1d = torch.linspace(0, 1, max_dim, device=device, dtype=dtype)[:height]


    # broadcast to create 2D linspace grids
    posed_latents_ch0 = height_1d.reshape(height, 1).repeat(1, width)
    posed_latents_ch1 = max_dim_height_1d.reshape(height, 1).repeat(1, width)
    posed_latents_ch2 = width_1d.reshape(1, width).repeat(height, 1)
    posed_latents_ch3 = max_dim_width_1d.reshape(1, width).repeat(height, 1)

    # stack and repeat
    posed_latents = torch.stack([posed_latents_ch0, posed_latents_ch1, posed_latents_ch2, posed_latents_ch3], dim=0) # (4, height, width)
    posed_latents = posed_latents.unsqueeze(0).repeat(batch_size, 1, 1, 1) # (batch_size, 4, height, width)
    posed_latents = torch.cat([latents, posed_latents], dim=1) # (batch_size, in_channels + 4, height, width)

    return posed_latents
