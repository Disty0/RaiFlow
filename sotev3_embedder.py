from typing import Tuple, Union

import torch

from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def SoteDiffusionV3PosEmbed1D(shape: Union[torch.Size, Tuple[int]], device: torch.device, dtype: torch.dtype, secondary_seq_len: int, base_seq_len: int, sigmas: torch.FloatTensor) -> torch.FloatTensor:
    batch_size, seq_len, _ = shape
    ones = torch.ones((batch_size, seq_len, 1), device=device, dtype=dtype)

    # Create 1D linspace tensors on the target device
    posed_embeds_ch0 = torch.linspace(start=0, end=1, steps=seq_len, device=device, dtype=dtype)
    posed_embeds_ch1 = torch.linspace(start=0, end=1, steps=(seq_len + secondary_seq_len), device=device, dtype=dtype)[:seq_len]
    posed_embeds_ch2 = ones * (seq_len / base_seq_len)
    posed_embeds_ch3 = ones * sigmas

    # stack and repeat for batch_size
    posed_embeds = torch.stack([posed_embeds_ch0, posed_embeds_ch1], dim=1)
    posed_embeds = posed_embeds.unsqueeze(0).repeat(batch_size, 1, 1)

    posed_embeds = torch.cat([posed_embeds, posed_embeds_ch2, posed_embeds_ch3], dim=2)
    return posed_embeds


def SoteDiffusionV3PosEmbed2D(shape: Union[torch.Size, Tuple[int]], device: torch.device, dtype: torch.dtype) -> torch.FloatTensor:
    batch_size, _, height, width = shape
    max_dim = max(width, height)

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
    return posed_latents


def pack_2d_latents_to_1d(latents: torch.FloatTensor, patch_size: int) -> torch.FloatTensor:
    batch_size, channels, height, width  = latents.shape
    packed_latents = latents.view(batch_size, channels, (height // patch_size), patch_size, (width // patch_size), patch_size)
    packed_latents = packed_latents.permute(0, 2, 4, 1, 3, 5)
    packed_latents = packed_latents.reshape(batch_size, (height // patch_size) * (width // patch_size), (channels * patch_size * patch_size))
    return packed_latents


def unpack_1d_latents_to_2d(latents: torch.FloatTensor, patch_size: int, original_height: int, original_widht: int) -> torch.FloatTensor:
        batch_size, _, channels = latents.shape

        latents = latents.view(
            batch_size,
            (original_height // patch_size),
            (original_widht // patch_size),
            (channels // (patch_size * patch_size)),
            patch_size,
            patch_size
        )

        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, (channels // (patch_size * patch_size)), original_height, original_widht)
        return latents
