from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.models.embeddings import get_1d_rotary_pos_embed

from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def RaiFlowPosEmbed1D(shape: Union[torch.Size, Tuple[int]], device: torch.device, dtype: torch.dtype, secondary_seq_len: int, base_seq_len: int, sigmas: torch.FloatTensor) -> torch.FloatTensor:
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


def RaiFlowPosEmbed2D(shape: Union[torch.Size, Tuple[int]], device: torch.device, dtype: torch.dtype) -> torch.FloatTensor:
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


# Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
# removed batch_size argument as it is unused by this function
def prepare_latent_image_ids(height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


class FluxPosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    # modified from diffusers in RaiFlow to add dtype control
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor, freqs_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        if freqs_dtype is None:
            is_mps = ids.device.type == "mps"
            is_npu = ids.device.type == "npu"
            freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin
