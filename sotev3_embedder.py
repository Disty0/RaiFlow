import torch

from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def SoteDiffusionV3PosEmbed1D(embeds: torch.Tensor, sigmas: torch.Tensor, latents_seq_len: int, base_seq_len: int):
    batch_size, seq_len, _ = embeds.shape

    posed_embeds = torch.ones((batch_size, seq_len, 4), device=embeds.device, dtype=embeds.dtype)
    posed_embeds[:, :, 0] = torch.linspace(start=0, end=1, steps=seq_len, device=embeds.device, dtype=embeds.dtype)
    posed_embeds[:, :, 1] = torch.linspace(start=0, end=1, steps=(seq_len + latents_seq_len), device=embeds.device, dtype=embeds.dtype)[:seq_len]
    posed_embeds[:, :, 2] = posed_embeds[:, :, 2] * (seq_len / base_seq_len)
    posed_embeds[:, :, 3] = posed_embeds[:, :, 3] * sigmas.expand(batch_size, seq_len)

    embeds = torch.cat([embeds, posed_embeds], dim=2)
    return embeds


def SoteDiffusionV3PosEmbed2D(latents):
    batch_size, _, height, width = latents.shape
    max_dim = max(width, height)
    posed_latents = torch.zeros((batch_size, 4, height, width))
    for x in range(width):
        posed_latents[:, 0, :, x] = torch.linspace(start=0, end=1, steps=height)
        posed_latents[:, 1, :, x] = torch.linspace(start=0, end=1, steps=max_dim)[:height]
    for y in range(height):
        posed_latents[:, 2, y, :] = torch.linspace(start=0, end=1, steps=width)
        posed_latents[:, 3, y, :] = torch.linspace(start=0, end=1, steps=max_dim)[:width]
    posed_latents = posed_latents.to(latents.device, dtype=latents.dtype)
    posed_latents = torch.cat([latents, posed_latents], dim=1)
    return posed_latents


def SoteDiffusionV3PatchEmbed2D(latents: torch.Tensor, sigmas: torch.Tensor, patch_size: int, embeds_seq_len: int, base_seq_len: int):
    batch_size, in_channels, height, width = latents.shape
    patched_latents = latents.reshape(
        (
            batch_size,
            int((width/patch_size) * (height/patch_size)),
            int(in_channels*patch_size*patch_size)
        )
    )

    batch_size, seq_len, _ = patched_latents.shape

    posed_embeds = torch.ones((batch_size, seq_len, 4), device=latents.device, dtype=latents.dtype)
    posed_embeds[:, :, 0] = torch.linspace(start=0, end=1, steps=seq_len, device=latents.device, dtype=latents.dtype)
    posed_embeds[:, :, 1] = torch.linspace(start=0, end=1, steps=(seq_len + embeds_seq_len), device=latents.device, dtype=latents.dtype)[embeds_seq_len:]
    posed_embeds[:, :, 2] = posed_embeds[:, :, 2] * (seq_len / base_seq_len)
    posed_embeds[:, :, 3] = posed_embeds[:, :, 3] * sigmas.expand(batch_size, seq_len)

    patched_latents = torch.cat([patched_latents, posed_embeds], dim=2)
    return patched_latents
