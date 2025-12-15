import torch
from torch import nn

fp16_max = 65504


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
        self.norm_latent_embedder = nn.RMSNorm(dim_out, eps=eps, elementwise_affine=elementwise_affine)

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
        with torch.autocast(device_type=hidden_states.device.type, enabled=False): # force fp32
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
            hidden_states = self.norm_latent_embedder(hidden_states.clamp(-fp16_max, fp16_max))
            hidden_states = hidden_states.to(dtype=dtype, memory_format=torch.contiguous_format)
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
        self.norm_text_embedder = nn.RMSNorm(dim_out, eps=eps, elementwise_affine=elementwise_affine)

    def forward(
        self,
        encoder_hidden_states: torch.FloatTensor,
        timestep: torch.FloatTensor,
        dtype: torch.dtype,
        latents_seq_len: int,
        encoder_seq_len: int,
        batch_size: int,
    ) -> torch.FloatTensor:
        encoder_hidden_states = self.token_embedding(encoder_hidden_states)
        with torch.autocast(device_type=encoder_hidden_states.device.type, enabled=False): # force fp32
            with torch.no_grad():
                posed_encoder_1d = RaiFlowPosEmbed1D(
                    batch_size=batch_size,
                    seq_len=encoder_seq_len,
                    device=encoder_hidden_states.device,
                    dtype=torch.float32,
                    secondary_seq_len=latents_seq_len,
                    base_seq_len=self.base_seq_len,
                    timestep=timestep,
                    is_latent=False,
                )

            encoder_hidden_states = torch.cat([encoder_hidden_states.to(dtype=torch.float32), posed_encoder_1d], dim=2)
            encoder_hidden_states = self.text_embedder_proj(encoder_hidden_states.transpose(-1,-2)).transpose(-1,-2)
            encoder_hidden_states = self.norm_text_embedder(encoder_hidden_states.clamp(-fp16_max, fp16_max))
            encoder_hidden_states = encoder_hidden_states.to(dtype=dtype, memory_format=torch.contiguous_format)
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
        self.norm_unembed = nn.RMSNorm(dim, eps=eps, elementwise_affine=elementwise_affine)
        self.unembedder_proj = nn.Conv2d(dim, dim_out, 3, padding=1, bias=bias)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        height: int,
        width: int,
    ) -> torch.FloatTensor:
        with torch.autocast(device_type=hidden_states.device.type, enabled=False): # force fp32
            hidden_states = self.norm_unembed(hidden_states.to(dtype=torch.float32).clamp(-fp16_max, fp16_max))
            hidden_states = hidden_states.transpose(-1,-2).unflatten(-1, (height//self.patch_size, width//self.patch_size))
            hidden_states = torch.nn.functional.pixel_shuffle(self.unembedder_proj(hidden_states), self.patch_size)
            return hidden_states


def RaiFlowPosEmbed1D(batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype, secondary_seq_len: int, base_seq_len: int, timestep: torch.FloatTensor, is_latent: bool) -> torch.FloatTensor:
    global_pos = torch.linspace(start=0, end=1, steps=(seq_len + secondary_seq_len), device=device, dtype=dtype)
    if is_latent:
        global_pos = global_pos[secondary_seq_len:]
    else:
        global_pos = global_pos[:seq_len]

    pos_embeds = torch.stack(
        [
            torch.linspace(start=0, end=1, steps=seq_len, device=device, dtype=dtype),
            global_pos,
        ],
        dim=1,
    ).unsqueeze(0).repeat(batch_size, 1, 1)
    del global_pos

    shape = (batch_size, seq_len, 1)
    pos_embeds = torch.cat(
        [
            pos_embeds,
            torch.full(shape, (seq_len / base_seq_len), device=device, dtype=dtype),
            timestep.expand(shape)
        ],
        dim=2,
    )
    return pos_embeds


def RaiFlowPosEmbed2D(batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.FloatTensor:
    pos_x, pos_y = torch.meshgrid(
        torch.linspace(0, 1, height, device=device, dtype=dtype),
        torch.linspace(0, 1, width, device=device, dtype=dtype),
        indexing="ij",
    )

    max_dim_linspace = torch.linspace(0, 1, max(width, height), device=device, dtype=dtype)
    relative_pos_x, relative_pos_y = torch.meshgrid(max_dim_linspace[:height], max_dim_linspace[:width], indexing="ij")
    del max_dim_linspace

    pos_embeds = torch.stack([pos_x, relative_pos_x, pos_y, relative_pos_y], dim=0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    del pos_x, pos_y, relative_pos_x, relative_pos_y
    return pos_embeds
