import torch
from torch import nn

fp16_max = 65504


class RaiFlowLatentEmbedder(nn.Module):
    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        base_seq_len: int,
        max_freqs: int,
        inner_dim: int,
        eps: float = 1e-5,
        bias: bool = True,
        elementwise_affine: bool = True,
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
                posed_latents_2d = RaiFlowPosEmbed2D(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    max_freqs=self.max_freqs,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                posed_latents_1d = RaiFlowPosEmbed1D(
                    batch_size=batch_size,
                    seq_len=latents_seq_len,
                    secondary_seq_len=encoder_seq_len,
                    base_seq_len=self.base_seq_len,
                    max_freqs=self.max_freqs,
                    is_latent=True,
                    device=hidden_states.device,
                    dtype=torch.float32,
                )

            hidden_states = torch.cat([hidden_states.to(dtype=torch.float32), posed_latents_2d], dim=1)
            hidden_states = torch.nn.functional.pixel_unshuffle(hidden_states, self.patch_size)
            hidden_states = hidden_states.flatten(-2,-1).transpose(-1,-2).contiguous()
            hidden_states = torch.cat([hidden_states, posed_latents_1d], dim=2)
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
        eps: float = 1e-5,
        bias: bool = True,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.base_seq_len = base_seq_len
        self.max_freqs = max_freqs

        dim_in = (3 * self.max_freqs * 2) + self.embedding_dim
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, pad_token_id)
        self.text_embedder_proj = nn.Linear(dim_in, inner_dim, bias=bias)

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
                posed_encoder_1d = RaiFlowPosEmbed1D(
                    batch_size=batch_size,
                    seq_len=encoder_seq_len,
                    secondary_seq_len=latents_seq_len,
                    base_seq_len=self.base_seq_len,
                    max_freqs=self.max_freqs,
                    is_latent=False,
                    device=encoder_hidden_states.device,
                    dtype=torch.float32,
                )

            encoder_hidden_states = torch.cat([encoder_hidden_states.to(dtype=torch.float32), posed_encoder_1d], dim=2)
            encoder_hidden_states = self.text_embedder_proj(encoder_hidden_states).to(dtype=dtype)
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


def RaiFlowPosEmbed1D(batch_size: int, seq_len: int, secondary_seq_len: int, base_seq_len: int, max_freqs: int, is_latent: bool, device: torch.device, dtype: torch.dtype) -> torch.FloatTensor:
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


def RaiFlowPosEmbed2D(batch_size: int, height: int, width: int, max_freqs: int, device: torch.device, dtype: torch.dtype) -> torch.FloatTensor:
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
