from typing import List

import math
import torch
import torchvision
import numpy as np

from PIL import Image
from diffusers.utils import CONFIG_NAME
from diffusers.image_processor import PipelineImageInput
from diffusers.configuration_utils import ConfigMixin, register_to_config
from transformers import ImageProcessingMixin

# more readable but 2x slower version of this code is available here: https://github.com/Disty0/torch-jpeg


@torch.no_grad()
def rgb_to_ycbcr_tensor(image: torch.ByteTensor) -> torch.FloatTensor:
    if image.dtype != torch.float32:
        img = image.to(dtype=torch.float32).div_(255)
    else:
        img = image / 255
    y = (img[:,:,:,0] * 0.299).add_(img[:,:,:,1], alpha=0.587).add_(img[:,:,:,2], alpha=0.114)
    cb = (img[:,:,:,0] * -0.168935).add_(img[:,:,:,1], alpha=-0.331665).add_(img[:,:,:,2], alpha=0.50059).add_(0.5)
    cr = (img[:,:,:,0] * 0.499813).add_(img[:,:,:,1], alpha=-0.418531).add_(img[:,:,:,2], alpha=-0.081282).add_(0.5)
    ycbcr = torch.add(-1, torch.stack([y,cb,cr], dim=1), alpha=2)
    return ycbcr


@torch.no_grad()
def ycbcr_tensor_to_rgb(ycbcr: torch.FloatTensor) -> torch.ByteTensor:
    ycbcr_img = (ycbcr / 2)
    y = ycbcr_img[:,0,:,:].add_(0.5)
    cb = ycbcr_img[:,1,:,:]
    cr = ycbcr_img[:,2,:,:]

    r = (cr * 1.402525).add_(y)
    g = (cb * -0.343730).add_(cr, alpha=-0.714401).add_(y)
    b = (cb * 1.769905).add_(cr, alpha=0.000013).add_(y)
    rgb = torch.stack([r,g,b], dim=-1).mul_(255).round_().clamp_(0,255).to(dtype=torch.uint8)
    return rgb


@torch.no_grad()
def encode_single_channel_dct_2d(img: torch.FloatTensor, block_size: int=16, norm: str="ortho") -> torch.FloatTensor:
    batch_size, height, width = img.shape
    h_blocks = int(height//block_size)
    w_blocks = int(width//block_size)

    # batch_size, h_blocks, w_blocks, block_size_h, block_size_w
    dct_tensor = img.view(batch_size, h_blocks, block_size, w_blocks, block_size).transpose(2,3).to(dtype=torch.float32)
    dct_tensor = dct_2d(dct_tensor, norm=norm)

    # batch_size, combined_block_size, h_blocks, w_blocks
    dct_tensor = dct_tensor.reshape(batch_size, h_blocks, w_blocks, block_size*block_size).permute(0,3,1,2)
    return dct_tensor


@torch.no_grad()
def decode_single_channel_dct_2d(img: torch.FloatTensor, norm: str="ortho") -> torch.FloatTensor:
    batch_size, combined_block_size, h_blocks, w_blocks = img.shape
    block_size = int(math.sqrt(combined_block_size))
    height = int(h_blocks*block_size)
    width = int(w_blocks*block_size)

    img_tensor = img.permute(0,2,3,1).view(batch_size, h_blocks, w_blocks, block_size, block_size)
    img_tensor = idct_2d(img_tensor, norm=norm)
    img_tensor = img_tensor.permute(0,1,3,2,4).reshape(batch_size, height, width)
    return img_tensor


@torch.no_grad()
def encode_jpeg_tensor(img: torch.FloatTensor, block_size: int=16, cbcr_downscale: int=2, norm: str="ortho") -> torch.FloatTensor:
    img = img[:, :, :(img.shape[-2]//block_size)*block_size, :(img.shape[-1]//block_size)*block_size] # crop to a multiply of block_size
    cbcr_block_size = block_size//cbcr_downscale
    _, _, height, width = img.shape
    downsample = torchvision.transforms.Resize((height//cbcr_downscale, width//cbcr_downscale), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    down_img = downsample(img[:, 1:,:,:])
    y = encode_single_channel_dct_2d(img[:, 0, :,:], block_size=block_size, norm=norm)
    cb = encode_single_channel_dct_2d(down_img[:, 0, :,:], block_size=cbcr_block_size, norm=norm)
    cr = encode_single_channel_dct_2d(down_img[:, 1, :,:], block_size=cbcr_block_size, norm=norm)
    ycbcr = torch.cat([y,cb,cr], dim=1)
    return ycbcr


@torch.no_grad()
def decode_jpeg_tensor(jpeg_img: torch.FloatTensor, block_size: int=16, cbcr_downscale: int=2, norm: str="ortho") -> torch.FloatTensor:
    _, _, h_blocks, w_blocks = jpeg_img.shape
    y_block_size = block_size*block_size
    cbcr_block_size = int((block_size//cbcr_downscale) ** 2)
    cr_start = y_block_size + cbcr_block_size
    y = jpeg_img[:, :y_block_size]
    cb = jpeg_img[:, y_block_size:cr_start]
    cr = jpeg_img[:, cr_start:]
    y = decode_single_channel_dct_2d(y, norm=norm)
    cb = decode_single_channel_dct_2d(cb, norm=norm)
    cr = decode_single_channel_dct_2d(cr, norm=norm)
    upsample = torchvision.transforms.Resize((h_blocks*block_size, w_blocks*block_size), interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    cb = upsample(cb)
    cr = upsample(cr)
    ycbcr = torch.stack([y,cb,cr], dim=1)
    return ycbcr


def process_image_input(images: PipelineImageInput) -> torch.ByteTensor:
    if isinstance(images, list):
        combined_images = []
        for img in images:
            if isinstance(img, Image.Image):
                img = torch.from_numpy(np.asarray(img).copy()).unsqueeze(0)
                combined_images.append(img)
            elif isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                combined_images.append(img)
            elif isinstance(img, torch.Tensor):
                if img.ndim == 3:
                    img = img.unsqueeze(0)
                combined_images.append(img)
            else:
                raise RuntimeError(f"Invalid input! Given: {type(img)} should be in ('torch.Tensor', 'np.ndarray', 'PIL.Image.Image')")
        combined_images = torch.cat(combined_images, dim=0)
    elif isinstance(images, Image.Image):
        combined_images = torch.from_numpy(np.asarray(images).copy()).unsqueeze(0)
    elif isinstance(images, np.ndarray):
        combined_images = torch.from_numpy(images)
        if combined_images.ndim == 3:
            combined_images = combined_images.unsqueeze(0)
    elif isinstance(images, torch.Tensor):
        combined_images = images
        if combined_images.ndim == 3:
            combined_images = combined_images.unsqueeze(0)
    else:
        raise RuntimeError(f"Invalid input! Given: {type(images)} should be in ('torch.Tensor', 'np.ndarray', 'PIL.Image.Image')")
    return combined_images


class RaiFlowImageEncoder(ImageProcessingMixin, ConfigMixin):

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        block_size: int = 16,
        cbcr_downscale: int = 2,
        norm: str = "ortho",
        latents_std: List[float] = None,
        latents_mean: List[float] = None,
    ):
        self.block_size = block_size
        self.cbcr_downscale = cbcr_downscale
        self.norm = norm
        self.latents_std = latents_std
        self.latents_mean = latents_mean

    @torch.no_grad()
    def encode(self, images: PipelineImageInput, device: str="cpu") -> torch.FloatTensor:
        """
        Encode RGB 0-255 image to RaiFlow Latents.

        Args:
            image (`PIL.Image.Image`, `np.ndarray` or `torch.Tensor`):
                The image input, can be a PIL image, numpy array or pytorch tensor.
                Must be an RGB image or a list of RGB images with 0-255 range and (batch_size, height, width, channels) shape.

        Returns:
            `torch.Tensor`:
                The encoded RaiFlow Latents.
        """

        combined_images = process_image_input(images).to(device)
        latents = rgb_to_ycbcr_tensor(combined_images)
        latents = encode_jpeg_tensor(latents, block_size=self.block_size, cbcr_downscale=self.cbcr_downscale, norm=self.norm)

        if self.latents_mean is not None:
            latents = latents - torch.tensor(self.latents_mean, device=device, dtype=torch.float32).view(1,-1,1,1)
        if self.latents_std is not None:
            latents = latents / torch.tensor(self.latents_std, device=device, dtype=torch.float32).view(1,-1,1,1)

        return latents

    @torch.no_grad()
    def decode(self, latents: torch.FloatTensor, return_type: str="pil") -> PipelineImageInput:
        latents = latents.to(dtype=torch.float32)
        if self.latents_std is not None:
            latents_std = torch.tensor(self.latents_std, device=latents.device, dtype=torch.float32).view(1,-1,1,1)
            if self.latents_mean is not None:
                latents_mean = torch.tensor(self.latents_mean, device=latents.device, dtype=torch.float32).view(1,-1,1,1)
                latents = torch.addcmul(latents_mean, latents, latents_std)
            else:
                latents = latents * latents_std
        elif self.latents_mean is not None:
            latents = latents + torch.tensor(self.latents_mean, device=latents.device, dtype=torch.float32).view(1,-1,1,1)

        images = decode_jpeg_tensor(latents, block_size=self.block_size, cbcr_downscale=self.cbcr_downscale, norm=self.norm)
        images = ycbcr_tensor_to_rgb(images)

        if return_type == "pt":
            return images
        elif return_type == "np":
            return images.detach().cpu().numpy()
        elif return_type == "pil":
            image_list = []
            for i in range(images.shape[0]):
                image_list.append(Image.fromarray(images[i].detach().cpu().numpy()))
            return image_list
        else:
            raise RuntimeError(f"Invalid return_type! Given: {return_type} should be in ('pt', 'np', 'pil')")


# DCT functions are modified from https://github.com/zh217/torch-dct/blob/master/torch_dct/_dct.py

@torch.no_grad()
def dct(x, norm=None):
    x_shape = x.shape
    N = x_shape[-1]

    x = x.contiguous().view(-1, N)
    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :].mul_(math.pi / (2 * N))
    W_r = torch.cos(k)
    n_W_i = -torch.sin(k)

    V = torch.addcmul((Vc[:, :, 0] * W_r), Vc[:, :, 1], n_W_i)
    if norm == "ortho":
        V[:, 0].mul_(0.5 / math.sqrt(N))
        V[:, 1:].mul_(0.5 / math.sqrt(N / 2))

    V = V.view(x_shape).mul_(2)
    return V


@torch.no_grad()
def idct(X, norm=None):
    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, N).div_(2)
    if norm == "ortho":
        X_v[:, 0].mul_(math.sqrt(N) * 2)
        X_v[:, 1:].mul_(math.sqrt(N / 2) * 2)

    k = torch.arange(N, dtype=X.dtype, device=X.device)[None, :].mul_(math.pi / (2 * N))
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_i = torch.cat([X_v.new_zeros((X_v.shape[0], 1)), -(X_v.flip([1])[:, :-1])], dim=1)
    V_r = torch.addcmul((X_v * W_r), V_t_i, -W_i)
    V_i = torch.addcmul((X_v * W_i), V_t_i, W_r)

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] = v[:, :N - (N // 2)]
    x[:, 1::2] = v.flip([1])[:, :N // 2]

    x = x.view(x_shape)
    return x


@torch.no_grad()
def dct_2d(x, norm=None):
    X1 = dct(x, norm=norm).transpose_(-1, -2)
    X2 = dct(X1, norm=norm).transpose_(-1, -2)
    return X2


@torch.no_grad()
def idct_2d(X, norm=None):
    x1 = idct(X, norm=norm).transpose_(-1, -2)
    x2 = idct(x1, norm=norm).transpose_(-1, -2)
    return x2
