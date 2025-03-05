from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from diffusers.utils import BaseOutput


@dataclass
class SoteDiffusionV3PipelineOutput(BaseOutput):
    """
    Output class for Sote Diffusion V3 pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`.
            PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


@dataclass
class SoteDiffusionV3Transformer2DModelOutput(BaseOutput):
    """
    The output of [`Transformer2DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`.
            The flowmatch prediction calculated from noise and latent.
        noise (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`.
            The epsilon prediction output conditioned on the `encoder_hidden_states` input.
        latent (`torch.Tensor` of shape `(batch_size, num_channels, height, width)`.
            The x0 prediction output conditioned on the `encoder_hidden_states` input.
    """

    sample: "torch.Tensor"  # noqa: F821
    noise: "torch.Tensor"  # noqa: F821
    latent: "torch.Tensor"  # noqa: F821
