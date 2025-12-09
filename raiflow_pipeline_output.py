from dataclasses import dataclass
from typing import List, Union

import numpy as np
import PIL.Image

from diffusers.utils import BaseOutput


@dataclass
class RaiFlowPipelineOutput(BaseOutput):
    """
    Output class for RaiFlow pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width, num_channels)`.
            PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]  # noqa: F821
    x0_pred: "torch.FloatTensor"  # noqa: F821


@dataclass
class RaiFlowTransformer2DModelOutput(BaseOutput):
    """
    The output of [`RaiFlowTransformer2DModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: "torch.FloatTensor"  # noqa: F821
    x0_pred: "torch.FloatTensor"  # noqa: F821
