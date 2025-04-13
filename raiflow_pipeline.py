import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from diffusers.models.autoencoders import AutoencoderKL
from .raiflow_image_encoder import RaiFlowImageEncoder

from .raiflow_transformer import RaiFlowTransformer2DModel
from .raiflow_pipeline_output import RaiFlowPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import RaiFlowPipeline

        >>> pipe = RaiFlowPipeline.from_pretrained(
        ...     "Disty0/RaiFlow", torch_dtype=torch.float16
        ... )
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt).images[0]
        >>> image.save("raiflow.png")
        ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class RaiFlowPipeline(DiffusionPipeline):
    r"""
    Args:
        transformer ([`RaiFlowTransformer2DModel`]):
            Conditional Transformer (EMMDit) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        text_encoder ([`Qwen2VLForConditionalGeneration`]):
            [Qwen2VLForConditionalGeneration](https://huggingface.co/docs/transformers/main/model_doc/qwen2_vl#transformers.Qwen2VLForConditionalGeneration),
            specifically the [2B](https://huggingface.co/Qwen/Qwen2-VL-2B) variant.
        tokenizer (`Qwen2VLProcessor`):
            Tokenizer of class
            [Qwen2VLProcessor](https://huggingface.co/docs/transformers/main/model_doc/qwen2_vl#transformers.Qwen2VLProcessor).
        image_encoder ([`RaiFlowImageEncoder`], *optional*):
            RaiFlow JPEG encoder to encode and decode images to and from latent representations.
        vae ([`AutoencoderKL`], *optional*):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["image_encoder", "vae"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds", "noise_pred", "encoder_hidden_states"]

    def __init__(
        self,
        transformer: RaiFlowTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: Qwen2VLForConditionalGeneration,
        tokenizer: Qwen2VLProcessor,
        image_encoder: RaiFlowImageEncoder = None,
        vae: AutoencoderKL = None,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            vae=vae,
        )

        self.patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 2
        )

        if getattr(self, "vae", None) is not None:
            self.latent_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.image_processor = VaeImageProcessor(vae_scale_factor=self.latent_scale_factor * self.patch_size)
        elif getattr(self, "image_encoder", None) is not None:
            self.latent_scale_factor = self.image_encoder.config.block_size
        else:
            self.latent_scale_factor = 8

        self.default_sample_size = (
            self.transformer.config.sample_size if getattr(self, "transformer", None) is not None else 128
        )

    def _get_qwen2_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_images: Optional[PipelineImageInput] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        max_sequence_length: int = 1024,
    ) -> List[torch.FloatTensor]:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_images is not None and not isinstance(prompt_images, list):
            prompt_images = [prompt_images]

        inputs = self.tokenizer(
            text=prompt.copy(), # tokenizer overwrites
            images=prompt_images,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

        input_ids = inputs.input_ids
        untruncated_ids = self.tokenizer(text=prompt.copy(), images=prompt_images, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= input_ids.shape[-1] and not torch.equal(input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length :])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_embeds = self.text_encoder(**inputs, output_hidden_states=True).hidden_states[-1]
        prompt_embeds = prompt_embeds.to(device, dtype=dtype)

        attention_mask = inputs["attention_mask"].to(device, dtype=dtype)
        prompt_embeds = prompt_embeds * attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)
        prompt_embeds_list = []
        for i in range(prompt_embeds.size(0)):
            count = 0
            for j in reversed(attention_mask[i]):
                if j == 0:
                    break
                count += 1
            count = max(count,1)
            prompt_embeds_list.append(prompt_embeds[i, -count:])

        return prompt_embeds_list

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_images: Optional[PipelineImageInput] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_images: Optional[PipelineImageInput] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 1024,
        min_sequence_length: int = 256,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_images ('PipelineImageInput', *optional*):
                images to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt (`str` or `List[str]`, *optional*):
                The image or images not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:

            prompt_embeds = self._get_qwen2_prompt_embeds(
                prompt=prompt,
                prompt_images=prompt_images,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            # Offload all models if negative embeds are provided
            if negative_prompt_embeds is not None:
                self.maybe_free_model_hooks()

            max_len = 0
            for embed in prompt_embeds:
                max_len = max(max_len, embed.shape[0])
            max_len = max(max_len, min_sequence_length)
            if max_len % 256 != 0: # make it a multiple of 256
                max_len +=  256 - (max_len % 256)

            embed_dim = prompt_embeds[0].shape[-1]
            for i in range(len(prompt_embeds)):
                seq_len = prompt_embeds[i].shape[0]
                if seq_len != max_len:
                    prompt_embeds[i] = torch.cat(
                        [
                            prompt_embeds[i],
                            torch.ones((max_len-seq_len, embed_dim), device=prompt_embeds[i].device, dtype=prompt_embeds[i].dtype)
                        ],
                        dim=0,
                    )

            prompt_embeds = torch.stack(prompt_embeds, dim=0)

            _, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""

            # normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_qwen2_prompt_embeds(
                prompt=negative_prompt,
                prompt_images=negative_prompt_images,
                max_sequence_length=max_sequence_length,
                device=device,
            )

            # Offload all models
            self.maybe_free_model_hooks()

            max_len = 0
            for embed in negative_prompt_embeds:
                max_len = max(max_len, embed.shape[0])
            max_len = max(max_len, prompt_embeds.shape[1])
            if max_len % 256 != 0: # make it a multiple of 256
                max_len +=  256 - (max_len % 256)

            embed_dim = negative_prompt_embeds[0].shape[-1]
            for i in range(len(negative_prompt_embeds)):
                seq_len = negative_prompt_embeds[i].shape[0]
                if seq_len != max_len:
                    negative_prompt_embeds[i] = torch.cat(
                        [
                            negative_prompt_embeds[i],
                            torch.ones((max_len-seq_len, embed_dim), device=negative_prompt_embeds[i].device, dtype=negative_prompt_embeds[i].dtype)
                        ],
                        dim=0,
                    )

            negative_prompt_embeds = torch.stack(negative_prompt_embeds, dim=0)

            _, seq_len, _ = negative_prompt_embeds.shape
            # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            if negative_prompt_embeds.shape[1] > prompt_embeds.shape[1]:
                batch_size, seq_len, embed_dim = prompt_embeds.shape
                prompt_embeds = torch.cat(
                        [
                            prompt_embeds,
                            torch.ones((batch_size, negative_prompt_embeds.shape[1]-seq_len, embed_dim), device=prompt_embeds[i].device, dtype=prompt_embeds[i].dtype)
                        ],
                        dim=1,
                    )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_images=None,
        negative_prompt=None,
        negative_prompt_images=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
        min_sequence_length=None,
    ):
        if (
            height % (self.latent_scale_factor * self.patch_size) != 0
            or width % (self.latent_scale_factor * self.patch_size) != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.latent_scale_factor * self.patch_size} but are {height} and {width}."
                f"You can use height {height - height % (self.latent_scale_factor * self.patch_size)} and width {width - width % (self.latent_scale_factor * self.patch_size)}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

        if max_sequence_length is not None and not isinstance(max_sequence_length, int):
            raise ValueError(
                f"`max_sequence_length` must be an integer but got: {type(max_sequence_length)}"
            )

        if min_sequence_length is not None and not isinstance(min_sequence_length, int):
            raise ValueError(
                f"`min_sequence_length` must be an integer but got: {type(min_sequence_length)}"
            )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.latent_scale_factor,
            int(width) // self.latent_scale_factor,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype).to(device, dtype=dtype) # xpu returns float32

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def raiflow_x0_pred_guidance_scale(self):
        return self._raiflow_x0_pred_guidance_scale

    @property
    def raiflow_guidence_base_shift(self):
        return self._raiflow_guidence_base_shift

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_images: Optional[PipelineImageInput] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        raiflow_x0_pred_guidance_scale: float = 1.0,
        raiflow_guidence_base_shift: float = 4.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_images: Optional[PipelineImageInput] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 1024,
        min_sequence_length: int = 256,
        mu: Optional[float] = None,
    ) -> Union[RaiFlowPipelineOutput, Tuple[PipelineImageInput]]:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_images ('PipelineImageInput', *optional*):
                The image or images to guide the image generation.
            height (`int`, *optional*, defaults to self.transformer.config.sample_size * self.latent_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_size * self.latent_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            raiflow_x0_pred_guidance_scale (`float`, *optional*, defaults to 4.0):
                Guidance scale used for classifier free guidence via the final image predictions.
            raiflow_guidence_base_shift (`float`, *optional*, defaults to 4.0):
                Base shift used to scale the cfg value on the first timestep.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            negative_prompt_images (`PipelineImageInput`, *optional*):
                The image or images not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.raiflow.RaiFlowPipelineOutput`] instead of
                a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 1024): Maximum sequence length to use with the `prompt`.
            min_sequence_length (`int` defaults to 128): Minimum sequence length to use with the `prompt`.
            mu (`float`, *optional*): `mu` value used for `dynamic_shifting`.

        Examples:

        Returns:
            [`~pipelines.raiflow.RaiFlowPipelineOutput`] or `tuple`:
            [`~pipelines.raiflow.RaiFlowPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.latent_scale_factor
        width = width or self.default_sample_size * self.latent_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            prompt_images=prompt_images,
            negative_prompt=negative_prompt,
            negative_prompt_images=negative_prompt_images,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            min_sequence_length=min_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._raiflow_guidence_base_shift = raiflow_guidence_base_shift
        self._raiflow_x0_pred_guidance_scale = raiflow_x0_pred_guidance_scale

        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_images=prompt_images,
            negative_prompt=negative_prompt,
            negative_prompt_images=negative_prompt_images,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            min_sequence_length=min_sequence_length,
        )

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            _, _, latent_height, latent_width = latents.shape
            image_seq_len = latent_height * latent_width
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.16),
            )
            scheduler_kwargs["mu"] = mu
        elif mu is not None:
            scheduler_kwargs["mu"] = mu
        if self.scheduler.config.get("use_scaled_shifting", None):
            scheduler_kwargs["resolution"] = width * height
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        latents_dtype = latents.dtype

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred, encoder_hidden_states = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                    flip_target=True,
                )
                noise_pred = noise_pred.float()

                # perform guidances
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = (noise_pred_text * self.guidance_scale) - (noise_pred_uncond * (self.guidance_scale - 1))
                    #noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    """
                    current_sigma = self.scheduler.sigmas[self.scheduler.step_index or i]
                    if t == self.scheduler.config.num_train_timesteps and self.raiflow_guidence_base_shift > 0:
                        # downscale cfg at the first step to fix everything becoming black issue
                        downscaled_guidance_scale = (self.guidance_scale / 2) / (self.raiflow_guidence_base_shift / self.scheduler.shift)
                    else:
                        # halve the cfg scale because we are using double cfg
                        downscaled_guidance_scale = self.guidance_scale / 2

                    if downscaled_guidance_scale > 1:
                        noise_pred_text_cfg = (noise_pred_text * downscaled_guidance_scale) - (noise_pred_uncond * (downscaled_guidance_scale - 1))
                        noise_pred_uncond_cfg = (noise_pred_uncond * downscaled_guidance_scale) - (noise_pred_text * (downscaled_guidance_scale - 1))
                        x0_pred_guidance_scale = self.raiflow_x0_pred_guidance_scale
                    else:
                        noise_pred_text_cfg = noise_pred_text
                        noise_pred_uncond_cfg = noise_pred_uncond
                        x0_pred_guidance_scale = self.raiflow_x0_pred_guidance_scale + downscaled_guidance_scale

                    x0_pred_text = latents - (noise_pred_text_cfg * current_sigma)
                    x0_pred_uncond = latents - (noise_pred_uncond_cfg * current_sigma)

                    noise_pred = noise_pred_text_cfg - x0_pred_guidance_scale * ((x0_pred_text - x0_pred_uncond) * current_sigma)
                    """

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t.float(), latents.float(), return_dict=False)[0]
                latents = latents.to(latents_dtype)
                noise_pred = noise_pred.to(latents_dtype) # for callback

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # Offload all models before decoding
        self.maybe_free_model_hooks()

        if output_type == "latent":
            image = latents
        elif getattr(self, "vae", None) is not None:
            latents = (latents.float() / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            if self.vae.config.force_upcast and latents_dtype == torch.float16:
                self.vae = self.vae.to(dtype=torch.float32)
                image = self.vae.decode(latents, return_dict=False)[0]
                self.vae = self.vae.to(dtype=latents_dtype)
            else:
                latents = latents.to(latents_dtype)
                image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        elif getattr(self, "image_encoder", None) is not None:
            if latents.device.type in {"xpu", "mps"}:
                latents = latents.to("cpu")
            image = self.image_encoder.decode(latents, return_type=output_type)
        else:
            raise RuntimeError("Neither a VAE or an Image Encoder is found to decode the latents")

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, encoder_hidden_states)

        return RaiFlowPipelineOutput(images=image, encoder_hidden_states=encoder_hidden_states)
