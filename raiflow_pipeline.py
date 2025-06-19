import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from diffusers.image_processor import PipelineImageInput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from transformers import Qwen2Tokenizer
from .raiflow_image_encoder import RaiFlowImageEncoder

from .raiflow_embedder import prepare_latent_image_ids, prepare_text_embed_ids
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
        tokenizer (`Qwen2Tokenizer`):
            Tokenizer of class
            [Qwen2Tokenizer](https://huggingface.co/docs/transformers/main/model_doc/qwen2#transformers.Qwen2Tokenizer).
        image_encoder ([`RaiFlowImageEncoder`], *optional*):
            RaiFlow JPEG encoder to encode and decode images to and from latent representations.
    """

    model_cpu_offload_seq = "transformer"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds", "noise_pred", "hidden_states", "encoder_hidden_states"]

    def __init__(
        self,
        transformer: RaiFlowTransformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        tokenizer: Qwen2Tokenizer,
        image_encoder: RaiFlowImageEncoder,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            scheduler=scheduler,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
        )

        self.patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) is not None else 2
        )

        if getattr(self, "image_encoder", None) is not None:
            self.latent_scale_factor = self.image_encoder.config.block_size
        else:
            self.latent_scale_factor = 16

        self.default_sample_size = (
            self.transformer.config.sample_size if getattr(self, "transformer", None) is not None else 128
        )

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        max_sequence_length: int = 1024,
        pad_to_multiple_of: int = 256,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            max_sequence_length (`int` defaults to 1024): Maximum sequence length to use with the `prompt`.
            pad_to_multiple_of (`int` defaults to 256): Pad the sequence length to a multiple of this value`.
        """
        device = device or self._execution_device

        prompt = prompt or self.tokenizer.decode(self.transformer.config.pad_token_id)
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or self.tokenizer.decode(self.transformer.config.pad_token_id)
            negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt_len = len(negative_prompt)

            if negative_prompt_len == 1 and batch_size != 1:
                negative_prompt = negative_prompt * batch_size
            elif negative_prompt_len != batch_size:
                raise ValueError(f"number of prompts and negative prompts must be the same but got {batch_size} and {negative_prompt_len}")

            negative_prompt.extend(prompt)
            prompt = negative_prompt

        prompt_embeds = self.tokenizer(
            text=prompt,
            padding="longest",
            pad_to_multiple_of=pad_to_multiple_of,
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt"
        ).input_ids

        if num_images_per_prompt != 1:
            # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
            batch_size, seq_len = prompt_embeds.shape
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
            prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len)
            prompt_embeds = prompt_embeds.to(device)

        return prompt_embeds

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
        pad_to_multiple_of=None,
    ):
        if (
            height % (self.latent_scale_factor * self.patch_size) != 0
            or width % (self.latent_scale_factor * self.patch_size) != 0
        ):
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.latent_scale_factor * self.patch_size} but are {height} and {width}."
                f"You can use height {height - height % (self.latent_scale_factor * self.patch_size)} and width {width - width % (self.latent_scale_factor * self.patch_size)}."
            )

        if isinstance(prompt, list) and isinstance(negative_prompt, list) and len(negative_prompt) != 1 and len(prompt) != len(negative_prompt):
            raise ValueError(f"number of prompts and negative prompts must be the same but got {len(prompt)} and {len(negative_prompt)}")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if max_sequence_length is not None and not isinstance(max_sequence_length, int):
            raise ValueError(
                f"`max_sequence_length` must be an integer but got: {type(max_sequence_length)}"
            )

        if pad_to_multiple_of is not None and not isinstance(pad_to_multiple_of, int):
            raise ValueError(
                f"`pad_to_multiple_of` must be an integer but got: {type(pad_to_multiple_of)}"
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
        prompt: Union[str, List[str]],
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        raiflow_x0_pred_guidance_scale: float = 1.0,
        raiflow_guidence_base_shift: float = 4.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        combined_rotary_emb: Optional[Tuple[torch.FloatTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.FloatTensor]] = None,
        text_rotary_emb: Optional[Tuple[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = None,
        pad_to_multiple_of: int = None,
        mu: Optional[float] = None,
    ) -> Union[RaiFlowPipelineOutput, Tuple[PipelineImageInput]]:
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation.
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
            raiflow_x0_pred_guidance_scale (`float`, *optional*, defaults to 1.0):
                Guidance scale used for classifier free guidence via the final image predictions.
            raiflow_guidence_base_shift (`float`, *optional*, defaults to 4.0):
                Base shift used to scale the cfg value on the first timestep.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            combined_rotary_emb (Tuple[`torch.FloatTensor`] of shape `(combined_sequence_len, 3)`, *optional*):
                Used for rotary positional embeddings. combined_sequence_len is encoder_seq_len + latents_seq_len.
            image_rotary_emb (Tuple[`torch.FloatTensor`] of shape `(latents_seq_len, 3)`, *optional*):
                Used for rotary positional embeddings for the latents.
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
            max_sequence_length (`int` defaults to None): Maximum sequence length to use with the `prompt`.
            pad_to_multiple_of (`int` defaults to None): Pad the sequence length to a multiple of this value`.
            mu (`float`, *optional*): `mu` value used for `dynamic_shifting`.

        Examples:

        Returns:
            [`~pipelines.raiflow.RaiFlowPipelineOutput`] or `tuple`:
            [`~pipelines.raiflow.RaiFlowPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        height = height or self.default_sample_size * self.latent_scale_factor
        width = width or self.default_sample_size * self.latent_scale_factor
        max_sequence_length = max_sequence_length or self.transformer.config.encoder_max_sequence_length
        pad_to_multiple_of = pad_to_multiple_of or self.transformer.config.pad_to_multiple_of

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        self._guidance_scale = guidance_scale
        self._raiflow_guidence_base_shift = raiflow_guidence_base_shift
        self._raiflow_x0_pred_guidance_scale = raiflow_x0_pred_guidance_scale

        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        device = self._execution_device
        dtype = self.transformer.embed_tokens.weight.dtype # pipe can be quantized

        prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )

        _, _, latent_height, latent_width = latents.shape
        _, encoder_seq_len = prompt_embeds.shape
        encoder_seq_len = encoder_seq_len + 2

        padded_height = latent_height + 2
        padded_width = latent_width + 2

        patched_height = padded_height // self.patch_size
        patched_width = padded_width // self.patch_size
        latents_seq_len = patched_height * patched_width

        # 5. Prepare timesteps
        scheduler_kwargs = {}
        if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
            mu = calculate_shift(
                latents_seq_len,
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

        if combined_rotary_emb is None:
            txt_ids = prepare_text_embed_ids(encoder_seq_len, device, dtype)
            img_ids = prepare_latent_image_ids(patched_height, patched_width, device, dtype)
            combined_ids = torch.cat((txt_ids, img_ids), dim=0)
            combined_rotary_emb = self.transformer.pos_embed(combined_ids, freqs_dtype=torch.float32)

        if image_rotary_emb is None:
            image_rotary_emb = (combined_rotary_emb[0][encoder_seq_len :], combined_rotary_emb[1][encoder_seq_len :])
        if text_rotary_emb is None:
            text_rotary_emb = (combined_rotary_emb[0][: encoder_seq_len], combined_rotary_emb[1][: encoder_seq_len])

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timesteps = self.scheduler.sigmas[self.scheduler.step_index or i].to(device, dtype=dtype).expand(latent_model_input.shape[0])

                noise_pred, hidden_states, encoder_hidden_states = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    combined_rotary_emb=combined_rotary_emb,
                    image_rotary_emb=image_rotary_emb,
                    text_rotary_emb=text_rotary_emb,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )
                noise_pred = noise_pred.float()

                # perform guidances
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = (noise_pred_text * self.guidance_scale) - (noise_pred_uncond * (self.guidance_scale - 1))
                    #noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    """
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

                    x0_pred_text = latents - (noise_pred_text_cfg * timesteps)
                    x0_pred_uncond = latents - (noise_pred_uncond_cfg * timesteps)

                    noise_pred = noise_pred_text_cfg - x0_pred_guidance_scale * ((x0_pred_text - x0_pred_uncond) * timesteps)
                    """

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t.float(), latents.float(), return_dict=False)[0]
                latents = latents.to(dtype)
                noise_pred = noise_pred.to(dtype) # for callback

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # Offload all models
        self.maybe_free_model_hooks()

        if output_type == "latent":
            image = latents
        else:
            if latents.device.type in {"xpu", "mps"}:
                latents = latents.to("cpu")
            image = self.image_encoder.decode(latents, return_type=output_type)

        if not return_dict:
            return (image, hidden_states, encoder_hidden_states)

        return RaiFlowPipelineOutput(images=image, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)
