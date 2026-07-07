"""Qwen-Image inference strategy.

Loads a single-file Qwen-Image transformer ``.safetensors`` checkpoint and
assembles a full :class:`QwenImagePipeline` by borrowing the VAE, text encoder,
tokenizer and scheduler from a base Qwen-Image diffusers repo (default
``Qwen/Qwen-Image``). diffusers exposes ``QwenImageTransformer2DModel.from_single_file``
but *not* ``QwenImagePipeline.from_single_file``, so the pipeline has to be built
component-by-component.

Qwen-Image is driven by ``true_cfg_scale`` (classifier-free guidance), not the
usual ``guidance_scale``.

Optional ``model_config.parameters``:
  - ``base_repo``: repo id / local path providing the non-transformer components
    (default ``Qwen/Qwen-Image``).
  - ``pipeline_class``: diffusers pipeline class name (default ``QwenImagePipeline``).
  - ``single_file``: explicit checkpoint filename inside the model path; when
    omitted the first ``*.safetensors`` found is used.
  - ``text_encoder_repo`` / ``tokenizer_repo`` / ``text_encoder_class``: swap in a
    custom (e.g. uncensored) Qwen2.5-VL-compatible text encoder. See the module
    plan for the architecture-compatibility constraint.
  - ``lightning``: force-enable (``true``) or disable (``false``) distilled
    few-step defaults. When omitted it is inferred from the model name / filename.
  - ``num_inference_steps`` / ``true_cfg_scale``: explicit generation defaults that
    override the standard *and* Lightning-inferred defaults.

Generation defaults depend on whether the checkpoint is Lightning-distilled:
standard → 30 steps, ``true_cfg_scale`` 4.0; Lightning → 8 steps,
``true_cfg_scale`` 1.0 (higher CFG burns distilled models).
"""

import glob
import logging
import os
from typing import Optional

from PIL import Image

from ..base import InferenceStrategy
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)

DEFAULT_BASE_REPO = "Qwen/Qwen-Image"

# Substrings in a model name / checkpoint filename that indicate a distilled
# few-step ("Lightning"-style) checkpoint. These want low step counts and
# true_cfg_scale ~1.0 — standard CFG/step defaults burn or waste compute.
_LIGHTNING_MARKERS = ("lightning", "lightx", "lcm", "hyper", "turbo")


class QwenImageStrategy(InferenceStrategy):
    """Strategy for single-file Qwen-Image checkpoints (assembled pipeline)."""

    def _is_lightning(self, params: dict) -> bool:
        """Whether this checkpoint is a distilled few-step (Lightning) variant.

        Honors an explicit ``parameters.lightning`` flag; otherwise infers from
        the model name and single-file filename.
        """
        if "lightning" in params:
            return bool(params["lightning"])
        haystack = f"{self.model_config.name} {params.get('single_file', '')}".lower()
        # Strip explicit "no-lightning" tokens first, otherwise the "lightx"
        # marker would false-match filenames like "..._nolightx-bf16.safetensors".
        for negative in ("nolightx", "no-lightx", "no_lightx", "nolightning"):
            haystack = haystack.replace(negative, "")
        return any(marker in haystack for marker in _LIGHTNING_MARKERS)

    def _find_checkpoint(self, model_config: ModelConfig, params: dict) -> str:
        """Locate the single-file transformer checkpoint inside the model path."""
        if params.get("single_file"):
            ckpt = os.path.join(model_config.path, params["single_file"])
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"single_file not found: {ckpt}")
            return ckpt
        candidates = sorted(
            glob.glob(os.path.join(model_config.path, "**", "*.safetensors"), recursive=True)
        )
        if not candidates:
            raise FileNotFoundError(
                f"No .safetensors checkpoint found under {model_config.path}"
            )
        if len(candidates) > 1:
            logger.warning(
                f"Multiple .safetensors found, using first: {candidates[0]} "
                f"(set parameters.single_file to override)"
            )
        return candidates[0]

    def load(self, model_config: ModelConfig, device: str) -> bool:
        try:
            import diffusers
            from diffusers import QwenImageTransformer2DModel
            from ...config.settings import settings

            self.device = device
            self.model_config = model_config

            params = model_config.parameters or {}
            base_repo = params.get("base_repo", DEFAULT_BASE_REPO)
            pipeline_class_name = params.get("pipeline_class", "QwenImagePipeline")
            pipeline_cls = getattr(diffusers, pipeline_class_name, None)
            if pipeline_cls is None:
                logger.error(
                    f"Pipeline class '{pipeline_class_name}' not found in diffusers. "
                    "You may need to upgrade: pip install --upgrade diffusers"
                )
                return False

            ckpt_path = self._find_checkpoint(model_config, params)
            dtype = self._get_dtype(device, prefer_bf16=True)
            cache_dir = str(settings.cache_dir)

            # 1) Transformer from the user's single-file checkpoint. The config is
            #    pulled from the base repo's transformer/ subfolder.
            logger.info(f"Loading Qwen-Image transformer from single file: {ckpt_path}")
            transformer = QwenImageTransformer2DModel.from_single_file(
                ckpt_path,
                config=base_repo,
                subfolder="transformer",
                torch_dtype=dtype,
                cache_dir=cache_dir,
            )

            # 2) Optional custom (e.g. uncensored) text encoder / tokenizer override.
            overrides = {}
            te_repo = params.get("text_encoder_repo")
            if te_repo:
                import transformers

                te_class_name = params.get(
                    "text_encoder_class", "Qwen2_5_VLForConditionalGeneration"
                )
                te_cls = getattr(transformers, te_class_name, None)
                if te_cls is None:
                    logger.error(
                        f"Text encoder class '{te_class_name}' not found in transformers."
                    )
                    return False
                logger.info(f"Loading custom text encoder from: {te_repo}")
                overrides["text_encoder"] = te_cls.from_pretrained(
                    te_repo, torch_dtype=dtype, cache_dir=cache_dir
                )
                # Tokenizer usually unchanged; fall back to the base repo's tokenizer
                # if the custom repo doesn't ship one.
                tok_repo = params.get("tokenizer_repo", te_repo)
                try:
                    overrides["tokenizer"] = transformers.Qwen2Tokenizer.from_pretrained(
                        tok_repo, cache_dir=cache_dir
                    )
                except Exception as e:
                    logger.info(
                        f"No tokenizer at {tok_repo} ({e}); using base repo tokenizer"
                    )

            # 3) Assemble the pipeline from the base repo, overriding components.
            logger.info(f"Assembling {pipeline_class_name} from base repo: {base_repo}")
            self.pipeline = pipeline_cls.from_pretrained(
                base_repo,
                transformer=transformer,
                torch_dtype=dtype,
                cache_dir=cache_dir,
                **overrides,
            )

            self._move_to_device(device)
            self._apply_memory_optimizations()

            logger.info(f"Qwen-Image model {model_config.name} loaded on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Qwen-Image model: {e}")
            return False

    def generate(
        self,
        prompt: str,
        negative_prompt: str = " ",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        params = self.model_config.parameters or {}
        # Lightning-distilled checkpoints want few steps and true_cfg_scale ~1.0.
        # Explicit call args and config params still override these defaults.
        lightning = self._is_lightning(params)
        default_steps = 8 if lightning else 30
        default_cfg = 1.0 if lightning else 4.0

        steps = (
            num_inference_steps
            if num_inference_steps is not None
            else params.get("num_inference_steps", default_steps)
        )
        # Qwen-Image uses true_cfg_scale, not guidance_scale. Accept an explicit
        # override via kwargs, else the caller's guidance_scale, else the default.
        true_cfg = kwargs.get("true_cfg_scale")
        if true_cfg is None:
            true_cfg = guidance_scale if guidance_scale is not None else params.get(
                "true_cfg_scale", default_cfg
            )

        generator, used_seed = self._make_generator(seed, self.device)

        gen_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": steps,
            "true_cfg_scale": true_cfg,
            "width": width,
            "height": height,
            "generator": generator,
        }

        step_cb = self._diffusers_step_callback(kwargs.get("progress_callback"), steps)
        if step_cb is not None:
            gen_kwargs["callback_on_step_end"] = step_cb

        try:
            logger.info(
                f"Generating Qwen-Image ({'lightning' if lightning else 'standard'}): "
                f"steps={steps}, true_cfg_scale={true_cfg}, seed={used_seed}"
            )
            output = self.pipeline(**gen_kwargs)
            return output.images[0]
        except Exception as e:
            logger.error(f"Qwen-Image generation failed: {e}")
            return self._create_error_image(str(e), prompt)
