"""MLX inference strategy for Apple Silicon.

Routes through `mflux <https://github.com/filipstrand/mflux>`_ — a pure
MLX implementation of FLUX-family models (and a few others). On
M-series Macs MLX typically runs **2-3× faster** than the PyTorch+MPS
path that the other strategies in this package use, because MLX targets
the Metal/Neural Engine stack natively without the PyTorch wrapper.

This is an **Apple-Silicon-only strategy.** It refuses to load on
Linux/Windows or Intel Macs.

Registry entries opt in via::

    model_type: "mlx"
    parameters:
      mlx_variant: "flux1"       # currently: "flux1" (txt2img)
      mlx_model_name: "schnell"  # passed to mflux's ModelConfig.from_name():
                                 #   "schnell" | "dev" | "krea-dev"
      quantize: 8                # int | null. 4 or 8 bits via MLX quant.

      # Generation defaults (overridable per-call)
      num_inference_steps: 4
      guidance_scale: 0.0
      max_sequence_length: 256

Tracking issue: https://github.com/LocalKinAI/ollamadiffuser/issues/7
"""
from __future__ import annotations

import logging
import platform
import random
from typing import Optional

from PIL import Image

from ..base import InferenceStrategy
from ...config.settings import ModelConfig

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Public constants
# --------------------------------------------------------------------------

# Variants this strategy can route. Adding a new mflux-supported family
# is a matter of (a) extending this dict and (b) plumbing the variant
# through ``_resolve_model_and_config``.
SUPPORTED_MLX_VARIANTS = frozenset({
    "flux1",          # FLUX.1 schnell / dev / krea-dev (text-to-image)
    "flux1-kontext",  # FLUX.1 Kontext (image editing)
    "flux2",          # FLUX.2 klein 4B/9B (text-to-image)
    "z_image",        # Z-Image / Z-Image-Turbo (text-to-image)
    "qwen-image",     # Qwen-Image / Qwen-Image-Edit
})

# mflux quantization values it actually accepts. None means "no quant".
_VALID_QUANTIZE = (None, 4, 8)


# --------------------------------------------------------------------------
# Apple-Silicon detection
# --------------------------------------------------------------------------

def is_apple_silicon() -> bool:
    """Return True iff we're running on macOS arm64."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


# --------------------------------------------------------------------------
# Strategy
# --------------------------------------------------------------------------

class MLXStrategy(InferenceStrategy):
    """Apple-Silicon-native inference via mflux.

    Currently supports FLUX.1 text-to-image (schnell / dev / krea-dev).
    Other mflux model families (Flux2, Z-Image, Qwen-Image) are wired
    via the same ``mlx_variant`` parameter — see the constant
    ``SUPPORTED_MLX_VARIANTS``.

    Note: mflux uses MLX arrays under the hood, not PyTorch tensors.
    The base class's ``unload()`` calls ``pipeline.to("cpu")`` which
    doesn't apply here, so we override it. ``load_lora_runtime`` from
    the base also assumes diffusers — mflux LoRAs are passed at
    construction time, so we override that too with a clear error.
    """

    def __init__(self) -> None:
        super().__init__()
        # Track the mflux model on a separate attribute so the base
        # class's `pipeline`-based checks still work for "loaded?" queries
        # — we store the same object in both ``self.pipeline`` and
        # ``self._mlx_model`` to keep is_loaded semantics intact while
        # still letting MLX-specific code path use the typed attribute.
        self._mlx_model = None
        # Cached for capability checks in generate() (e.g. Kontext needs image).
        self._variant: Optional[str] = None

    # ----- Loading ------------------------------------------------------

    def load(self, model_config: ModelConfig, device: str) -> bool:
        if not is_apple_silicon():
            logger.error(
                "MLXStrategy requires macOS on Apple Silicon (arm64). "
                f"Detected platform: {platform.system()}/{platform.machine()}. "
                "Use the FLUX / GenericPipeline strategies instead on this host."
            )
            return False

        params = model_config.parameters or {}
        variant = params.get("mlx_variant", "flux1")
        mlx_model_name = params.get("mlx_model_name")
        quantize = params.get("quantize")

        if variant not in SUPPORTED_MLX_VARIANTS:
            logger.error(
                f"Unknown mlx_variant={variant!r}. "
                f"Supported: {sorted(SUPPORTED_MLX_VARIANTS)}"
            )
            return False
        if not mlx_model_name:
            logger.error(
                "MLXStrategy requires parameters.mlx_model_name in the "
                "registry entry (e.g. 'schnell' or 'dev')."
            )
            return False
        if quantize not in _VALID_QUANTIZE:
            logger.error(
                f"parameters.quantize must be one of {_VALID_QUANTIZE}, "
                f"got {quantize!r}."
            )
            return False

        try:
            model_cls, mflux_config = self._resolve_model_and_config(
                variant, mlx_model_name
            )
        except ImportError as e:
            logger.error(
                "mflux is not installed. Install with: "
                "pip install 'ollamadiffuser[mlx]'  (or: pip install mflux). "
                f"Original: {e}"
            )
            return False
        except ValueError as e:
            logger.error(str(e))
            return False

        logger.info(
            f"Loading mflux {variant} model '{mlx_model_name}' "
            f"(quantize={quantize})..."
        )
        try:
            self._mlx_model = model_cls(quantize=quantize, model_config=mflux_config)
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}", exc_info=True)
            return False

        # Cache the variant so generate() can branch on capabilities.
        self._variant = variant
        # Mirror onto self.pipeline so base class's is_loaded property works.
        self.pipeline = self._mlx_model
        self.model_config = model_config
        # MLX runs on unified memory; we report "mps" for compatibility with
        # caller code paths that branch on device.
        self.device = "mps"
        logger.info(
            f"MLX model '{model_config.name}' loaded "
            f"({variant} / {mlx_model_name} / quantize={quantize})"
        )
        return True

    @staticmethod
    def _resolve_model_and_config(variant: str, mlx_model_name: str):
        """Return ``(ModelClass, mflux_model_config)`` for a registry entry.

        All four mflux families share the constructor signature
        ``Cls(quantize=..., model_config=...)`` but the ``ModelConfig``
        comes from variant-specific factories, e.g.
        ``ModelConfig.flux2_klein_4b()``.

        ``mlx_model_name`` selects the specific config within a family:
          flux1         → "schnell" | "dev" | "krea-dev"
          flux1-kontext → "dev"  (Kontext is single-config in mflux today)
          flux2         → "klein-4b" | "klein-9b"
                          | "klein-base-4b" | "klein-base-9b"
          z_image       → "z-image" | "z-image-turbo"
          qwen-image    → "qwen-image" | "qwen-image-edit"

        Imports are local so this module is safe to import on non-MLX
        platforms.
        """
        if variant == "flux1":
            from mflux.models.flux.variants.txt2img.flux import Flux1
            from mflux.models.common.config.model_config import ModelConfig
            mc = ModelConfig.from_name(model_name=mlx_model_name, base_model=None)
            return Flux1, mc

        if variant == "flux1-kontext":
            from mflux.models.flux.variants.kontext.flux_kontext import Flux1Kontext
            from mflux.models.common.config.model_config import ModelConfig
            # mflux only ships dev_kontext() today; accept "dev" as the alias.
            if mlx_model_name not in ("dev", "kontext"):
                raise ValueError(
                    f"flux1-kontext: mlx_model_name must be 'dev' or 'kontext', "
                    f"got {mlx_model_name!r}"
                )
            return Flux1Kontext, ModelConfig.dev_kontext()

        if variant == "flux2":
            from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
            from mflux.models.common.config.model_config import ModelConfig
            factories = {
                "klein-4b": ModelConfig.flux2_klein_4b,
                "klein-9b": ModelConfig.flux2_klein_9b,
                "klein-base-4b": ModelConfig.flux2_klein_base_4b,
                "klein-base-9b": ModelConfig.flux2_klein_base_9b,
            }
            if mlx_model_name not in factories:
                raise ValueError(
                    f"flux2: mlx_model_name must be one of {sorted(factories)}, "
                    f"got {mlx_model_name!r}"
                )
            return Flux2Klein, factories[mlx_model_name]()

        if variant == "z_image":
            from mflux.models.z_image.variants.z_image import ZImage
            from mflux.models.common.config.model_config import ModelConfig
            factories = {
                "z-image": ModelConfig.z_image,
                "z-image-turbo": ModelConfig.z_image_turbo,
            }
            if mlx_model_name not in factories:
                raise ValueError(
                    f"z_image: mlx_model_name must be one of {sorted(factories)}, "
                    f"got {mlx_model_name!r}"
                )
            return ZImage, factories[mlx_model_name]()

        if variant == "qwen-image":
            from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
            from mflux.models.common.config.model_config import ModelConfig
            factories = {
                "qwen-image": ModelConfig.qwen_image,
                "qwen-image-edit": ModelConfig.qwen_image_edit,
            }
            if mlx_model_name not in factories:
                raise ValueError(
                    f"qwen-image: mlx_model_name must be one of {sorted(factories)}, "
                    f"got {mlx_model_name!r}"
                )
            return QwenImage, factories[mlx_model_name]()

        raise ValueError(f"Unhandled mlx_variant: {variant}")

    # ----- Generation ---------------------------------------------------

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        if self._mlx_model is None:
            raise RuntimeError("Model not loaded — call load() first")

        params = self.model_config.parameters if self.model_config else {}
        params = params or {}

        # Resolve generation params, allowing per-call override of registry defaults.
        steps = (
            num_inference_steps
            if num_inference_steps is not None
            else int(params.get("num_inference_steps", 4))
        )
        guidance = (
            guidance_scale
            if guidance_scale is not None
            else float(params.get("guidance_scale", 0.0))
        )

        # Seed (mflux requires an explicit int — no None allowed).
        used_seed = seed if seed is not None else random.randint(0, 2**31 - 1)

        # mflux's generate_image accepts img2img via image_path + image_strength.
        # We accept the same convention from our existing strategies: kwargs["image"]
        # may be a PIL Image (in-memory) or a path str.
        image_arg = kwargs.get("image")
        image_strength = kwargs.get("strength")
        image_path = self._materialize_image_path(image_arg) if image_arg else None

        # Kontext is an image-editing model — it REQUIRES an input image.
        # Fail loud rather than letting mflux error mid-pipeline.
        if self._variant == "flux1-kontext" and image_path is None:
            raise ValueError(
                "flux1-kontext is an image-editing model — "
                "pass image=<PIL.Image or path> to generate()."
            )

        # Negative prompt: mflux accepts None or str.
        neg = negative_prompt or None

        # Build the kwargs we'd LIKE to pass; filter to what this variant's
        # generate_image() actually accepts (Flux2Klein has no negative_prompt
        # parameter, for instance).
        candidate_kwargs = {
            "seed": int(used_seed),
            "prompt": prompt,
            "num_inference_steps": int(steps),
            "height": int(height),
            "width": int(width),
            "guidance": float(guidance),
            "image_path": image_path,
            "image_strength": image_strength,
            "negative_prompt": neg,
        }
        try:
            import inspect
            accepted = set(
                inspect.signature(self._mlx_model.generate_image).parameters.keys()
            )
        except (TypeError, ValueError):
            accepted = set(candidate_kwargs.keys())  # fall back to everything

        gen_kwargs = {}
        for key, val in candidate_kwargs.items():
            if key in accepted:
                gen_kwargs[key] = val
            elif val not in (None, ""):
                logger.debug(
                    f"MLX variant {self._variant!r} doesn't accept "
                    f"{key!r}={val!r}; dropping."
                )

        logger.info(
            f"MLX generating ({self._variant}): "
            f"steps={steps}, guidance={guidance}, seed={used_seed}, "
            f"size={width}x{height}"
        )
        try:
            generated = self._mlx_model.generate_image(**gen_kwargs)
            # generated is mflux.utils.generated_image.GeneratedImage; .image is PIL.
            return self._sanitize_image(generated.image)
        except Exception as e:
            logger.error(f"MLX generation failed: {e}", exc_info=True)
            return self._create_error_image(str(e), prompt)

    @staticmethod
    def _materialize_image_path(image_arg) -> Optional[str]:
        """Normalize an image kwarg to a path string mflux can read."""
        if image_arg is None:
            return None
        if isinstance(image_arg, str):
            return image_arg
        # PIL Image → save to a tempfile.
        if hasattr(image_arg, "save"):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.close()
            image_arg.save(tmp.name)
            return tmp.name
        raise TypeError(
            f"Unsupported image type for MLX img2img: {type(image_arg).__name__}"
        )

    # ----- Lifecycle ----------------------------------------------------

    def unload(self) -> None:
        """Release mflux model. Base class's torch-specific cleanup is skipped."""
        if self._mlx_model is not None:
            del self._mlx_model
            self._mlx_model = None
        self.pipeline = None
        self.model_config = None
        self.current_lora = None
        self._variant = None
        logger.info("MLX model unloaded")

    # ----- Unsupported on this backend ----------------------------------

    def load_lora_runtime(self, repo_id, weight_name=None, scale=1.0) -> bool:
        """LoRA at runtime is not yet wired for the MLX backend.

        mflux expects ``lora_paths`` and ``lora_scales`` at construction
        time, not at runtime. Supporting hot-swap would require
        reconstructing the Flux1 instance — deferred until users ask.
        """
        logger.error(
            "Runtime LoRA loading is not supported on the MLX backend yet. "
            "mflux takes lora_paths/lora_scales at construction time. "
            "Track via https://github.com/LocalKinAI/ollamadiffuser/issues/7"
        )
        return False

    def unload_lora(self) -> bool:
        return False

    def get_info(self):
        info = super().get_info()
        info["backend"] = "mlx"
        info["mflux_variant"] = (
            (self.model_config.parameters or {}).get("mlx_variant")
            if self.model_config
            else None
        )
        return info
