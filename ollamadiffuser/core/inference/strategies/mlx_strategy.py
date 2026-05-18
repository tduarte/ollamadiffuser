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
# through ``_resolve_model_class``.
SUPPORTED_MLX_VARIANTS = frozenset({"flux1"})

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
            model_cls = self._resolve_model_class(variant)
        except ImportError as e:
            logger.error(
                "mflux is not installed. Install with: "
                "pip install 'ollamadiffuser[mlx]'  (or: pip install mflux). "
                f"Original: {e}"
            )
            return False

        logger.info(
            f"Loading mflux {variant} model '{mlx_model_name}' "
            f"(quantize={quantize})..."
        )
        try:
            self._mlx_model = model_cls.from_name(mlx_model_name, quantize=quantize)
        except Exception as e:
            logger.error(f"Failed to load MLX model: {e}", exc_info=True)
            return False

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
    def _resolve_model_class(variant: str):
        """Return the mflux class for a given variant.

        Imports are local so this module can be imported on non-MLX
        platforms without pulling mflux in.
        """
        if variant == "flux1":
            from mflux.models.flux.variants.txt2img.flux import Flux1
            return Flux1
        # Future variants (mflux already ships these under different
        # subpackages; wire them when registry entries land):
        #   - "flux2" → mflux.models.flux2.variants.txt2img.flux2.Flux2
        #   - "z_image" → mflux.models.z_image.variants.txt2img.z_image.ZImage
        #   - "flux1-kontext" → mflux.models.flux.variants.kontext.*
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

        # Negative prompt: mflux accepts None or str.
        neg = negative_prompt or None

        logger.info(
            f"MLX generating: steps={steps}, guidance={guidance}, "
            f"seed={used_seed}, size={width}x{height}"
        )
        try:
            generated = self._mlx_model.generate_image(
                seed=int(used_seed),
                prompt=prompt,
                num_inference_steps=int(steps),
                height=int(height),
                width=int(width),
                guidance=float(guidance),
                image_path=image_path,
                image_strength=image_strength,
                negative_prompt=neg,
            )
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
