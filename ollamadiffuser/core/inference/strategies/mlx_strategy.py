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
# is a matter of (a) extending this dict, (b) plumbing the variant
# through ``_resolve_model_and_config``, and (c) listing any required
# input kwargs in ``_VARIANT_REQUIRED_INPUTS``.
SUPPORTED_MLX_VARIANTS = frozenset({
    "flux1",             # FLUX.1 schnell / dev / krea-dev (text-to-image)
    "flux1-kontext",     # FLUX.1 Kontext (image editing)
    "flux1-fill",        # FLUX.1 Fill (inpaint / outpaint)
    "flux1-redux",       # FLUX.1 Redux (image-to-image variation)
    "flux1-depth",       # FLUX.1 Depth (depth-conditioned generation)
    "flux1-controlnet",  # FLUX.1 ControlNet (canny / upscaler)
    "flux2",             # FLUX.2 klein 4B/9B (text-to-image)
    "z_image",           # Z-Image / Z-Image-Turbo (text-to-image)
    "qwen-image",        # Qwen-Image / Qwen-Image-Edit
})

# Per-variant required-input kwargs (passed to ``generate()``). If any
# of these are absent, ``generate()`` raises ValueError early instead of
# letting mflux crash deep inside the pipeline.
_VARIANT_REQUIRED_INPUTS = {
    "flux1-kontext":    ["image"],
    "flux1-fill":       ["image", "mask_image"],
    "flux1-redux":      ["redux_images"],
    "flux1-depth":      ["image"],
    "flux1-controlnet": ["control_image"],
}

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
    doesn't apply here, so we override it.

    LoRAs: mflux takes ``lora_paths`` / ``lora_scales`` at **construction
    time**, not at runtime — there is no hot-swap. So ``load_lora_runtime``
    transparently **reloads** the mflux model with the LoRA applied (and
    ``unload_lora`` reloads it without). This works for FLUX.1, FLUX.2/Klein
    and Z-Image LoRAs. The reload is fast relative to generation because the
    weights are already cached locally.
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
            self._mlx_model = self._construct_mlx_model(
                model_cls, mflux_config, quantize)
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

        if variant == "flux1-fill":
            from mflux.models.flux.variants.fill.flux_fill import Flux1Fill
            from mflux.models.common.config.model_config import ModelConfig
            # Two factories: dev_fill (general) and dev_fill_catvton (try-on).
            factories = {
                "dev": ModelConfig.dev_fill,
                "fill": ModelConfig.dev_fill,
                "catvton": ModelConfig.dev_fill_catvton,
            }
            if mlx_model_name not in factories:
                raise ValueError(
                    f"flux1-fill: mlx_model_name must be one of {sorted(factories)}, "
                    f"got {mlx_model_name!r}"
                )
            return Flux1Fill, factories[mlx_model_name]()

        if variant == "flux1-redux":
            from mflux.models.flux.variants.redux.flux_redux import Flux1Redux
            from mflux.models.common.config.model_config import ModelConfig
            if mlx_model_name not in ("dev", "redux"):
                raise ValueError(
                    f"flux1-redux: mlx_model_name must be 'dev' or 'redux', "
                    f"got {mlx_model_name!r}"
                )
            return Flux1Redux, ModelConfig.dev_redux()

        if variant == "flux1-depth":
            from mflux.models.flux.variants.depth.flux_depth import Flux1Depth
            from mflux.models.common.config.model_config import ModelConfig
            if mlx_model_name not in ("dev", "depth"):
                raise ValueError(
                    f"flux1-depth: mlx_model_name must be 'dev' or 'depth', "
                    f"got {mlx_model_name!r}"
                )
            return Flux1Depth, ModelConfig.dev_depth()

        if variant == "flux1-controlnet":
            from mflux.models.flux.variants.controlnet.flux_controlnet import Flux1Controlnet
            from mflux.models.common.config.model_config import ModelConfig
            factories = {
                "canny":             ModelConfig.dev_controlnet_canny,
                "canny-dev":         ModelConfig.dev_controlnet_canny,
                "canny-schnell":     ModelConfig.schnell_controlnet_canny,
                "upscaler":          ModelConfig.dev_controlnet_upscaler,
                "upscaler-dev":      ModelConfig.dev_controlnet_upscaler,
            }
            if mlx_model_name not in factories:
                raise ValueError(
                    f"flux1-controlnet: mlx_model_name must be one of "
                    f"{sorted(factories)}, got {mlx_model_name!r}"
                )
            return Flux1Controlnet, factories[mlx_model_name]()

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

    @staticmethod
    def _construct_mlx_model(model_cls, mflux_config, quantize,
                             lora_paths=None, lora_scales=None):
        """Instantiate an mflux model, optionally with construction-time LoRAs.

        mflux's model constructors all accept ``lora_paths`` / ``lora_scales``
        (see ``Flux1``, ``Flux2Klein``, ``ZImage``). We only pass them when a
        LoRA is requested so the no-LoRA path stays byte-for-byte identical to
        the original ``model_cls(quantize=..., model_config=...)`` call.
        """
        kwargs = {"quantize": quantize, "model_config": mflux_config}
        if lora_paths:
            kwargs["lora_paths"] = list(lora_paths)
            kwargs["lora_scales"] = (
                list(lora_scales) if lora_scales is not None
                else [1.0] * len(lora_paths)
            )
        return model_cls(**kwargs)

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

        # mflux's variants accept different image-conditioning kwargs.
        # We accept the same convention from our existing strategies:
        #   kwargs["image"]         → image_path           (img2img, kontext, fill, depth)
        #   kwargs["mask_image"]    → masked_image_path    (fill / inpainting)
        #   kwargs["depth_image"]   → depth_image_path     (depth, optional)
        #   kwargs["control_image"] → controlnet_image_path (controlnet)
        #   kwargs["redux_images"]  → redux_image_paths    (redux, list)
        # Each may be a PIL Image, a path str, or (for redux) a list thereof.
        image_path           = self._materialize_image_path(kwargs.get("image"))
        masked_image_path    = self._materialize_image_path(kwargs.get("mask_image"))
        depth_image_path     = self._materialize_image_path(kwargs.get("depth_image"))
        controlnet_image_path = self._materialize_image_path(kwargs.get("control_image"))
        redux_image_paths    = self._materialize_image_path_list(kwargs.get("redux_images"))

        # Per-variant required-input enforcement. Fail loud here rather
        # than letting mflux crash mid-pipeline with a cryptic error.
        required = _VARIANT_REQUIRED_INPUTS.get(self._variant, [])
        provided = {
            "image":         image_path,
            "mask_image":    masked_image_path,
            "depth_image":   depth_image_path,
            "control_image": controlnet_image_path,
            "redux_images":  redux_image_paths,
        }
        missing = [k for k in required if not provided.get(k)]
        if missing:
            raise ValueError(
                f"MLX variant {self._variant!r} requires kwargs {missing}; "
                f"pass them to generate(), e.g. generate(prompt, image=<PIL|path>)."
            )

        # Negative prompt: mflux accepts None or str.
        neg = negative_prompt or None

        # Build the kwargs we'd LIKE to pass; filter to what this variant's
        # generate_image() actually accepts (Flux2Klein has no negative_prompt
        # parameter, for instance; Flux1Fill has masked_image_path).
        candidate_kwargs = {
            "seed": int(used_seed),
            "prompt": prompt,
            "num_inference_steps": int(steps),
            "height": int(height),
            "width": int(width),
            "guidance": float(guidance),
            "image_path": image_path,
            "masked_image_path": masked_image_path,
            "depth_image_path": depth_image_path,
            "controlnet_image_path": controlnet_image_path,
            "redux_image_paths": redux_image_paths,
            "image_strength": kwargs.get("strength"),
            "controlnet_strength": kwargs.get("controlnet_strength"),
            "redux_image_strengths": kwargs.get("redux_image_strengths"),
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

    @classmethod
    def _materialize_image_path_list(cls, images_arg) -> Optional[list]:
        """Normalize a list-of-images kwarg (PIL or paths) to list-of-paths."""
        if images_arg is None:
            return None
        if not isinstance(images_arg, (list, tuple)):
            # Allow scalar PIL/path too — wrap in a list.
            images_arg = [images_arg]
        return [cls._materialize_image_path(x) for x in images_arg if x is not None]

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

    # ----- LoRA (construction-time, via transparent reload) -------------

    def load_lora_runtime(self, repo_id, weight_name=None, scale=1.0) -> bool:
        """Apply a LoRA by reloading the mflux model with it.

        mflux takes ``lora_paths`` / ``lora_scales`` at construction time, so we
        rebuild the model with the LoRA rather than hot-swapping. ``repo_id`` +
        ``weight_name`` are resolved to a local ``.safetensors`` file (mflux
        cannot load from a bare HF repo id). On any failure the existing
        (base or previously-LoRA'd) model is left untouched.
        """
        if self._mlx_model is None:
            logger.error("MLX model not loaded — call load() before applying a LoRA.")
            return False

        lora_file = self._resolve_lora_file(repo_id, weight_name)
        if lora_file is None:
            logger.error(
                "Could not resolve a local LoRA weight file (repo_id=%r, "
                "weight_name=%r). mflux applies LoRAs from local .safetensors "
                "files — install it first, e.g. "
                "`ollamadiffuser hf pull <repo> --weight-name <file>`.",
                repo_id, weight_name,
            )
            return False

        try:
            new_model = self._rebuild_current_model(
                lora_paths=[str(lora_file)], lora_scales=[float(scale)])
        except Exception as e:
            logger.error(f"Failed to reload MLX model with LoRA: {e}", exc_info=True)
            return False

        # Swap in only after a successful build so a failure is non-destructive.
        self._mlx_model = new_model
        self.pipeline = new_model
        self.current_lora = {
            "repo_id": repo_id,
            "weight_name": weight_name,
            "scale": scale,
            "path": str(lora_file),
            "loaded": True,
        }
        logger.info(f"MLX LoRA applied from {lora_file} (scale={scale})")
        return True

    def unload_lora(self) -> bool:
        """Remove the LoRA by reloading the mflux model without any adapters."""
        if self._mlx_model is None:
            return False
        if not self.current_lora:
            return True  # nothing to do
        try:
            self._mlx_model = self._rebuild_current_model()
        except Exception as e:
            logger.error(f"Failed to reload MLX model without LoRA: {e}", exc_info=True)
            return False
        self.pipeline = self._mlx_model
        self.current_lora = None
        logger.info("MLX LoRA unloaded (model reloaded without adapters)")
        return True

    def _rebuild_current_model(self, lora_paths=None, lora_scales=None):
        """Re-instantiate the current mflux model, optionally with LoRAs.

        Resolves the class/config from the loaded model's registry parameters,
        so it produces the same model as :meth:`load` did (plus any LoRAs).
        """
        params = (self.model_config.parameters if self.model_config else {}) or {}
        variant = params.get("mlx_variant", "flux1")
        mlx_model_name = params.get("mlx_model_name")
        quantize = params.get("quantize")
        model_cls, mflux_config = self._resolve_model_and_config(variant, mlx_model_name)
        return self._construct_mlx_model(
            model_cls, mflux_config, quantize,
            lora_paths=lora_paths, lora_scales=lora_scales)

    @staticmethod
    def _resolve_lora_file(repo_id, weight_name=None):
        """Resolve (repo_id, weight_name) to a local ``.safetensors`` file.

        Handles the shapes lora_manager passes: a directory + filename (the
        common case, since the weight is already downloaded locally), a bare
        file path, or a directory containing a single weight file. Returns a
        ``Path`` or None if nothing local can be found.
        """
        from pathlib import Path

        base = Path(repo_id)
        if weight_name:
            cand = base / weight_name
            if cand.is_file():
                return cand
        if base.is_file():
            return base
        if base.is_dir():
            weights = sorted(base.glob("*.safetensors"))
            if weight_name:
                exact = [p for p in weights if p.name == weight_name]
                if exact:
                    return exact[0]
            elif len(weights) == 1:
                return weights[0]
        return None

    def get_info(self):
        info = super().get_info()
        info["backend"] = "mlx"
        info["mflux_variant"] = (
            (self.model_config.parameters or {}).get("mlx_variant")
            if self.model_config
            else None
        )
        return info
