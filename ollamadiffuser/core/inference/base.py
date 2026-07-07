"""Base inference strategy for OllamaDiffuser"""

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from ..config.settings import ModelConfig

logger = logging.getLogger(__name__)


# Unified safety checker kwargs - use these in all from_pretrained calls
SAFETY_DISABLED_KWARGS = {
    "safety_checker": None,
    "requires_safety_checker": False,
    "feature_extractor": None,
}


class InferenceStrategy(ABC):
    """Abstract base class for all model inference strategies"""

    def __init__(self):
        self.pipeline = None
        self.model_config: Optional[ModelConfig] = None
        self.device: Optional[str] = None
        self.current_lora = None

    @abstractmethod
    def load(self, model_config: ModelConfig, device: str) -> bool:
        """Load the model pipeline"""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution",
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Image.Image:
        """Generate an image"""
        pass

    def unload(self) -> None:
        """Unload model and free memory"""
        if self.pipeline:
            self.pipeline = self.pipeline.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del self.pipeline
            self.pipeline = None
        self.model_config = None
        self.current_lora = None
        logger.info("Model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self.pipeline is not None

    def get_info(self) -> Dict[str, Any]:
        """Return model information"""
        if not self.model_config:
            return {}
        return {
            "name": self.model_config.name,
            "type": self.model_config.model_type,
            "device": self.device,
            "variant": self.model_config.variant,
            "parameters": self.model_config.parameters,
        }

    def _make_generator(self, seed: Optional[int], device: str) -> tuple:
        """Create a torch Generator with the given or random seed. Returns (generator, seed)."""
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        if device == "cpu":
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
        logger.info(f"Using seed: {seed}")
        return generator, seed

    def _get_dtype(self, device: str, prefer_bf16: bool = False) -> torch.dtype:
        """Get appropriate dtype for the given device"""
        if device == "cpu":
            return torch.float32
        if prefer_bf16:
            return torch.bfloat16
        return torch.float16

    def _move_to_device(self, device: str) -> bool:
        """Move pipeline to device with fallback to CPU"""
        try:
            self.pipeline = self.pipeline.to(device)
            logger.info(f"Pipeline moved to {device}")
            return True
        except Exception as e:
            logger.warning(f"Failed to move pipeline to {device}: {e}")
            if device != "cpu":
                logger.info("Falling back to CPU")
                self.device = "cpu"
                self.pipeline = self.pipeline.to("cpu")
                return True
            raise

    def _apply_memory_optimizations(self):
        """Apply common memory optimizations"""
        # Attention slicing splits attention into chunks to save VRAM.
        # Skip on MPS: unified memory makes it unnecessary, and it causes
        # NaN in float16 UNet output for some SDXL models (e.g. RealVisXL).
        if self.device != "mps" and hasattr(self.pipeline, "enable_attention_slicing"):
            self.pipeline.enable_attention_slicing()
            logger.info("Enabled attention slicing")
        if hasattr(self.pipeline, "enable_vae_tiling"):
            self.pipeline.enable_vae_tiling()
            logger.info("Enabled VAE tiling")
        if hasattr(self.pipeline, "enable_vae_slicing"):
            self.pipeline.enable_vae_slicing()
            logger.info("Enabled VAE slicing")

    def load_lora_runtime(
        self, repo_id: str, weight_name: str = None, scale: float = 1.0
    ) -> bool:
        """Load LoRA weights at runtime.

        Some community (kohya) LoRAs trip diffusers' text-encoder loader with an
        IndexError in get_peft_kwargs. When that happens we fall back to loading
        the UNet weights only — that carries the visual concept, and only the
        (usually minor) text-encoder adjustment is dropped.
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        try:
            # Replace any previously-loaded LoRA so switching LoRAs works
            # (peft errors if the 'default' adapter already exists).
            if self.current_lora:
                self.unload_lora()
            try:
                if weight_name:
                    self.pipeline.load_lora_weights(repo_id, weight_name=weight_name)
                else:
                    self.pipeline.load_lora_weights(repo_id)
            except (IndexError, ValueError, KeyError) as e:
                unet_sd = self._lora_unet_state_dict(repo_id, weight_name)
                if not unet_sd:
                    raise
                logger.warning(
                    f"Full LoRA load failed ({type(e).__name__}: {e}); "
                    "retrying with UNet weights only (text-encoder part skipped)"
                )
                self.pipeline.load_lora_weights(unet_sd, adapter_name="default")
            if hasattr(self.pipeline, "set_adapters") and scale != 1.0:
                self.pipeline.set_adapters(["default"], adapter_weights=[scale])
            self.current_lora = {
                "repo_id": repo_id,
                "weight_name": weight_name,
                "scale": scale,
                "loaded": True,
            }
            logger.info(f"LoRA loaded from {repo_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            return False

    @staticmethod
    def _lora_unet_state_dict(repo_id: str, weight_name: str = None):
        """Load only the UNet tensors of a local kohya-format .safetensors LoRA.

        Returns None for non-local sources or non-safetensors files, so the
        caller re-raises the original error rather than silently no-op'ing.
        """
        from pathlib import Path

        path = Path(repo_id)
        if weight_name:
            path = path / weight_name
        if not path.is_file() or path.suffix != ".safetensors":
            return None
        try:
            from safetensors.torch import load_file

            raw = load_file(str(path))
        except Exception:
            return None
        return {k: v for k, v in raw.items() if k.startswith("lora_unet")}

    def unload_lora(self) -> bool:
        """Unload LoRA weights"""
        if not self.pipeline:
            return False
        try:
            if hasattr(self.pipeline, "unload_lora_weights"):
                self.pipeline.unload_lora_weights()
                self.current_lora = None
                logger.info("LoRA unloaded")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unload LoRA: {e}")
            return False

    def load_textual_inversion(self, path: str, token: Optional[str] = None) -> bool:
        """Load a textual-inversion embedding into the pipeline.

        ``token`` is the prompt trigger word for the embedding (defaults to the
        embedding's own stored token). No-op-safe on pipelines that don't
        support textual inversion.
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        if not hasattr(self.pipeline, "load_textual_inversion"):
            logger.warning("This pipeline does not support textual inversion")
            return False
        try:
            if token:
                self.pipeline.load_textual_inversion(path, token=token)
            else:
                self.pipeline.load_textual_inversion(path)
            logger.info(f"Loaded textual inversion from {path} (token={token})")
            return True
        except Exception as e:
            logger.error(f"Failed to load textual inversion: {e}")
            return False

    def attach_vae(self, path: str) -> bool:
        """Replace the pipeline's VAE with a single-file VAE from ``path``.

        Persists until the model is reloaded. The original VAE is remembered so
        :meth:`restore_vae` can put it back.
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded")
        try:
            from diffusers import AutoencoderKL

            dtype = self._get_dtype(self.device or "cpu")
            vae = AutoencoderKL.from_single_file(path, torch_dtype=dtype)
            vae = vae.to(self.device)
            if not hasattr(self, "_original_vae"):
                self._original_vae = getattr(self.pipeline, "vae", None)
            self.pipeline.vae = vae
            logger.info(f"Attached VAE from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to attach VAE: {e}")
            return False

    def restore_vae(self) -> bool:
        """Restore the pipeline's original VAE if one was replaced."""
        if not self.pipeline or not getattr(self, "_original_vae", None):
            return False
        self.pipeline.vae = self._original_vae
        del self._original_vae
        logger.info("Restored original VAE")
        return True

    @staticmethod
    def _sanitize_image(image: Image.Image) -> Image.Image:
        """Clamp NaN/Inf pixels to avoid 'invalid value encountered in cast' on MPS."""
        arr = np.array(image, dtype=np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            logger.warning("NaN/Inf detected in generated image — clamping to valid range")
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        return image

    def _create_error_image(self, error_msg: str, prompt: str) -> Image.Image:
        """Create an error placeholder image"""
        from PIL import ImageDraw, ImageFont

        img = Image.new("RGB", (512, 512), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        draw.text((10, 10), f"Error: {error_msg}", fill=(255, 0, 0), font=font)
        prompt_display = prompt[:50] + "..." if len(prompt) > 50 else prompt
        draw.text(
            (10, 30), f"Prompt: {prompt_display}", fill=(0, 0, 0), font=font
        )
        return img
