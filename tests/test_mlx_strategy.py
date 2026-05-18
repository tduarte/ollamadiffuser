"""Tests for the MLX inference strategy (issue #7).

These tests run on any platform — when not on Apple Silicon they
exercise the platform-check refusal path; when on Apple Silicon they
mock the mflux model class so no weights need to download.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ollamadiffuser.core.config.settings import ModelConfig
from ollamadiffuser.core.inference.engine import _get_strategy
from ollamadiffuser.core.inference.strategies import mlx_strategy
from ollamadiffuser.core.inference.strategies.mlx_strategy import (
    MLXStrategy,
    SUPPORTED_MLX_VARIANTS,
    is_apple_silicon,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _make_config(**overrides) -> ModelConfig:
    """Build a ModelConfig that points the strategy at mflux."""
    base = {
        "name": "flux.1-schnell-mlx",
        "path": "/tmp/unused",
        "model_type": "mlx",
        "variant": "mlx-q8",
        "parameters": {
            "mlx_variant": "flux1",
            "mlx_model_name": "schnell",
            "quantize": 8,
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
        },
    }
    base.update(overrides)
    # ModelConfig accepts arbitrary kwargs depending on the codebase's settings
    # module; tolerate either dataclass or simple-namespace shapes.
    try:
        return ModelConfig(**base)
    except TypeError:
        # Fall back to a generic namespace if ModelConfig is restrictive.
        from types import SimpleNamespace
        return SimpleNamespace(**base)


def _stub_flux_class():
    """Build a MagicMock that acts like mflux's Flux1 class.

    Returns (cls_mock, instance_mock).
    """
    instance = MagicMock(name="Flux1Instance")
    # generate_image returns an object whose .image attribute is a PIL.Image
    fake_pil = Image.new("RGB", (64, 64), color=(0, 128, 255))
    generated = MagicMock(name="GeneratedImage", image=fake_pil)
    instance.generate_image.return_value = generated

    cls = MagicMock(name="Flux1Class")
    cls.from_name.return_value = instance
    return cls, instance


# --------------------------------------------------------------------------
# Dispatch + module constants
# --------------------------------------------------------------------------

class TestDispatch:
    def test_engine_dispatches_mlx_model_type(self):
        strategy = _get_strategy("mlx")
        assert isinstance(strategy, MLXStrategy)

    def test_supported_variants_constant(self):
        assert "flux1" in SUPPORTED_MLX_VARIANTS

    def test_is_apple_silicon_returns_bool(self):
        assert isinstance(is_apple_silicon(), bool)


# --------------------------------------------------------------------------
# Platform refusal
# --------------------------------------------------------------------------

class TestPlatformGuard:
    def test_refuses_non_apple_silicon(self):
        s = MLXStrategy()
        config = _make_config()
        with patch.object(mlx_strategy, "is_apple_silicon", return_value=False):
            assert s.load(config, device="cuda") is False
        assert s.pipeline is None
        assert s.is_loaded is False


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------

@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only runs on Apple Silicon")
class TestConfigValidation:
    def test_rejects_unknown_variant(self):
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": "nonsense",
            "mlx_model_name": "schnell",
        })
        assert s.load(config, device="mps") is False
        assert s.is_loaded is False

    def test_rejects_missing_model_name(self):
        s = MLXStrategy()
        config = _make_config(parameters={"mlx_variant": "flux1"})
        assert s.load(config, device="mps") is False

    def test_rejects_invalid_quantize(self):
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": "flux1",
            "mlx_model_name": "schnell",
            "quantize": 3,  # neither None, 4, nor 8
        })
        assert s.load(config, device="mps") is False


# --------------------------------------------------------------------------
# Loading + generation (mflux mocked out)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only runs on Apple Silicon")
class TestLoadAndGenerate:
    def test_load_calls_from_name_with_correct_args(self):
        cls_mock, _ = _stub_flux_class()
        s = MLXStrategy()
        with patch.object(MLXStrategy, "_resolve_model_class", return_value=cls_mock):
            ok = s.load(_make_config(), device="mps")
        assert ok is True
        cls_mock.from_name.assert_called_once_with("schnell", quantize=8)
        assert s.is_loaded
        assert s.device == "mps"

    def test_load_reports_failure_on_mflux_import_error(self):
        s = MLXStrategy()
        def raise_import(*a, **kw):
            raise ImportError("mflux missing")
        with patch.object(MLXStrategy, "_resolve_model_class", side_effect=raise_import):
            assert s.load(_make_config(), device="mps") is False
        assert s.is_loaded is False

    def test_generate_forwards_params_and_returns_pil(self):
        cls_mock, inst_mock = _stub_flux_class()
        s = MLXStrategy()
        with patch.object(MLXStrategy, "_resolve_model_class", return_value=cls_mock):
            s.load(_make_config(), device="mps")

        out = s.generate(
            prompt="A dog",
            num_inference_steps=4,
            guidance_scale=0.0,
            width=512,
            height=512,
            seed=42,
        )
        assert isinstance(out, Image.Image)
        inst_mock.generate_image.assert_called_once()
        called = inst_mock.generate_image.call_args.kwargs
        assert called["prompt"] == "A dog"
        assert called["seed"] == 42
        assert called["num_inference_steps"] == 4
        assert called["guidance"] == 0.0
        assert called["height"] == 512
        assert called["width"] == 512

    def test_generate_uses_registry_defaults_when_not_overridden(self):
        cls_mock, inst_mock = _stub_flux_class()
        s = MLXStrategy()
        with patch.object(MLXStrategy, "_resolve_model_class", return_value=cls_mock):
            s.load(_make_config(), device="mps")

        s.generate(prompt="hi", seed=1)
        called = inst_mock.generate_image.call_args.kwargs
        # Registry says num_inference_steps=4, guidance_scale=0.0
        assert called["num_inference_steps"] == 4
        assert called["guidance"] == 0.0

    def test_generate_without_seed_gets_random_int(self):
        cls_mock, inst_mock = _stub_flux_class()
        s = MLXStrategy()
        with patch.object(MLXStrategy, "_resolve_model_class", return_value=cls_mock):
            s.load(_make_config(), device="mps")

        s.generate(prompt="hi")  # no seed
        called = inst_mock.generate_image.call_args.kwargs
        assert isinstance(called["seed"], int)
        assert 0 <= called["seed"] < 2**31

    def test_generate_before_load_raises(self):
        s = MLXStrategy()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            s.generate(prompt="hi")

    def test_unload_clears_state(self):
        cls_mock, _ = _stub_flux_class()
        s = MLXStrategy()
        with patch.object(MLXStrategy, "_resolve_model_class", return_value=cls_mock):
            s.load(_make_config(), device="mps")
        assert s.is_loaded
        s.unload()
        assert s.is_loaded is False
        assert s._mlx_model is None
        assert s.pipeline is None


# --------------------------------------------------------------------------
# Unsupported features (clear-error contract)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only runs on Apple Silicon")
class TestUnsupported:
    def test_runtime_lora_returns_false_with_log(self, caplog):
        cls_mock, _ = _stub_flux_class()
        s = MLXStrategy()
        with patch.object(MLXStrategy, "_resolve_model_class", return_value=cls_mock):
            s.load(_make_config(), device="mps")

        with caplog.at_level("ERROR"):
            assert s.load_lora_runtime("some/repo") is False
        assert any("not supported" in r.message.lower() for r in caplog.records)

    def test_get_info_reports_backend_mlx(self):
        cls_mock, _ = _stub_flux_class()
        s = MLXStrategy()
        with patch.object(MLXStrategy, "_resolve_model_class", return_value=cls_mock):
            s.load(_make_config(), device="mps")
        info = s.get_info()
        assert info["backend"] == "mlx"
        assert info["mflux_variant"] == "flux1"
