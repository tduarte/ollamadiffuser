"""Tests for the Qwen-Image inference strategy.

These run on any platform: the diffusers/transformers classes are mocked so no
weights are downloaded and no real pipeline is built.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from PIL import Image

from ollamadiffuser.core.config.settings import ModelConfig
from ollamadiffuser.core.inference.engine import _get_strategy
from ollamadiffuser.core.inference.strategies.qwen_strategy import QwenImageStrategy


def _make_config(path, **param_overrides) -> ModelConfig:
    params = {"num_inference_steps": 30, "true_cfg_scale": 4.0}
    params.update(param_overrides)
    return ModelConfig(
        name="qwen-custom",
        path=str(path),
        model_type="qwen",
        parameters=params,
    )


def _write_ckpt(tmp_path, name="qwen_image_custom.safetensors"):
    ckpt = tmp_path / name
    ckpt.write_bytes(b"\x00" * 16)  # non-empty dummy
    return ckpt


class _MockDiffusers:
    """Stand-in for the diffusers module with the two classes the strategy uses."""

    def __init__(self):
        self.transformer_obj = MagicMock(name="transformer")
        self.pipe_instance = MagicMock(name="pipeline")
        self.pipe_instance.to.return_value = self.pipe_instance

        self.QwenImageTransformer2DModel = MagicMock()
        self.QwenImageTransformer2DModel.from_single_file.return_value = self.transformer_obj

        self.QwenImagePipeline = MagicMock()
        self.QwenImagePipeline.from_pretrained.return_value = self.pipe_instance


def _install_modules(monkeypatch, diffusers_mock, transformers_mock=None):
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_mock)
    if transformers_mock is not None:
        monkeypatch.setitem(sys.modules, "transformers", transformers_mock)


def test_dispatch_returns_qwen_strategy():
    assert isinstance(_get_strategy("qwen"), QwenImageStrategy)


def test_load_assembles_pipeline_with_transformer_override(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    diffusers_mock = _MockDiffusers()
    _install_modules(monkeypatch, diffusers_mock)

    strategy = QwenImageStrategy()
    ok = strategy.load(_make_config(tmp_path), "cpu")

    assert ok is True
    # Transformer loaded from the single file with base-repo config.
    sf_kwargs = diffusers_mock.QwenImageTransformer2DModel.from_single_file.call_args
    assert sf_kwargs.kwargs["config"] == "Qwen/Qwen-Image"
    assert sf_kwargs.kwargs["subfolder"] == "transformer"
    # Pipeline assembled from base repo with the transformer override.
    fp_args, fp_kwargs = diffusers_mock.QwenImagePipeline.from_pretrained.call_args
    assert fp_args[0] == "Qwen/Qwen-Image"
    assert fp_kwargs["transformer"] is diffusers_mock.transformer_obj
    # No custom text encoder was requested.
    assert "text_encoder" not in fp_kwargs


def test_load_missing_checkpoint_returns_false(tmp_path, monkeypatch):
    # No .safetensors written -> _find_checkpoint raises -> load returns False.
    diffusers_mock = _MockDiffusers()
    _install_modules(monkeypatch, diffusers_mock)

    strategy = QwenImageStrategy()
    assert strategy.load(_make_config(tmp_path), "cpu") is False


def test_load_with_custom_text_encoder(tmp_path, monkeypatch):
    _write_ckpt(tmp_path)
    diffusers_mock = _MockDiffusers()

    transformers_mock = MagicMock()
    te_obj = MagicMock(name="text_encoder")
    transformers_mock.Qwen2_5_VLForConditionalGeneration.from_pretrained.return_value = te_obj
    tok_obj = MagicMock(name="tokenizer")
    transformers_mock.Qwen2Tokenizer.from_pretrained.return_value = tok_obj

    _install_modules(monkeypatch, diffusers_mock, transformers_mock)

    strategy = QwenImageStrategy()
    ok = strategy.load(
        _make_config(tmp_path, text_encoder_repo="someone/uncensored-qwen2.5-vl"),
        "cpu",
    )

    assert ok is True
    transformers_mock.Qwen2_5_VLForConditionalGeneration.from_pretrained.assert_called_once()
    _, fp_kwargs = diffusers_mock.QwenImagePipeline.from_pretrained.call_args
    assert fp_kwargs["text_encoder"] is te_obj
    assert fp_kwargs["tokenizer"] is tok_obj


def test_generate_passes_true_cfg_scale(monkeypatch):
    strategy = QwenImageStrategy()
    strategy.device = "cpu"
    strategy.model_config = MagicMock()
    strategy.model_config.parameters = {"num_inference_steps": 25, "true_cfg_scale": 3.0}

    output = MagicMock()
    output.images = [Image.new("RGB", (8, 8))]
    strategy.pipeline = MagicMock(return_value=output)

    img = strategy.generate("a cat", seed=123)

    assert isinstance(img, Image.Image)
    call_kwargs = strategy.pipeline.call_args.kwargs
    assert call_kwargs["true_cfg_scale"] == 3.0
    assert call_kwargs["num_inference_steps"] == 25
    assert "guidance_scale" not in call_kwargs  # Qwen uses true_cfg_scale only


def test_generate_explicit_true_cfg_override(monkeypatch):
    strategy = QwenImageStrategy()
    strategy.device = "cpu"
    strategy.model_config = MagicMock()
    strategy.model_config.parameters = {}

    output = MagicMock()
    output.images = [Image.new("RGB", (8, 8))]
    strategy.pipeline = MagicMock(return_value=output)

    strategy.generate("a dog", seed=1, true_cfg_scale=7.5)
    assert strategy.pipeline.call_args.kwargs["true_cfg_scale"] == 7.5


def _lightning_strategy(name="qwen", parameters=None):
    strategy = QwenImageStrategy()
    strategy.device = "cpu"
    strategy.model_config = MagicMock()
    strategy.model_config.name = name
    strategy.model_config.parameters = parameters or {}
    output = MagicMock()
    output.images = [Image.new("RGB", (8, 8))]
    strategy.pipeline = MagicMock(return_value=output)
    return strategy


def test_is_lightning_inferred_from_filename():
    strategy = QwenImageStrategy()
    strategy.model_config = MagicMock()
    strategy.model_config.name = "qwen-rapid"
    assert strategy._is_lightning(
        {"single_file": "Qwen-Rapid-AIO-NSFW-v19_lightx-bf16.safetensors"}
    ) is True
    # "nolightx" contains the "lightx" substring but must NOT be treated as Lightning.
    assert strategy._is_lightning(
        {"single_file": "Qwen-Rapid-AIO-NSFW-v19_nolightx-bf16.safetensors"}
    ) is False
    strategy.model_config.name = "qwen-standard"
    assert strategy._is_lightning({"single_file": "model.safetensors"}) is False


def test_lightning_defaults_applied():
    # Explicit flag -> lightning defaults (8 steps, true_cfg_scale 1.0)
    strategy = _lightning_strategy(parameters={"lightning": True})
    strategy.generate("a cat", seed=1)
    kw = strategy.pipeline.call_args.kwargs
    assert kw["num_inference_steps"] == 8
    assert kw["true_cfg_scale"] == 1.0


def test_lightning_flag_false_uses_standard_defaults():
    # Even a "lightx" name is overridden by an explicit lightning=False.
    strategy = _lightning_strategy(
        name="qwen-lightx", parameters={"lightning": False}
    )
    strategy.generate("a cat", seed=1)
    kw = strategy.pipeline.call_args.kwargs
    assert kw["num_inference_steps"] == 30
    assert kw["true_cfg_scale"] == 4.0


def test_lightning_config_params_override_defaults():
    # Config params beat the Lightning defaults.
    strategy = _lightning_strategy(
        parameters={"lightning": True, "num_inference_steps": 6, "true_cfg_scale": 1.5}
    )
    strategy.generate("a cat", seed=1)
    kw = strategy.pipeline.call_args.kwargs
    assert kw["num_inference_steps"] == 6
    assert kw["true_cfg_scale"] == 1.5
