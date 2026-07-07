"""Tests for per-step generation progress callbacks.

Covers the shared base helpers, engine passthrough, and the diffusers
`callback_on_step_end` wiring (incl. the generic-strategy signature guard).
The MLX mflux-registry path is tested in test_mlx_strategy.py.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from PIL import Image

from ollamadiffuser.core.inference.base import InferenceStrategy
from ollamadiffuser.core.inference.engine import InferenceEngine
from ollamadiffuser.core.config.settings import ModelConfig


class _Dummy(InferenceStrategy):
    def load(self, *a, **k):
        return True

    def generate(self, *a, **k):
        return Image.new("RGB", (8, 8))


# ----- base helpers --------------------------------------------------------

def test_diffusers_step_callback_reports_1based_and_returns_kwargs():
    seen = []
    cb = _Dummy()._diffusers_step_callback(lambda s, t: seen.append((s, t)), 40)
    # diffusers calls positionally: (pipe, step, timestep, callback_kwargs); step is 0-based
    out = cb(None, 11, 0, {"latents": 123})
    assert seen == [(12, 40)]
    assert out == {"latents": 123}


def test_diffusers_step_callback_none_when_no_callback():
    assert _Dummy()._diffusers_step_callback(None, 40) is None


def test_safe_progress_swallows_errors_and_none():
    def boom(step, total):
        raise ValueError("nope")
    # Neither of these should raise.
    InferenceStrategy._safe_progress(boom, 1, 8)
    InferenceStrategy._safe_progress(None, 1, 8)


def test_safe_progress_passes_message_to_3arg_callback():
    seen = []
    InferenceStrategy._safe_progress(
        lambda step, total, message=None: seen.append((step, total, message)),
        8, 8, "Decoding image…")
    assert seen == [(8, 8, "Decoding image…")]


def test_safe_progress_tolerates_legacy_2arg_callback():
    seen = []
    # A 2-arg callback must still work (message dropped, no crash).
    InferenceStrategy._safe_progress(
        lambda step, total: seen.append((step, total)), 3, 8, "ignored")
    assert seen == [(3, 8)]


# ----- engine passthrough --------------------------------------------------

def test_engine_forwards_progress_callback_to_strategy():
    captured = {}

    class Stub(InferenceStrategy):
        def load(self, *a, **k):
            return True

        def generate(self, prompt, **kwargs):
            captured["cb"] = kwargs.get("progress_callback")
            return Image.new("RGB", (8, 8))

    engine = InferenceEngine()
    engine._strategy = Stub()
    cb = lambda s, t: None
    engine.generate_image("x", progress_callback=cb)
    assert captured["cb"] is cb


# ----- diffusers strategy wiring (qwen as representative) -------------------

def _qwen_with_mock_pipeline():
    from ollamadiffuser.core.inference.strategies.qwen_strategy import QwenImageStrategy
    s = QwenImageStrategy()
    s.device = "cpu"
    s.model_config = MagicMock()
    s.model_config.name = "qwen-custom"
    s.model_config.parameters = {}
    out = MagicMock()
    out.images = [Image.new("RGB", (8, 8))]
    s.pipeline = MagicMock(return_value=out)
    return s


def test_qwen_adds_callback_on_step_end_when_progress_given():
    s = _qwen_with_mock_pipeline()
    s.generate("a cat", num_inference_steps=8, seed=1,
               progress_callback=lambda step, total: None)
    assert "callback_on_step_end" in s.pipeline.call_args.kwargs


def test_qwen_omits_callback_when_no_progress():
    s = _qwen_with_mock_pipeline()
    s.generate("a cat", num_inference_steps=8, seed=1)
    assert "callback_on_step_end" not in s.pipeline.call_args.kwargs


# ----- generic strategy signature guard ------------------------------------

class _PipeWithCb:
    def __call__(self, prompt, num_inference_steps=1, guidance_scale=1.0,
                 width=8, height=8, generator=None, negative_prompt=None,
                 callback_on_step_end=None):
        self.received = {"callback_on_step_end": callback_on_step_end}
        return SimpleNamespace(images=[Image.new("RGB", (8, 8))])


class _PipeNoCb:
    def __call__(self, prompt, num_inference_steps=1, guidance_scale=1.0,
                 width=8, height=8, generator=None, negative_prompt=None):
        return SimpleNamespace(images=[Image.new("RGB", (8, 8))])


def _generic_with(pipe):
    from ollamadiffuser.core.inference.strategies.generic_strategy import GenericPipelineStrategy
    s = GenericPipelineStrategy()
    s.device = "cpu"
    s.model_config = ModelConfig(name="g", path="/x", model_type="generic", parameters={})
    s.pipeline = pipe
    return s


def test_generic_adds_callback_when_pipeline_accepts_it():
    pipe = _PipeWithCb()
    s = _generic_with(pipe)
    s.generate("a cat", num_inference_steps=4, seed=1,
               progress_callback=lambda step, total: None)
    assert callable(pipe.received["callback_on_step_end"])


def test_generic_skips_callback_when_pipeline_lacks_it():
    pipe = _PipeNoCb()
    s = _generic_with(pipe)
    # Must not raise a TypeError for an unexpected callback_on_step_end kwarg.
    img = s.generate("a cat", num_inference_steps=4, seed=1,
                     progress_callback=lambda step, total: None)
    assert isinstance(img, Image.Image)
