"""Tests for the MLX inference strategy (issue #7, Phases 1 + 2).

These tests run on any platform — when not on Apple Silicon they
exercise the platform-check refusal path; when on Apple Silicon they
mock the mflux model class so no weights need to download.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ollamadiffuser.core.config.settings import ModelConfig
from ollamadiffuser.core.inference.engine import _get_strategy
from ollamadiffuser.core.inference.strategies import mlx_strategy
from ollamadiffuser.core.inference.strategies.mlx_strategy import (
    MLXStrategy,
    SUPPORTED_MLX_VARIANTS,
    _MfluxProgress,
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
    try:
        return ModelConfig(**base)
    except TypeError:
        from types import SimpleNamespace
        return SimpleNamespace(**base)


def _stub_resolution(*, supports_negative=True):
    """Return (cls_mock, instance_mock, mflux_config_mock).

    ``instance_mock.generate_image`` is a plain function so that
    ``inspect.signature()`` returns the exact parameters the strategy
    is allowed to forward — MLXStrategy filters kwargs against that
    signature. Set ``supports_negative=False`` to simulate a class
    like Flux2Klein that has no ``negative_prompt`` parameter.
    Call records are kept on ``instance_mock._call_kwargs`` (a list of
    dicts) — assert against the last entry.
    """
    fake_pil = Image.new("RGB", (64, 64), color=(0, 128, 255))
    generated_obj = MagicMock(name="GeneratedImage", image=fake_pil)

    call_log: list = []

    if supports_negative:
        def fake_generate_image(
            seed, prompt, num_inference_steps=4, height=1024,
            width=1024, guidance=4.0, image_path=None,
            image_strength=None, scheduler="linear",
            negative_prompt=None,
        ):
            call_log.append({
                "seed": seed,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
                "guidance": guidance,
                "image_path": image_path,
                "image_strength": image_strength,
                "scheduler": scheduler,
                "negative_prompt": negative_prompt,
            })
            return generated_obj
    else:
        def fake_generate_image(
            seed, prompt, num_inference_steps=4, height=1024,
            width=1024, guidance=1.0, image_path=None,
            image_strength=None, scheduler="flow_match_euler_discrete",
        ):
            call_log.append({
                "seed": seed,
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
                "guidance": guidance,
                "image_path": image_path,
                "image_strength": image_strength,
                "scheduler": scheduler,
            })
            return generated_obj

    instance = MagicMock(name="MfluxInstance")
    instance.generate_image = fake_generate_image
    instance._call_kwargs = call_log  # for test assertions

    cls = MagicMock(name="MfluxClass", return_value=instance)
    mflux_config = MagicMock(name="MfluxModelConfig")
    return cls, instance, mflux_config


# --------------------------------------------------------------------------
# Dispatch + module constants
# --------------------------------------------------------------------------

class TestDispatch:
    def test_engine_dispatches_mlx_model_type(self):
        strategy = _get_strategy("mlx")
        assert isinstance(strategy, MLXStrategy)

    def test_supported_variants_constant(self):
        # Phase 1 + Phase 2 + Phase 2.5 — nine variants total.
        assert SUPPORTED_MLX_VARIANTS == frozenset({
            "flux1", "flux1-kontext",
            "flux1-fill", "flux1-redux", "flux1-depth", "flux1-controlnet",
            "flux2", "z_image", "qwen-image",
        })

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
            "quantize": 3,
        })
        assert s.load(config, device="mps") is False


# --------------------------------------------------------------------------
# Resolution per variant (no actual mflux call — verifies dispatch logic)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only runs on Apple Silicon")
class TestVariantResolution:
    def test_flux1_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1", "schnell")
        assert cls.__name__ == "Flux1"

    def test_flux1_kontext_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-kontext", "dev")
        assert cls.__name__ == "Flux1Kontext"

    def test_flux1_kontext_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux1-kontext"):
            MLXStrategy._resolve_model_and_config("flux1-kontext", "bogus")

    def test_flux2_klein_4b_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux2", "klein-4b")
        assert cls.__name__ == "Flux2Klein"

    def test_flux2_klein_9b_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux2", "klein-9b")
        assert cls.__name__ == "Flux2Klein"

    def test_flux2_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux2"):
            MLXStrategy._resolve_model_and_config("flux2", "klein-99b")

    def test_z_image_turbo_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("z_image", "z-image-turbo")
        assert cls.__name__ == "ZImage"

    def test_z_image_rejects_bad_name(self):
        with pytest.raises(ValueError, match="z_image"):
            MLXStrategy._resolve_model_and_config("z_image", "z-image-pro")

    def test_qwen_image_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("qwen-image", "qwen-image")
        assert cls.__name__ == "QwenImage"

    def test_qwen_image_edit_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("qwen-image", "qwen-image-edit")
        assert cls.__name__ == "QwenImage"

    def test_qwen_image_rejects_bad_name(self):
        with pytest.raises(ValueError, match="qwen-image"):
            MLXStrategy._resolve_model_and_config("qwen-image", "qwen-bogus")

    # --- Phase 2.5: additional FLUX.1 family variants ---

    def test_flux1_fill_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-fill", "dev")
        assert cls.__name__ == "Flux1Fill"

    def test_flux1_fill_catvton_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-fill", "catvton")
        assert cls.__name__ == "Flux1Fill"

    def test_flux1_fill_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux1-fill"):
            MLXStrategy._resolve_model_and_config("flux1-fill", "bogus")

    def test_flux1_redux_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-redux", "dev")
        assert cls.__name__ == "Flux1Redux"

    def test_flux1_redux_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux1-redux"):
            MLXStrategy._resolve_model_and_config("flux1-redux", "bogus")

    def test_flux1_depth_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-depth", "dev")
        assert cls.__name__ == "Flux1Depth"

    def test_flux1_depth_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux1-depth"):
            MLXStrategy._resolve_model_and_config("flux1-depth", "bogus")

    def test_flux1_controlnet_canny_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-controlnet", "canny")
        assert cls.__name__ == "Flux1Controlnet"

    def test_flux1_controlnet_upscaler_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config("flux1-controlnet", "upscaler")
        assert cls.__name__ == "Flux1Controlnet"

    def test_flux1_controlnet_canny_schnell_resolves(self):
        cls, _ = MLXStrategy._resolve_model_and_config(
            "flux1-controlnet", "canny-schnell"
        )
        assert cls.__name__ == "Flux1Controlnet"

    def test_flux1_controlnet_rejects_bad_name(self):
        with pytest.raises(ValueError, match="flux1-controlnet"):
            MLXStrategy._resolve_model_and_config("flux1-controlnet", "pose")


# --------------------------------------------------------------------------
# Loading + generation (mflux mocked out)
# --------------------------------------------------------------------------

@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only runs on Apple Silicon")
class TestLoadAndGenerate:
    def test_load_calls_class_constructor_with_quantize_and_config(self):
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            ok = s.load(_make_config(), device="mps")
        assert ok is True
        cls_mock.assert_called_once_with(quantize=8, model_config=mflux_config_mock)
        assert s.is_loaded
        assert s.device == "mps"
        assert s._variant == "flux1"

    def test_mflux_import_and_construct_share_one_nonmain_thread(self):
        """Regression: the mflux import (``_resolve_model_and_config``) and model
        construction must run on the SAME worker thread, and that thread must not
        be the main thread.

        MLX creates its Metal GPU stream on the thread that first imports it. If
        the import happens on the main thread but construction/generation run on
        the strategy's worker thread, the worker has no ``Stream(gpu, 0)`` and
        mflux aborts the whole process during VAE decode ("There is no
        Stream(gpu, 0) in current thread"). Keeping import + construction on one
        worker thread is the fix.
        """
        import threading

        seen = {}
        cls_mock, _, cfg_mock = _stub_resolution()

        def rec_resolve(variant, name):
            seen["resolve"] = threading.get_ident()
            return (cls_mock, cfg_mock)

        def rec_construct(model_cls, mflux_config, quantize,
                          lora_paths=None, lora_scales=None):
            seen["construct"] = threading.get_ident()
            return model_cls(quantize=quantize, model_config=mflux_config)

        s = MLXStrategy()
        with patch.object(MLXStrategy, "_resolve_model_and_config", rec_resolve), \
             patch.object(MLXStrategy, "_construct_mlx_model", rec_construct):
            assert s.load(_make_config(), device="mps") is True

        assert "resolve" in seen and "construct" in seen
        # Import and construction ran on the same thread ...
        assert seen["resolve"] == seen["construct"]
        # ... and it was NOT the main thread (where MLX must never be imported).
        assert seen["resolve"] != threading.get_ident()

    def test_load_reports_failure_on_mflux_import_error(self):
        s = MLXStrategy()

        def raise_import(*a, **kw):
            raise ImportError("mflux missing")
        with patch.object(
            MLXStrategy, "_resolve_model_and_config", side_effect=raise_import
        ):
            assert s.load(_make_config(), device="mps") is False
        assert s.is_loaded is False

    def test_load_reports_failure_on_bad_model_name(self):
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": "flux2",
            "mlx_model_name": "klein-bogus",
        })
        # Don't mock — let resolution actually raise ValueError.
        assert s.load(config, device="mps") is False

    def test_generate_forwards_params_and_returns_pil(self):
        cls_mock, inst_mock, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
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
        assert len(inst_mock._call_kwargs) == 1
        called = inst_mock._call_kwargs[-1]
        assert called["prompt"] == "A dog"
        assert called["seed"] == 42
        assert called["num_inference_steps"] == 4
        assert called["guidance"] == 0.0
        assert called["height"] == 512
        assert called["width"] == 512

    def test_generate_filters_kwargs_for_variants_without_negative_prompt(self):
        """Flux2Klein has no negative_prompt — strategy must not pass it."""
        cls_mock, inst_mock, mflux_config_mock = _stub_resolution(supports_negative=False)
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": "flux2",
            "mlx_model_name": "klein-4b",
            "quantize": 8,
        })
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(config, device="mps")

        # If the strategy fails to filter, this raises TypeError because
        # our fake fn has no `negative_prompt` parameter.
        s.generate(prompt="hi", negative_prompt="ugly", seed=1)
        called = inst_mock._call_kwargs[-1]
        # And the call was actually made (no exception).
        assert called["prompt"] == "hi"

    def test_kontext_without_image_raises(self):
        """flux1-kontext is an image-editor — must reject if no image given."""
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": "flux1-kontext",
            "mlx_model_name": "dev",
            "quantize": 8,
        })
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(config, device="mps")

        with pytest.raises(ValueError, match="requires kwargs"):
            s.generate(prompt="make it sunset")  # no image=

    @pytest.mark.parametrize("variant,model_name,missing_kwargs,passed_kwargs", [
        ("flux1-fill", "dev", ["image", "mask_image"], {}),
        ("flux1-fill", "dev", ["mask_image"], {"image": "/tmp/x.png"}),
        ("flux1-redux", "dev", ["redux_images"], {}),
        ("flux1-depth", "dev", ["image"], {}),
        ("flux1-controlnet", "canny", ["control_image"], {}),
    ])
    def test_variant_required_input_enforcement(
        self, variant, model_name, missing_kwargs, passed_kwargs
    ):
        """Each variant's required-input contract is enforced before mflux call."""
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        config = _make_config(parameters={
            "mlx_variant": variant,
            "mlx_model_name": model_name,
            "quantize": 8,
        })
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(config, device="mps")

        with pytest.raises(ValueError, match="requires kwargs"):
            s.generate(prompt="x", **passed_kwargs)

    def test_redux_accepts_scalar_or_list(self):
        """redux_images may be a single PIL/path or a list of them."""
        from ollamadiffuser.core.inference.strategies.mlx_strategy import (
            MLXStrategy as _S,
        )
        # Scalar PIL → list of 1
        pil = Image.new("RGB", (8, 8))
        out = _S._materialize_image_path_list(pil)
        assert isinstance(out, list) and len(out) == 1

        # List of PIL → list of paths
        out2 = _S._materialize_image_path_list([pil, pil])
        assert isinstance(out2, list) and len(out2) == 2

        # None → None
        assert _S._materialize_image_path_list(None) is None

    def test_generate_without_seed_gets_random_int(self):
        cls_mock, inst_mock, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(_make_config(), device="mps")

        s.generate(prompt="hi")
        called = inst_mock._call_kwargs[-1]
        assert isinstance(called["seed"], int)
        assert 0 <= called["seed"] < 2**31

    def test_generate_before_load_raises(self):
        s = MLXStrategy()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            s.generate(prompt="hi")

    def test_unload_clears_state_and_variant(self):
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(_make_config(), device="mps")
        assert s.is_loaded
        assert s._variant == "flux1"
        s.unload()
        assert s.is_loaded is False
        assert s._mlx_model is None
        assert s.pipeline is None
        assert s._variant is None


# --------------------------------------------------------------------------
# LoRA (construction-time, via transparent reload)
# --------------------------------------------------------------------------

class TestResolveLoRAFile:
    """_resolve_lora_file is pure/path-based — runs on any platform."""

    def test_dir_plus_weight_name(self, tmp_path):
        f = tmp_path / "adapter.safetensors"
        f.write_bytes(b"x")
        assert MLXStrategy._resolve_lora_file(str(tmp_path), "adapter.safetensors") == f

    def test_bare_file_path(self, tmp_path):
        f = tmp_path / "adapter.safetensors"
        f.write_bytes(b"x")
        assert MLXStrategy._resolve_lora_file(str(f)) == f

    def test_dir_single_weight_autopicks(self, tmp_path):
        f = tmp_path / "only.safetensors"
        f.write_bytes(b"x")
        assert MLXStrategy._resolve_lora_file(str(tmp_path)) == f

    def test_unresolvable_returns_none(self, tmp_path):
        # HF repo id (not local) with a filename that isn't present
        assert MLXStrategy._resolve_lora_file("org/repo", "missing.safetensors") is None
        # dir with multiple weights and no weight_name → ambiguous → None
        (tmp_path / "a.safetensors").write_bytes(b"x")
        (tmp_path / "b.safetensors").write_bytes(b"x")
        assert MLXStrategy._resolve_lora_file(str(tmp_path)) is None


class TestThreadPinning:
    """All mflux/MLX work must run on ONE dedicated thread (Metal streams are
    thread-affine), even when callers invoke from different pool threads — as
    the API server / MCP do via asyncio.to_thread."""

    def test_run_uses_single_named_worker_thread(self):
        import threading

        s = MLXStrategy()
        names = []
        s._run(lambda: names.append(threading.current_thread().name))
        s._run(lambda: names.append(threading.current_thread().name))
        # Invoke from a *different* caller thread — must still pin to one worker.
        t = threading.Thread(
            target=lambda: s._run(lambda: names.append(threading.current_thread().name)))
        t.start()
        t.join()
        assert all(n.startswith("mlx") for n in names)  # the dedicated pool
        assert len(set(names)) == 1                       # always the same thread
        assert names[0] != threading.current_thread().name  # not the caller's
        s.unload()

    def test_run_propagates_exceptions_to_caller(self):
        s = MLXStrategy()
        with pytest.raises(ValueError, match="boom"):
            s._run(lambda: (_ for _ in ()).throw(ValueError("boom")))
        s.unload()

    def test_unload_shuts_down_executor(self):
        s = MLXStrategy()
        s._run(lambda: None)
        assert s._executor is not None
        s.unload()
        assert s._executor is None


@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only runs on Apple Silicon")
class TestLoRA:
    def test_load_lora_reloads_model_with_lora_paths(self, tmp_path):
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(_make_config(), device="mps")
            lora = tmp_path / "adapter.safetensors"
            lora.write_bytes(b"x")
            ok = s.load_lora_runtime(str(tmp_path), "adapter.safetensors", scale=0.8)
        assert ok is True
        # The model was rebuilt with construction-time LoRA args.
        last = cls_mock.call_args
        assert last.kwargs["lora_paths"] == [str(lora)]
        assert last.kwargs["lora_scales"] == [0.8]
        assert s.current_lora["scale"] == 0.8
        assert s.current_lora["path"] == str(lora)
        assert s.is_loaded

    def test_load_lora_unresolvable_returns_false_nondestructive(self):
        cls_mock, inst_mock, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(_make_config(), device="mps")
            base_model = s._mlx_model
            ok = s.load_lora_runtime("org/repo-not-local", "missing.safetensors")
        assert ok is False
        assert s.current_lora is None
        assert s._mlx_model is base_model  # untouched

    def test_unload_lora_reloads_without_adapters(self, tmp_path):
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(_make_config(), device="mps")
            lora = tmp_path / "a.safetensors"
            lora.write_bytes(b"x")
            assert s.load_lora_runtime(str(tmp_path), "a.safetensors") is True
            assert s.current_lora is not None
            assert s.unload_lora() is True
        assert s.current_lora is None
        # Final construction carried no LoRA kwargs.
        assert "lora_paths" not in cls_mock.call_args.kwargs

    def test_load_lora_before_load_returns_false(self):
        s = MLXStrategy()
        assert s.load_lora_runtime("some/dir", "x.safetensors") is False


@pytest.mark.skipif(not is_apple_silicon(), reason="MLX only runs on Apple Silicon")
class TestBackendInfo:
    def test_get_info_reports_backend_mlx(self):
        cls_mock, _, mflux_config_mock = _stub_resolution()
        s = MLXStrategy()
        with patch.object(
            MLXStrategy,
            "_resolve_model_and_config",
            return_value=(cls_mock, mflux_config_mock),
        ):
            s.load(_make_config(), device="mps")
        info = s.get_info()
        assert info["backend"] == "mlx"
        assert info["mflux_variant"] == "flux1"


# --------------------------------------------------------------------------
# Per-step progress (mflux callback registry) — uses fakes, runs anywhere
# --------------------------------------------------------------------------

class _FakeRegistry:
    def __init__(self):
        self.in_loop = []
        self.after_loop = []

    def register(self, cb):
        if hasattr(cb, "call_in_loop"):
            self.in_loop.append(cb)
        if hasattr(cb, "call_after_loop"):
            self.after_loop.append(cb)


class _FakeMLXModel:
    def __init__(self):
        self.callbacks = _FakeRegistry()

    def generate_image(self, seed, prompt, num_inference_steps=4, height=1024,
                       width=1024, guidance=0.0, **kw):
        return SimpleNamespace(image=Image.new("RGB", (8, 8)))


class TestMLXProgress:
    def _strategy(self):
        s = MLXStrategy()
        s._mlx_model = _FakeMLXModel()
        s._variant = "qwen-image"
        s.device = "mps"
        s.model_config = _make_config(parameters={
            "mlx_variant": "qwen-image", "mlx_model_name": "qwen-image",
            "quantize": 8, "num_inference_steps": 8, "guidance_scale": 4.0,
        })
        return s

    def test_registers_progress_callback_and_reports_1based(self):
        s = self._strategy()
        seen = []
        s.generate(prompt="x", num_inference_steps=8,
                   progress_callback=lambda step, total, message=None: seen.append(
                       (step, total, message)))
        in_regs = [c for c in s._mlx_model.callbacks.in_loop if isinstance(c, _MfluxProgress)]
        after_regs = [c for c in s._mlx_model.callbacks.after_loop if isinstance(c, _MfluxProgress)]
        assert len(in_regs) == 1
        assert len(after_regs) == 1  # decode-phase reporter
        config = SimpleNamespace(num_inference_steps=8)
        # Per-step hook: 1-based step, no message.
        in_regs[0].call_in_loop(t=0, seed=1, prompt="x", latents=None,
                                config=config, time_steps=None)
        # After-loop hook: signals the VAE decode phase with a message.
        after_regs[0].call_after_loop(seed=1, prompt="x", latents=None, config=config)
        assert seen == [(1, 8, None), (8, 8, "Decoding image…")]

    def test_no_duplicate_registration_across_calls(self):
        s = self._strategy()
        for _ in range(3):
            s.generate(prompt="x", num_inference_steps=8,
                       progress_callback=lambda step, total, message=None: None)
        in_regs = [c for c in s._mlx_model.callbacks.in_loop if isinstance(c, _MfluxProgress)]
        after_regs = [c for c in s._mlx_model.callbacks.after_loop if isinstance(c, _MfluxProgress)]
        assert len(in_regs) == 1
        assert len(after_regs) == 1

    def test_no_registration_without_callback(self):
        s = self._strategy()
        s.generate(prompt="x", num_inference_steps=8)
        regs = [c for c in s._mlx_model.callbacks.in_loop if isinstance(c, _MfluxProgress)]
        assert len(regs) == 0
