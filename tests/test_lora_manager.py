"""Tests for LoRAManager name resolution (tolerant lookup)."""

from ollamadiffuser.core.utils.lora_manager import LoRAManager


def _mgr(config):
    # Bypass __init__ (which touches the filesystem) for a pure config test.
    m = LoRAManager.__new__(LoRAManager)
    m.config = config
    return m


def test_resolve_exact_and_normalized():
    m = _mgr({"my-cool-lora": {"weight_name": "My Cool LoRA.safetensors"}})
    assert m.resolve_lora_name("my-cool-lora") == "my-cool-lora"      # exact
    assert m.resolve_lora_name("My Cool LoRA") == "my-cool-lora"      # spaces + case
    assert m.resolve_lora_name("my_cool_lora") == "my-cool-lora"      # underscores
    assert m.resolve_lora_name("MY COOL LORA") == "my-cool-lora"      # upper


def test_resolve_via_weight_filename():
    # Agent passes the on-disk filename (with spaces); key is a dash slug.
    m = _mgr({"slug-x": {"weight_name": "Original Name With Spaces.safetensors"}})
    assert m.resolve_lora_name("Original Name With Spaces") == "slug-x"
    assert m.resolve_lora_name("original-name-with-spaces") == "slug-x"


def test_resolve_unknown_returns_none():
    m = _mgr({"a-b": {"weight_name": "a b.safetensors"}})
    assert m.resolve_lora_name("totally-different") is None
    assert m.resolve_lora_name("") is None


def test_is_installed_and_get_info_tolerant():
    m = _mgr({"pony-realism": {"weight_name": "Pony Realism.safetensors", "base_model": "Pony"}})
    assert m.is_lora_installed("Pony Realism") is True
    assert m.get_lora_info("pony realism")["base_model"] == "Pony"
    assert m.is_lora_installed("nope") is False


def test_lora_unet_only_filter(tmp_path):
    """The UNet-only fallback keeps lora_unet_* and drops text-encoder tensors."""
    import torch
    from safetensors.torch import save_file
    from ollamadiffuser.core.inference.base import InferenceStrategy

    p = tmp_path / "x.safetensors"
    save_file({
        "lora_unet_down_blocks_0.lora_down.weight": torch.zeros(2, 2),
        "lora_unet_down_blocks_0.lora_up.weight": torch.zeros(2, 2),
        "lora_te1_text_model.lora_down.weight": torch.zeros(2, 2),
        "lora_te2_text_model.lora_down.weight": torch.zeros(2, 2),
    }, str(p))

    sd = InferenceStrategy._lora_unet_state_dict(str(tmp_path), "x.safetensors")
    assert set(sd) == {
        "lora_unet_down_blocks_0.lora_down.weight",
        "lora_unet_down_blocks_0.lora_up.weight",
    }
    # Non-local / non-safetensors sources return None so the caller re-raises.
    assert InferenceStrategy._lora_unet_state_dict(str(tmp_path), "missing.safetensors") is None
    assert InferenceStrategy._lora_unet_state_dict("org/repo", None) is None
