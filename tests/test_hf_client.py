"""Tests for the Hugging Face search / install client and manager."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ollamadiffuser.core.utils import hf_client as hc
from ollamadiffuser.core.utils.hf_client import (
    HuggingFaceError,
    HuggingFaceManager,
)


# --- helpers ----------------------------------------------------------------

def _model(id, downloads=0, likes=0, pipeline_tag=None, tags=None):
    return SimpleNamespace(
        id=id, downloads=downloads, likes=likes,
        pipeline_tag=pipeline_tag, tags=tags or [],
    )


def _fake_api(models=None, files=None, info=None):
    """Return a MagicMock standing in for HfApi()."""
    api = MagicMock()
    api.list_models.return_value = iter(models or [])
    api.list_repo_files.return_value = files or []
    api.model_info.return_value = info
    return api


# --- base model / normalization ---------------------------------------------

@pytest.mark.parametrize("tags,expected", [
    (["lora", "base_model:org/Foo"], "org/Foo"),
    (["base_model:adapter:org/Bar"], "org/Bar"),
    (["base_model:org/Plain", "base_model:adapter:org/Plain"], "org/Plain"),
    (["lora", "diffusers"], None),
    ([], None),
])
def test_base_model_from_tags(tags, expected):
    assert hc.base_model_from_tags(tags) == expected


def test_normalize_model_shape():
    m = _model("org/flux-lora", downloads=100, likes=9, pipeline_tag="text-to-image",
               tags=["lora", "diffusers", "base_model:black-forest-labs/FLUX.2-klein-9B"])
    row = hc._normalize_model(m)
    assert row["repo_id"] == "org/flux-lora"
    assert row["downloads"] == 100
    assert row["likes"] == 9
    assert row["pipeline_tag"] == "text-to-image"
    assert row["is_lora"] is True
    assert row["base_model"] == "black-forest-labs/FLUX.2-klein-9B"


# --- search -----------------------------------------------------------------

def test_search_builds_lora_filter_and_normalizes():
    api = _fake_api(models=[
        _model("a/lora1", downloads=5, tags=["lora"]),
        _model("b/lora2", downloads=3, tags=["lora", "base_model:org/Base"]),
    ])
    with patch.object(hc, "_api", return_value=api):
        rows = hc.search("flux", model_type="lora", limit=10)
    # filter passed the 'lora' tag and sorted by downloads
    _, kwargs = api.list_models.call_args
    assert kwargs["filter"] == ["lora"]
    assert kwargs["sort"] == "downloads"
    assert kwargs["search"] == "flux"
    assert [r["repo_id"] for r in rows] == ["a/lora1", "b/lora2"]


def test_search_checkpoint_and_base_model_filter():
    api = _fake_api(models=[])
    with patch.object(hc, "_api", return_value=api):
        hc.search("flux", model_type="checkpoint",
                  base_model="black-forest-labs/FLUX.2-klein-9B")
    _, kwargs = api.list_models.call_args
    assert "diffusers" in kwargs["filter"]
    assert "base_model:black-forest-labs/FLUX.2-klein-9B" in kwargs["filter"]


def test_search_include_files_lists_weights():
    api = _fake_api(
        models=[_model("org/klein-lora", tags=["lora"])],
        files=["README.md", "Flux Klein - NSFW v2.safetensors", "config.json"],
    )
    with patch.object(hc, "_api", return_value=api):
        rows = hc.search("klein", include_files=True)
    assert rows[0]["lora_weights"] == ["Flux Klein - NSFW v2.safetensors"]


def test_search_wraps_errors():
    api = MagicMock()
    api.list_models.side_effect = RuntimeError("boom")
    with patch.object(hc, "_api", return_value=api):
        with pytest.raises(HuggingFaceError, match="search failed"):
            hc.search("x")


# --- file discovery / info --------------------------------------------------

def test_list_lora_weights_filters_safetensors():
    api = _fake_api(files=["a.safetensors", "b.bin", "c.SAFETENSORS", "readme.md"])
    with patch.object(hc, "_api", return_value=api):
        weights = hc.list_lora_weights("org/repo")
    assert weights == ["a.safetensors", "c.SAFETENSORS"]


def test_list_repo_files_swallows_errors():
    api = MagicMock()
    api.list_repo_files.side_effect = RuntimeError("nope")
    with patch.object(hc, "_api", return_value=api):
        assert hc.list_repo_files("org/repo") == []


def test_get_model_info_combines_metadata_and_files():
    info = _model("org/klein", downloads=42, tags=["lora", "base_model:org/Base"])
    api = _fake_api(files=["w.safetensors", "x.json"], info=info)
    with patch.object(hc, "_api", return_value=api):
        row = hc.get_model_info("org/klein")
    assert row["repo_id"] == "org/klein"
    assert row["base_model"] == "org/Base"
    assert row["lora_weights"] == ["w.safetensors"]
    assert "x.json" in row["files"]


def test_get_model_info_wraps_errors():
    api = MagicMock()
    api.model_info.side_effect = RuntimeError("404")
    with patch.object(hc, "_api", return_value=api):
        with pytest.raises(HuggingFaceError, match="Could not fetch"):
            hc.get_model_info("org/missing")


# --- manager: pull_lora -----------------------------------------------------

def test_pull_lora_autoselects_single_weight():
    mgr = HuggingFaceManager()
    lm = MagicMock()
    lm.pull_lora.return_value = True
    lm.config = {"diroverflo_FLux_Klein_9B_NSFW": {}}
    with patch.object(hc, "list_lora_weights", return_value=["Flux Klein - NSFW v2.safetensors"]), \
         patch.object(HuggingFaceManager, "_discover_base_model",
                      return_value="black-forest-labs/FLUX.2-klein-base-9B"), \
         patch("ollamadiffuser.core.utils.lora_manager.lora_manager", lm):
        res = mgr.pull_lora("diroverflo/FLux_Klein_9B_NSFW")
    # weight auto-selected and forwarded to lora_manager.pull_lora
    _, kwargs = lm.pull_lora.call_args
    assert kwargs["weight_name"] == "Flux Klein - NSFW v2.safetensors"
    assert res["content_category"] == "lora"
    assert res["base_model"] == "black-forest-labs/FLUX.2-klein-base-9B"
    # base model enriched onto the stored record
    stored = lm.config["diroverflo_FLux_Klein_9B_NSFW"]
    assert stored["base_model"] == "black-forest-labs/FLUX.2-klein-base-9B"
    assert stored["source"] == "huggingface"


def test_pull_lora_requires_weight_when_multiple():
    mgr = HuggingFaceManager()
    lm = MagicMock()
    with patch.object(hc, "list_lora_weights", return_value=["a.safetensors", "b.safetensors"]), \
         patch("ollamadiffuser.core.utils.lora_manager.lora_manager", lm):
        with pytest.raises(HuggingFaceError, match="weight-name"):
            mgr.pull_lora("org/multi")
    lm.pull_lora.assert_not_called()


def test_pull_lora_raises_when_download_fails():
    mgr = HuggingFaceManager()
    lm = MagicMock()
    lm.pull_lora.return_value = False
    with patch.object(hc, "list_lora_weights", return_value=["only.safetensors"]), \
         patch.object(HuggingFaceManager, "_discover_base_model", return_value=None), \
         patch("ollamadiffuser.core.utils.lora_manager.lora_manager", lm):
        with pytest.raises(HuggingFaceError, match="Failed to download"):
            mgr.pull_lora("org/repo")


# --- manager: pull_model ----------------------------------------------------

def test_pull_model_registers_and_downloads():
    mgr = HuggingFaceManager()
    registry = MagicMock()
    registry.add_model.return_value = True
    mm = MagicMock()
    mm.pull_model.return_value = True
    with patch("ollamadiffuser.core.config.model_registry.model_registry", registry), \
         patch("ollamadiffuser.core.models.manager.model_manager", mm):
        res = mgr.pull_model("black-forest-labs/FLUX.1-schnell", model_type="flux",
                             alias="my-flux")
    name, cfg = registry.add_model.call_args.args
    assert name == "my-flux"
    assert cfg == {"repo_id": "black-forest-labs/FLUX.1-schnell", "model_type": "flux"}
    mm.pull_model.assert_called_once()
    assert res["model_type"] == "flux"
    assert res["name"] == "my-flux"


def test_pull_model_requires_type():
    mgr = HuggingFaceManager()
    with pytest.raises(HuggingFaceError, match="requires"):
        mgr.pull_model("org/repo", model_type="")


def test_pull_model_raises_on_download_failure():
    mgr = HuggingFaceManager()
    registry = MagicMock()
    registry.add_model.return_value = True
    mm = MagicMock()
    mm.pull_model.return_value = False
    with patch("ollamadiffuser.core.config.model_registry.model_registry", registry), \
         patch("ollamadiffuser.core.models.manager.model_manager", mm):
        with pytest.raises(HuggingFaceError, match="Failed to download"):
            mgr.pull_model("org/repo", model_type="sdxl")
