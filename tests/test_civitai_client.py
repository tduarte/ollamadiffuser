"""Tests for the CivitAI / CivitAI Red client and manager."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ollamadiffuser.core.utils import civitai_client as cc
from ollamadiffuser.core.utils.civitai_client import (
    CivitaiError,
    CivitaiManager,
    VersionInfo,
)


# --- helpers ----------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None, chunks=None):
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {}
        self._chunks = chunks or []

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def iter_content(self, chunk_size=1):
        for c in self._chunks:
            yield c

    def close(self):
        pass


def _version_json(**over):
    data = {
        "id": 999,
        "modelId": 111,
        "name": "v1.0",
        "baseModel": "SDXL 1.0",
        "trainedWords": ["magicword"],
        "model": {"name": "Cool Model", "type": "Checkpoint", "nsfw": False,
                  "description": "<p>A <b>great</b> model</p>", "tags": ["anime"]},
        "files": [
            {"name": "cool.safetensors", "sizeKB": 5000, "type": "Model",
             "primary": True, "downloadUrl": "https://civitai.com/api/download/models/999"},
        ],
    }
    data.update(over)
    return data


# --- parsing ----------------------------------------------------------------

@pytest.mark.parametrize("ref,expected", [
    ("https://civitai.com/models/12345", (cc.DEFAULT_BASE, 12345, None)),
    ("https://civitai.com/models/12345?modelVersionId=999", (cc.DEFAULT_BASE, 12345, 999)),
    ("https://civitai.red/models/7?modelVersionId=8", (cc.RED_BASE, 7, 8)),
    ("https://civitai.com/api/download/models/555", (cc.DEFAULT_BASE, None, 555)),
    ("https://civitai.com/api/v1/model-versions/321", (cc.DEFAULT_BASE, None, 321)),
    ("4242", (cc.DEFAULT_BASE, None, 4242)),
])
def test_parse_ref(ref, expected):
    r = cc.parse_civitai_ref(ref)
    assert (r.base_url, r.model_id, r.version_id) == expected


def test_parse_bare_id_red_flag():
    assert cc.parse_civitai_ref("4242", red=True).base_url == cc.RED_BASE
    assert cc.parse_civitai_ref("4242", red=True).is_red is True


def test_parse_invalid():
    with pytest.raises(CivitaiError):
        cc.parse_civitai_ref("not-a-ref")
    with pytest.raises(CivitaiError):
        cc.parse_civitai_ref("")


# --- mapping ----------------------------------------------------------------

@pytest.mark.parametrize("bm,mt", [
    ("SD 1.5", "sd15"), ("SD 1.4", "sd15"), ("SDXL 1.0", "sdxl"),
    ("Pony", "sdxl"), ("Illustrious", "sdxl"), ("NoobAI", "sdxl"),
    ("SD 3.5", "sd3"), ("Flux.1 D", "flux"), ("Unknownverse", None), (None, None),
])
def test_map_base_model(bm, mt):
    assert cc.map_base_model(bm) == mt


@pytest.mark.parametrize("t,cat", [
    ("Checkpoint", "checkpoint"), ("LORA", "lora"), ("LoCon", "lora"),
    ("LyCORIS", "lora"), ("TextualInversion", "embedding"), ("VAE", "vae"),
    ("Wildcards", None), (None, None),
])
def test_map_civitai_type(t, cat):
    assert cc.map_civitai_type(t) == cat


# --- file selection ---------------------------------------------------------

def _vi(files):
    return VersionInfo(base_url=cc.DEFAULT_BASE, version_id=1, model_id=1, name="x",
                       civitai_type="Checkpoint", content_category="checkpoint",
                       model_type="sdxl", base_model="SDXL 1.0", files=files)


def test_select_primary_prefers_flag():
    v = _vi([{"name": "a", "sizeKB": 100, "type": "Model", "primary": False},
             {"name": "b", "sizeKB": 5000, "type": "Model", "primary": True}])
    assert cc.select_primary_file(v)["name"] == "b"


def test_select_primary_fallback_type_model():
    v = _vi([{"name": "cfg", "sizeKB": 1, "type": "Config", "primary": False},
             {"name": "a", "sizeKB": 100, "type": "Model", "primary": False}])
    assert cc.select_primary_file(v)["name"] == "a"


def test_select_primary_fallback_largest():
    v = _vi([{"name": "a", "sizeKB": 100, "type": "Other"},
             {"name": "b", "sizeKB": 9000, "type": "Other"}])
    assert cc.select_primary_file(v)["name"] == "b"


def test_select_primary_empty():
    with pytest.raises(CivitaiError):
        cc.select_primary_file(_vi([]))


# --- resolve / normalize ----------------------------------------------------

def test_resolve_version():
    with patch.object(cc, "requests") as req:
        req.get.return_value = FakeResponse(json_data=_version_json())
        req.RequestException = Exception
        ref = cc.CivitaiRef(base_url=cc.DEFAULT_BASE, version_id=999)
        v = cc.resolve(ref)
    assert v.content_category == "checkpoint"
    assert v.model_type == "sdxl"
    assert v.trained_words == ["magicword"]
    assert v.description == "A great model"  # html stripped
    assert v.nsfw is False


def test_resolve_model_uses_latest_version():
    model_json = {"id": 111, "name": "Cool", "type": "LORA", "nsfw": True,
                  "modelVersions": [_version_json(id=5, baseModel="Pony")]}
    with patch.object(cc, "requests") as req:
        req.get.return_value = FakeResponse(json_data=model_json)
        req.RequestException = Exception
        v = cc.resolve(cc.CivitaiRef(base_url=cc.DEFAULT_BASE, model_id=111))
    assert v.version_id == 5
    assert v.civitai_type == "LORA"
    assert v.content_category == "lora"


# --- search -----------------------------------------------------------------

def test_search_normalization():
    payload = {"items": [
        {"id": 1, "name": "M1", "type": "Checkpoint", "nsfw": False,
         "stats": {"downloadCount": 42},
         "modelVersions": [{"id": 10, "baseModel": "SDXL 1.0"}]},
    ]}
    with patch.object(cc, "requests") as req:
        req.get.return_value = FakeResponse(json_data=payload)
        req.RequestException = Exception
        rows = cc.search("m1")
    assert rows[0]["version_id"] == 10
    assert rows[0]["base_model"] == "SDXL 1.0"
    assert rows[0]["download_count"] == 42


# --- download: resume / restart / red fallback ------------------------------

def test_download_fresh(tmp_path):
    dest = tmp_path / "m.safetensors"
    seen = []
    with patch.object(cc, "requests") as req:
        req.RequestException = Exception
        req.get.return_value = FakeResponse(
            status_code=200, headers={"Content-Length": "6"}, chunks=[b"abc", b"def"])
        cc.download_file("http://x/y", dest, base_url=cc.DEFAULT_BASE,
                         progress_callback=seen.append)
    assert dest.read_bytes() == b"abcdef"
    # a well-formed Ollama-style progress line was emitted
    assert any(m.startswith("pulling ") and "%" in m for m in seen)


def test_download_resume_206(tmp_path):
    dest = tmp_path / "m.safetensors"
    dest.write_bytes(b"abc")  # 3 bytes already present
    with patch.object(cc, "requests") as req:
        req.RequestException = Exception
        resp = FakeResponse(status_code=206, headers={"Content-Length": "3"}, chunks=[b"def"])
        req.get.return_value = resp
        cc.download_file("http://x/y", dest, base_url=cc.DEFAULT_BASE)
    assert dest.read_bytes() == b"abcdef"
    # Range header requested resume from byte 3
    _, kwargs = req.get.call_args
    assert kwargs["headers"].get("Range") == "bytes=3-"


def test_download_restart_when_range_ignored(tmp_path):
    dest = tmp_path / "m.safetensors"
    dest.write_bytes(b"OLD")
    with patch.object(cc, "requests") as req:
        req.RequestException = Exception
        req.get.return_value = FakeResponse(
            status_code=200, headers={"Content-Length": "6"}, chunks=[b"abc", b"def"])
        cc.download_file("http://x/y", dest, base_url=cc.DEFAULT_BASE)
    assert dest.read_bytes() == b"abcdef"  # overwritten, not appended


def test_download_red_token_fallback(tmp_path):
    dest = tmp_path / "m.safetensors"
    denied = FakeResponse(status_code=403)
    ok = FakeResponse(status_code=200, headers={"Content-Length": "3"}, chunks=[b"abc"])
    with patch.object(cc, "requests") as req, \
         patch.object(cc.settings, "civitai_api_key", "SECRET"):
        req.RequestException = Exception
        req.get.side_effect = [denied, ok]
        cc.download_file("https://civitai.red/api/download/models/1", dest,
                         base_url=cc.RED_BASE)
    assert dest.read_bytes() == b"abc"
    # second attempt appended ?token=<key> and dropped the bearer header
    second_url = req.get.call_args_list[1].args[0]
    assert "token=SECRET" in second_url
    assert "Authorization" not in req.get.call_args_list[1].kwargs["headers"]


# --- manager: pull checkpoint / lora ----------------------------------------

def test_pull_checkpoint_registers(tmp_path):
    mgr = CivitaiManager()
    added = {}
    with patch.object(cc, "requests") as req, \
         patch.object(cc, "download_file") as dl, \
         patch.object(cc.settings, "get_model_path", lambda n: tmp_path / n), \
         patch.object(cc.settings, "models", {}), \
         patch.object(cc.settings, "add_model", lambda mc: added.update({"mc": mc})):
        req.get.return_value = FakeResponse(json_data=_version_json())
        req.RequestException = Exception
        res = mgr.pull("999")
    assert dl.called
    mc = added["mc"]
    assert mc.model_type == "sdxl"
    assert mc.parameters["single_file"] == "cool.safetensors"
    assert mc.parameters["source"] == "civitai"
    assert mc.parameters["trained_words"] == ["magicword"]
    assert res["content_category"] == "checkpoint"


def test_pull_checkpoint_blocked_when_disallowed(tmp_path):
    # allow_checkpoints=False (the MCP policy) must reject a checkpoint ref
    # before anything downloads.
    mgr = CivitaiManager()
    with patch.object(cc, "requests") as req, \
         patch.object(cc, "download_file") as dl:
        req.get.return_value = FakeResponse(json_data=_version_json())
        req.RequestException = Exception
        with pytest.raises(CivitaiError, match="checkpoint"):
            mgr.pull("999", allow_checkpoints=False)
    assert not dl.called


def test_pull_lora_allowed_when_checkpoints_disallowed(tmp_path):
    # Disallowing checkpoints must NOT block LoRA downloads.
    mgr = CivitaiManager()
    lm = MagicMock()
    lm.lora_dir_for.return_value = tmp_path / "loras" / "cool-model-v1-0"
    lora_json = _version_json(model={"name": "Cool Model", "type": "LORA", "nsfw": False})
    with patch.object(cc, "requests") as req, \
         patch.object(cc, "download_file") as dl, \
         patch("ollamadiffuser.core.utils.lora_manager.lora_manager", lm):
        req.get.return_value = FakeResponse(json_data=lora_json)
        req.RequestException = Exception
        res = mgr.pull("999", allow_checkpoints=False)
    assert dl.called
    assert res["content_category"] == "lora"


def test_pull_checkpoint_needs_model_type_when_unmapped(tmp_path):
    mgr = CivitaiManager()
    with patch.object(cc, "requests") as req, patch.object(cc, "download_file"):
        req.get.return_value = FakeResponse(json_data=_version_json(baseModel="Weirdverse"))
        req.RequestException = Exception
        with pytest.raises(CivitaiError, match="model-type"):
            mgr.pull("999")


def test_pull_flux_experimental_gate(tmp_path):
    mgr = CivitaiManager()
    with patch.object(cc, "requests") as req, patch.object(cc, "download_file"):
        req.get.return_value = FakeResponse(json_data=_version_json(baseModel="Flux.1 D"))
        req.RequestException = Exception
        with pytest.raises(CivitaiError, match="experimental"):
            mgr.pull("999")


def test_pull_lora_uses_lora_manager(tmp_path):
    mgr = CivitaiManager()
    lm = MagicMock()
    lm.lora_dir_for.return_value = tmp_path / "loras" / "cool-model-v1-0"
    lora_json = _version_json(model={"name": "Cool Model", "type": "LORA", "nsfw": False})
    with patch.object(cc, "requests") as req, \
         patch.object(cc, "download_file") as dl, \
         patch("ollamadiffuser.core.utils.lora_manager.lora_manager", lm):
        req.get.return_value = FakeResponse(json_data=lora_json)
        req.RequestException = Exception
        res = mgr.pull("999")
    assert dl.called
    assert lm.register_downloaded_lora.called
    _, kwargs = lm.register_downloaded_lora.call_args
    assert kwargs["source"] == "civitai"
    assert kwargs["trained_words"] == ["magicword"]
    assert res["content_category"] == "lora"


# --- local import -----------------------------------------------------------

def test_read_sidecar_civitai_info(tmp_path):
    model = tmp_path / "mylora.safetensors"
    model.write_bytes(b"x")
    (tmp_path / "mylora.civitai.info").write_text(json.dumps({
        "baseModel": "SD 1.5", "trainedWords": ["trig"],
        "model": {"type": "LORA", "description": "<p>hi</p>"}}))
    meta = cc.read_sidecar(model)
    assert meta["content_category"] == "lora"
    assert meta["model_type"] == "sd15"
    assert meta["trained_words"] == ["trig"]
    assert meta["description"] == "hi"


def test_read_sidecar_agentimg_metadata_json(tmp_path):
    model = tmp_path / "ponyRealism_V23ULTRA.safetensors"
    model.write_bytes(b"x")
    # agentimg/Stability-Matrix shape: nested `civitai` block + top-level base_model.
    (tmp_path / "ponyRealism_V23ULTRA.metadata.json").write_text(json.dumps({
        "file_name": "ponyRealism_V23ULTRA",
        "base_model": "Pony",
        "from_civitai": True,
        "civitai": {
            "baseModel": "Pony",
            "trainedWords": ["score_9", "score_8_up"],
            "model": {"type": "Checkpoint", "description": "<p>pony</p>"},
        },
    }))
    meta = cc.read_sidecar(model)
    assert meta["content_category"] == "checkpoint"
    assert meta["model_type"] == "sdxl"          # Pony -> sdxl
    assert meta["trained_words"] == ["score_9", "score_8_up"]
    assert meta["description"] == "pony"


def test_import_local_checkpoint_in_place(tmp_path):
    f = tmp_path / "checkpoints" / "sdxl_cool.safetensors"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"weights")
    (f.parent / "sdxl_cool.civitai.info").write_text(json.dumps({
        "baseModel": "SDXL 1.0", "model": {"type": "Checkpoint"}}))
    mgr = CivitaiManager()
    added = {}
    with patch.object(cc.settings, "add_model", lambda mc: added.update({"mc": mc})):
        results = mgr.import_local(str(f))
    mc = added["mc"]
    assert mc.model_type == "sdxl"
    assert mc.parameters["single_file"] == "sdxl_cool.safetensors"
    assert mc.path == str(f.parent)  # registered in place, no copy
    assert results[0]["content_category"] == "checkpoint"


def test_pull_embedding_uses_embedding_manager(tmp_path):
    mgr = CivitaiManager()
    em = MagicMock()
    em.embedding_dir_for.return_value = tmp_path / "emb"
    emb_json = _version_json(
        model={"name": "Bad Hands", "type": "TextualInversion", "nsfw": False},
        trainedWords=["badhands"])
    with patch.object(cc, "requests") as req, \
         patch.object(cc, "download_file") as dl, \
         patch("ollamadiffuser.core.utils.embedding_manager.embedding_manager", em):
        req.get.return_value = FakeResponse(json_data=emb_json)
        req.RequestException = Exception
        res = mgr.pull("999")
    assert dl.called and em.register_downloaded_embedding.called
    _, kwargs = em.register_downloaded_embedding.call_args
    assert kwargs["token"] == "badhands"       # trigger token = first trained word
    assert res["content_category"] == "embedding"


def test_pull_vae_uses_vae_manager(tmp_path):
    mgr = CivitaiManager()
    vm = MagicMock()
    vm.vae_dir_for.return_value = tmp_path / "vae"
    vae_json = _version_json(model={"name": "Nice VAE", "type": "VAE", "nsfw": False})
    with patch.object(cc, "requests") as req, \
         patch.object(cc, "download_file") as dl, \
         patch("ollamadiffuser.core.utils.vae_manager.vae_manager", vm):
        req.get.return_value = FakeResponse(json_data=vae_json)
        req.RequestException = Exception
        res = mgr.pull("999")
    assert dl.called and vm.register_downloaded_vae.called
    assert res["content_category"] == "vae"


def test_import_local_embedding_by_folder(tmp_path):
    f = tmp_path / "embeddings" / "badhands.pt"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"emb")
    mgr = CivitaiManager()
    em = MagicMock()
    with patch("ollamadiffuser.core.utils.embedding_manager.embedding_manager", em):
        results = mgr.import_local(str(f), do_lookup=False)
    assert em.register_downloaded_embedding.called
    assert results[0]["content_category"] == "embedding"
    _, kwargs = em.register_downloaded_embedding.call_args
    assert kwargs["in_place"] is True


def test_import_local_no_lookup_requires_flags(tmp_path):
    f = tmp_path / "mystery.safetensors"
    f.write_bytes(b"weights")
    mgr = CivitaiManager()
    # No sidecar, no lookup, no folder hint -> skipped with guidance
    results = mgr.import_local(str(f), do_lookup=False)
    assert "skipped" in results[0]

    # With explicit flags it registers in place.
    added = {}
    with patch.object(cc.settings, "add_model", lambda mc: added.update({"mc": mc})):
        results = mgr.import_local(str(f), content_type="checkpoint",
                                   model_type="sd15", do_lookup=False)
    assert added["mc"].model_type == "sd15"
    assert added["mc"].parameters["single_file"] == "mystery.safetensors"
