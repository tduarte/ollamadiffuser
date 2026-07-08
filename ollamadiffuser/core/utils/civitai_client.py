#!/usr/bin/env python3
"""
CivitAI / CivitAI Red client and download manager.

This module lets ollamadiffuser pull models directly from civitai.com and
civitai.red (the mature-content sibling that shares CivitAI's account/DB), in
addition to the existing Hugging Face path. It handles:

- Parsing CivitAI references (model URLs, ``?modelVersionId=``, direct
  ``/api/download/models/{id}`` URLs, and bare numeric ids).
- Resolving model/version metadata via the public REST API, plus a hash lookup
  used to identify already-downloaded local files.
- Mapping CivitAI ``baseModel`` strings to ollamadiffuser ``model_type`` values.
- Keyword search.
- A streaming, resumable, token-authenticated download that reports progress in
  the same "Ollama style" the CLI already renders.
- A dispatcher (:class:`CivitaiManager`) that routes a resolved model to the
  right storage/registration path by CivitAI type (Checkpoint / LoRA / ...),
  and a local-import path for files the user already downloaded.

CivitAI models are single-file ``.safetensors`` checkpoints, which the SDXL and
SD1.5 inference strategies load via ``from_single_file`` (see
``core/inference/strategies``). We therefore register them with
``parameters["single_file"]`` set, and never touch the static model registry.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urlparse, parse_qs

import requests

from ..config.settings import settings, ModelConfig

logger = logging.getLogger(__name__)

# --- Constants ---------------------------------------------------------------

DEFAULT_BASE = "https://civitai.com"
RED_BASE = "https://civitai.red"
API_PATH = "/api/v1"
_API_TIMEOUT = 30          # seconds, metadata requests
_CONNECT_TIMEOUT = 30      # seconds, download connection
_CHUNK = 1024 * 1024       # 1 MiB streaming chunk
_DOWNLOAD_RETRIES = 3

# File extensions we treat as model weights during local import.
_WEIGHT_EXTS = (".safetensors", ".ckpt", ".pt", ".bin", ".pth")


class CivitaiError(RuntimeError):
    """Raised for CivitAI API / download failures that callers should surface."""


@dataclass
class CivitaiRef:
    """A parsed reference to a CivitAI model or version."""
    base_url: str = DEFAULT_BASE
    model_id: Optional[int] = None
    version_id: Optional[int] = None

    @property
    def is_red(self) -> bool:
        return "civitai.red" in self.base_url


# --- baseModel -> model_type mapping ----------------------------------------
# Substring rules, evaluated in order; the first hit wins. Order matters so that
# specific families (SDXL/Pony/Illustrious) are matched before the generic
# "sd 1" fallback. Unknown base models return None (caller warns / asks for an
# explicit --model-type).
_BASE_MODEL_RULES = [
    ("pony", "sdxl"),
    ("illustrious", "sdxl"),
    ("noobai", "sdxl"),
    ("sdxl", "sdxl"),
    ("sd 3", "sd3"),
    ("stable diffusion 3", "sd3"),
    ("flux", "flux"),
    ("sd 1", "sd15"),
    ("sd1", "sd15"),
    ("stable diffusion 1", "sd15"),
]

# CivitAI model.type -> our internal content category.
_TYPE_MAP = {
    "checkpoint": "checkpoint",
    "lora": "lora",
    "locon": "lora",
    "lycoris": "lora",
    "dora": "lora",
    "textualinversion": "embedding",
    "embedding": "embedding",
    "vae": "vae",
}


def map_base_model(base_model: Optional[str]) -> Optional[str]:
    """Map a CivitAI ``baseModel`` string to an ollamadiffuser model_type."""
    if not base_model:
        return None
    b = base_model.strip().lower()
    for needle, model_type in _BASE_MODEL_RULES:
        if needle in b:
            return model_type
    return None


def map_civitai_type(civitai_type: Optional[str]) -> Optional[str]:
    """Map a CivitAI ``model.type`` to our content category (checkpoint/lora/...)."""
    if not civitai_type:
        return None
    return _TYPE_MAP.get(civitai_type.strip().lower())


# --- Reference parsing -------------------------------------------------------

def parse_civitai_ref(ref: str, red: bool = False) -> CivitaiRef:
    """Parse a user-supplied CivitAI reference.

    Accepts:
      - ``https://civitai.com/models/{id}`` (optionally ``?modelVersionId={vid}``)
      - ``https://civitai.red/models/{id}`` (mature-content domain)
      - ``https://civitai.com/api/download/models/{vid}`` (direct download URL)
      - ``https://civitai.com/api/v1/model-versions/{vid}`` (API URL)
      - a bare integer, treated as a **model version id** (matches CivitAI's own
        download URL semantics)

    ``red=True`` forces the civitai.red base when the ref is a bare id or a
    sch/host-less value.
    """
    if ref is None:
        raise CivitaiError("Empty CivitAI reference")
    ref = ref.strip()
    if not ref:
        raise CivitaiError("Empty CivitAI reference")

    # Bare numeric id -> version id.
    if ref.isdigit():
        base = RED_BASE if red else DEFAULT_BASE
        return CivitaiRef(base_url=base, version_id=int(ref))

    parsed = urlparse(ref)
    if not parsed.scheme:
        # Not a URL and not a plain int -> unusable.
        raise CivitaiError(
            f"Unrecognized CivitAI reference: {ref!r}. "
            "Pass a civitai.com/civitai.red URL or a numeric model-version id."
        )

    host = (parsed.netloc or "").lower()
    if "civitai.red" in host or red:
        base = RED_BASE
    else:
        base = DEFAULT_BASE

    path = parsed.path or ""
    qs = parse_qs(parsed.query or "")

    model_id: Optional[int] = None
    version_id: Optional[int] = None

    # Order matters: the download and API paths also contain the substring
    # "/models/<id>", so match the more specific version paths first and only
    # fall back to the model-page path.
    m = re.search(r"/api/download/models/(\d+)", path)
    if m:
        version_id = int(m.group(1))
    elif (m := re.search(r"/model-versions/(\d+)", path)):
        version_id = int(m.group(1))
    elif (m := re.search(r"/models/(\d+)", path)):
        model_id = int(m.group(1))

    if "modelVersionId" in qs:
        try:
            version_id = int(qs["modelVersionId"][0])
        except (ValueError, IndexError):
            pass

    if model_id is None and version_id is None:
        raise CivitaiError(f"Could not find a model or version id in URL: {ref}")

    return CivitaiRef(base_url=base, model_id=model_id, version_id=version_id)


# --- HTTP helpers ------------------------------------------------------------

def _api_key() -> Optional[str]:
    return settings.civitai_api_key


def _auth_headers(api_key: Optional[str] = None) -> Dict[str, str]:
    key = api_key if api_key is not None else _api_key()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


def _api_get(base_url: str, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GET a CivitAI API endpoint and return parsed JSON."""
    url = f"{base_url}{API_PATH}{endpoint}"
    try:
        resp = requests.get(
            url, params=params, headers=_auth_headers(), timeout=_API_TIMEOUT
        )
    except requests.RequestException as e:
        raise CivitaiError(f"CivitAI request failed ({url}): {e}") from e
    if resp.status_code == 404:
        raise CivitaiError(f"Not found on CivitAI: {url}")
    if resp.status_code in (401, 403):
        raise CivitaiError(
            f"CivitAI denied access ({resp.status_code}) for {url}. "
            "This model may require a login/API key — set CIVITAI_API_KEY."
        )
    if resp.status_code != 200:
        raise CivitaiError(f"CivitAI returned {resp.status_code} for {url}")
    try:
        return resp.json()
    except ValueError as e:
        raise CivitaiError(f"CivitAI returned invalid JSON for {url}: {e}") from e


def get_version(base_url: str, version_id: int) -> Dict[str, Any]:
    """GET /api/v1/model-versions/{id}."""
    return _api_get(base_url, f"/model-versions/{version_id}")


def get_model(base_url: str, model_id: int) -> Dict[str, Any]:
    """GET /api/v1/models/{id}."""
    return _api_get(base_url, f"/models/{model_id}")


def get_version_by_hash(base_url: str, sha256: str) -> Optional[Dict[str, Any]]:
    """GET /api/v1/model-versions/by-hash/{hash}; None if not found."""
    try:
        return _api_get(base_url, f"/model-versions/by-hash/{sha256}")
    except CivitaiError:
        return None


# --- Metadata normalization --------------------------------------------------

def _strip_html(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


@dataclass
class VersionInfo:
    """Normalized view of a CivitAI model version, source-agnostic."""
    base_url: str
    version_id: Optional[int]
    model_id: Optional[int]
    name: str
    civitai_type: Optional[str]        # raw CivitAI model.type
    content_category: Optional[str]    # checkpoint / lora / embedding / vae
    model_type: Optional[str]          # sd15 / sdxl / flux / sd3 / None
    base_model: Optional[str]
    files: List[Dict[str, Any]] = field(default_factory=list)
    trained_words: List[str] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    nsfw: bool = False


def _normalize_version(base_url: str, version: Dict[str, Any],
                       model: Optional[Dict[str, Any]] = None) -> VersionInfo:
    model = model or version.get("model") or {}
    civitai_type = model.get("type")
    base_model = version.get("baseModel")
    model_name = model.get("name") or version.get("name") or "civitai-model"
    version_name = version.get("name") or ""
    full_name = f"{model_name} {version_name}".strip()
    return VersionInfo(
        base_url=base_url,
        version_id=version.get("id"),
        model_id=version.get("modelId") or (model.get("id") if model else None),
        name=full_name,
        civitai_type=civitai_type,
        content_category=map_civitai_type(civitai_type),
        model_type=map_base_model(base_model),
        base_model=base_model,
        files=version.get("files", []) or [],
        trained_words=version.get("trainedWords", []) or [],
        description=_strip_html(model.get("description")),
        tags=model.get("tags", []) or [],
        nsfw=bool(model.get("nsfw", False)),
    )


def resolve(ref: CivitaiRef) -> VersionInfo:
    """Resolve a parsed ref into normalized :class:`VersionInfo`."""
    if ref.version_id is not None:
        version = get_version(ref.base_url, ref.version_id)
        return _normalize_version(ref.base_url, version)
    if ref.model_id is not None:
        model = get_model(ref.base_url, ref.model_id)
        versions = model.get("modelVersions") or []
        if not versions:
            raise CivitaiError(f"Model {ref.model_id} has no versions")
        return _normalize_version(ref.base_url, versions[0], model=model)
    raise CivitaiError("Reference has neither a model id nor a version id")


def select_primary_file(version: VersionInfo) -> Dict[str, Any]:
    """Pick the file to download from a version's files list."""
    files = version.files
    if not files:
        raise CivitaiError(f"No downloadable files for '{version.name}'")
    for f in files:
        if f.get("primary"):
            return f
    for f in files:
        if (f.get("type") or "").lower() == "model":
            return f
    return max(files, key=lambda f: f.get("sizeKB") or 0)


def file_download_url(base_url: str, version: VersionInfo, file_info: Dict[str, Any]) -> str:
    url = file_info.get("downloadUrl")
    if url:
        return url
    if version.version_id is not None:
        return f"{base_url}/api/download/models/{version.version_id}"
    raise CivitaiError("File has no download URL and version id is unknown")


# --- Search ------------------------------------------------------------------

def search(query: str, types: Optional[str] = None, base_model: Optional[str] = None,
           limit: int = 20, nsfw: bool = False, base_url: str = DEFAULT_BASE
           ) -> List[Dict[str, Any]]:
    """Keyword search. Returns normalized rows for display."""
    params: Dict[str, Any] = {"query": query, "limit": max(1, min(limit, 100))}
    if types:
        params["types"] = types
    if base_model:
        params["baseModel"] = base_model
    params["nsfw"] = "true" if nsfw else "false"

    data = _api_get(base_url, "/models", params=params)
    rows: List[Dict[str, Any]] = []
    for item in data.get("items", []) or []:
        versions = item.get("modelVersions") or []
        latest = versions[0] if versions else {}
        stats = item.get("stats") or {}
        rows.append({
            "model_id": item.get("id"),
            "name": item.get("name"),
            "type": item.get("type"),
            "base_model": latest.get("baseModel"),
            "version_id": latest.get("id"),
            "download_count": stats.get("downloadCount", 0),
            "nsfw": bool(item.get("nsfw", False)),
        })
    return rows


# --- Streaming download ------------------------------------------------------

def _slugify(text: str, fallback: str = "civitai-model") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", (text or "").strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-.")
    return slug or fallback


def _filename_from_headers(resp: requests.Response) -> Optional[str]:
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
    if m:
        return os.path.basename(m.group(1).strip())
    return None


def download_file(url: str, dest_path: Path, *, base_url: str = DEFAULT_BASE,
                  api_key: Optional[str] = None, progress_callback: Optional[Callable] = None,
                  expected_size: Optional[int] = None, display_name: Optional[str] = None) -> Path:
    """Stream ``url`` to ``dest_path`` with resume, auth, and progress.

    - Follows redirects (CivitAI redirects the download to a CDN URL).
    - Sends ``Authorization: Bearer`` when a key is available. On civitai.red,
      where bearer auth is known-unreliable, falls back to a ``?token=`` query
      parameter on 401/403.
    - Resumes a partial file via a Range request.
    - Emits the Ollama-style progress string that ``OllamaStyleProgress`` renders.
    """
    from .download_utils import EnhancedProgressTracker  # lazy: pulls in hub deps

    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    is_red = "civitai.red" in base_url
    key = api_key if api_key is not None else _api_key()
    display_name = display_name or dest_path.name

    last_err: Optional[Exception] = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        try:
            resume_from = dest_path.stat().st_size if dest_path.exists() else 0
            headers: Dict[str, str] = {}
            if key:
                headers["Authorization"] = f"Bearer {key}"
            if resume_from:
                headers["Range"] = f"bytes={resume_from}-"

            req_url = url
            resp = requests.get(req_url, headers=headers, stream=True,
                                allow_redirects=True, timeout=_CONNECT_TIMEOUT)

            # civitai.red bearer-auth fallback: retry with ?token= query param.
            if resp.status_code in (401, 403) and is_red and key:
                resp.close()
                sep = "&" if "?" in req_url else "?"
                req_url = f"{url}{sep}token={key}"
                retry_headers = {k: v for k, v in headers.items() if k != "Authorization"}
                resp = requests.get(req_url, headers=retry_headers, stream=True,
                                    allow_redirects=True, timeout=_CONNECT_TIMEOUT)

            if resp.status_code in (401, 403):
                resp.close()
                raise CivitaiError(
                    f"CivitAI denied the download ({resp.status_code}). "
                    "Set CIVITAI_API_KEY (and ensure the account can access this model)."
                )

            mode = "ab"
            if resume_from and resp.status_code == 200:
                # Server ignored the Range request -> start over.
                resume_from = 0
                mode = "wb"
            elif resp.status_code not in (200, 206):
                resp.close()
                raise CivitaiError(f"Unexpected HTTP {resp.status_code} downloading {display_name}")

            # Total size = already-have + remaining content-length.
            content_length = resp.headers.get("Content-Length")
            remaining = int(content_length) if content_length is not None else None
            if expected_size:
                total = expected_size
            elif remaining is not None:
                total = resume_from + remaining
            else:
                total = 0

            tracker = EnhancedProgressTracker(total_files=1, progress_callback=progress_callback)
            tracker.start_file(display_name, total)

            downloaded = resume_from
            if resume_from and progress_callback and total:
                tracker.update_file_progress(display_name, downloaded, total)

            with open(dest_path, mode) as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        tracker.update_file_progress(display_name, downloaded, total)
            resp.close()
            tracker.complete_file(display_name)
            return dest_path

        except (requests.RequestException, CivitaiError) as e:
            last_err = e
            logger.warning("Download attempt %d/%d failed: %s", attempt, _DOWNLOAD_RETRIES, e)
            if isinstance(e, CivitaiError) and "denied" in str(e):
                break  # auth failures won't fix themselves on retry
            if attempt < _DOWNLOAD_RETRIES:
                time.sleep(min(30 * attempt, 90))

    raise CivitaiError(f"Failed to download {display_name}: {last_err}")


def sha256_file(path: Path) -> str:
    """Compute the SHA256 of a file (used for CivitAI hash lookup)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Local-import sidecar parsing -------------------------------------------

def _first(d: Dict[str, Any], *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def read_sidecar(model_file: Path) -> Optional[Dict[str, Any]]:
    """Read a metadata sidecar written by Stability Matrix / Civitai Helper.

    Looks for ``<stem>.civitai.info``, ``<stem>.json``, ``<stem>.cm-info.json``
    and ``<name>.json`` next to the model file. Returns a normalized dict
    (``civitai_type``, ``base_model``, ``trained_words``, ``description``) or
    None if no readable sidecar is found. Fully offline.
    """
    stem = model_file.parent / model_file.stem
    candidates = [
        Path(f"{stem}.metadata.json"),   # Stability Matrix / agentimg
        Path(f"{stem}.civitai.info"),    # Civitai Helper
        Path(f"{stem}.cm-info.json"),
        Path(f"{stem}.json"),
        model_file.with_name(model_file.name + ".json"),
    ]
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            with open(cand, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (ValueError, OSError) as e:
            logger.debug("Could not read sidecar %s: %s", cand, e)
            continue
        return _normalize_sidecar(data)
    return None


def _normalize_sidecar(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the varied sidecar schemas into our common shape.

    Handles Civitai Helper `.civitai.info` (raw version JSON), Stability Matrix
    `.cm-info.json` (PascalCase), and agentimg `.metadata.json` (which nests the
    CivitAI version JSON under a ``civitai`` key with top-level ``base_model``).
    """
    civ = data.get("civitai") if isinstance(data.get("civitai"), dict) else {}
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    if not model and isinstance(civ.get("model"), dict):
        model = civ["model"]
    civitai_type = (_first(data, "type") or model.get("type") or civ.get("type")
                    or _first(data, "ModelType"))
    base_model = (_first(data, "baseModel", "BaseModel", "base_model",
                         "sd version", "sd_version") or civ.get("baseModel"))
    trained = _first(data, "trainedWords", "TrainedWords", "activation text",
                     "activation_text", default=None)
    if not trained:
        trained = civ.get("trainedWords") or []
    if isinstance(trained, str):
        trained = [w.strip() for w in trained.split(",") if w.strip()]
    description = _strip_html(
        _first(data, "description", "Description", default="")
        or model.get("description", "") or (civ.get("description") or ""))
    return {
        "civitai_type": civitai_type,
        "content_category": map_civitai_type(civitai_type),
        "base_model": base_model,
        "model_type": map_base_model(base_model),
        "trained_words": list(trained) if trained else [],
        "description": description,
    }


# --- Dispatcher --------------------------------------------------------------

_EXPERIMENTAL_TYPES = {"flux", "sd3"}


class CivitaiManager:
    """Orchestrates CivitAI pulls and local imports across content types.

    Shared by the CLI and the MCP server so both dispatch identically. Only the
    Phase-1 content types (Checkpoint, LoRA) are wired here; embeddings and VAEs
    are handled by their dedicated managers in a later phase.
    """

    # -- public API ---------------------------------------------------------

    def pull(self, ref_str: str, *, model_type: Optional[str] = None,
             alias: Optional[str] = None, force: bool = False, red: bool = False,
             allow_experimental: bool = False, allow_checkpoints: bool = True,
             progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Resolve a CivitAI ref, download it, and register it.

        Returns a small result dict: ``{name, content_category, model_type, path}``.

        ``allow_checkpoints`` gates full base-model (checkpoint) downloads. It
        defaults to True for the CLI; callers that only permit add-ons (e.g. the
        MCP server) pass False to reject checkpoints before anything downloads.
        """
        ref = parse_civitai_ref(ref_str, red=red)
        version = resolve(ref)
        category = version.content_category
        if category is None:
            raise CivitaiError(
                f"Unsupported CivitAI type: {version.civitai_type!r}"
            )
        if category == "checkpoint" and not allow_checkpoints:
            raise CivitaiError(
                "Downloading full base-model checkpoints is not permitted here; "
                "only LoRAs, embeddings, and VAEs can be downloaded. "
                "Use the ollamadiffuser CLI to install checkpoints."
            )
        if category == "checkpoint":
            return self._pull_checkpoint(version, model_type=model_type, alias=alias,
                                         force=force, allow_experimental=allow_experimental,
                                         progress_callback=progress_callback)
        if category == "lora":
            return self._pull_lora(version, alias=alias, force=force,
                                   progress_callback=progress_callback)
        if category == "embedding":
            return self._pull_embedding(version, alias=alias,
                                        progress_callback=progress_callback)
        if category == "vae":
            return self._pull_vae(version, alias=alias,
                                  progress_callback=progress_callback)
        raise CivitaiError(f"Unhandled content category: {category}")

    def import_local(self, path: str, *, content_type: Optional[str] = None,
                     model_type: Optional[str] = None, alias: Optional[str] = None,
                     recursive: bool = False, do_lookup: bool = True,
                     base_url: str = DEFAULT_BASE,
                     progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Register already-downloaded local files in place (no copy/download)."""
        p = Path(path).expanduser()
        if not p.exists():
            raise CivitaiError(f"Path does not exist: {p}")

        if p.is_dir():
            pattern = "**/*" if recursive else "*"
            files = [f for f in sorted(p.glob(pattern))
                     if f.is_file() and f.suffix.lower() in _WEIGHT_EXTS]
        else:
            files = [p]
        if not files:
            raise CivitaiError(f"No weight files ({', '.join(_WEIGHT_EXTS)}) found under {p}")

        results: List[Dict[str, Any]] = []
        for f in files:
            try:
                results.append(self._import_one(
                    f, content_type=content_type, model_type=model_type,
                    alias=alias if len(files) == 1 else None,
                    do_lookup=do_lookup, base_url=base_url,
                    progress_callback=progress_callback))
            except CivitaiError as e:
                logger.warning("Skipping %s: %s", f.name, e)
                results.append({"file": str(f), "skipped": str(e)})
        return results

    # -- checkpoint ---------------------------------------------------------

    def _pull_checkpoint(self, version: VersionInfo, *, model_type: Optional[str],
                         alias: Optional[str], force: bool, allow_experimental: bool,
                         progress_callback: Optional[Callable]) -> Dict[str, Any]:
        resolved_type = model_type or version.model_type
        if resolved_type is None:
            raise CivitaiError(
                f"Could not map baseModel {version.base_model!r} to a model type. "
                "Re-run with --model-type sd15|sdxl|sd3|flux."
            )
        if resolved_type in _EXPERIMENTAL_TYPES and not allow_experimental:
            raise CivitaiError(
                f"{resolved_type.upper()} single-file checkpoints from CivitAI are "
                "experimental (often missing text encoders/VAE). Re-run with "
                "--experimental to attempt it anyway."
            )

        name = self._unique_name(alias or _slugify(version.name), version, force)
        file_info = select_primary_file(version)
        filename = file_info.get("name") or f"{name}.safetensors"
        target_dir = settings.get_model_path(name)
        dest = target_dir / filename

        url = file_download_url(version.base_url, version, file_info)
        size = int((file_info.get("sizeKB") or 0) * 1024) or None
        if progress_callback:
            progress_callback(f"📦 CivitAI checkpoint: {version.name}")
            progress_callback(f"🔗 {version.base_url} (version {version.version_id})")
        download_file(url, dest, base_url=version.base_url,
                      progress_callback=progress_callback, expected_size=size,
                      display_name=filename)

        params = self._metadata_params(version, single_file=filename)
        self._write_sidecar(target_dir, version)
        model_config = ModelConfig(
            name=name, path=str(target_dir), model_type=resolved_type,
            variant=None, components=None, parameters=params,
        )
        settings.add_model(model_config)
        logger.info("Registered CivitAI checkpoint '%s' (%s)", name, resolved_type)
        return {"name": name, "content_category": "checkpoint",
                "model_type": resolved_type, "path": str(target_dir)}

    # -- lora ---------------------------------------------------------------

    def _pull_lora(self, version: VersionInfo, *, alias: Optional[str], force: bool,
                   progress_callback: Optional[Callable]) -> Dict[str, Any]:
        from .lora_manager import lora_manager

        name = alias or _slugify(version.name)
        file_info = select_primary_file(version)
        filename = file_info.get("name") or f"{name}.safetensors"
        target_dir = lora_manager.lora_dir_for(name)
        dest = target_dir / filename

        url = file_download_url(version.base_url, version, file_info)
        size = int((file_info.get("sizeKB") or 0) * 1024) or None
        if progress_callback:
            progress_callback(f"📦 CivitAI LoRA: {version.name}")
        download_file(url, dest, base_url=version.base_url,
                      progress_callback=progress_callback, expected_size=size,
                      display_name=filename)

        lora_manager.register_downloaded_lora(
            name, dest, source="civitai", trained_words=version.trained_words,
            base_model=version.base_model, description=version.description)
        logger.info("Registered CivitAI LoRA '%s'", name)
        return {"name": name, "content_category": "lora",
                "model_type": version.model_type, "path": str(target_dir),
                "trained_words": version.trained_words}

    # -- embedding ----------------------------------------------------------

    def _pull_embedding(self, version: VersionInfo, *, alias: Optional[str],
                        progress_callback: Optional[Callable]) -> Dict[str, Any]:
        from .embedding_manager import embedding_manager

        name = alias or _slugify(version.name)
        file_info = select_primary_file(version)
        filename = file_info.get("name") or f"{name}.pt"
        target_dir = embedding_manager.embedding_dir_for(name)
        dest = target_dir / filename

        url = file_download_url(version.base_url, version, file_info)
        size = int((file_info.get("sizeKB") or 0) * 1024) or None
        if progress_callback:
            progress_callback(f"📦 CivitAI embedding: {version.name}")
        download_file(url, dest, base_url=version.base_url,
                      progress_callback=progress_callback, expected_size=size,
                      display_name=filename)

        # The trigger token is the first trained word, else the slug name.
        token = (version.trained_words[0] if version.trained_words else name)
        embedding_manager.register_downloaded_embedding(
            name, dest, token=token, source="civitai",
            base_model=version.base_model, trained_words=version.trained_words)
        logger.info("Registered CivitAI embedding '%s' (token=%s)", name, token)
        return {"name": name, "content_category": "embedding",
                "model_type": version.model_type, "path": str(target_dir),
                "trained_words": version.trained_words, "token": token}

    # -- vae ----------------------------------------------------------------

    def _pull_vae(self, version: VersionInfo, *, alias: Optional[str],
                  progress_callback: Optional[Callable]) -> Dict[str, Any]:
        from .vae_manager import vae_manager

        name = alias or _slugify(version.name)
        file_info = select_primary_file(version)
        filename = file_info.get("name") or f"{name}.safetensors"
        target_dir = vae_manager.vae_dir_for(name)
        dest = target_dir / filename

        url = file_download_url(version.base_url, version, file_info)
        size = int((file_info.get("sizeKB") or 0) * 1024) or None
        if progress_callback:
            progress_callback(f"📦 CivitAI VAE: {version.name}")
        download_file(url, dest, base_url=version.base_url,
                      progress_callback=progress_callback, expected_size=size,
                      display_name=filename)

        vae_manager.register_downloaded_vae(
            name, dest, source="civitai", base_model=version.base_model)
        logger.info("Registered CivitAI VAE '%s'", name)
        return {"name": name, "content_category": "vae",
                "model_type": version.model_type, "path": str(target_dir)}

    # -- local import -------------------------------------------------------

    def _import_one(self, f: Path, *, content_type: Optional[str], model_type: Optional[str],
                    alias: Optional[str], do_lookup: bool, base_url: str,
                    progress_callback: Optional[Callable]) -> Dict[str, Any]:
        meta = read_sidecar(f) or {}
        source = "civitai-local-sidecar" if meta else "civitai-local"
        if not meta and do_lookup:
            if progress_callback:
                progress_callback(f"🔍 Identifying {f.name} via hash lookup...")
            try:
                info = get_version_by_hash(base_url, sha256_file(f))
            except Exception as e:  # hashing/network best-effort
                logger.debug("Hash lookup failed for %s: %s", f.name, e)
                info = None
            if info:
                v = _normalize_version(base_url, info)
                meta = {
                    "civitai_type": v.civitai_type,
                    "content_category": v.content_category,
                    "base_model": v.base_model,
                    "model_type": v.model_type,
                    "trained_words": v.trained_words,
                    "description": v.description,
                }
                source = "civitai-local-hash"

        category = content_type or meta.get("content_category")
        resolved_type = model_type or meta.get("model_type")

        # Heuristic fallback from the parent folder name (loras/, checkpoints/...).
        if category is None:
            parent = f.parent.name.lower()
            if "lora" in parent:
                category = "lora"
            elif "embed" in parent or parent in ("ti", "textualinversion"):
                category = "embedding"
            elif "vae" in parent:
                category = "vae"
            elif any(k in parent for k in ("checkpoint", "models", "stable-diffusion")):
                category = "checkpoint"
        if category is None:
            raise CivitaiError(
                f"could not determine type for {f.name}; pass --type checkpoint|lora|embedding|vae")

        name = alias or _slugify(f.stem)
        trained = meta.get("trained_words", [])
        base_model = meta.get("base_model")

        if category == "checkpoint":
            if resolved_type is None:
                raise CivitaiError(
                    f"could not determine model type for {f.name}; pass --model-type")
            params = {
                "single_file": f.name,
                "source": source,
                "base_model": base_model,
                "trained_words": trained,
                "description": meta.get("description", ""),
                "civitai_type": meta.get("civitai_type"),
            }
            settings.add_model(ModelConfig(
                name=name, path=str(f.parent), model_type=resolved_type,
                variant=None, components=None, parameters=params))
            return {"file": str(f), "name": name, "content_category": "checkpoint",
                    "model_type": resolved_type, "source": source}

        if category == "lora":
            from .lora_manager import lora_manager
            lora_manager.register_downloaded_lora(
                name, f, source=source, trained_words=trained, base_model=base_model,
                in_place=True)
            return {"file": str(f), "name": name, "content_category": "lora",
                    "model_type": resolved_type, "source": source}

        if category == "embedding":
            from .embedding_manager import embedding_manager
            token = trained[0] if trained else name
            embedding_manager.register_downloaded_embedding(
                name, f, token=token, source=source, base_model=base_model,
                trained_words=trained, in_place=True)
            return {"file": str(f), "name": name, "content_category": "embedding",
                    "model_type": resolved_type, "source": source}

        if category == "vae":
            from .vae_manager import vae_manager
            vae_manager.register_downloaded_vae(
                name, f, source=source, base_model=base_model, in_place=True)
            return {"file": str(f), "name": name, "content_category": "vae",
                    "model_type": resolved_type, "source": source}

        raise CivitaiError(f"import of '{category}' not supported yet")

    # -- helpers ------------------------------------------------------------

    def _unique_name(self, name: str, version: VersionInfo, force: bool) -> str:
        if name not in settings.models or force:
            return name
        suffix = f"-v{version.version_id}" if version.version_id else "-civitai"
        candidate = f"{name}{suffix}"
        n = 2
        while candidate in settings.models:
            candidate = f"{name}{suffix}-{n}"
            n += 1
        return candidate

    def _metadata_params(self, version: VersionInfo, single_file: str) -> Dict[str, Any]:
        return {
            "single_file": single_file,
            "source": "civitai",
            "civitai_version_id": version.version_id,
            "civitai_model_id": version.model_id,
            "base_model": version.base_model,
            "trained_words": version.trained_words,
            "description": version.description,
            "tags": version.tags,
            "nsfw": version.nsfw,
        }

    def _write_sidecar(self, target_dir: Path, version: VersionInfo) -> None:
        """Persist full metadata next to the model so nothing is lost."""
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "source": "civitai",
                "base_url": version.base_url,
                "model_id": version.model_id,
                "version_id": version.version_id,
                "name": version.name,
                "civitai_type": version.civitai_type,
                "base_model": version.base_model,
                "model_type": version.model_type,
                "trained_words": version.trained_words,
                "description": version.description,
                "tags": version.tags,
                "nsfw": version.nsfw,
                "imported_at": datetime.now().isoformat(),
            }
            with open(target_dir / "metadata.json", "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False)
        except OSError as e:
            logger.debug("Could not write metadata sidecar: %s", e)


# Module-level singleton (mirrors settings/model_manager/lora_manager style).
civitai_manager = CivitaiManager()
