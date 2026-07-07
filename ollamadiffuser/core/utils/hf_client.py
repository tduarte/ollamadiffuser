#!/usr/bin/env python3
"""
Hugging Face Hub search / install client and manager.

This is the Hugging Face sibling of ``civitai_client``. Where CivitAI ships
single-file ``.safetensors`` checkpoints, Hugging Face hosts both full diffusers
pipelines (multi-folder repos) and LoRA adapter repos (one or more
``.safetensors`` weight files). This module handles:

- Keyword **search** via ``HfApi().list_models(search=..., filter=..., sort=...)``,
  filtering for LoRAs (``lora`` tag), full diffusers models (``diffusers`` tag),
  and/or a specific ``base_model:<repo>`` tag. Results are normalized into small
  display rows (repo id, downloads, likes, tags, pipeline tag, base model).
- **File discovery**: an HF LoRA repo may contain multiple ``.safetensors``
  files, so we expose ``list_lora_weights`` / ``get_model_info`` (via
  ``HfApi().list_repo_files``) so the user/agent can pick the exact weight.
- A dispatcher (:class:`HuggingFaceManager`) shared by the CLI and MCP server:
  it installs a LoRA by delegating to the existing HF-based
  ``lora_manager.pull_lora`` (enriching the stored record with the discovered
  base model), and installs a full model by registering it in the model registry
  and downloading it through the normal ``model_manager.pull_model`` pipeline.

No new download code lives here — LoRAs go through ``robust_file_download`` (via
``lora_manager``) and full models through ``robust_snapshot_download`` (via
``model_manager``), exactly like the Hugging Face path already used elsewhere.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Callable

from ..config.settings import settings

logger = logging.getLogger(__name__)

# Weight-file extensions we treat as LoRA/model weights when listing a repo.
_WEIGHT_EXTS = (".safetensors",)

# --type value -> Hugging Face filter tag. "checkpoint"/"model" both mean a full
# diffusers pipeline (multi-folder repo), which carries the ``diffusers`` tag.
_TYPE_FILTER = {
    "lora": "lora",
    "checkpoint": "diffusers",
    "model": "diffusers",
}


class HuggingFaceError(RuntimeError):
    """Raised for Hugging Face search/install failures callers should surface."""


# --- HfApi helpers -----------------------------------------------------------

def _api():
    """Return an :class:`HfApi` bound to the configured token (for gated repos).

    Imported lazily so importing this module stays cheap and never fails when
    ``huggingface_hub`` is somehow unavailable until a call is actually made.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover - hub is a hard dependency
        raise HuggingFaceError(
            f"huggingface_hub is not installed: {e}"
        ) from e
    return HfApi(token=settings.hf_token or None)


def _slugify(text: str, fallback: str = "hf-model") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", (text or "").strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-.")
    return slug or fallback


def slug_for_repo(repo_id: str) -> str:
    """Local registry name for a repo id (``org/name`` -> ``org_name``)."""
    return repo_id.replace("/", "_")


# --- Normalization -----------------------------------------------------------

def base_model_from_tags(tags: List[str]) -> Optional[str]:
    """Extract the base model from HF tags.

    HF marks a LoRA's base with ``base_model:<repo>`` and
    ``base_model:adapter:<repo>``. We prefer the plain form and strip the
    ``adapter:`` marker so callers get ``org/Model`` either way.
    """
    plain = None
    adapter = None
    for t in tags or []:
        if not isinstance(t, str) or not t.startswith("base_model:"):
            continue
        rest = t[len("base_model:"):]
        if rest.startswith("adapter:"):
            adapter = rest[len("adapter:"):]
        else:
            plain = rest
    return plain or adapter


def _normalize_model(m: Any) -> Dict[str, Any]:
    """Normalize an HfApi ModelInfo into a display row."""
    tags = list(getattr(m, "tags", None) or [])
    return {
        "repo_id": getattr(m, "id", None) or getattr(m, "modelId", None),
        "downloads": getattr(m, "downloads", 0) or 0,
        "likes": getattr(m, "likes", 0) or 0,
        "pipeline_tag": getattr(m, "pipeline_tag", None),
        "tags": tags,
        "base_model": base_model_from_tags(tags),
        "is_lora": "lora" in tags,
    }


# --- File discovery ----------------------------------------------------------

def list_repo_files(repo_id: str) -> List[str]:
    """Return every file path in a repo (best-effort; [] on error)."""
    try:
        return list(_api().list_repo_files(repo_id))
    except Exception as e:  # network / not-found / gated
        logger.debug("list_repo_files(%s) failed: %s", repo_id, e)
        return []


def list_lora_weights(repo_id: str) -> List[str]:
    """Return the ``.safetensors`` weight files in a repo (for weight selection)."""
    return [f for f in list_repo_files(repo_id)
            if f.lower().endswith(_WEIGHT_EXTS)]


# --- Search ------------------------------------------------------------------

def search(query: str, model_type: Optional[str] = None,
           base_model: Optional[str] = None, limit: int = 20,
           include_files: bool = False) -> List[Dict[str, Any]]:
    """Keyword search over Hugging Face models.

    ``model_type`` filters to ``lora`` or ``checkpoint`` (full diffusers repo);
    ``base_model`` narrows to a specific ``base_model:<repo>`` tag. Results are
    sorted by download count. When ``include_files`` is set, each row is
    enriched with its ``lora_weights`` list (one extra API call per row — off by
    default so search stays fast).
    """
    filters: List[str] = []
    if model_type:
        tag = _TYPE_FILTER.get(model_type.strip().lower())
        if tag:
            filters.append(tag)
    if base_model:
        bm = base_model.strip()
        filters.append(bm if bm.startswith("base_model:") else f"base_model:{bm}")

    kwargs: Dict[str, Any] = {
        "search": query,
        "sort": "downloads",
        "limit": max(1, min(limit, 100)),
    }
    if filters:
        kwargs["filter"] = filters

    api = _api()
    try:
        models = list(api.list_models(**kwargs))
    except Exception as e:
        raise HuggingFaceError(f"Hugging Face search failed: {e}") from e

    rows = [_normalize_model(m) for m in models]
    if include_files:
        for r in rows:
            r["lora_weights"] = list_lora_weights(r["repo_id"])
    return rows


def get_model_info(repo_id: str) -> Dict[str, Any]:
    """Fetch normalized metadata + file listing for a single repo."""
    api = _api()
    try:
        m = api.model_info(repo_id)
    except Exception as e:
        raise HuggingFaceError(f"Could not fetch '{repo_id}' from Hugging Face: {e}") from e
    row = _normalize_model(m)
    files = list_repo_files(repo_id)
    row["files"] = files
    row["lora_weights"] = [f for f in files if f.lower().endswith(_WEIGHT_EXTS)]
    return row


# --- Dispatcher --------------------------------------------------------------

class HuggingFaceManager:
    """Install LoRAs and full models from Hugging Face.

    Shared by the CLI and the MCP server so both dispatch identically, mirroring
    :class:`~ollamadiffuser.core.utils.civitai_client.CivitaiManager`. Downloads
    reuse the existing HF-based helpers rather than any new code:
      - LoRA   -> ``lora_manager.pull_lora`` (robust_file_download)
      - model  -> ``model_manager.pull_model`` (robust_snapshot_download)
    """

    # -- LoRA ---------------------------------------------------------------

    def pull_lora(self, repo_id: str, *, weight_name: Optional[str] = None,
                  alias: Optional[str] = None,
                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Install a LoRA adapter repo from Hugging Face.

        If ``weight_name`` is omitted and the repo has exactly one
        ``.safetensors`` file, it is picked automatically; if it has several,
        we raise with the list so the caller can choose.
        """
        from .lora_manager import lora_manager

        weights = list_lora_weights(repo_id)
        if weight_name is None:
            if len(weights) == 1:
                weight_name = weights[0]
                if progress_callback:
                    progress_callback(f"📄 Selected weight file: {weight_name}")
            elif len(weights) > 1:
                raise HuggingFaceError(
                    f"'{repo_id}' contains {len(weights)} weight files; pass one "
                    f"with --weight-name. Options: {', '.join(weights)}"
                )
            # len == 0: fall through and let pull_lora do a full snapshot.

        ok = lora_manager.pull_lora(
            repo_id, weight_name=weight_name, alias=alias,
            progress_callback=progress_callback)
        if not ok:
            raise HuggingFaceError(f"Failed to download LoRA '{repo_id}'")

        name = alias or slug_for_repo(repo_id)
        # Enrich the stored record with the discovered base model so list_loras /
        # find_loras can filter by base model like the CivitAI-sourced LoRAs do.
        base_model = self._discover_base_model(repo_id)
        info = lora_manager.config.get(name)
        if info is not None:
            info["source"] = "huggingface"
            if base_model and not info.get("base_model"):
                info["base_model"] = base_model
            lora_manager._save_config()

        logger.info("Registered Hugging Face LoRA '%s' (weight=%s)", name, weight_name)
        return {"name": name, "content_category": "lora", "repo_id": repo_id,
                "weight_name": weight_name, "base_model": base_model,
                "available_weights": weights}

    # -- Full model ---------------------------------------------------------

    def pull_model(self, repo_id: str, *, model_type: str,
                   alias: Optional[str] = None, variant: Optional[str] = None,
                   force: bool = False,
                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Register an HF full diffusers model and download it.

        Registers ``repo_id`` + ``model_type`` in the runtime model registry
        (mirroring ``registry add``) and downloads it through the normal
        ``model_manager.pull_model`` pipeline, so it then behaves like any other
        installed model (``run`` / ``load_model`` / ``generate_image``).
        """
        from ..config.model_registry import model_registry
        from ..models.manager import model_manager

        if not model_type:
            raise HuggingFaceError(
                "Installing a full model requires --type (e.g. flux, sdxl, sd15, sd3, mlx)."
            )
        name = alias or slug_for_repo(repo_id)
        cfg: Dict[str, Any] = {"repo_id": repo_id, "model_type": model_type}
        if variant:
            cfg["variant"] = variant
        if not model_registry.add_model(name, cfg):
            raise HuggingFaceError(f"Could not register '{name}' in the model registry")

        if progress_callback:
            progress_callback(f"📦 Hugging Face model: {repo_id} ({model_type})")
        ok = model_manager.pull_model(name, force=force, progress_callback=progress_callback)
        if not ok:
            raise HuggingFaceError(f"Failed to download model '{repo_id}'")

        logger.info("Installed Hugging Face model '%s' (%s)", name, model_type)
        return {"name": name, "content_category": "checkpoint",
                "model_type": model_type, "repo_id": repo_id}

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _discover_base_model(repo_id: str) -> Optional[str]:
        """Best-effort base-model lookup for a repo (None on any failure)."""
        try:
            return get_model_info(repo_id).get("base_model")
        except HuggingFaceError:
            return None


# Module-level singleton (mirrors civitai_manager / lora_manager style).
hf_manager = HuggingFaceManager()
