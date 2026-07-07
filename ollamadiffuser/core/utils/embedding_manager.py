#!/usr/bin/env python3
"""
Textual-inversion embedding manager.

Stores embeddings under ``~/.ollamadiffuser/embeddings/<name>/`` with an
``embeddings.json`` registry, and loads them into the currently-loaded model.
Mirrors :class:`LoRAManager`'s shape. Downloads themselves are handled by
``civitai_client`` (or by importing local files in place).
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manager for textual-inversion embeddings."""

    def __init__(self):
        self.embedding_dir = settings.config_dir / "embeddings"
        self.embedding_dir.mkdir(exist_ok=True)
        self.config_file = self.embedding_dir / "embeddings.json"
        self._load_config()

    def _load_config(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load embeddings config: {e}")
                self.config = {}
        else:
            self.config = {}

    def _save_config(self):
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save embeddings config: {e}")

    def _dir_size(self, path: Path) -> int:
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

    def _fmt(self, n: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"

    def embedding_dir_for(self, name: str) -> Path:
        """Return (creating if needed) the storage directory for an embedding."""
        path = self.embedding_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def register_downloaded_embedding(self, name: str, file_path, token: str,
                                      source: str, base_model: Optional[str] = None,
                                      trained_words: Optional[List[str]] = None,
                                      in_place: bool = False) -> bool:
        """Register an embedding from a local single file (CivitAI or import)."""
        file_path = Path(file_path)
        self.config[name] = {
            "token": token,
            "weight_name": file_path.name,
            "path": str(file_path.parent),
            "source": source,
            "base_model": base_model,
            "trained_words": trained_words or [],
            "in_place": in_place,
            "downloaded_at": datetime.now().isoformat(),
            "size": self._fmt(self._dir_size(file_path.parent)),
        }
        self._save_config()
        logger.info(f"Registered embedding '{name}' (token={token}, source={source})")
        return True

    def load_embedding(self, name: str) -> bool:
        """Load a registered embedding into the currently-loaded model."""
        from ..models.manager import model_manager

        if name not in self.config:
            logger.error(f"Embedding {name} not found")
            return False
        if not model_manager.is_model_loaded():
            logger.error("No model is currently loaded")
            return False

        info = self.config[name]
        weight = Path(info["path"]) / info["weight_name"]
        if not weight.exists():
            logger.error(f"Embedding file does not exist: {weight}")
            return False

        engine = model_manager.loaded_model
        return engine.load_textual_inversion(str(weight), token=info.get("token"))

    def list_embeddings(self) -> Dict[str, dict]:
        return dict(self.config)

    def get_embedding_info(self, name: str) -> Optional[dict]:
        return self.config.get(name)

    def remove_embedding(self, name: str) -> bool:
        if name not in self.config:
            logger.error(f"Embedding {name} not found")
            return False
        info = self.config[name]
        path = Path(info["path"])
        # Never delete files imported in place (user's own directory).
        if info.get("in_place"):
            logger.info(f"Embedding {name} imported in place; unregistering only")
        elif path.exists() and path.parent == self.embedding_dir:
            shutil.rmtree(path)
        del self.config[name]
        self._save_config()
        return True


# Module-level singleton.
embedding_manager = EmbeddingManager()
