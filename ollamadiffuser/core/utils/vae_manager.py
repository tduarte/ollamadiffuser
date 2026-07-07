#!/usr/bin/env python3
"""
VAE manager.

Stores VAEs under ``~/.ollamadiffuser/vaes/<name>/`` with a ``vaes.json``
registry, and attaches a VAE to the currently-loaded model's pipeline. Mirrors
:class:`LoRAManager`'s shape. Downloads are handled by ``civitai_client`` (or
by importing local files in place).
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..config.settings import settings

logger = logging.getLogger(__name__)


class VaeManager:
    """Manager for standalone VAEs."""

    def __init__(self):
        self.vae_dir = settings.config_dir / "vaes"
        self.vae_dir.mkdir(exist_ok=True)
        self.config_file = self.vae_dir / "vaes.json"
        self._load_config()

    def _load_config(self):
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load vaes config: {e}")
                self.config = {}
        else:
            self.config = {}

    def _save_config(self):
        try:
            with open(self.config_file, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save vaes config: {e}")

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

    def vae_dir_for(self, name: str) -> Path:
        """Return (creating if needed) the storage directory for a VAE."""
        path = self.vae_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def register_downloaded_vae(self, name: str, file_path, source: str,
                                base_model: Optional[str] = None,
                                in_place: bool = False) -> bool:
        """Register a VAE from a local single file (CivitAI or import)."""
        file_path = Path(file_path)
        self.config[name] = {
            "weight_name": file_path.name,
            "path": str(file_path.parent),
            "source": source,
            "base_model": base_model,
            "in_place": in_place,
            "downloaded_at": datetime.now().isoformat(),
            "size": self._fmt(self._dir_size(file_path.parent)),
        }
        self._save_config()
        logger.info(f"Registered VAE '{name}' (source={source})")
        return True

    def attach_vae(self, name: str) -> bool:
        """Attach a registered VAE to the currently-loaded model."""
        from ..models.manager import model_manager

        if name not in self.config:
            logger.error(f"VAE {name} not found")
            return False
        if not model_manager.is_model_loaded():
            logger.error("No model is currently loaded")
            return False

        info = self.config[name]
        weight = Path(info["path"]) / info["weight_name"]
        if not weight.exists():
            logger.error(f"VAE file does not exist: {weight}")
            return False

        engine = model_manager.loaded_model
        return engine.attach_vae(str(weight))

    def restore_default_vae(self) -> bool:
        """Restore the loaded model's original VAE."""
        from ..models.manager import model_manager

        if not model_manager.is_model_loaded():
            return False
        return model_manager.loaded_model.restore_vae()

    def list_vaes(self) -> Dict[str, dict]:
        return dict(self.config)

    def get_vae_info(self, name: str) -> Optional[dict]:
        return self.config.get(name)

    def remove_vae(self, name: str) -> bool:
        if name not in self.config:
            logger.error(f"VAE {name} not found")
            return False
        info = self.config[name]
        path = Path(info["path"])
        if info.get("in_place"):
            logger.info(f"VAE {name} imported in place; unregistering only")
        elif path.exists() and path.parent == self.vae_dir:
            shutil.rmtree(path)
        del self.config[name]
        self._save_config()
        return True


# Module-level singleton.
vae_manager = VaeManager()
