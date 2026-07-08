#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) manager for downloading and managing LoRA weights
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Callable
import logging
from datetime import datetime
from huggingface_hub import hf_hub_download, login

from ..config.settings import settings
from .download_utils import robust_file_download

logger = logging.getLogger(__name__)


def parse_suggested_weight(text: Optional[str]) -> Optional[float]:
    """Best-effort extract a suggested LoRA weight/strength from free text.

    CivitAI exposes no structured recommended-weight, but authors often write it in the
    description ("recommended weight: 0.7", "strength 0.6-0.8"). Returns the value (midpoint
    of a range) when a weight/strength keyword sits near a plausible number, else None.
    """
    if not text:
        return None
    low = text.lower()
    # A weight/strength keyword followed (within a few chars) by a number or range.
    m = re.search(r"(?:weight|strength)\D{0,12}?(\d(?:\.\d+)?)(?:\s*[-–to]{1,3}\s*(\d(?:\.\d+)?))?",
                  low)
    if not m:
        return None
    try:
        lo = float(m.group(1))
        hi = float(m.group(2)) if m.group(2) else lo
    except ValueError:
        return None
    val = (lo + hi) / 2.0
    # Plausible LoRA weights only (ignore stray numbers like step counts).
    return round(val, 2) if 0.0 < val <= 2.0 else None

class LoRAManager:
    """Manager for LoRA weights"""
    
    def __init__(self):
        self.lora_dir = settings.config_dir / "loras"
        self.lora_dir.mkdir(exist_ok=True)
        self.config_file = self.lora_dir / "loras.json"
        self.current_lora = None
        self._load_config()
    
    def _load_config(self):
        """Load LoRA configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load LoRA config: {e}")
                self.config = {}
        else:
            self.config = {}
    
    def _save_config(self):
        """Save LoRA configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save LoRA config: {e}")
    
    def _get_lora_path(self, lora_name: str) -> Path:
        """Get path for LoRA storage"""
        return self.lora_dir / lora_name
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory"""
        total_size = 0
        try:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"Failed to calculate directory size: {e}")
        return total_size
    
    def _is_server_running(self) -> bool:
        """Check if the API server is running"""
        try:
            import requests
            response = requests.get(f"http://{settings.server.host}:{settings.server.port}/api/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _try_load_lora_via_api(self, lora_name: str, scale: float = 1.0) -> bool:
        """Try to load LoRA via API server"""
        try:
            if not self._is_server_running():
                return False
            
            # Resolve tolerant of spaces/dashes/case, then check it exists
            resolved = self.resolve_lora_name(lora_name)
            if resolved is None:
                logger.error(f"LoRA {lora_name} not found")
                return False
            lora_name = resolved

            lora_info = self.config[lora_name]
            
            import requests
            
            # Prepare API request. Send the resolved local weight path (not the
            # bare repo_id) so the server-side load_lora_runtime can resolve a
            # real file — required for MLX (mflux), harmless for diffusers.
            load_source, weight_name = self._resolve_load_source(lora_info)
            api_data = {
                "lora_name": lora_name,
                "repo_id": load_source,
                "scale": scale
            }

            if weight_name:
                api_data["weight_name"] = weight_name
            
            # Make API request to load LoRA
            response = requests.post(
                f"http://{settings.server.host}:{settings.server.port}/api/lora/load",
                json=api_data,
                timeout=30
            )
            
            if response.status_code == 200:
                self.current_lora = lora_name
                logger.info(f"LoRA {lora_name} loaded successfully via API with scale {scale}")
                return True
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load LoRA via API: {e}")
            return False
    
    def _try_unload_lora_via_api(self) -> bool:
        """Try to unload LoRA via API server"""
        try:
            if not self._is_server_running():
                return False
            
            import requests
            
            # Make API request to unload LoRA
            response = requests.post(
                f"http://{settings.server.host}:{settings.server.port}/api/lora/unload",
                timeout=30
            )
            
            if response.status_code == 200:
                self.current_lora = None
                logger.info("LoRA unloaded successfully via API")
                return True
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unload LoRA via API: {e}")
            return False
    
    def _resolve_load_source(self, lora_info: Dict) -> tuple:
        """Return ``(load_source, weight_name)`` to pass to ``load_lora_runtime``.

        Prefer the already-downloaded local weight file: MLX (mflux) needs a real
        filesystem path, and diffusers avoids a re-download. Fall back to the
        stored ``repo_id`` (e.g. a bare Hugging Face repo id) only when the local
        file is missing. Shared by the in-process path (:meth:`load_lora`) and the
        API path (:meth:`_try_load_lora_via_api`) so both resolve identically.
        """
        weight_name = lora_info.get("weight_name")
        load_source = lora_info.get("repo_id")
        path = lora_info.get("path")
        if weight_name and path and (Path(path) / weight_name).is_file():
            load_source = str(path)
        return load_source, weight_name

    def pull_lora(self, repo_id: str, weight_name: Optional[str] = None,
                  alias: Optional[str] = None, progress_callback: Optional[Callable] = None) -> bool:
        """Download LoRA weights from Hugging Face Hub"""
        try:
            # Determine local name
            lora_name = alias if alias else repo_id.replace('/', '_')
            lora_path = self._get_lora_path(lora_name)
            
            # Check if already exists
            if lora_name in self.config and lora_path.exists():
                if progress_callback:
                    progress_callback(f"✅ LoRA {lora_name} already exists")
                logger.info(f"LoRA {lora_name} already exists")
                return True
            
            # Create directory
            lora_path.mkdir(exist_ok=True)
            
            # Ensure HuggingFace token is set
            if settings.hf_token:
                login(token=settings.hf_token)
                if progress_callback:
                    progress_callback(f"🔑 Authenticated with HuggingFace")
            
            if progress_callback:
                progress_callback(f"📥 Downloading LoRA from {repo_id}")
            
            # Download specific weight file or all files
            if weight_name:
                # Download specific file
                downloaded_file = robust_file_download(
                    repo_id=repo_id,
                    filename=weight_name,
                    local_dir=str(lora_path),
                    cache_dir=str(settings.cache_dir),
                    progress_callback=progress_callback
                )
                
                # Store metadata
                lora_info = {
                    "repo_id": repo_id,
                    "weight_name": weight_name,
                    "path": str(lora_path),
                    "downloaded_at": datetime.now().isoformat(),
                    "size": self._format_size(self._get_directory_size(lora_path))
                }
            else:
                # Download all files (fallback)
                from .download_utils import robust_snapshot_download
                robust_snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(lora_path),
                    cache_dir=str(settings.cache_dir),
                    progress_callback=progress_callback
                )
                
                # Store metadata
                lora_info = {
                    "repo_id": repo_id,
                    "path": str(lora_path),
                    "downloaded_at": datetime.now().isoformat(),
                    "size": self._format_size(self._get_directory_size(lora_path))
                }
            
            # Update configuration
            self.config[lora_name] = lora_info
            self._save_config()
            
            logger.info(f"LoRA {lora_name} downloaded successfully")
            if progress_callback:
                progress_callback(f"✅ LoRA {lora_name} downloaded successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download LoRA: {e}")
            if progress_callback:
                progress_callback(f"❌ Failed to download LoRA: {e}")
            
            # Clean up failed download
            if lora_path.exists():
                try:
                    shutil.rmtree(lora_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up failed download: {cleanup_error}")
            
            return False
    
    def lora_dir_for(self, lora_name: str) -> Path:
        """Return (creating if needed) the storage directory for a named LoRA."""
        lora_path = self._get_lora_path(lora_name)
        lora_path.mkdir(parents=True, exist_ok=True)
        return lora_path

    @staticmethod
    def _norm_name(s: str) -> str:
        """Normalize a LoRA name for tolerant matching (drop spaces/dashes/case)."""
        return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

    def resolve_lora_name(self, name: str) -> Optional[str]:
        """Resolve a user/agent-supplied name to an actual registry key.

        Registry keys are slugs (dashes), but callers often pass the human name
        with spaces or the original filename. Match exactly first, then by a
        normalized form of the key and the stored weight filename.
        """
        if name in self.config:
            return name
        target = self._norm_name(name)
        if not target:
            return None
        for key, info in self.config.items():
            if self._norm_name(key) == target:
                return key
            weight = info.get("weight_name", "")
            if weight and self._norm_name(Path(weight).stem) == target:
                return key
        return None

    def register_downloaded_lora(self, lora_name: str, file_path, source: str,
                                 trained_words: Optional[List[str]] = None,
                                 base_model: Optional[str] = None,
                                 in_place: bool = False,
                                 description: Optional[str] = None) -> bool:
        """Register a LoRA from a local single file (from CivitAI or import).

        Stores ``repo_id``/``weight_name`` pointing at the exact local file so the
        existing ``load_lora`` weight-file branch loads it unchanged. ``in_place``
        marks files that live outside the managed LoRA dir (do not delete on rm).
        ``description`` (when available) is scanned for a suggested weight/strength.
        """
        file_path = Path(file_path)
        parent = str(file_path.parent)
        self.config[lora_name] = {
            "repo_id": parent,          # local dir acts as the load source
            "weight_name": file_path.name,
            "path": parent,
            "source": source,
            "trained_words": trained_words or [],
            "base_model": base_model,
            "in_place": in_place,
            "downloaded_at": datetime.now().isoformat(),
            "size": self._format_size(self._get_directory_size(file_path.parent)),
            "suggested_weight": parse_suggested_weight(description),
        }
        self._save_config()
        logger.info(f"Registered LoRA '{lora_name}' from {file_path} (source={source})")
        return True

    def load_lora(self, lora_name: str, scale: float = 1.0) -> bool:
        """Load LoRA weights into the current model"""
        try:
            from ..models.manager import model_manager
            
            # Check if model is loaded locally
            if not model_manager.is_model_loaded():
                # Try to load via API if server is running
                if self._try_load_lora_via_api(lora_name, scale):
                    return True
                logger.error("No model is currently loaded")
                return False
            
            # Resolve tolerant of spaces/dashes/case, then check it exists
            resolved = self.resolve_lora_name(lora_name)
            if resolved is None:
                logger.error(f"LoRA {lora_name} not found")
                return False
            lora_name = resolved

            lora_info = self.config[lora_name]
            lora_path = Path(lora_info["path"])
            
            if not lora_path.exists():
                logger.error(f"LoRA path does not exist: {lora_path}")
                return False
            
            # Get the inference engine
            engine = model_manager.loaded_model
            if not engine:
                logger.error("No inference engine available")
                return False
            
            # Load LoRA weights
            if "weight_name" in lora_info:
                # Prefer the already-downloaded local file (MLX needs a real path;
                # diffusers avoids a re-download). Same resolution as the API path.
                load_source, weight_name = self._resolve_load_source(lora_info)
                success = engine.load_lora_runtime(
                    repo_id=load_source,
                    weight_name=weight_name,
                    scale=scale
                )
            else:
                # Load from local directory
                weight_files = list(lora_path.glob("*.safetensors"))
                if not weight_files:
                    weight_files = list(lora_path.glob("*.bin"))
                
                if not weight_files:
                    logger.error(f"No weight files found in {lora_path}")
                    return False
                
                # Use the first weight file found
                weight_file = weight_files[0]
                success = engine.load_lora_runtime(
                    repo_id=str(lora_path),
                    weight_name=weight_file.name,
                    scale=scale
                )
            
            if success:
                self.current_lora = lora_name
                logger.info(f"LoRA {lora_name} loaded successfully with scale {scale}")
                return True
            else:
                logger.error(f"Failed to load LoRA {lora_name}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            return False
    
    def unload_lora(self) -> bool:
        """Unload current LoRA weights"""
        try:
            from ..models.manager import model_manager
            
            # Check if model is loaded locally
            if not model_manager.is_model_loaded():
                # Try to unload via API if server is running
                if self._try_unload_lora_via_api():
                    return True
                logger.error("No model is currently loaded")
                return False
            
            # Get the inference engine
            engine = model_manager.loaded_model
            if not engine:
                logger.error("No inference engine available")
                return False
            
            # Unload LoRA weights
            success = engine.unload_lora()
            
            if success:
                self.current_lora = None
                logger.info("LoRA weights unloaded successfully")
                return True
            else:
                logger.error("Failed to unload LoRA weights")
                return False
                
        except Exception as e:
            logger.error(f"Failed to unload LoRA: {e}")
            return False
    
    def remove_lora(self, lora_name: str) -> bool:
        """Remove LoRA weights"""
        try:
            # Check if LoRA exists
            if lora_name not in self.config:
                logger.error(f"LoRA {lora_name} not found")
                return False
            
            # Unload if currently loaded
            if self.current_lora == lora_name:
                self.unload_lora()
            
            # Remove files
            lora_info = self.config[lora_name]
            lora_path = Path(lora_info["path"])

            # Never delete files that were imported in place (they live in the
            # user's own directory, outside the managed LoRA store).
            if lora_info.get("in_place"):
                logger.info(f"LoRA {lora_name} imported in place; unregistering only, keeping files")
            elif lora_path.exists():
                shutil.rmtree(lora_path)

            # Remove from configuration
            del self.config[lora_name]
            self._save_config()
            
            logger.info(f"LoRA {lora_name} removed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove LoRA: {e}")
            return False
    
    def list_installed_loras(self) -> Dict[str, Dict]:
        """List all installed LoRA weights"""
        return self.config.copy()
    
    def get_current_lora(self) -> Optional[str]:
        """Get currently loaded LoRA name"""
        return self.current_lora
    
    def get_lora_info(self, lora_name: str) -> Optional[Dict]:
        """Get information about a specific LoRA (tolerant of spaces/dashes/case)."""
        resolved = self.resolve_lora_name(lora_name)
        return self.config.get(resolved) if resolved else None

    def is_lora_installed(self, lora_name: str) -> bool:
        """Check if LoRA is installed (tolerant of spaces/dashes/case)."""
        return self.resolve_lora_name(lora_name) is not None

# Global LoRA manager instance
lora_manager = LoRAManager() 