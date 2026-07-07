"""OllamaDiffuser MCP Server - Model Context Protocol integration."""

import asyncio
import io
import logging
import sys
from typing import Optional

from ..core.models.manager import model_manager
from ..core.config.settings import settings

logger = logging.getLogger(__name__)


def _model_trigger_words(model_name: Optional[str]):
    """Trigger words stored on a model's config (CivitAI checkpoints), if any."""
    if not model_name:
        return []
    cfg = settings.models.get(model_name)
    if not cfg or not cfg.parameters:
        return []
    return cfg.parameters.get("trained_words") or []


def _ensure_mcp():
    """Check that the mcp package is available."""
    try:
        import mcp  # noqa: F401

        return True
    except ImportError:
        logger.error(
            "MCP package not installed. Install with: pip install 'ollamadiffuser[mcp]'"
        )
        return False


def create_mcp_server():
    """Create and configure the MCP server with all tools."""
    from mcp.server.fastmcp import FastMCP, Image

    mcp_server = FastMCP(
        "OllamaDiffuser",
        instructions=(
            "Local AI image generation via Stable Diffusion, FLUX, and 30+ models"
        ),
    )

    @mcp_server.tool()
    async def generate_image(
        prompt: str,
        model: Optional[str] = None,
        negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution",
        width: int = 1024,
        height: int = 1024,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        use_trigger_words: bool = True,
    ) -> Image:
        """Generate an image from a text prompt using a local diffusion model.

        Args:
            prompt: Text description of the desired image
            model: Model to use (auto-loads if needed). Leave empty to use current model.
            negative_prompt: What to avoid in the image
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of inference steps (model-specific default if omitted)
            guidance_scale: Guidance scale (model-specific default if omitted)
            seed: Random seed for reproducibility
            use_trigger_words: Auto-prepend the model's CivitAI trigger words if
                they are not already present in the prompt (recommended).
        """
        if model and model_manager.get_current_model() != model:
            if not model_manager.is_model_installed(model):
                raise ValueError(
                    f"Model '{model}' is not installed. "
                    f"Install it first: ollamadiffuser pull {model}"
                )
            logger.info(f"Loading model: {model}")
            success = await asyncio.to_thread(model_manager.load_model, model)
            if not success:
                raise RuntimeError(f"Failed to load model '{model}'")

        if not model_manager.is_model_loaded():
            raise RuntimeError(
                "No model loaded. Load one with: load_model('model-name') "
                "or pass model= parameter"
            )

        # Auto-inject CivitAI trigger words for the loaded model when missing.
        if use_trigger_words:
            words = _model_trigger_words(model_manager.get_current_model())
            missing = [w for w in words if w and w.lower() not in prompt.lower()]
            if missing:
                prompt = ", ".join(missing) + ", " + prompt
                logger.info(f"Injected trigger words: {missing}")

        engine = model_manager.loaded_model
        result = await asyncio.to_thread(
            engine.generate_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
        )

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return Image(data=buf.getvalue(), format="png")

    @mcp_server.tool()
    async def list_models() -> str:
        """List all available and installed image generation models.

        Returns a formatted list showing which models are available to download
        and which are already installed locally.
        """
        available = model_manager.list_available_models()
        installed = model_manager.list_installed_models()
        current = model_manager.get_current_model()

        # Include locally-registered models (e.g. from CivitAI) that are not in
        # the static registry, so imported/pulled models always show up.
        names = sorted(set(available) | set(installed))

        lines = ["Available models:"]
        for name in names:
            status_parts = []
            if name in installed:
                status_parts.append("installed")
            if name == current:
                status_parts.append("loaded")
            suffix = f" ({', '.join(status_parts)})" if status_parts else ""

            # Annotate installed models with type + base model for selection.
            detail = ""
            cfg = settings.models.get(name)
            if cfg:
                params = cfg.parameters or {}
                bits = [cfg.model_type]
                if params.get("base_model"):
                    bits.append(str(params["base_model"]))
                detail = f" [{', '.join(b for b in bits if b)}]"
            lines.append(f"  - {name}{suffix}{detail}")

        lines.append(f"\nInstalled: {len(installed)}/{len(names)}")
        if current:
            lines.append(f"Currently loaded: {current}")

        return "\n".join(lines)

    @mcp_server.tool()
    async def load_model(model_name: str) -> str:
        """Load a specific image generation model into memory.

        Args:
            model_name: Name of the model to load (must be installed first)
        """
        if not model_manager.is_model_installed(model_name):
            installed = model_manager.list_installed_models()
            return (
                f"Model '{model_name}' is not installed. "
                f"Installed models: {', '.join(installed) if installed else 'none'}. "
                f"Use 'ollamadiffuser pull {model_name}' to install it first."
            )

        success = await asyncio.to_thread(model_manager.load_model, model_name)
        if success:
            return f"Model '{model_name}' loaded successfully"
        return f"Failed to load model '{model_name}'"

    @mcp_server.tool()
    async def get_status() -> str:
        """Get the current status of OllamaDiffuser.

        Returns device info, loaded model, and installed model count.
        """
        is_loaded = model_manager.is_model_loaded()
        current = model_manager.get_current_model()
        installed = model_manager.list_installed_models()

        lines = ["OllamaDiffuser Status:"]
        lines.append(f"  Model loaded: {'yes' if is_loaded else 'no'}")
        if current:
            lines.append(f"  Current model: {current}")
        lines.append(f"  Installed models: {len(installed)}")

        if is_loaded and model_manager.loaded_model:
            engine = model_manager.loaded_model
            info = engine.get_model_info()
            if info:
                lines.append(f"  Device: {info.get('device', 'unknown')}")
                lines.append(f"  Model type: {info.get('type', 'unknown')}")

        return "\n".join(lines)

    @mcp_server.tool()
    async def get_model_details(model_name: str) -> str:
        """Get full metadata for an installed model to help choose/use it.

        Includes model type, base model, trigger words, source, and description
        when available (populated for models pulled or imported from CivitAI).

        Args:
            model_name: Name of an installed model.
        """
        cfg = settings.models.get(model_name)
        if cfg is None:
            info = model_manager.get_model_info(model_name)
            if not info:
                return f"Model '{model_name}' is not installed or unknown."
            lines = [f"{model_name}", f"  Model type: {info.get('model_type', 'unknown')}"]
            if info.get("repo_id"):
                lines.append(f"  Repository: {info['repo_id']}")
            return "\n".join(lines)

        params = cfg.parameters or {}
        lines = [f"{model_name}", f"  Model type: {cfg.model_type}"]
        if params.get("base_model"):
            lines.append(f"  Base model: {params['base_model']}")
        if params.get("source"):
            lines.append(f"  Source: {params['source']}")
        words = params.get("trained_words")
        if words:
            lines.append(f"  Trigger words: {', '.join(words)}")
        if params.get("tags"):
            lines.append(f"  Tags: {', '.join(params['tags'][:12])}")
        if params.get("nsfw"):
            lines.append("  Mature content: yes")
        if params.get("description"):
            lines.append(f"  Description: {params['description'][:400]}")
        lines.append(f"  Path: {cfg.path}")
        return "\n".join(lines)

    @mcp_server.tool()
    async def search_civitai(
        query: str,
        model_type: Optional[str] = None,
        base_model: Optional[str] = None,
        limit: int = 10,
        nsfw: bool = False,
        red: bool = False,
    ) -> str:
        """Search CivitAI for downloadable models.

        Args:
            query: Keyword to search for.
            model_type: Optional CivitAI type filter (Checkpoint, LORA, TextualInversion, VAE).
            base_model: Optional base-model filter (e.g. 'SDXL 1.0', 'Pony', 'SD 1.5').
            limit: Maximum number of results.
            nsfw: Include mature content (uses civitai.red when red=True).
            red: Search via civitai.red instead of civitai.com.

        Returns a list including each result's version id, which download_civitai_model accepts.
        """
        from ..core.utils import civitai_client

        base = civitai_client.RED_BASE if red else civitai_client.DEFAULT_BASE
        try:
            rows = await asyncio.to_thread(
                civitai_client.search, query, model_type, base_model, limit, nsfw, base
            )
        except civitai_client.CivitaiError as e:
            return f"Search failed: {e}"

        if not rows:
            return f"No CivitAI results for '{query}'."
        lines = [f"CivitAI results for '{query}':"]
        for r in rows:
            nsfw_tag = " [NSFW]" if r.get("nsfw") else ""
            lines.append(
                f"  - version {r.get('version_id')}: {r.get('name')} "
                f"({r.get('type')}, {r.get('base_model') or '?'}, "
                f"{r.get('download_count', 0)} downloads){nsfw_tag}"
            )
        lines.append("\nDownload one with download_civitai_model(url_or_id=<version id>).")
        return "\n".join(lines)

    @mcp_server.tool()
    async def download_civitai_model(
        url_or_id: str,
        model_type: Optional[str] = None,
        alias: Optional[str] = None,
        red: bool = False,
        experimental: bool = False,
    ) -> str:
        """Download and install a model from CivitAI / CivitAI Red.

        Args:
            url_or_id: A civitai.com/civitai.red URL or a numeric model-version id.
            model_type: Override the inferred type for checkpoints (sd15, sdxl, sd3, flux).
            alias: Local name to register the model under.
            red: Treat a bare id / host-less ref as civitai.red.
            experimental: Attempt FLUX/SD3 single-file checkpoints (may be incomplete).

        The API key is read from CIVITAI_API_KEY / settings; it is never passed here.
        """
        from ..core.utils.civitai_client import civitai_manager, CivitaiError

        try:
            result = await asyncio.to_thread(
                lambda: civitai_manager.pull(
                    url_or_id, model_type=model_type, alias=alias, red=red,
                    allow_experimental=experimental)
            )
        except CivitaiError as e:
            return f"Download failed: {e}"

        name = result["name"]
        category = result["content_category"]
        lines = [f"Installed '{name}' ({result.get('model_type') or category})."]
        if result.get("trained_words"):
            lines.append(f"Trigger words: {', '.join(result['trained_words'])}")
        if category == "checkpoint":
            lines.append(f"Load and use it with: load_model('{name}') then generate_image(...).")
        elif category == "lora":
            lines.append(f"Load it onto a running model with the lora CLI, then generate.")
        elif category == "embedding":
            lines.append(f"Apply it with load_embedding('{name}'), then use its trigger word in prompts.")
        elif category == "vae":
            lines.append(f"Apply it with attach_vae('{name}') while a model is loaded.")
        return "\n".join(lines)

    @mcp_server.tool()
    async def load_embedding(name: str) -> str:
        """Load an installed textual-inversion embedding into the current model.

        Args:
            name: Name of an installed embedding (see list after downloading).
        """
        from ..core.utils.embedding_manager import embedding_manager

        if not model_manager.is_model_loaded():
            return "No model loaded. Load a model first with load_model(...)."
        ok = await asyncio.to_thread(embedding_manager.load_embedding, name)
        if not ok:
            return f"Failed to load embedding '{name}' (is it installed?)."
        info = embedding_manager.get_embedding_info(name) or {}
        token = info.get("token")
        hint = f" Use '{token}' in your prompt to trigger it." if token else ""
        return f"Embedding '{name}' loaded.{hint}"

    @mcp_server.tool()
    async def attach_vae(name: str) -> str:
        """Attach an installed VAE to the current model (until it is reloaded).

        Args:
            name: Name of an installed VAE.
        """
        from ..core.utils.vae_manager import vae_manager

        if not model_manager.is_model_loaded():
            return "No model loaded. Load a model first with load_model(...)."
        ok = await asyncio.to_thread(vae_manager.attach_vae, name)
        return f"VAE '{name}' attached." if ok else f"Failed to attach VAE '{name}'."

    return mcp_server


def main(
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 9000,
):
    """Entry point for the MCP server.

    Args:
        transport: Transport type - "stdio", "sse", or "streamable-http".
        host: Bind address for network transports.
        port: Port for network transports.
    """
    if not _ensure_mcp():
        sys.exit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    server = create_mcp_server()

    if transport == "stdio":
        server.run(transport="stdio")
    elif transport in ("sse", "streamable-http"):
        import uvicorn

        app = server.sse_app() if transport == "sse" else server.streamable_http_app()
        uvicorn.run(app, host=host, port=port)
    else:
        logger.error(f"Unknown transport: {transport}. Use stdio, sse, or streamable-http.")
        sys.exit(1)


if __name__ == "__main__":
    main()
