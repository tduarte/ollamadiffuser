"""OllamaDiffuser MCP Server - Model Context Protocol integration."""

import asyncio
import io
import logging
import os
import sys
from typing import Optional

from ..core.models.manager import model_manager
from ..core.config.settings import settings
from ..core.config.model_registry import model_registry
from ..core.config import model_guide as _model_guide

logger = logging.getLogger(__name__)


# Curated negative-prompt terms that suppress the most common AI anatomy errors
# (hands/fingers/limbs/eyes). Merged into the negative prompt when
# avoid_anatomy_errors is on.
_ANATOMY_NEGATIVE = (
    "bad anatomy, bad hands, bad proportions, deformed, disfigured, mutation, mutated, "
    "extra fingers, fewer fingers, missing fingers, fused fingers, extra digit, fewer digits, "
    "malformed hands, poorly drawn hands, extra limbs, missing limbs, extra arms, extra legs, "
    "deformed eyes, cross-eyed, long neck"
)


def _merge_negatives(negative_prompt: str, add_anatomy: bool) -> str:
    """Append the anatomy negatives to the user's negative prompt, de-duped."""
    if not add_anatomy:
        return negative_prompt
    have = {t.strip().lower() for t in (negative_prompt or "").split(",") if t.strip()}
    extra = [t.strip() for t in _ANATOMY_NEGATIVE.split(",")
             if t.strip() and t.strip().lower() not in have]
    if not extra:
        return negative_prompt
    return (negative_prompt + ", " + ", ".join(extra)).strip(", ") if negative_prompt \
        else ", ".join(extra)


def _model_trigger_words(model_name: Optional[str]):
    """Trigger words to auto-inject: the loaded model's plus any loaded LoRA's."""
    words = []
    if model_name:
        cfg = settings.models.get(model_name)
        if cfg and cfg.parameters:
            words += cfg.parameters.get("trained_words") or []
    # Add the currently-loaded LoRA's trigger words (this is where they matter most).
    try:
        from ..core.utils.lora_manager import lora_manager
        active = lora_manager.current_lora
        if active and active in lora_manager.config:
            words += lora_manager.config[active].get("trained_words") or []
    except Exception:
        pass
    # De-dupe, preserve order.
    seen, out = set(), []
    for w in words:
        if w and w.lower() not in seen:
            seen.add(w.lower()); out.append(w)
    return out


def _model_type_and_params(model_name: str):
    """Resolve (model_type, parameters) for a model from its installed config or the
    static registry — the inputs the model_guide helpers need."""
    cfg = settings.models.get(model_name)
    reg = model_registry.get_model(model_name)
    model_type = (cfg.model_type if cfg else None) or (reg or {}).get("model_type")
    params = (cfg.parameters if cfg else None) or (reg or {}).get("parameters") or {}
    return model_type, params


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
    from mcp.server.fastmcp import Context, FastMCP, Image

    mcp_server = FastMCP(
        "OllamaDiffuser",
        instructions=(
            "Local AI image generation via Stable Diffusion, SDXL/Pony, FLUX, and MLX models.\n"
            "IMPORTANT: discover models and LoRAs through THESE TOOLS — do not search the "
            "filesystem. Models and LoRAs are registered with ollamadiffuser (in "
            "~/.ollamadiffuser), not just files on disk, and each carries metadata you need:\n"
            "  - model_guide(name?): what each base model is good for, a quality score, whether "
            "it needs Danbooru or Pony 'score_' tags, its recommended settings, and how far you "
            "can tune them. CALL model_guide('<model>') BEFORE generating with a model — "
            "generate_image REFUSES models you have not consulted, and the first generation must "
            "use the recommended settings it gives.\n"
            "  - list_models / get_model_details(name): installed models, their type, base model, "
            "trigger words, description.\n"
            "  - list_loras(base_model?) or find_loras(query): installed LoRAs with base model and "
            "trigger words.\n"
            "  - search_civitai / download_civitai_model: fetch new LoRAs (and embeddings/VAEs) "
            "from CivitAI. Full base-model checkpoints CANNOT be downloaded via MCP — they must "
            "be installed with the ollamadiffuser CLI.\n"
            "  - search_huggingface / install_hf_lora: fetch new LoRAs from Hugging Face (another "
            "source). HF LoRA repos can hold several .safetensors files — search with "
            "show_files=True (or get the list from the result) and pass weight_name.\n"
            "Typical flow: model_guide(<model>) -> load_model -> (optional) apply_lora / "
            "load_embedding / attach_vae -> generate_image. generate_image auto-injects the "
            "loaded model's and LoRA's trigger words. Each model's prompt/tag style and settings "
            "come from model_guide — e.g. Pony/Illustrious SDXL checkpoints need "
            "'score_9, score_8_up, score_7_up' and Danbooru tags, while FLUX/Klein want plain "
            "natural language; model_guide tells you which. The ANATOMY REVIEW tuning suggestions "
            "below are subordinate to the per-model tuning ranges model_guide returns.\n"
            "PROMPT QUALITY: before generating, if the user's prompt is short or vague (e.g. "
            "'a cat', 'a warrior'), first ENRICH it into a detailed prompt — a well-specified "
            "prompt improves results more than any parameter. Preserve the user's intent and any "
            "details they gave; never swap in a different subject. Work through this framework, "
            "adding what fits (skip what the user clearly doesn't want): (1) subject + key "
            "attributes (age, species/material, expression, clothing); (2) action/pose (dynamic vs "
            "static); (3) environment/setting (location, background, time period); (4) composition "
            "(shot type — close-up/wide/aerial, camera angle, framing, rule of thirds); (5) "
            "lighting (direction, soft/hard, source e.g. golden hour/studio/neon, mood); (6) color "
            "palette (dominant tones, contrast, saturation); (7) style/medium (photograph, oil "
            "painting, 3D render, anime, or a specific artist/era); (8) for photoreal, camera/lens "
            "(focal length, depth of field, film grain, aspect ratio); (9) mood/atmosphere; (10) "
            "detail/quality tags ('highly detailed', 'sharp focus', '8k') — these help some models "
            "more than others. Keep model/LoRA trigger words at the front. Surface the enhanced "
            "prompt to the user as part of the pre-generation confirmation summary (see CONFIRM "
            "BEFORE GENERATING). If the user gave an already-detailed prompt or asked for exact "
            "wording, use it verbatim.\n"
            "CONFIRM BEFORE GENERATING: image generation is slow, so do NOT call generate_image "
            "for a user-driven run until you have shown the user a summary of exactly what you "
            "will run and they have approved it. The summary must include: the final (enriched) "
            "prompt and the negative prompt; the model; any LoRA/embedding/VAE and its scale; and "
            "the key parameters (steps, guidance_scale, width x height, seed). Then ask the user "
            "to proceed, and only call generate_image once they confirm. This gate covers "
            "user-driven generations — a new request, or a change the user asks for (different "
            "prompt, model, LoRA, or params). It does NOT cover your OWN automatic anatomy-fix "
            "iteration (see ANATOMY REVIEW) — the seed changes, LoRA application, and "
            "guidance/step tweaks you make to fix a bad image proceed WITHOUT re-asking; just "
            "briefly note what you changed. Exception: if the user explicitly told you to "
            "generate without confirming (e.g. 'just generate', 'don't ask'), proceed and still "
            "briefly state what you ran.\n"
            "ANATOMY REVIEW: after each generate_image, LOOK AT the returned image and check "
            "hands/fingers (count + shape), limbs, eyes and faces — these are where diffusion "
            "models fail most. If you see extra/fused fingers, malformed hands, extra limbs or "
            "warped faces, FIX IT and regenerate: (1) change the seed (cheapest fix — most anatomy "
            "errors are seed-specific), (2) keep avoid_anatomy_errors=True, (3) if it's a "
            "Pony/SDXL model, apply_lora() an installed hands/anatomy LoRA (find_loras('hand')), "
            "(4) lower guidance_scale to ~5 and raise steps to ~30. Iterate up to a few times, "
            "then return the best result and note any remaining issue."
        ),
    )

    # Server-lifetime enforcement state (see the model_guide gate in generate_image):
    #   _briefed   — models whose model_guide(<name>) the agent has consulted.
    #   _generated — models that have completed at least one generation this session.
    # These are quantized local models that degrade badly with wrong settings, so the
    # gate REQUIRES the guide before generating and forces recommended settings on the
    # first generation. State resets on server restart; the agent simply re-briefs.
    _briefed: set = set()
    _generated: set = set()

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
        avoid_anatomy_errors: bool = True,
        ctx: Context = None,
    ) -> Image:
        """Generate an image from a local diffusion model; returns the finished image.

        Generation is slow. For a user-driven run, only call this AFTER you have
        summarized what you will generate (final enriched prompt + negative prompt,
        model, any LoRA/embedding/VAE + scale, and key params) and the user has
        confirmed — see the CONFIRM BEFORE GENERATING guidance in the server
        instructions. Your own automatic anatomy-fix rerolls do not need confirmation.

        This call blocks until the image is fully rendered. Progress steps count
        the denoising loop and are followed by a "Decoding image…" (VAE) phase
        that has no step count — reaching the last step (e.g. 30/30) does NOT mean
        it is done. Always wait for this tool to return the image; do not assume
        completion from the progress messages.

        If `prompt` is short or vague, first expand it into a detailed prompt
        (subject + attributes, action, setting, composition, lighting, color,
        style/medium, mood, quality tags) while preserving the user's intent —
        see the PROMPT QUALITY guidance in the server instructions. A
        well-specified prompt improves results more than any parameter here.

        After generating, LOOK AT the returned image and check anatomy —
        hands/fingers (count and shape), limbs, eyes, faces. If something is
        wrong, regenerate to fix it: try a different `seed` first (cheapest),
        keep `avoid_anatomy_errors=True`, and if the model is Pony/SDXL consider
        apply_lora() with an installed hands/anatomy LoRA (find_loras("hand")).
        Lowering guidance_scale slightly (~5) and raising steps (~30) also helps.

        Args:
            prompt: Text description of the desired image
            model: Model to use (auto-loads if needed). Leave empty to use current model.
            negative_prompt: What to avoid in the image
            width: Image width in pixels
            height: Image height in pixels
            steps: Number of inference steps (model-specific default if omitted)
            guidance_scale: Guidance scale (model-specific default if omitted)
            seed: Random seed for reproducibility. Change it to reroll an image
                whose anatomy came out wrong.
            use_trigger_words: Auto-prepend the model's CivitAI trigger words if
                they are not already present in the prompt (recommended).
            avoid_anatomy_errors: Merge curated anatomy terms (bad hands, extra
                fingers, deformed, ...) into the negative prompt (recommended).
        """
        negative_prompt = _merge_negatives(negative_prompt, avoid_anatomy_errors)
        # Decide whether we must (re)load. get_current_model() returns the
        # *persisted* current-model name, which survives an unload/restart even
        # when nothing is resident. So "name matches current" is NOT sufficient
        # to skip loading — we must also confirm something is actually in memory.
        # Otherwise generate_image(model='x') after a respawn wrongly skips the
        # load and then fails with "No model loaded".
        want = model or model_manager.get_current_model()

        # --- model_guide gate -------------------------------------------------
        # Quantized local models need the right settings. Require the agent to
        # consult model_guide(<model>) before generating, and force the model's
        # recommended settings on the FIRST generation. Later generations are
        # free (the anatomy-fix reroll loop tunes seed/guidance/steps itself).
        if want:
            _mt, _params = _model_type_and_params(want)
            _rec = _model_guide.recommended_settings(want, _mt, _params)
            if want not in _briefed:
                raise ValueError(
                    f"Blocked: call model_guide('{want}') before generating with it — "
                    f"these are quantized local models that need specific settings.\n\n"
                    f"{_model_guide.format_full(want, _mt, _params)}\n\n"
                    f"Then retry generate_image using the recommended settings above."
                )
            if want not in _generated:
                _rs, _rg = _rec.get("steps"), _rec.get("guidance_scale")
                if _rs is not None:
                    if steps is None:
                        steps = _rs
                    elif steps != _rs:
                        raise ValueError(
                            f"First generation with '{want}' must use the recommended "
                            f"steps={_rs} (you passed steps={steps}). Recommended: "
                            f"steps={_rs}, guidance_scale={_rg}. Tune on a later attempt."
                        )
                if _rg is not None:
                    if guidance_scale is None:
                        guidance_scale = _rg
                    elif abs(float(guidance_scale) - float(_rg)) > 1e-6:
                        raise ValueError(
                            f"First generation with '{want}' must use the recommended "
                            f"guidance_scale={_rg} (you passed guidance_scale={guidance_scale}). "
                            f"Recommended: steps={_rs}, guidance_scale={_rg}. "
                            f"Tune on a later attempt."
                        )

        need_load = bool(want) and (
            not model_manager.is_model_loaded()
            or model_manager.get_current_model() != want
        )
        if need_load:
            if not model_manager.is_model_installed(want):
                raise ValueError(
                    f"Model '{want}' is not installed. "
                    f"Install it first: ollamadiffuser pull {want}"
                )
            logger.info(f"Loading model: {want}")
            success = await asyncio.to_thread(model_manager.load_model, want)
            if not success:
                raise RuntimeError(f"Failed to load model '{want}'")

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
        # Per-step progress -> MCP progress notifications. Generation runs in a
        # worker thread; the callback fires there, so we hop back to the event
        # loop with run_coroutine_threadsafe. Fire-and-forget (never block the
        # denoise loop on the notification). report_progress is a no-op unless
        # the client sent a progressToken. A progress notification also signals
        # "still working", which MCP clients use to reset request timeouts.
        progress_callback = None
        if ctx is not None:
            loop = asyncio.get_running_loop()

            def progress_callback(step, total, message=None):
                try:
                    fut = asyncio.run_coroutine_threadsafe(
                        ctx.report_progress(
                            progress=step, total=total,
                            message=message or f"Step {step}/{total}"),
                        loop,
                    )
                    fut.add_done_callback(lambda f: f.exception())
                except Exception:
                    pass

        result = await asyncio.to_thread(
            engine.generate_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
            progress_callback=progress_callback,
        )

        # First generation for this model succeeded — subsequent calls may tune freely.
        if want:
            _generated.add(want)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return Image(data=buf.getvalue(), format="png")

    @mcp_server.tool()
    async def model_guide(model_name: Optional[str] = None) -> str:
        """Guidance for choosing and driving a base model. CALL THIS FIRST.

        These are quantized local models that behave very differently — some want
        plain natural-language prompts, others need Danbooru/Pony 'score_' tags;
        some are distilled to 4-8 steps at guidance ~1, others want 30 steps at
        guidance ~7. generate_image REQUIRES that you call model_guide(<model>)
        for the model you intend to use before it will generate, and it forces the
        model's recommended settings on the first generation.

        Args:
            model_name: Omit to browse a catalog of all available base models
                (family, quality, prompt style, what each is good for). Pass a
                specific model name to get its full guide — what it's good for, a
                quality score, whether it needs Danbooru/score tags, the
                recommended settings, and how far you can tune them. Calling it
                with a name marks that model as briefed so generate_image accepts it.
        """
        if model_name:
            if not (settings.models.get(model_name) or model_registry.get_model(model_name)):
                return (f"Model '{model_name}' is unknown. Call model_guide() with no "
                        f"argument to see the available base models.")
            mt, params = _model_type_and_params(model_name)
            _briefed.add(model_name)
            return (_model_guide.format_full(model_name, mt, params) +
                    "\n\nUse these recommended settings for the FIRST generation; "
                    "tune within the room above only on later attempts.")

        # No arg: catalog of available + installed base models.
        available = model_registry.get_available_models()      # name -> registry dict
        installed = settings.models                            # name -> ModelConfig
        names = sorted(set(available) | set(installed))
        lines = ["Available base models — call model_guide('<name>') before using one:"]
        for name in names:
            mt, params = _model_type_and_params(name)
            lines.append(_model_guide.catalog_line(name, mt, params))
        return "\n".join(lines)

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
            # "loaded" means actually resident in memory — not merely the
            # persisted current-model name (which survives unload/restart).
            if name == current and model_manager.is_model_loaded():
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
            if model_manager.is_model_loaded():
                lines.append(f"Currently loaded: {current}")
            else:
                lines.append(f"Current model (not resident, will load on use): {current}")

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
        alias: Optional[str] = None,
        red: bool = False,
    ) -> str:
        """Download and install a LoRA, embedding, or VAE from CivitAI / CivitAI Red.

        Full base-model checkpoints cannot be downloaded here — the ref is
        rejected before anything downloads. Install checkpoints with the
        ollamadiffuser CLI instead.

        Args:
            url_or_id: A civitai.com/civitai.red URL or a numeric model-version id.
            alias: Local name to register the LoRA/embedding/VAE under.
            red: Treat a bare id / host-less ref as civitai.red.

        The API key is read from CIVITAI_API_KEY / settings; it is never passed here.
        """
        from ..core.utils.civitai_client import civitai_manager, CivitaiError

        try:
            result = await asyncio.to_thread(
                lambda: civitai_manager.pull(
                    url_or_id, alias=alias, red=red, allow_checkpoints=False)
            )
        except CivitaiError as e:
            return f"Download failed: {e}"

        name = result["name"]
        category = result["content_category"]
        lines = [f"Installed '{name}' ({result.get('model_type') or category})."]
        if result.get("trained_words"):
            lines.append(f"Trigger words: {', '.join(result['trained_words'])}")
        if category == "lora":
            lines.append(f"Apply it to the loaded model with apply_lora('{name}'), then generate.")
        elif category == "embedding":
            lines.append(f"Apply it with load_embedding('{name}'), then use its trigger word in prompts.")
        elif category == "vae":
            lines.append(f"Apply it with attach_vae('{name}') while a model is loaded.")
        return "\n".join(lines)

    @mcp_server.tool()
    async def search_huggingface(
        query: str,
        model_type: Optional[str] = None,
        base_model: Optional[str] = None,
        limit: int = 10,
        show_files: bool = False,
    ) -> str:
        """Search Hugging Face Hub for downloadable models and LoRAs.

        Args:
            query: Keyword to search for.
            model_type: Optional filter: 'lora' (adapters) or 'checkpoint' (full diffusers model).
            base_model: Optional base-model tag filter, e.g. 'black-forest-labs/FLUX.2-klein-9B'.
            limit: Maximum number of results.
            show_files: Also list each result's .safetensors weight file(s). Use this to find the
                weight_name to pass to install_hf_lora when a LoRA repo has multiple files.

        Returns rows with each repo id; install a LoRA with install_hf_lora(repo_id, weight_name?).
        Full base models cannot be installed here — use the ollamadiffuser CLI for those.
        """
        from ..core.utils import hf_client

        try:
            rows = await asyncio.to_thread(
                hf_client.search, query, model_type, base_model, limit, show_files
            )
        except hf_client.HuggingFaceError as e:
            return f"Search failed: {e}"

        if not rows:
            return f"No Hugging Face results for '{query}'."
        lines = [f"Hugging Face results for '{query}':"]
        for r in rows:
            kind = "LoRA" if r.get("is_lora") else (r.get("pipeline_tag") or "model")
            line = (
                f"  - {r.get('repo_id')} ({kind}, base: {r.get('base_model') or '?'}, "
                f"{r.get('downloads', 0)} downloads, {r.get('likes', 0)} likes)"
            )
            if show_files:
                weights = r.get("lora_weights") or []
                if weights:
                    line += f"\n      weights: {', '.join(weights)}"
            lines.append(line)
        lines.append("\nInstall a LoRA with install_hf_lora(repo_id=..., weight_name=...). "
                     "Full base models must be installed with the ollamadiffuser CLI.")
        return "\n".join(lines)

    @mcp_server.tool()
    async def install_hf_lora(
        repo_id: str,
        weight_name: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> str:
        """Download and install a LoRA from Hugging Face.

        Args:
            repo_id: Hugging Face repo, e.g. 'diroverflo/FLux_Klein_9B_NSFW'.
            weight_name: The .safetensors weight file to install. Optional if the repo has exactly
                one; required (with an error listing options) if it has several. Use
                search_huggingface(show_files=True) to discover the file names.
            alias: Local name to register the LoRA under (defaults to the repo id slug).

        After installing, apply it to the loaded model with apply_lora('<name>').
        """
        from ..core.utils.hf_client import hf_manager, HuggingFaceError

        try:
            result = await asyncio.to_thread(
                lambda: hf_manager.pull_lora(repo_id, weight_name=weight_name, alias=alias)
            )
        except HuggingFaceError as e:
            return f"Install failed: {e}"

        name = result["name"]
        lines = [f"Installed LoRA '{name}' from {repo_id}."]
        if result.get("weight_name"):
            lines.append(f"Weight file: {result['weight_name']}")
        if result.get("base_model"):
            lines.append(f"Base model: {result['base_model']}")
        lines.append(f"Apply it to the loaded model with apply_lora('{name}').")
        return "\n".join(lines)

    @mcp_server.tool()
    async def list_loras(base_model: Optional[str] = None) -> str:
        """List installed LoRAs the agent can apply to a model.

        Shows each LoRA's base model and trigger words so you can pick one that
        matches the loaded checkpoint and know what to put in the prompt.

        Args:
            base_model: Optional filter, e.g. 'SDXL 1.0', 'Pony', 'SD 1.5'.
        """
        from ..core.utils.lora_manager import lora_manager

        loras = lora_manager.list_installed_loras()
        if not loras:
            return "No LoRAs installed. Download some with download_civitai_model(...)."
        current = lora_manager.get_current_lora()
        lines = ["Installed LoRAs:"]
        shown = 0
        for name, info in sorted(loras.items()):
            bm = info.get("base_model")
            if base_model and (bm or "").lower() != base_model.lower():
                continue
            shown += 1
            tw = info.get("trained_words") or []
            tags = " [loaded]" if name == current else ""
            trig = f" — triggers: {', '.join(tw)}" if tw else ""
            lines.append(f"  - {name}{tags} ({bm or 'unknown base'}){trig}")
        if shown == 0:
            return f"No LoRAs match base model '{base_model}'."
        lines.append("\nApply one to the loaded model with apply_lora('<name>').")
        return "\n".join(lines)

    @mcp_server.tool()
    async def find_loras(query: str = "", base_model: Optional[str] = None,
                         limit: int = 30) -> str:
        """Search installed LoRAs by keyword — use this instead of browsing the filesystem.

        Matches the query against each LoRA's name, trigger words, and base model.
        LoRAs live in ollamadiffuser's registry (~/.ollamadiffuser/loras), so a
        filesystem `find` will miss their metadata; this returns it.

        Args:
            query: Keyword(s) to match (name / trigger words / base model). Empty lists all.
            base_model: Optional exact base-model filter (e.g. 'Pony', 'SDXL 1.0').
            limit: Max results to return.
        """
        from ..core.utils.lora_manager import lora_manager

        loras = lora_manager.list_installed_loras()
        if not loras:
            return "No LoRAs installed. Download some with download_civitai_model(...)."
        q = query.lower().strip()
        current = lora_manager.get_current_lora()
        rows = []
        for name, info in loras.items():
            bm = info.get("base_model") or ""
            if base_model and bm.lower() != base_model.lower():
                continue
            tw = info.get("trained_words") or []
            haystack = " ".join([name, bm, " ".join(tw)]).lower()
            if q and q not in haystack and not any(tok in haystack for tok in q.split()):
                continue
            rows.append((name, bm, tw, name == current))
        if not rows:
            suffix = f" for base model '{base_model}'" if base_model else ""
            return f"No LoRAs match '{query}'{suffix}."
        rows.sort(key=lambda r: r[0])
        lines = [f"{len(rows)} matching LoRA(s):"]
        for name, bm, tw, loaded in rows[:limit]:
            tag = " [loaded]" if loaded else ""
            trig = f" — triggers: {', '.join(tw)}" if tw else ""
            lines.append(f"  - {name}{tag} ({bm or 'unknown base'}){trig}")
        if len(rows) > limit:
            lines.append(f"  ... and {len(rows) - limit} more (narrow with query or base_model).")
        lines.append("\nApply one to the loaded model with apply_lora('<name>').")
        return "\n".join(lines)

    @mcp_server.tool()
    async def apply_lora(name: str, scale: float = 1.0) -> str:
        """Apply an installed LoRA to the currently-loaded model.

        Its trigger words are then auto-injected into generate_image prompts.

        Args:
            name: LoRA name (see list_loras).
            scale: LoRA strength (default 1.0).
        """
        from ..core.utils.lora_manager import lora_manager

        if not model_manager.is_model_loaded():
            return "No model loaded. Load a model first with load_model(...)."
        # Resolve tolerant of spaces/dashes/case (agents often pass the display name).
        resolved = lora_manager.resolve_lora_name(name)
        if resolved is None:
            return (f"No LoRA matches '{name}'. Use list_loras() or find_loras('{name}') "
                    "to see the exact registered names.")
        ok = await asyncio.to_thread(lora_manager.load_lora, resolved, scale)
        if not ok:
            return (f"Found LoRA '{resolved}' but it failed to load onto the current model "
                    "— it may be for a different base model (check its base with get_model_details / list_loras).")
        info = lora_manager.get_lora_info(resolved) or {}
        tw = info.get("trained_words") or []
        hint = f" Trigger words to add to the prompt: {', '.join(tw)}." if tw else ""
        return f"LoRA '{resolved}' applied at scale {scale}.{hint}"

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

    # Guarantee the client can always terminate us. Generation runs on a
    # background (daemon) thread; if the client SIGTERMs us mid-run — e.g. it
    # gave up on a slow call and is restarting the server — exit hard so we
    # can't linger as an orphaned process holding a multi-GB model in memory.
    if transport == "stdio":
        import signal

        def _hard_exit(_signum, _frame):
            os._exit(0)

        try:
            signal.signal(signal.SIGTERM, _hard_exit)
        except (ValueError, OSError):
            pass  # e.g. not running in the main thread

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
