"""OllamaDiffuser MCP Server - Model Context Protocol integration."""

import asyncio
import base64
import io
import logging
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image as PILImage

from ..core.models.manager import model_manager
from ..core.config.settings import settings
from ..core.config.model_registry import model_registry
from ..core.config import model_guide as _model_guide

logger = logging.getLogger(__name__)


# Generated images are written here so the agent can (a) reuse an earlier result as an
# input_image/control_image path, and (b) tell the user where the file is. `from_last`
# still works for the immediate previous result without a path.
_OUTPUT_DIR = Path(tempfile.gettempdir()) / "ollamadiffuser-outputs"


def _save_output(image: "PILImage.Image", model: Optional[str]) -> str:
    """Persist a generated image to the outputs dir; return its absolute path."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    slug = (model or "image").replace("/", "_").replace(" ", "_")
    path = _OUTPUT_DIR / f"{slug}_{stamp}.png"
    image.save(path, format="PNG")
    return str(path)


# Curated negative-prompt terms that suppress the most common AI anatomy errors
# (hands/fingers/limbs/eyes). Merged into the negative prompt when
# avoid_anatomy_errors is on.
_ANATOMY_NEGATIVE = (
    "bad anatomy, bad hands, bad proportions, deformed, disfigured, mutation, mutated, "
    "extra fingers, fewer fingers, missing fingers, fused fingers, extra digit, fewer digits, "
    "malformed hands, poorly drawn hands, extra limbs, missing limbs, extra arms, extra legs, "
    "deformed eyes, cross-eyed, long neck"
)


def _append_negatives(negative_prompt: str, terms: str) -> str:
    """Append comma-separated `terms` to `negative_prompt`, de-duped case-insensitively."""
    have = {t.strip().lower() for t in (negative_prompt or "").split(",") if t.strip()}
    extra = [t.strip() for t in terms.split(",")
             if t.strip() and t.strip().lower() not in have]
    if not extra:
        return negative_prompt
    return (negative_prompt + ", " + ", ".join(extra)).strip(", ") if negative_prompt \
        else ", ".join(extra)


def _merge_negatives(negative_prompt: str, add_anatomy: bool) -> str:
    """Append the anatomy negatives to the user's negative prompt, de-duped."""
    if not add_anatomy:
        return negative_prompt
    return _append_negatives(negative_prompt, _ANATOMY_NEGATIVE)


def _load_image_path(ref: str, arg_name: str) -> "PILImage.Image":
    """Load a caller-supplied image for img2img/control from any convenient form:
    a local file path, an http(s) URL, a data: URI, or raw base64 bytes.

    This lets a client hand over a user-provided image directly (URL or base64) without
    first writing it to disk. Generated images are also saved to disk (their path is in
    each generate_image response). Agents that fabricate a path for the previous result
    are steered to from_last instead of hitting a bare error.
    """
    ref = (ref or "").strip()

    # data: URI (base64-encoded image)
    if ref.startswith("data:"):
        try:
            b64 = ref.split(",", 1)[1]
            return PILImage.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        except Exception as e:
            raise ValueError(f"{arg_name}: could not decode data: URI ({e}).")

    # http(s) URL — download it
    if ref.startswith(("http://", "https://")):
        try:
            import requests
            resp = requests.get(ref, timeout=30)
            resp.raise_for_status()
            return PILImage.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            raise ValueError(f"{arg_name}: could not fetch image from URL ({e}).")

    # Local file path
    if os.path.isfile(ref):
        return PILImage.open(ref).convert("RGB")

    # Raw base64 fallback (long, scheme-less, not a file)
    if len(ref) > 100 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", ref or ""):
        try:
            return PILImage.open(io.BytesIO(base64.b64decode(ref))).convert("RGB")
        except Exception:
            pass

    raise ValueError(
        f"{arg_name}={ref[:60]!r}… is not a readable image. Accepted: a local file "
        f"path, an http(s) URL, a data: URI, or base64 bytes. To reuse the PREVIOUS "
        f"generation, use from_last='init' (img2img/edit) or from_last='control' "
        f"(restyle/upscale) instead of inventing a path."
    )


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


def _checkpoint_arch(model_name: Optional[str]):
    """Normalized architecture of a checkpoint for LoRA compatibility (sdxl/sd15/sd3/flux/
    qwen/z-image), or None if unknown. SDXL/Pony/Illustrious all collapse to 'sdxl'."""
    if not model_name:
        return None
    mt, params = _model_type_and_params(model_name)
    mt = (mt or "").lower()
    if mt == "mlx":
        variant = str(params.get("mlx_variant") or "").lower()
        if variant.startswith("flux"):
            return "flux"
        if "qwen" in variant:
            return "qwen"
        if "z_image" in variant or "z-image" in variant:
            return "z-image"
        return "flux"
    if mt in ("sdxl", "controlnet_sdxl"):
        return "sdxl"
    if mt in ("sd15", "controlnet_sd15"):
        return "sd15"
    return mt or None


def _lora_arch(base_model: Optional[str]):
    """Normalized architecture a LoRA targets, from its CivitAI base_model string; None if
    unknown (e.g. HF LoRAs that carry no base_model)."""
    if not base_model:
        return None
    from ..core.utils.civitai_client import map_base_model
    return map_base_model(base_model)


def _compat_note(lora_base_model: Optional[str], loaded_arch: Optional[str]) -> str:
    """Short compatibility marker for a LoRA vs the loaded checkpoint arch, or '' when no
    model is loaded / either arch is unknown."""
    if not loaded_arch:
        return ""
    la = _lora_arch(lora_base_model)
    if la is None:
        return "  (base unknown — may not apply)"
    return f"  ✓ fits {loaded_arch}" if la == loaded_arch else \
        f"  ✗ needs {la} (loaded: {loaded_arch})"


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
            "it needs Danbooru or Pony 'score_' tags, HOW to prompt it (word order, (tag:1.2) "
            "weighting, BREAK, labeled Subject:/Pose:/Camera: sections, negative-prompt support), "
            "a realism recipe for realism-tuned checkpoints, its recommended settings, and how "
            "far you can tune them. CALL model_guide('<model>') BEFORE generating with a model — "
            "generate_image REFUSES models you have not consulted, and the first generation must "
            "use the recommended settings it gives.\n"
            "  - recommend_model(need?): the models that can do what you need (img2img, "
            "controlnet/restyle, upscale, edit, realism, ...) RANKED by quality — use it to pick "
            "the best model for a task when several qualify.\n"
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
            "IMAGE-TO-IMAGE / EDIT / UPSCALE: generate_image can also take an image. model_guide "
            "reports each model's image_ops. Multi-pass pipeline: draft with a cheap/fast model "
            "(txt2img) -> refine/restyle by feeding that image back (generate_image with "
            "from_last='init' + strength: ~0.3 keeps composition and improves quality, ~0.6 "
            "restyles) -> optional instruction EDIT with an edit model (flux.1-kontext-dev-mlx or "
            "qwen-image-edit-mlx: from_last='init' and describe the change in the prompt) -> "
            "optional UPSCALE (load flux.1-controlnet-upscaler-mlx, from_last='control', set a "
            "larger width/height). QUALITY DIRECTION: refine LAST with the HIGHER-quality model — "
            "e.g. draft on SDXL then refine on Klein for quality; only go Klein->Pony when the "
            "GOAL is an anime restyle, not higher fidelity. Each pass is its own generate_image "
            "call and needs its model briefed via model_guide first.\n"
            "CONTROLNET RESTYLE (change style, KEEP structure): to restyle a photo while holding "
            "its composition, load a FLUX control model and pass the SOURCE image via "
            "control_image (or from_last='control'), with the TARGET STYLE in the prompt. On MLX "
            "the edge/depth map is extracted for you — pass a NORMAL photo, not a pre-made map. "
            "WHICH MODEL: DEFAULT to DEPTH (flux.1-depth-dev-mlx) for most restyles — it keeps the "
            "3D layout but lets style/materials/lighting change freely (photo->painting, "
            "day->night, real->anime). Use CANNY (flux.1-controlnet-canny-mlx) only when exact "
            "OUTLINES must survive — architecture, products, logos, text, line art — accepting a "
            "less dramatic restyle. Call model_guide on the one you pick for the full recipe. Dial "
            "tightness: canny via controlnet_conditioning_scale (higher = hug edges), depth via "
            "`strength` (lower = less source bleed / more restyle). This is NOT img2img (which "
            "drifts from the whole image) — ControlNet holds structure explicitly. Heads-up: canny "
            "is only a ~3GB adapter and needs the FLUX.1-dev base (~20GB) installed; depth is a "
            "self-contained model. Match width/height to the SOURCE's aspect ratio (a square "
            "1024x1024 on a non-square source distorts it). And do NOT pass `strength` to the "
            "depth model — it generates fresh from the depth map; a strength value turns the "
            "output into noise.\n"
            "REALISM & ORDER OF OPERATIONS: the PROMPT and NEGATIVE tags are the DOMINANT quality "
            "lever — LoRAs and guidance are secondary polish. For a realism-tuned Pony/SDXL "
            "checkpoint that still looks cartoonish, first fix the PROMPT (add photo/realistic "
            "tags, drop 'source_anime') and NEGATIVES (cartoon, anime, illustration, 3d render, "
            "plastic skin — generate_image auto-adds these when it detects a realism checkpoint) "
            "before reaching for a LoRA. Follow model_guide's realism recipe. Apply LoRAs at "
            "MODERATE weight (0.5-0.8, not 1.0). Note: many top 'photorealism' LoRAs are FLUX or "
            "plain-SDXL and are INCOMPATIBLE with a Pony checkpoint — apply_lora blocks those, "
            "and list_loras/search_civitai flag them.\n"
            "A/B COMPARE: to attribute a quality change correctly, fix a `seed`, generate a "
            "baseline, then change ONE thing (apply_lora, or a prompt tweak) and regenerate with "
            "the SAME seed. Changing prompt + LoRA + seed at once makes the result impossible to "
            "attribute. Establish a prompt-only baseline before adding LoRAs.\n"
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
            "the key parameters (steps, guidance_scale, width x height, seed). For an image "
            "conditioned pass (img2img refine / edit / upscale) also state the pass type, the "
            "source image (the previous result via from_last, or a path), and the strength (or the "
            "target size for an upscale). Then ask the user "
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
    # The most recent generated image, for chaining passes (from_last=). Holds
    # {"image": PIL.Image, "model": name}. In-memory; resets on server restart.
    _last_image: dict = {}

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
        avoid_cartoon: Optional[bool] = None,
        input_image: Optional[str] = None,
        control_image: Optional[str] = None,
        from_last: Optional[str] = None,
        strength: float = 0.75,
        controlnet_conditioning_scale: float = 1.0,
        ctx: Context = None,
    ):
        """Generate an image from a local diffusion model.

        Returns the finished image inline AND a line with the path it was saved to. To
        feed a generated image into a later img2img/edit/upscale pass, prefer
        from_last='init'/'control' (chains the immediate previous result), or pass the
        SAVED PATH from a prior response. For a USER-PROVIDED image, input_image /
        control_image accept a local path, an http(s) URL, a data: URI, or base64 bytes
        directly — so you don't need to save it first. Don't invent a path.


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
            width: Image width in pixels. For img2img / edit / ControlNet, MATCH the
                source image's aspect ratio (the source is resized to width x height,
                so a square 1024x1024 on a portrait/landscape source distorts it).
            height: Image height in pixels. See width — keep the source's aspect ratio
                for image-conditioned passes.
            steps: Number of inference steps (model-specific default if omitted)
            guidance_scale: Guidance scale (model-specific default if omitted)
            seed: Random seed for reproducibility. Change it to reroll an image
                whose anatomy came out wrong.
            use_trigger_words: Auto-prepend the model's CivitAI trigger words if
                they are not already present in the prompt (recommended).
            avoid_anatomy_errors: Merge curated anatomy terms (bad hands, extra
                fingers, deformed, ...) into the negative prompt (recommended).
            avoid_cartoon: Merge anti-cartoon negatives (cartoon, anime, 3d render,
                illustration, plastic skin, ...) for realism work. None = auto: on
                when the checkpoint is realism-oriented AND uses negatives; True/False
                forces it. No effect on models that ignore negatives (FLUX/Klein/turbo)
                — steer those toward realism with positive phrasing instead.
            input_image: The init/source image for img2img (refine/restyle) or an edit
                model (kontext / qwen-image-edit — put the change as an instruction in
                `prompt`). Accepts a local file path, an http(s) URL, a data: URI, or
                base64 bytes — so a user-provided image works directly. Only for models
                whose model_guide image_ops include img2img or edit.
            control_image: The SOURCE image for a ControlNet/upscaler model — canny
                (edge-locked restyle), depth (layout-locked restyle), or the upscaler
                (low-res source). Same accepted forms as input_image (path/URL/data
                URI/base64). Pass a NORMAL photo; the edge/depth map is extracted for you
                on MLX. Only for models with an upscale/canny/depth op. TARGET STYLE goes
                in `prompt`.
            from_last: Reuse the previous generation instead of a path — "init" feeds
                it as `input_image` (img2img/edit), "control" as `control_image`
                (restyle/upscale). Great for draft -> refine -> restyle/upscale chains.
            strength: img2img denoise strength with an init image (low ~0.2-0.4 refines,
                high ~0.5-0.7 restyles). Ignored by the depth model (which generates
                fresh from the depth map).
            controlnet_conditioning_scale: How tightly a canny/upscaler control_image is
                held (~0.5-0.9; lower = looser structure, more style freedom).
        """
        negative_prompt = _merge_negatives(negative_prompt, avoid_anatomy_errors)
        # Decide whether we must (re)load. get_current_model() returns the
        # *persisted* current-model name, which survives an unload/restart even
        # when nothing is resident. So "name matches current" is NOT sufficient
        # to skip loading — we must also confirm something is actually in memory.
        # Otherwise generate_image(model='x') after a respawn wrongly skips the
        # load and then fails with "No model loaded".
        want = model or model_manager.get_current_model()

        # Resolved image inputs for img2img/edit (init_img) and control/upscale (ctrl_img).
        init_img = None
        ctrl_img = None

        # --- model_guide gate -------------------------------------------------
        # Quantized local models need the right settings. Require the agent to
        # consult model_guide(<model>) before generating, and force the model's
        # recommended settings on the FIRST generation. Later generations are
        # free (the anatomy-fix reroll loop tunes seed/guidance/steps itself).
        if want:
            _mt, _params = _model_type_and_params(want)
            _rec = _model_guide.recommended_settings(want, _mt, _params)

            # Anti-cartoon negatives for realism work. Auto when the checkpoint reads as
            # realism-oriented AND the model actually responds to negatives; skipped for
            # models that ignore negatives (FLUX/Klein/turbo — steer those positively).
            _fam = _model_guide.resolve_family(want, _mt, _params)
            _uses_neg = _model_guide.negatives_mode(_fam) != "ignored"
            _want_cartoon = avoid_cartoon if avoid_cartoon is not None else \
                (_uses_neg and _model_guide.is_realism(want, _params))
            if _want_cartoon and _uses_neg:
                terms = _model_guide.ANTI_CARTOON_NEGATIVE
                if _fam == "pony":
                    terms += ", " + _model_guide.PONY_SCORE_NEGATIVE
                negative_prompt = _append_negatives(negative_prompt, terms)

            if want not in _briefed:
                raise ValueError(
                    f"Blocked: call model_guide('{want}') before generating with it — "
                    f"these are quantized local models that need specific settings.\n\n"
                    f"{_model_guide.format_full(want, _mt, _params)}\n\n"
                    f"Then retry generate_image using the recommended settings above."
                )

            # Resolve image inputs (from_last chains the previous result; paths are loaded).
            if from_last is not None and from_last not in ("init", "control"):
                raise ValueError("from_last must be 'init' or 'control'.")
            if from_last:
                last = _last_image.get("image")
                # Fall back to the on-disk copy if the in-memory result was dropped.
                if last is None and _last_image.get("path") and os.path.isfile(_last_image["path"]):
                    last = PILImage.open(_last_image["path"]).convert("RGB")
                if last is None:
                    raise ValueError(
                        "from_last was set but there is no previous image to chain. "
                        "Generate one first, or pass input_image/control_image as a path "
                        "(the saved path is in the previous generate_image response)."
                    )
            if input_image:
                init_img = _load_image_path(input_image, "input_image")
            elif from_last == "init":
                init_img = last
            if control_image:
                ctrl_img = _load_image_path(control_image, "control_image")
            elif from_last == "control":
                ctrl_img = last

            # Validate the requested op against the model's capabilities.
            _ops = _model_guide.image_ops(want, _mt, _params)
            if init_img is not None and not _model_guide.accepts_init_image(_ops):
                raise ValueError(
                    f"'{want}' does not support an init/edit image (its image ops are "
                    f"{_ops}). Use a model with img2img or edit, then retry — see "
                    f"model_guide('{want}')."
                )
            if ctrl_img is not None and not _model_guide.accepts_control_image(_ops):
                raise ValueError(
                    f"'{want}' does not take a control_image (its image ops are {_ops}). "
                    f"For upscaling use flux.1-controlnet-upscaler-mlx; for edges/depth use "
                    f"the canny/depth models — see model_guide('{want}')."
                )
            image_conditioned = init_img is not None or ctrl_img is not None

            # Force recommended settings on the FIRST pure-txt2img generation. Image-
            # conditioned passes (img2img/edit/upscale) are exempt from the hard check —
            # strength/edit/control change the optimal step count — but omitted params
            # still default to the recommended values.
            if want not in _generated:
                _rs, _rg = _rec.get("steps"), _rec.get("guidance_scale")
                if _rs is not None:
                    if steps is None:
                        steps = _rs
                    elif steps != _rs and not image_conditioned:
                        raise ValueError(
                            f"First generation with '{want}' must use the recommended "
                            f"steps={_rs} (you passed steps={steps}). Recommended: "
                            f"steps={_rs}, guidance_scale={_rg}. Tune on a later attempt."
                        )
                if _rg is not None:
                    if guidance_scale is None:
                        guidance_scale = _rg
                    elif abs(float(guidance_scale) - float(_rg)) > 1e-6 and not image_conditioned:
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
            # Loading a multi-GB model is slow and silent. Emit a keepalive heartbeat so
            # the client's request timeout keeps resetting (and the user sees progress)
            # until the denoise loop takes over.
            load_fut = asyncio.ensure_future(
                asyncio.to_thread(model_manager.load_model, want))
            elapsed = 0
            while True:
                done, _ = await asyncio.wait({load_fut}, timeout=3)
                if load_fut in done:
                    break
                elapsed += 3
                if ctx is not None:
                    try:
                        await ctx.report_progress(
                            progress=elapsed, total=None,
                            message=f"Loading {want}… {elapsed}s")
                    except Exception:
                        pass
            success = load_fut.result()
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

        gen_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            seed=seed,
            progress_callback=progress_callback,
        )
        # Image conditioning: init image → img2img/edit (+strength); control image →
        # ControlNet/upscaler. The engine forwards these to the active strategy.
        if init_img is not None:
            gen_kwargs["image"] = init_img
            gen_kwargs["strength"] = strength
        if ctrl_img is not None:
            # Route the control source per the model's control_spec: FLUX depth wants it on
            # `image` (mflux computes depth), canny/upscaler on `control_image` with the
            # strength under mflux's `controlnet_strength`. Falls back to plain control_image
            # for SD ControlNets (unchanged).
            spec = _model_guide.control_spec(want, _mt, _params) if want else None
            if spec:
                gen_kwargs[spec["source_kwarg"]] = ctrl_img
                if spec["strength_kwarg"]:
                    gen_kwargs[spec["strength_kwarg"]] = controlnet_conditioning_scale
                # Depth routes the source to `image`, but mflux's depth model ALWAYS starts
                # from full noise and uses the image only for the depth map — it never does
                # img2img. A positive image_strength there truncates the denoise schedule on
                # pure noise and yields static, so force it off (full schedule).
                if spec["source_kwarg"] == "image":
                    gen_kwargs["strength"] = None
            else:
                gen_kwargs["control_image"] = ctrl_img
                gen_kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale

        result = await asyncio.to_thread(engine.generate_image, **gen_kwargs)

        # First generation for this model succeeded — subsequent calls may tune freely.
        if want:
            _generated.add(want)
        # Persist to disk so this result can be reused by path, and remember it for from_last=.
        saved_path = await asyncio.to_thread(_save_output, result, want)
        _last_image["image"] = result
        _last_image["model"] = want
        _last_image["path"] = saved_path

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        note = (f"Saved to: {saved_path}\n"
                f"Reuse it as a next pass with from_last='init' (img2img/edit) or "
                f"from_last='control' (restyle/upscale), or pass this path as "
                f"input_image/control_image.")
        return [Image(data=buf.getvalue(), format="png"), note]

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
    async def recommend_model(need: Optional[str] = None) -> str:
        """Rank models by quality for a given capability — pick the BEST one for the job.

        Capability comes first (a model must be able to do what you need), then the quality
        score breaks ties between models that can. Installed models are ranked ahead of
        not-yet-installed ones at equal quality.

        Args:
            need: What you need the model to do. One of: 'txt2img', 'img2img', 'edit',
                'upscale', 'controlnet' (or 'canny'/'depth'/'restyle'), 'realism', or free
                text. Omit to see the overall quality board across all models.
        """
        q = (need or "").lower().strip()

        def matches(ops, realism, good_for, family):
            if not q:
                return True
            if q in ("img2img", "txt2img", "edit", "upscale", "canny", "depth"):
                return q in ops
            if q in ("controlnet", "control", "restyle", "style", "structure"):
                return any(o in ops for o in ("canny", "depth", "control"))
            if q in ("realism", "photoreal", "photorealistic", "photo", "realistic"):
                return realism
            return q in good_for.lower() or q in family

        installed = set(settings.models.keys())
        available = set(model_registry.get_available_models().keys())
        rows = []
        for name in installed | available:
            mt, params = _model_type_and_params(name)
            g = _model_guide.guide_for(name, mt, params)
            realism = bool(g.get("realism")) or _model_guide.is_realism(name, params)
            if not matches(g["image_ops"], realism, g["good_for"], g["family"]):
                continue
            rows.append((g["quality"], name in installed, name, g))
        if not rows:
            return (f"No models match '{need}'. Try 'img2img', 'controlnet', 'upscale', "
                    "'edit', or 'realism', or call model_guide() to browse everything.")
        # Highest quality first; installed ahead of not-installed at equal quality.
        rows.sort(key=lambda r: (-r[0], not r[1], r[2]))
        header = f"Best models for '{need}' (quality-ranked):" if need else \
            "Models by quality (highest first):"
        lines = [header]
        for i, (quality, is_inst, name, g) in enumerate(rows[:20], 1):
            tag = "installed" if is_inst else "not installed — pull via CLI"
            lines.append(f"  {i}. {name} [{g['family']}] q{quality}/10 "
                         f"({'/'.join(g['image_ops'])}; {tag}) — {g['good_for']}")
        lines.append("\nThen call model_guide('<name>') before generating with your pick.")
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
        loaded_arch = _checkpoint_arch(model_manager.get_current_model()) \
            if model_manager.is_model_loaded() else None
        lines = [f"CivitAI results for '{query}':"]
        for r in rows:
            nsfw_tag = " [NSFW]" if r.get("nsfw") else ""
            # Flag compatibility for LoRA-type results against the loaded checkpoint.
            compat = ""
            if loaded_arch and (r.get("type") or "").lower() in ("lora", "locon", "lycoris"):
                compat = _compat_note(r.get("base_model"), loaded_arch)
            lines.append(
                f"  - version {r.get('version_id')}: {r.get('name')} "
                f"({r.get('type')}, {r.get('base_model') or '?'}, "
                f"{r.get('download_count', 0)} downloads){nsfw_tag}{compat}"
            )
        if loaded_arch:
            lines.append(f"\n(✗ = incompatible with the loaded {loaded_arch} checkpoint — many "
                         "top 'photorealism' LoRAs are FLUX/SDXL and won't apply to it.)")
        lines.append("Download one with download_civitai_model(url_or_id=<version id>).")
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
        loaded_arch = _checkpoint_arch(model_manager.get_current_model()) \
            if model_manager.is_model_loaded() else None
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
            compat = _compat_note(bm, loaded_arch)
            lines.append(f"  - {name}{tags} ({bm or 'unknown base'}){compat}{trig}")
        if shown == 0:
            return f"No LoRAs match base model '{base_model}'."
        if loaded_arch:
            lines.append(f"\n(Compatibility shown vs the loaded {loaded_arch} checkpoint.)")
        lines.append("Apply one to the loaded model with apply_lora('<name>').")
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
        loaded_arch = _checkpoint_arch(model_manager.get_current_model()) \
            if model_manager.is_model_loaded() else None
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
            compat = _compat_note(bm, loaded_arch)
            lines.append(f"  - {name}{tag} ({bm or 'unknown base'}){compat}{trig}")
        if len(rows) > limit:
            lines.append(f"  ... and {len(rows) - limit} more (narrow with query or base_model).")
        if loaded_arch:
            lines.append(f"\n(Compatibility shown vs the loaded {loaded_arch} checkpoint.)")
        lines.append("Apply one to the loaded model with apply_lora('<name>').")
        return "\n".join(lines)

    @mcp_server.tool()
    async def apply_lora(name: str, scale: Optional[float] = None) -> str:
        """Apply an installed LoRA to the currently-loaded model.

        Its trigger words are then auto-injected into generate_image prompts.
        Remember the order of operations: the prompt/negative tags are the dominant
        quality lever; a LoRA is secondary polish — apply it at a MODERATE weight and
        A/B it against a fixed seed (see the instructions) rather than trusting 1.0.

        Args:
            name: LoRA name (see list_loras).
            scale: LoRA strength. Omit to use the LoRA's suggested weight if known, else
                ~0.7. Style/realism LoRAs usually want 0.5-0.8 (1.0 tends to over-process
                and add artifacts); character/concept LoRAs can go higher.
        """
        from ..core.utils.lora_manager import lora_manager

        if not model_manager.is_model_loaded():
            return "No model loaded. Load a model first with load_model(...)."
        # Resolve tolerant of spaces/dashes/case (agents often pass the display name).
        resolved = lora_manager.resolve_lora_name(name)
        if resolved is None:
            return (f"No LoRA matches '{name}'. Use list_loras() or find_loras('{name}') "
                    "to see the exact registered names.")
        info = lora_manager.get_lora_info(resolved) or {}

        # Base-model compatibility: block cross-arch (e.g. FLUX LoRA on SDXL) — it silently
        # fails / corrupts output. Skip when the LoRA's base is unknown (HF LoRAs).
        loaded = model_manager.get_current_model()
        loaded_arch = _checkpoint_arch(loaded)
        lora_base = info.get("base_model")
        lora_a = _lora_arch(lora_base)
        if loaded_arch and lora_a and loaded_arch != lora_a:
            return (f"Incompatible: LoRA '{resolved}' is for {lora_base} ({lora_a}), but the "
                    f"loaded checkpoint '{loaded}' is {loaded_arch}. It won't apply correctly. "
                    f"Find a {loaded_arch} LoRA with find_loras(...) / list_loras().")

        # Resolve weight: explicit > per-LoRA suggested > moderate default.
        suggested = info.get("suggested_weight")
        used_scale = scale if scale is not None else (suggested if suggested else 0.7)

        ok = await asyncio.to_thread(lora_manager.load_lora, resolved, used_scale)
        if not ok:
            return (f"Found LoRA '{resolved}' but it failed to load onto the current model "
                    "— it may be for a different base model (check list_loras / get_model_details).")
        tw = info.get("trained_words") or []
        hint = f" Trigger words to add to the prompt: {', '.join(tw)}." if tw else ""
        if scale is None:
            why = (f"its suggested weight" if suggested else "a moderate default")
            note = (f" (used {why}; style/realism LoRAs usually 0.5-0.8, character/concept "
                    f"can go higher — A/B it at a fixed seed).")
        elif scale >= 0.9:
            note = (" (high weight — if the image looks over-processed/artifacted, drop to "
                    "~0.6-0.8 and A/B at the same seed).")
        else:
            note = ""
        return f"LoRA '{resolved}' applied at scale {used_scale}.{note}{hint}"

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
