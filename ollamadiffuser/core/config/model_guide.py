"""Curated, agent-facing guidance for the base models ollamadiffuser can run.

The static registry (``model_registry.py``) already knows each model's *recommended*
inference settings (``parameters.num_inference_steps`` / ``guidance_scale``). What it does
NOT capture is the softer knowledge an agent needs to choose and drive a model well: what
it's good for, how high-quality it is, whether it wants plain natural-language prompts vs
Danbooru/Pony-score tags, and how far the settings can be pushed before quality falls apart.

This module adds exactly that, keyed by model *family* (so all ~50 registry entries and any
locally-installed checkpoints resolve to one of a dozen curated guides), and exposes helpers
the MCP server uses to (a) surface the guide to the agent and (b) enforce recommended
settings on the first generation. Curated ``recommended`` settings are authoritative and
override the registry when it is wrong (e.g. FLUX.2 Klein, which the registry defaults to
28 steps / guidance 3.5 but actually wants ~8 / ~1.0); families that omit ``recommended``
fall back to the model's registry ``parameters``.
"""

from typing import Any, Dict, Optional

# prompt_style values: "natural" | "danbooru" | "pony-score" | "mixed"

_FAMILIES: Dict[str, Dict[str, Any]] = {
    "flux2-klein": {
        "good_for": "photorealism and natural-language scenes with strong prompt adherence; "
                    "fast on Apple Silicon (MLX). Decent in-image text.",
        "quality": 8,
        "prompt_style": "natural",
        "tags_note": "Plain natural-language sentences, can be long and descriptive. Do NOT "
                     "use Danbooru or Pony 'score_' tags — they degrade FLUX output.",
        # Registry defaults for the Klein MLX variants are wrong (28/3.5); mflux wants ~8/1.0.
        "recommended": {"steps": 8, "guidance_scale": 1.0, "size": "1024x1024"},
        "tuning": "steps 4-12 (step-distilled — beyond ~12 wastes time for no real gain); "
                  "guidance 1.0-2.0 (Klein is NOT guidance-distilled, so guidance bites hard "
                  "— above ~2.5 it over-saturates and 'fries' the image). Euler + simple.",
    },
    "flux2-dev": {
        "good_for": "highest-quality FLUX.2 base generation; photoreal images and reliable text.",
        "quality": 9,
        "prompt_style": "natural",
        "tags_note": "Natural language. No Danbooru/score tags.",
        "recommended": None,
        "tuning": "steps 20-30; guidance 3-5. Natural language.",
    },
    "flux-dev": {
        "good_for": "photoreal, coherent scenes and solid prompt adherence (FLUX.1-dev).",
        "quality": 8,
        "prompt_style": "natural",
        "tags_note": "Natural language. No Danbooru/score tags.",
        "recommended": None,  # registry: dev 20/3.0, mlx 28/3.5, gguf 28/1.0
        "tuning": "steps 20-28; guidance ~3.0-4.0 (guidance-distilled — keep near 3.5; very "
                  "high burns contrast). Natural language.",
    },
    "flux-schnell": {
        "good_for": "fast 1-4 step drafts and iteration (FLUX.1-schnell).",
        "quality": 6,
        "prompt_style": "natural",
        "tags_note": "Natural language. No Danbooru/score tags.",
        "recommended": None,  # registry: 4 steps / guidance 0
        "tuning": "steps 1-4 (step+guidance distilled — more steps don't help); guidance 0 "
                  "(CFG disabled). Natural language.",
    },
    "flux-kontext": {
        "good_for": "instruction-based image editing (needs an input image).",
        "quality": 8,
        "prompt_style": "natural",
        "tags_note": "Describe the edit in plain language ('change the sky to sunset').",
        "recommended": None,  # registry: 28 / 2.5
        "tuning": "steps 20-30; guidance 2.5-4.0.",
    },
    "sdxl-base": {
        "good_for": "versatile 1024px images across broad styles; huge LoRA ecosystem.",
        "quality": 7,
        "prompt_style": "mixed",
        "tags_note": "Natural language or comma-separated tags both work. No Pony 'score_' "
                     "tags (those are Pony-only).",
        "recommended": None,  # registry: 30-50 / 5-7.5
        "tuning": "steps 25-40; guidance 5-8 (classic CFG; ~7 is a safe middle). Any SDXL sampler.",
    },
    "sdxl-turbo": {
        "good_for": "ultra-fast 1-4 step SDXL (turbo / lightning variants).",
        "quality": 5,
        "prompt_style": "mixed",
        "tags_note": "Short prompts work best.",
        "recommended": None,  # registry: turbo 1/0, lightning 4/0
        "tuning": "steps 1-8 by variant (turbo=1, lightning=4); guidance 0-1 (distilled — CFG "
                  "off). Raising guidance breaks these models.",
    },
    "pony": {
        "good_for": "anime and stylized illustration / character art (Pony, Illustrious, "
                    "NoobAI — all SDXL-based).",
        "quality": 7,
        "prompt_style": "pony-score",
        "tags_note": "REQUIRES quality/score tags: prepend 'score_9, score_8_up, score_7_up' "
                     "(often with 'source_anime', 'rating_safe'). Uses Danbooru-style "
                     "comma-separated tags, NOT full sentences.",
        # Pony checkpoints arrive via CivitAI and may not carry settings; supply sane defaults.
        "recommended": {"steps": 28, "guidance_scale": 6.0, "size": "1024x1024"},
        "tuning": "steps 25-35; guidance 5-7. Comma-separated Danbooru tags with the score "
                  "tags at the very front.",
    },
    "sd15": {
        "good_for": "lightweight 512px base; fast, enormous LoRA ecosystem, lower fidelity.",
        "quality": 5,
        "prompt_style": "mixed",
        "tags_note": "Comma tags or short natural language. No score tags.",
        "recommended": None,  # registry: 30-50 / 7-7.5
        "tuning": "steps 25-40; guidance 6-8. Native ~512px (up to 768); larger needs hi-res fix.",
    },
    "sd35": {
        "good_for": "strong prompt adherence and in-image typography (Stable Diffusion 3.5).",
        "quality": 7,
        "prompt_style": "natural",
        "tags_note": "Natural language. No score tags.",
        "recommended": None,  # registry: large/medium 28 / 3.5-4.5
        "tuning": "steps 28-40; guidance 3.5-5.0. Natural language.",
    },
    "sd35-turbo": {
        "good_for": "4-step distilled Stable Diffusion 3.5 Large.",
        "quality": 6,
        "prompt_style": "natural",
        "tags_note": "Natural language.",
        "recommended": None,  # registry: 4 / 0
        "tuning": "steps 4-8; guidance 0-1 (distilled).",
    },
    "qwen": {
        "good_for": "excellent readable in-image text and bilingual (CN/EN) prompts (Qwen-Image).",
        "quality": 8,
        "prompt_style": "natural",
        "tags_note": "Natural language; unusually good at rendering exact text you quote.",
        "recommended": None,  # registry: 30 / 4
        "tuning": "steps 20-40; guidance 3-5.",
    },
    "chroma": {
        "good_for": "FLUX-based artistic / uncensored generation (Chroma).",
        "quality": 7,
        "prompt_style": "natural",
        "tags_note": "Natural language.",
        "recommended": None,  # registry: 28 / 4
        "tuning": "steps 26-40; guidance 3-5.",
    },
    "z-image-turbo": {
        "good_for": "fast 8-step distilled generation (Z-Image Turbo).",
        "quality": 6,
        "prompt_style": "natural",
        "tags_note": "Natural language.",
        "recommended": None,  # registry: 8 / 4-5
        "tuning": "steps 6-10; guidance 3.5-5.0 (distilled — don't push far).",
    },
}

# Generic fallback used when a model doesn't resolve to a curated family. Keyed loosely by
# model_type so the tuning hint is at least in the right ballpark.
_GENERIC_TUNING = {
    "flux": "steps 20-30; guidance 2-4 (natural language).",
    "sdxl": "steps 25-40; guidance 5-8.",
    "sd15": "steps 25-40; guidance 6-8.",
    "sd3": "steps 28-40; guidance 3.5-5.",
    "mlx": "follow the underlying FLUX/Qwen family; steps 8-28, guidance 1-4.",
    "hidream": "steps 16-30; guidance 2.5-4.",
    "video": "video model — steps 30-50; guidance 3-6; expect long runtimes.",
    "controlnet_sd15": "steps 25-40; guidance 6-8; needs a control image.",
    "controlnet_sdxl": "steps 25-40; guidance 5-8; needs a control image.",
}


def _norm_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return params or {}


def resolve_family(model_name: str,
                   model_type: Optional[str] = None,
                   params: Optional[Dict[str, Any]] = None) -> str:
    """Map a model name (+ type/params) to a curated family id, or '_generic'."""
    name = (model_name or "").lower()
    mt = (model_type or "").lower()
    base = str(_norm_params(params).get("base_model") or "").lower()

    # Pony/Illustrious/NoobAI collapse to SDXL by model_type, so detect them by their
    # CivitAI base_model string or name before falling through to the generic sdxl guide.
    if any(k in base for k in ("pony", "illustrious", "noobai")) or \
       any(k in name for k in ("pony", "illustrious", "noobai")):
        return "pony"

    if "klein" in name:
        return "flux2-klein"
    if "flux.2" in name or "flux2" in name:
        return "flux2-dev"
    if "schnell" in name:
        return "flux-schnell"
    if "kontext" in name:
        return "flux-kontext"
    if "qwen" in name:
        return "qwen"
    if "chroma" in name:
        return "chroma"
    if "z-image" in name or "z_image" in name:
        return "z-image-turbo"
    if "turbo" in name or "lightning" in name:
        if mt == "sd3" or "3.5" in name or "3-5" in name:
            return "sd35-turbo"
        return "sdxl-turbo"
    if "flux" in name or mt == "flux":
        return "flux-dev"  # schnell/klein/kontext already handled above
    if mt == "sdxl":
        return "sdxl-base"
    if mt == "sd15":
        return "sd15"
    if mt == "sd3" or "3.5" in name:
        return "sd35"
    return "_generic"


def recommended_settings(model_name: str,
                         model_type: Optional[str] = None,
                         params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolved {steps, guidance_scale, size}: curated when set, else registry params.

    Values may be None if neither the curated family nor the registry supplies them; the
    caller (the generate gate) skips forcing a param whose recommended value is None.
    """
    fam = resolve_family(model_name, model_type, params)
    entry = _FAMILIES.get(fam) or {}
    rec = entry.get("recommended") or {}
    p = _norm_params(params)
    steps = rec.get("steps", p.get("num_inference_steps"))
    guidance = rec.get("guidance_scale", p.get("guidance_scale"))
    size = rec.get("size", "1024x1024")
    return {"steps": steps, "guidance_scale": guidance, "size": size}


def guide_for(model_name: str,
              model_type: Optional[str] = None,
              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Full merged guide dict for a model."""
    fam = resolve_family(model_name, model_type, params)
    entry = _FAMILIES.get(fam)
    rec = recommended_settings(model_name, model_type, params)
    if entry is None:
        mt = (model_type or "").lower()
        return {
            "family": "generic",
            "good_for": f"{model_type or 'diffusion'} model (no curated guide yet).",
            "quality": 6,
            "prompt_style": "natural",
            "tags_note": "Use natural language unless the model card says otherwise; "
                         "no Pony score tags.",
            "tuning": _GENERIC_TUNING.get(mt,
                      "Start from the recommended settings; adjust steps +/-30% and "
                      "guidance +/-1-2 if results are poor."),
            "recommended": rec,
        }
    return {
        "family": fam,
        "good_for": entry["good_for"],
        "quality": entry["quality"],
        "prompt_style": entry["prompt_style"],
        "tags_note": entry["tags_note"],
        "tuning": entry["tuning"],
        "recommended": rec,
    }


def _fmt_settings(rec: Dict[str, Any]) -> str:
    steps = rec.get("steps")
    g = rec.get("guidance_scale")
    size = rec.get("size")
    parts = []
    parts.append(f"steps={steps}" if steps is not None else "steps=(model default)")
    parts.append(f"guidance_scale={g}" if g is not None else "guidance_scale=(model default)")
    if size:
        parts.append(f"size={size}")
    return ", ".join(parts)


def format_full(model_name: str,
                model_type: Optional[str] = None,
                params: Optional[Dict[str, Any]] = None) -> str:
    """Human-readable detailed guide for one model (used by the tool and the gate error)."""
    g = guide_for(model_name, model_type, params)
    lines = [
        f"{model_name}  [{g['family']}]",
        f"  Quality: {g['quality']}/10",
        f"  Good for: {g['good_for']}",
        f"  Prompt style: {g['prompt_style']} — {g['tags_note']}",
        f"  Recommended settings: {_fmt_settings(g['recommended'])}",
        f"  Tuning room: {g['tuning']}",
    ]
    return "\n".join(lines)


def catalog_line(model_name: str,
                 model_type: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None) -> str:
    """One-line summary for the browse-all catalog."""
    g = guide_for(model_name, model_type, params)
    return (f"  - {model_name} [{g['family']}] q{g['quality']}/10, "
            f"{g['prompt_style']} prompts — {g['good_for']}")
