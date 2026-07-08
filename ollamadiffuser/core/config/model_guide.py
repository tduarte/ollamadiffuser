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
        "good_for": "SDXL checkpoints that use the Pony score-tag prompting convention "
                    "(Pony, Illustrious, NoobAI and their derivatives). Style depends on the "
                    "specific checkpoint — ranges from anime/illustration to photorealistic "
                    "(e.g. Pony Realism, CyberRealistic Pony). Do NOT assume anime; follow the "
                    "loaded checkpoint's own description/trigger words for its target look.",
        "quality": 7,
        "prompt_style": "pony-score",
        "tags_note": "REQUIRES quality/score tags: prepend 'score_9, score_8_up, score_7_up'. "
                     "Uses Danbooru-style comma-separated tags, NOT full sentences. For a "
                     "realism checkpoint add photo tags (e.g. 'photorealistic, realistic, "
                     "photo') and skip 'source_anime'; for an anime checkpoint use "
                     "'source_anime'/'rating_safe' etc.",
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
    "qwen-edit": {
        "good_for": "instruction-based image editing (Qwen-Image-Edit) — change/add/remove "
                    "elements while keeping the rest; strong at text.",
        "quality": 8,
        "prompt_style": "natural",
        "tags_note": "Pass the source via input_image and DESCRIBE the change as an "
                     "instruction ('replace the sky with a sunset'). Not a strength-based "
                     "refine.",
        "recommended": None,  # registry: 30 / 4
        "tuning": "steps 20-40; guidance 3-5. Edit is driven by the instruction prompt.",
    },
    "flux-upscaler": {
        "good_for": "AI photo upscaling / detail enhancement (FLUX.1 Jasper Upscaler "
                    "ControlNet) — enlarge a low-res image while adding coherent detail.",
        "quality": 8,
        "prompt_style": "natural",
        "tags_note": "Pass the low-res source via control_image and set a LARGER target "
                     "width/height (~1.5-2x). A short prompt describing the subject helps.",
        "recommended": None,  # registry: 28 / 3.5
        "tuning": "steps 20-30; guidance 3-4; target size 1.5-2x the source. "
                  "controlnet_conditioning_scale ~0.6-1.0.",
    },
    "flux-canny": {
        "good_for": "structure-preserving restyle via EDGE control (Canny). PICK CANNY when "
                    "outlines/linework must stay EXACT — architecture, product shots, logos, "
                    "text/signage, comics/line art, or any case where silhouettes must not "
                    "drift. Tighter lock than depth, so LESS freedom for dramatic style changes.",
        "quality": 8,
        "prompt_style": "natural",
        "tags_note": "Pass the SOURCE photo via control_image (or from_last='control'); the "
                     "prompt = TARGET STYLE. On MLX the edge map is extracted for you — pass a "
                     "normal image, not a pre-made map. Natural language, no negatives. "
                     "NOTE: this model is only the ~3GB ControlNet adapter — it also needs the "
                     "FLUX.1-dev base (~20GB) present; first load is heavy.",
        "recommended": None,  # registry: 28 steps, guidance 3.5
        "tuning": "steps 20-30. controlnet_conditioning_scale ~0.5-0.9: HIGHER hugs the edges "
                  "tighter (safer structure, less restyle), LOWER frees the style. If the result "
                  "ignores your style, lower it; if it loses the composition, raise it.",
    },
    "flux-depth": {
        "good_for": "structure-preserving restyle via DEPTH control. PICK DEPTH for BIGGER style "
                    "changes — photo->painting, day->night, real->3D/anime — where you want to "
                    "keep the 3D LAYOUT and spatial arrangement but are happy to change materials, "
                    "textures, lighting and fine detail. Looser than canny; the go-to for most "
                    "'change the whole style, keep the scene' restyles.",
        "quality": 8,
        "prompt_style": "natural",
        "tags_note": "Pass the SOURCE photo via control_image (or from_last='control'); the "
                     "prompt = TARGET STYLE. On MLX the depth map is computed for you — pass a "
                     "normal image. Natural language, no negatives. Self-contained full model "
                     "(no separate base). Do NOT set `strength` — depth generates a FRESH image "
                     "from the depth map (there is no source-pixel blend), and a strength value "
                     "corrupts it into noise. Keep the source's aspect ratio in width/height.",
        "recommended": None,  # registry: 28 steps, guidance 10
        "tuning": "steps 20-30; guidance ~10 (registry default). The layout is locked by the "
                  "depth map — steer the LOOK entirely through the prompt (style, materials, "
                  "lighting). No source-bleed / strength knob here.",
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

# --- Realism recipe -------------------------------------------------------------------
# Prompt/negative tags are the DOMINANT realism lever for SDXL/Pony checkpoints; LoRAs and
# guidance are secondary. These are the concrete tag sets the guide hands the agent.
REALISM_POSITIVE_TAGS = ("photo, realistic, RAW photo, film grain, detailed skin, "
                         "natural lighting")
ANTI_CARTOON_NEGATIVE = ("cartoon, anime, 3d render, cgi, illustration, painting, "
                         "plastic skin, smooth skin")
PONY_SCORE_NEGATIVE = "score_6, score_5, score_4"  # Pony-only: down-weights low-quality tiers
_REALISM_POS_TOKENS = ("realism", "realistic", "photoreal", "photograph", "photo", "cinematic")
_ANIME_TOKENS = ("anime", "cartoon", "toon", "manga", "hentai", "waifu", "illustration")


def is_realism(model_name: str, params: Optional[Dict[str, Any]] = None) -> bool:
    """Whether an SDXL/Pony checkpoint is realism-oriented (vs anime/illustration).

    Scans the always-present signals (name, base_model, trained_words, description, tags).
    The name is the strongest signal (e.g. 'ponyrealism', 'cyberrealistic'); tags/description
    are best-effort (absent for locally-imported checkpoints).
    """
    p = _norm_params(params)
    name = (model_name or "").lower()
    hay = " ".join([
        name,
        str(p.get("description") or ""),
        " ".join(p.get("trained_words") or []),
        " ".join(p.get("tags") or []),
    ]).lower()
    pos = any(t in name for t in _REALISM_POS_TOKENS) or any(t in hay for t in _REALISM_POS_TOKENS)
    anime = any(t in name for t in _ANIME_TOKENS)
    # Name signals win ties; otherwise realism only when a positive token is present.
    if anime and not any(t in name for t in _REALISM_POS_TOKENS):
        return False
    return pos


# --- Per-model prompting mechanics ----------------------------------------------------
# Different text encoders reward different prompt construction. Families map to a prompt
# GROUP; each group carries capability flags + a concrete "how to prompt" recipe. `negatives`
# is per-family (some distilled models in a natural-language group ignore negatives entirely).
_PROMPT_GROUPS = {
    "clip-tags": {
        "format": "comma-separated tags",
        "supports_weighting": True,
        "supports_break": True,
        "supports_sections": False,
        "prompting": "Comma-separated tags, most important FIRST (earlier = more weight). "
                     "Emphasize with (tag:1.2) — keep weights 0.4-1.6 or it burns. Use BREAK "
                     "to split concepts into separate chunks (e.g. 'subject BREAK background'). "
                     "Negatives matter a lot here.",
    },
    "sd35": {
        "format": "natural language or tags",
        "supports_weighting": True,
        "supports_break": True,
        "supports_sections": False,
        "prompting": "Prefer natural descriptive sentences (SD3.5 was trained on them); tags "
                     "also work. Subject first. Keep negatives SHORT (<10 terms, only issues "
                     "you actually see) — they refine, they don't fix base quality.",
    },
    "flux-natural": {
        "format": "natural language",
        "supports_weighting": False,
        "supports_break": False,
        "supports_sections": False,
        "prompting": "Natural-language sentences, NOT tag lists. No (word:1.3) weighting — say "
                     "'with emphasis on X'. Earlier words weigh more; put the subject first, "
                     "then action, style, context, lighting.",
    },
    "flux2": {
        "format": "natural language, structured",
        "supports_weighting": False,
        "supports_break": False,
        "supports_sections": True,
        "prompting": "Natural language, subject -> action -> style -> context -> lighting -> "
                     "technical. You CAN use labeled sections or even a JSON prompt for precise "
                     "control, and a 'camera:' block for lens/framing. No negative prompts — "
                     "describe what you DO want (say 'a photograph', not 'not a cartoon').",
    },
    "qwen": {
        "format": "natural language, structured",
        "supports_weighting": False,
        "supports_break": False,
        "supports_sections": True,
        "prompting": "Structured labeled lines work great: 'Subject: / Pose: / Clothing: / "
                     "Camera: / Environment: / Lighting: / Mood:'. Subject first. For readable "
                     "in-image TEXT, wrap it in double quotes and put each text item on its own "
                     "line with a position. Short (1-3 sentences) or structured beats long.",
    },
    "z-natural": {
        "format": "long natural-language narrative",
        "supports_weighting": False,
        "supports_break": False,
        "supports_sections": False,
        "prompting": "Long, coherent natural-language description (30-120 words), NOT tag soup. "
                     "Layer it: subject/pose/appearance -> environment -> lighting/mood/camera. "
                     "Guidance is 0, so negatives are IGNORED — phrase every constraint "
                     "positively.",
    },
    "generic-natural": {
        "format": "natural language",
        "supports_weighting": False,
        "supports_break": False,
        "supports_sections": False,
        "prompting": "Describe the scene in natural language, subject first. Check the model "
                     "card for tag vs sentence preference and negative-prompt support.",
    },
}

_FAMILY_TO_GROUP = {
    "pony": "clip-tags", "sdxl-base": "clip-tags", "sdxl-turbo": "clip-tags", "sd15": "clip-tags",
    "sd35": "sd35", "sd35-turbo": "sd35",
    "flux-dev": "flux-natural", "flux-schnell": "flux-natural", "flux-kontext": "flux-natural",
    "chroma": "flux-natural", "flux-upscaler": "flux-natural",
    "flux-canny": "flux-natural", "flux-depth": "flux-natural",
    "flux2-klein": "flux2", "flux2-dev": "flux2",
    "qwen": "qwen", "qwen-edit": "qwen",
    "z-image-turbo": "z-natural",
}

# Families whose models ignore negative prompts (guidance-distilled / no-CFG). Overrides the
# group default. Everything else: clip-tags -> strong, sd35/qwen/flux -> weak.
_NEGATIVES_IGNORED = {"flux-schnell", "flux2-klein", "flux2-dev", "z-image-turbo",
                      "sdxl-turbo", "sd35-turbo"}
_NEGATIVES_STRONG = {"pony", "sdxl-base", "sd15"}


def negatives_mode(family: str) -> str:
    """How much a family's models respond to negative prompts: strong | weak | ignored."""
    if family in _NEGATIVES_IGNORED:
        return "ignored"
    if family in _NEGATIVES_STRONG:
        return "strong"
    return "weak"


def prompting_info(family: str) -> Dict[str, Any]:
    """Prompt-construction mechanics + capability flags for a family."""
    group = _PROMPT_GROUPS[_FAMILY_TO_GROUP.get(family, "generic-natural")]
    return {
        "format": group["format"],
        "supports_weighting": group["supports_weighting"],
        "supports_break": group["supports_break"],
        "supports_sections": group["supports_sections"],
        "negatives": negatives_mode(family),
        "prompting": group["prompting"],
    }


def realism_recipe(family: str, params: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Concrete realism tag advice for a realism-oriented SDXL/Pony checkpoint, else None."""
    neg = negatives_mode(family)
    lines = [f"Positive: add {REALISM_POSITIVE_TAGS}."]
    if family == "pony":
        lines.append("Keep 'score_9, score_8_up, score_7_up' first, but drop 'source_anime' "
                     "(it pushes illustration) — prefer 'source_pony' or no source tag.")
    if neg == "ignored":
        lines.append("This model IGNORES negatives — steer away from cartoon by POSITIVE "
                     "phrasing ('a photorealistic photograph, sharp focus'), not a negative list.")
    else:
        neg_terms = ANTI_CARTOON_NEGATIVE + (", " + PONY_SCORE_NEGATIVE if family == "pony" else "")
        lines.append(f"Negative: {neg_terms}.")
    return " ".join(lines)


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
    if "upscal" in name:
        return "flux-upscaler"
    if "canny" in name:
        return "flux-canny"
    if "depth" in name:
        return "flux-depth"
    if "qwen" in name and "edit" in name:
        return "qwen-edit"
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


# How to drive each image-conditioning op from the MCP generate_image tool.
IMAGE_OP_HELP = {
    "txt2img": "text-to-image (no input image).",
    "img2img": "refine/restyle: input_image (or from_last='init') + strength "
               "(~0.2-0.4 subtle refine, 0.5-0.7 restyle).",
    "edit": "instruction edit: input_image (or from_last='init') and DESCRIBE the change "
            "in the prompt; strength is minor.",
    "upscale": "AI upscale: control_image=<low-res> (or from_last='control') and set a "
               "LARGER target width/height (~1.5-2x).",
    "canny": "edge-conditioned: control_image=<canny edge map>.",
    "depth": "depth-conditioned: control_image=<depth map>.",
    "control": "structure-conditioned: control_image=<reference/control map>.",
    "inpaint": "inpaint — not exposed via MCP yet (use CLI/Web UI): needs image + mask.",
    "redux": "image variation — not exposed via MCP yet (use CLI/Web UI): needs reference image(s).",
}

# Which ops accept the img2img/edit "init image" slot vs the ControlNet "control image" slot.
_INIT_OPS = ("img2img", "edit")
_CONTROL_OPS = ("upscale", "canny", "depth", "control")


def image_ops(model_name: str,
              model_type: Optional[str] = None,
              params: Optional[Dict[str, Any]] = None) -> list:
    """Image-conditioning modes a model supports, most-relevant first.

    Decided by the MLX variant/model-name (authoritative for MLX, which is what most local
    installs use), else by diffusers model_type. Values are keys of IMAGE_OP_HELP.
    """
    name = (model_name or "").lower()
    mt = (model_type or "").lower()
    p = _norm_params(params)
    variant = str(p.get("mlx_variant") or "").lower()
    mlx_name = str(p.get("mlx_model_name") or "").lower()

    if variant:
        if variant == "flux1-kontext":
            return ["edit"]
        if variant == "flux1-fill":
            return ["inpaint"]
        if variant == "flux1-redux":
            return ["redux"]
        if variant == "flux1-depth":
            return ["depth"]
        if variant == "flux1-controlnet":
            if "upscal" in mlx_name:
                return ["upscale"]
            if "canny" in mlx_name:
                return ["canny"]
            return ["control"]
        if variant == "qwen-image":
            return ["edit"] if "edit" in mlx_name else ["txt2img", "img2img"]
        # flux1 (dev/schnell), flux2 (klein), z_image → text-to-image + img2img
        return ["txt2img", "img2img"]

    # Diffusers models (non-MLX).
    if mt in ("sdxl", "sd15", "sd3", "flux", "generic"):
        return ["txt2img", "img2img"]
    if mt in ("controlnet_sd15", "controlnet_sdxl"):
        return ["control"]
    # kontext/upscaler/etc. can also arrive as non-mlx names; fall back on name hints.
    if "kontext" in name or ("qwen" in name and "edit" in name):
        return ["edit"]
    if "upscal" in name:
        return ["upscale"]
    if "canny" in name:
        return ["canny"]
    if "depth" in name:
        return ["depth"]
    return ["txt2img"]


def accepts_init_image(ops) -> bool:
    """True if the model takes an img2img/edit source image."""
    return any(o in ops for o in _INIT_OPS)


def accepts_control_image(ops) -> bool:
    """True if the model takes a ControlNet-style control image (upscale/canny/depth)."""
    return any(o in ops for o in _CONTROL_OPS)


def control_spec(model_name: str,
                 model_type: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Routing for a control/ControlNet model, or None if the model isn't one.

    Different FLUX control models take the source image on a DIFFERENT engine kwarg and use a
    DIFFERENT strength kwarg:
      - depth (mflux flux1-depth): source -> ``image`` (depth is computed internally); no
        controlnet strength (guidance carries it).
      - canny / upscaler (mflux flux1-controlnet): source -> ``control_image``; strength ->
        ``controlnet_strength`` (mflux's name; canny extracts edges internally).
      - non-MLX / SD ControlNet: source -> ``control_image``; strength ->
        ``controlnet_conditioning_scale`` (unchanged, so the existing SD path is untouched).
    Returns ``{kind, source_kwarg, strength_kwarg}`` (strength_kwarg may be None).
    """
    ops = image_ops(model_name, model_type, params)
    if not (accepts_control_image(ops) or "depth" in ops):
        return None
    mt = (model_type or "").lower()
    variant = str(_norm_params(params).get("mlx_variant") or "").lower()
    if variant == "flux1-depth" or ("depth" in ops and mt == "mlx"):
        return {"kind": "depth", "source_kwarg": "image", "strength_kwarg": None}
    kind = next((o for o in ("canny", "upscale", "depth", "control") if o in ops), "control")
    strength_kwarg = "controlnet_strength" if mt == "mlx" else "controlnet_conditioning_scale"
    return {"kind": kind, "source_kwarg": "control_image", "strength_kwarg": strength_kwarg}


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
    ops = image_ops(model_name, model_type, params)
    prompting = prompting_info(fam)
    # Realism recipe only for SDXL/Pony-style checkpoints that read as realism-oriented.
    realism = fam in ("pony", "sdxl-base") and is_realism(model_name, params)
    recipe = realism_recipe(fam, params) if realism else None
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
            "image_ops": ops,
            "prompting": prompting,
            "realism": False,
            "recipe": None,
        }
    return {
        "family": fam,
        "good_for": entry["good_for"],
        "quality": entry["quality"],
        "prompt_style": entry["prompt_style"],
        "tags_note": entry["tags_note"],
        "tuning": entry["tuning"],
        "recommended": rec,
        "image_ops": ops,
        "prompting": prompting,
        "realism": realism,
        "recipe": recipe,
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
    ops = g["image_ops"]
    op_lines = [f"    - {op}: {IMAGE_OP_HELP.get(op, op)}" for op in ops]
    pr = g["prompting"]
    caps = []
    caps.append("weighting (word:1.3)" if pr["supports_weighting"] else "no weighting")
    caps.append("BREAK" if pr["supports_break"] else "no BREAK")
    caps.append("labeled sections/JSON" if pr["supports_sections"] else "no sections")
    caps.append(f"negatives: {pr['negatives']}")
    lines = [
        f"{model_name}  [{g['family']}]",
        f"  Quality: {g['quality']}/10",
        f"  Good for: {g['good_for']}",
        f"  Prompt style: {g['prompt_style']} — {g['tags_note']}",
        f"  Prompting ({pr['format']}; {', '.join(caps)}): {pr['prompting']}",
    ]
    if g.get("recipe"):
        lines.append(f"  Realism recipe: {g['recipe']}")
    lines += [
        f"  Recommended settings: {_fmt_settings(g['recommended'])}",
        f"  Tuning room: {g['tuning']}",
        f"  Image ops: {', '.join(ops)}",
        *op_lines,
    ]
    return "\n".join(lines)


def catalog_line(model_name: str,
                 model_type: Optional[str] = None,
                 params: Optional[Dict[str, Any]] = None) -> str:
    """One-line summary for the browse-all catalog."""
    g = guide_for(model_name, model_type, params)
    return (f"  - {model_name} [{g['family']}] q{g['quality']}/10, "
            f"{g['prompt_style']} prompts, ops:{'/'.join(g['image_ops'])} — {g['good_for']}")
