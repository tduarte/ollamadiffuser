"""Tests for MCP server tools."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage


def _mcp_available():
    try:
        import mcp  # noqa: F401

        return True
    except ImportError:
        return False


MCP_SKIP = pytest.mark.skipif(not _mcp_available(), reason="mcp package not installed")


@pytest.fixture
def mock_model_manager():
    mm = MagicMock()
    mm.is_model_loaded.return_value = False
    mm.get_current_model.return_value = None
    mm.list_available_models.return_value = [
        "flux.1-schnell",
        "dreamshaper",
        "stable-diffusion-1.5",
    ]
    mm.list_installed_models.return_value = ["dreamshaper"]
    mm.is_model_installed.return_value = False
    return mm


@MCP_SKIP
class TestImageLoading:
    def test_accepts_data_uri_path_and_errors(self, tmp_path):
        import base64
        import io as _io
        from ollamadiffuser.mcp.server import _load_image_path

        buf = _io.BytesIO()
        PILImage.new("RGB", (8, 8), "red").save(buf, format="PNG")
        raw = base64.b64encode(buf.getvalue()).decode()

        # data: URI
        img = _load_image_path("data:image/png;base64," + raw, "input_image")
        assert img.size == (8, 8)
        # local file path
        p = tmp_path / "x.png"
        PILImage.new("RGB", (8, 8), "blue").save(p)
        assert _load_image_path(str(p), "control_image").size == (8, 8)
        # bad reference -> helpful error mentioning the accepted forms
        with pytest.raises(ValueError, match="local file path.*URL.*data"):
            _load_image_path("/nope/invented.png", "input_image")


@MCP_SKIP
class TestAnatomyNegatives:
    def test_merge_adds_and_dedupes(self):
        from ollamadiffuser.mcp.server import _merge_negatives, _ANATOMY_NEGATIVE

        # empty base -> anatomy terms added
        assert "bad hands" in _merge_negatives("", True)
        # opt-out -> unchanged
        assert _merge_negatives("blurry", False) == "blurry"
        # existing term not duplicated
        merged = _merge_negatives("bad hands, blurry", True)
        assert merged.lower().count("bad hands") == 1
        # user terms preserved
        assert merged.startswith("bad hands, blurry")


@MCP_SKIP
class TestMCPServerCreation:
    def test_create_server(self, mock_model_manager):
        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            assert server is not None
            assert server.name == "OllamaDiffuser"

    def test_server_registers_expected_tools(self, mock_model_manager):
        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            tools = server._tool_manager._tools
            for expected in (
                "generate_image",
                "list_models",
                "load_model",
                "get_status",
                "get_model_details",
                "search_civitai",
                "download_civitai_model",
                "search_huggingface",
                "install_hf_lora",
                "list_loras",
                "find_loras",
                "apply_lora",
                "load_embedding",
                "attach_vae",
                "model_guide",
                "recommend_model",
            ):
                assert expected in tools
            # install_hf_model was removed: base-model checkpoints are CLI-only.
            assert "install_hf_model" not in tools
            assert len(tools) == 16


@MCP_SKIP
class TestListModelsTool:
    def test_list_models_output(self, mock_model_manager):
        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.run(
                server.call_tool("list_models", {})
            )
            text = content[0].text
            assert "dreamshaper" in text
            assert "installed" in text
            assert "flux.1-schnell" in text
            assert "Installed: 1/3" in text


@MCP_SKIP
class TestGetStatusTool:
    def test_status_no_model(self, mock_model_manager):
        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.run(
                server.call_tool("get_status", {})
            )
            text = content[0].text
            assert "Model loaded: no" in text
            assert "Installed models: 1" in text

    def test_status_with_model(self, mock_model_manager):
        mock_model_manager.is_model_loaded.return_value = True
        mock_model_manager.get_current_model.return_value = "dreamshaper"
        mock_engine = MagicMock()
        mock_engine.get_model_info.return_value = {
            "device": "mps",
            "type": "sd15",
        }
        mock_model_manager.loaded_model = mock_engine

        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.run(
                server.call_tool("get_status", {})
            )
            text = content[0].text
            assert "Model loaded: yes" in text
            assert "Current model: dreamshaper" in text
            assert "Device: mps" in text


@MCP_SKIP
class TestLoadModelTool:
    def test_load_not_installed(self, mock_model_manager):
        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.run(
                server.call_tool("load_model", {"model_name": "nonexistent"})
            )
            text = content[0].text
            assert "not installed" in text

    def test_load_success(self, mock_model_manager):
        mock_model_manager.is_model_installed.return_value = True
        mock_model_manager.load_model.return_value = True

        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.run(
                server.call_tool("load_model", {"model_name": "dreamshaper"})
            )
            text = content[0].text
            assert "loaded successfully" in text


@MCP_SKIP
class TestGenerateImageTool:
    def test_generate_no_model_raises(self, mock_model_manager):
        from mcp.server.fastmcp.exceptions import ToolError

        with patch(
            "ollamadiffuser.mcp.server.model_manager", mock_model_manager
        ):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            with pytest.raises(ToolError, match="No model loaded"):
                asyncio.run(
                    server.call_tool("generate_image", {"prompt": "test"})
                )


def _gen_manager(model):
    """A model_manager mock that reports `model` loaded and returns a real PIL image
    from generate_image, so the generate_image tool runs end-to-end."""
    mm = MagicMock()
    mm.get_current_model.return_value = model
    mm.is_model_loaded.return_value = True
    mm.is_model_installed.return_value = True
    mm.loaded_model.generate_image.return_value = PILImage.new("RGB", (8, 8), "blue")
    return mm


class TestModelGuideResolver:
    """model_guide has no MCP dependency — resolver/settings tests run unconditionally."""

    def test_family_and_recommended(self):
        from ollamadiffuser.core.config import model_guide as mg

        # Klein: registry says 28/3.5 but the curated guide overrides to 8/1.0.
        assert mg.resolve_family("flux.2-klein-9b-mlx", "mlx", {}) == "flux2-klein"
        rec = mg.recommended_settings(
            "flux.2-klein-9b-mlx", "mlx",
            {"num_inference_steps": 28, "guidance_scale": 3.5})
        assert rec["steps"] == 8
        assert rec["guidance_scale"] == 1.0

        # Pony detected via CivitAI base_model even though model_type is sdxl.
        assert mg.resolve_family("SomeMix", "sdxl", {"base_model": "Pony"}) == "pony"
        assert mg.guide_for("SomeMix", "sdxl", {"base_model": "Pony"})["prompt_style"] \
            == "pony-score"

        # Falls back to registry params when the family has no curated recommendation.
        rec2 = mg.recommended_settings(
            "stable-diffusion-1.5", "sd15",
            {"num_inference_steps": 50, "guidance_scale": 7.5})
        assert rec2["steps"] == 50 and rec2["guidance_scale"] == 7.5

    def test_image_ops(self):
        from ollamadiffuser.core.config import model_guide as mg

        def ops(name, mt, **p):
            return mg.image_ops(name, mt, p)

        assert ops("flux.2-klein-9b-mlx", "mlx",
                   mlx_variant="flux2", mlx_model_name="klein-9b") == ["txt2img", "img2img"]
        assert ops("qwen-image-edit-mlx", "mlx",
                   mlx_variant="qwen-image", mlx_model_name="qwen-image-edit") == ["edit"]
        assert ops("flux.1-kontext-dev-mlx", "mlx",
                   mlx_variant="flux1-kontext", mlx_model_name="dev") == ["edit"]
        assert ops("flux.1-controlnet-upscaler-mlx", "mlx",
                   mlx_variant="flux1-controlnet", mlx_model_name="upscaler") == ["upscale"]
        assert ops("flux.1-fill-dev-mlx", "mlx",
                   mlx_variant="flux1-fill", mlx_model_name="dev") == ["inpaint"]
        assert ops("stable-diffusion-xl-base", "sdxl") == ["txt2img", "img2img"]

        # init vs control slot helpers
        assert mg.accepts_init_image(["txt2img", "img2img"]) is True
        assert mg.accepts_control_image(["txt2img", "img2img"]) is False
        assert mg.accepts_control_image(["upscale"]) is True
        assert mg.accepts_init_image(["edit"]) is True

    def test_realism_detection(self):
        from ollamadiffuser.core.config import model_guide as mg

        assert mg.is_realism("ponyrealism_v23ultra", {"base_model": "Pony"}) is True
        assert mg.is_realism("cyberrealistic_pony", {"base_model": "Pony"}) is True
        assert mg.is_realism("SomeAnimeMix", {"base_model": "Pony", "tags": ["anime"]}) is False
        # Realism recipe drops source_anime and, for a model that uses negatives, lists them.
        recipe = mg.realism_recipe("pony", {})
        assert "source_pony" in recipe and "cartoon" in recipe

    def test_prompting_and_negatives(self):
        from ollamadiffuser.core.config import model_guide as mg

        assert mg.negatives_mode("pony") == "strong"
        assert mg.negatives_mode("flux2-klein") == "ignored"
        assert mg.negatives_mode("flux-schnell") == "ignored"
        assert mg.negatives_mode("z-image-turbo") == "ignored"
        pony = mg.prompting_info("pony")
        assert pony["supports_weighting"] and pony["supports_break"]
        klein = mg.prompting_info("flux2-klein")
        assert klein["supports_sections"] and not klein["supports_weighting"]
        assert klein["negatives"] == "ignored"

    def test_parse_suggested_weight(self):
        from ollamadiffuser.core.utils.lora_manager import parse_suggested_weight

        assert parse_suggested_weight("recommended weight: 0.7") == 0.7
        assert parse_suggested_weight("use strength 0.6-0.8") == 0.7
        assert parse_suggested_weight("a great detail lora") is None
        assert parse_suggested_weight("trained for 30 steps") is None

    def test_control_spec(self):
        from ollamadiffuser.core.config import model_guide as mg

        canny = mg.control_spec("flux.1-controlnet-canny-mlx", "mlx",
                                {"mlx_variant": "flux1-controlnet", "mlx_model_name": "canny"})
        assert canny == {"kind": "canny", "source_kwarg": "control_image",
                         "strength_kwarg": "controlnet_strength"}
        depth = mg.control_spec("flux.1-depth-dev-mlx", "mlx",
                                {"mlx_variant": "flux1-depth", "mlx_model_name": "dev"})
        assert depth == {"kind": "depth", "source_kwarg": "image", "strength_kwarg": None}
        # A non-control model returns None.
        assert mg.control_spec("flux.2-klein-9b-mlx", "mlx",
                               {"mlx_variant": "flux2", "mlx_model_name": "klein-9b"}) is None


@MCP_SKIP
class TestModelGuideTool:
    def test_catalog_lists_available(self, mock_model_manager):
        with patch("ollamadiffuser.mcp.server.model_manager", mock_model_manager):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.run(server.call_tool("model_guide", {}))
            text = content[0].text
            assert "model_guide('<name>')" in text
            assert "flux.1-schnell" in text  # a known registry model

    def test_named_guide_returns_recommended(self, mock_model_manager):
        with patch("ollamadiffuser.mcp.server.model_manager", mock_model_manager):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.run(
                server.call_tool("model_guide", {"model_name": "flux.1-schnell"}))
            text = content[0].text
            assert "Recommended settings" in text
            assert "flux-schnell" in text


@MCP_SKIP
class TestGenerateGuideGate:
    def test_unbriefed_generate_blocked(self):
        from mcp.server.fastmcp.exceptions import ToolError

        mm = _gen_manager("flux.1-schnell")
        with patch("ollamadiffuser.mcp.server.model_manager", mm):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            with pytest.raises(ToolError) as ei:
                asyncio.run(server.call_tool(
                    "generate_image", {"prompt": "x", "model": "flux.1-schnell"}))
            msg = str(ei.value)
            assert "model_guide" in msg
            assert "steps=4" in msg  # recommended settings inlined in the block error
            mm.loaded_model.generate_image.assert_not_called()

    def test_first_gen_fills_recommended(self):
        mm = _gen_manager("flux.1-schnell")
        with patch("ollamadiffuser.mcp.server.model_manager", mm):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            asyncio.run(server.call_tool("model_guide", {"model_name": "flux.1-schnell"}))
            asyncio.run(server.call_tool(
                "generate_image", {"prompt": "a cat", "model": "flux.1-schnell"}))
            kw = mm.loaded_model.generate_image.call_args.kwargs
            assert kw["num_inference_steps"] == 4
            assert kw["guidance_scale"] == 1.0

    def test_first_gen_rejects_off_recommended(self):
        from mcp.server.fastmcp.exceptions import ToolError

        mm = _gen_manager("flux.1-schnell")
        with patch("ollamadiffuser.mcp.server.model_manager", mm):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            asyncio.run(server.call_tool("model_guide", {"model_name": "flux.1-schnell"}))
            with pytest.raises(ToolError, match="must use the recommended"):
                asyncio.run(server.call_tool(
                    "generate_image",
                    {"prompt": "x", "model": "flux.1-schnell", "steps": 30}))
            mm.loaded_model.generate_image.assert_not_called()

    def test_second_gen_allows_tuning(self):
        mm = _gen_manager("flux.1-schnell")
        with patch("ollamadiffuser.mcp.server.model_manager", mm):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            asyncio.run(server.call_tool("model_guide", {"model_name": "flux.1-schnell"}))
            asyncio.run(server.call_tool(
                "generate_image", {"prompt": "x", "model": "flux.1-schnell"}))  # first
            asyncio.run(server.call_tool(
                "generate_image",
                {"prompt": "x", "model": "flux.1-schnell",
                 "steps": 8, "guidance_scale": 2.0}))  # tune freely
            kw = mm.loaded_model.generate_image.call_args.kwargs
            assert kw["num_inference_steps"] == 8
            assert kw["guidance_scale"] == 2.0


@MCP_SKIP
class TestOutputSaving:
    def test_generation_saved_and_path_returned(self):
        import os
        mm = _gen_manager("flux.1-schnell")
        with patch("ollamadiffuser.mcp.server.model_manager", mm):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            asyncio.run(server.call_tool("model_guide", {"model_name": "flux.1-schnell"}))
            res = asyncio.run(server.call_tool(
                "generate_image", {"prompt": "a cat", "model": "flux.1-schnell"}))
            # Flatten the returned content blocks and collect any text.
            def _texts(r):
                out = []
                for it in (r if isinstance(r, (list, tuple)) else [r]):
                    for x in (it if isinstance(it, (list, tuple)) else [it]):
                        if hasattr(x, "text"):
                            out.append(x.text)
                return out
            texts = [t for t in _texts(res) if "Saved to:" in t]
            assert texts, f"no saved-path text in result: {res!r}"
            saved = texts[0].split("Saved to: ")[1].splitlines()[0]
            assert os.path.isfile(saved)


@MCP_SKIP
class TestImageConditioning:
    def test_img2img_from_last(self):
        mm = _gen_manager("flux.1-schnell")
        with patch("ollamadiffuser.mcp.server.model_manager", mm):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            asyncio.run(server.call_tool("model_guide", {"model_name": "flux.1-schnell"}))
            asyncio.run(server.call_tool(
                "generate_image", {"prompt": "draft", "model": "flux.1-schnell"}))  # txt2img
            asyncio.run(server.call_tool(
                "generate_image",
                {"prompt": "refine", "model": "flux.1-schnell",
                 "from_last": "init", "strength": 0.35}))  # img2img off the last result
            kw = mm.loaded_model.generate_image.call_args.kwargs
            assert kw.get("image") is not None
            assert kw.get("strength") == 0.35

    def test_img2img_first_gen_not_forced(self):
        # An image-conditioned FIRST generation is exempt from the recommended-settings gate.
        mm = _gen_manager("flux.1-schnell")
        with patch("ollamadiffuser.mcp.server.model_manager", mm):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            asyncio.run(server.call_tool("model_guide", {"model_name": "flux.1-schnell"}))
            img = PILImage.new("RGB", (16, 16), "green")
            with patch("ollamadiffuser.mcp.server._load_image_path", return_value=img):
                asyncio.run(server.call_tool(
                    "generate_image",
                    {"prompt": "x", "model": "flux.1-schnell",
                     "input_image": "whatever.png", "steps": 22}))  # off-recommended, allowed
            kw = mm.loaded_model.generate_image.call_args.kwargs
            assert kw["num_inference_steps"] == 22
            assert kw.get("image") is not None

    def test_control_image_rejected_on_non_control_model(self):
        from mcp.server.fastmcp.exceptions import ToolError

        mm = _gen_manager("flux.1-schnell")
        with patch("ollamadiffuser.mcp.server.model_manager", mm):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            asyncio.run(server.call_tool("model_guide", {"model_name": "flux.1-schnell"}))
            img = PILImage.new("RGB", (16, 16), "green")
            with patch("ollamadiffuser.mcp.server._load_image_path", return_value=img):
                with pytest.raises(ToolError, match="does not take a control_image"):
                    asyncio.run(server.call_tool(
                        "generate_image",
                        {"prompt": "x", "model": "flux.1-schnell",
                         "control_image": "ctrl.png"}))
            mm.loaded_model.generate_image.assert_not_called()

    def test_from_last_without_prior_errors(self):
        from mcp.server.fastmcp.exceptions import ToolError

        mm = _gen_manager("flux.1-schnell")
        with patch("ollamadiffuser.mcp.server.model_manager", mm):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            asyncio.run(server.call_tool("model_guide", {"model_name": "flux.1-schnell"}))
            with pytest.raises(ToolError, match="no previous image"):
                asyncio.run(server.call_tool(
                    "generate_image",
                    {"prompt": "x", "model": "flux.1-schnell", "from_last": "init"}))


def _realism_pony_settings():
    from ollamadiffuser.core.config.settings import ModelConfig
    return {"ponyrealism": ModelConfig(
        name="ponyrealism", path="/x", model_type="sdxl",
        parameters={"base_model": "Pony", "num_inference_steps": 28, "guidance_scale": 6.0})}


@MCP_SKIP
class TestRealismNegatives:
    def _run(self, model, settings_models, args):
        import ollamadiffuser.mcp.server as srv
        mm = _gen_manager(model)
        with patch.object(srv, "model_manager", mm), \
                patch.object(srv.settings, "models", settings_models):
            server = srv.create_mcp_server()
            asyncio.run(server.call_tool("model_guide", {"model_name": model}))
            asyncio.run(server.call_tool("generate_image", {**args, "model": model}))
            return mm.loaded_model.generate_image.call_args.kwargs["negative_prompt"]

    def test_auto_merge_on_realism_pony(self):
        neg = self._run("ponyrealism", _realism_pony_settings(), {"prompt": "a woman"})
        assert "cartoon" in neg and "anime" in neg
        assert "score_6" in neg  # pony-only score negatives

    def test_opt_out(self):
        neg = self._run("ponyrealism", _realism_pony_settings(),
                        {"prompt": "a woman", "avoid_cartoon": False})
        assert "cartoon" not in neg

    def test_not_merged_when_negatives_ignored(self):
        # flux.1-schnell ignores negatives → never auto-merge anti-cartoon terms.
        neg = self._run("flux.1-schnell", {}, {"prompt": "a woman"})
        assert "cartoon" not in neg


@MCP_SKIP
class TestApplyLora:
    def _server(self, lora_info, loaded="ponyrealism"):
        import ollamadiffuser.mcp.server as srv
        mm = _gen_manager(loaded)
        lmm = MagicMock()
        lmm.resolve_lora_name.return_value = "thelora"
        lmm.get_lora_info.return_value = lora_info
        lmm.load_lora.return_value = True
        p1 = patch.object(srv, "model_manager", mm)
        p2 = patch.object(srv.settings, "models", _realism_pony_settings())
        p3 = patch("ollamadiffuser.core.utils.lora_manager.lora_manager", lmm)
        return srv, p1, p2, p3, lmm

    def test_incompatible_blocked(self):
        srv, p1, p2, p3, lmm = self._server(
            {"base_model": "FLUX.1", "trained_words": []})
        with p1, p2, p3:
            server = srv.create_mcp_server()
            content, _ = asyncio.run(server.call_tool("apply_lora", {"name": "thelora"}))
            assert "Incompatible" in content[0].text
            lmm.load_lora.assert_not_called()

    def test_default_weight_resolves(self):
        srv, p1, p2, p3, lmm = self._server(
            {"base_model": "Pony", "trained_words": ["zzz"], "suggested_weight": None})
        with p1, p2, p3:
            server = srv.create_mcp_server()
            content, _ = asyncio.run(server.call_tool("apply_lora", {"name": "thelora"}))
            assert "scale 0.7" in content[0].text
            assert lmm.load_lora.call_args.args[1] == 0.7

    def test_suggested_weight_used(self):
        srv, p1, p2, p3, lmm = self._server(
            {"base_model": "Pony", "trained_words": [], "suggested_weight": 0.55})
        with p1, p2, p3:
            server = srv.create_mcp_server()
            content, _ = asyncio.run(server.call_tool("apply_lora", {"name": "thelora"}))
            assert lmm.load_lora.call_args.args[1] == 0.55


def _control_settings(name, variant, mlx_name, guidance):
    from ollamadiffuser.core.config.settings import ModelConfig
    return {name: ModelConfig(
        name=name, path="/x", model_type="mlx",
        parameters={"mlx_variant": variant, "mlx_model_name": mlx_name,
                    "num_inference_steps": 28, "guidance_scale": guidance})}


@MCP_SKIP
class TestControlNetRouting:
    def _run(self, model, settings_models, args):
        import ollamadiffuser.mcp.server as srv
        mm = _gen_manager(model)
        img = PILImage.new("RGB", (16, 16), "green")
        with patch.object(srv, "model_manager", mm), \
                patch.object(srv.settings, "models", settings_models), \
                patch("ollamadiffuser.mcp.server._load_image_path", return_value=img):
            server = srv.create_mcp_server()
            asyncio.run(server.call_tool("model_guide", {"model_name": model}))
            asyncio.run(server.call_tool("generate_image", {**args, "model": model}))
            return mm.loaded_model.generate_image.call_args.kwargs

    def test_canny_routes_to_control_image(self):
        kw = self._run(
            "flux.1-controlnet-canny-mlx",
            _control_settings("flux.1-controlnet-canny-mlx", "flux1-controlnet", "canny", 3.5),
            {"prompt": "oil painting", "control_image": "src.png",
             "controlnet_conditioning_scale": 0.7})
        assert kw.get("control_image") is not None
        assert kw.get("controlnet_strength") == 0.7   # bridged to mflux's name
        assert "image" not in kw

    def test_depth_routes_to_image_without_strength(self):
        # Depth uses the source only for the depth map (mflux always starts from full
        # noise); a positive image_strength corrupts it, so strength must be forced off
        # even when the caller passes one.
        kw = self._run(
            "flux.1-depth-dev-mlx",
            _control_settings("flux.1-depth-dev-mlx", "flux1-depth", "dev", 10.0),
            {"prompt": "cyberpunk", "control_image": "src.png", "strength": 0.3})
        assert kw.get("image") is not None      # depth wants the source on `image`
        assert kw.get("strength") is None       # NOT 0.3 — img2img is off for depth
        assert "control_image" not in kw


@MCP_SKIP
class TestRecommendModel:
    def test_controlnet_ranked(self, mock_model_manager):
        with patch("ollamadiffuser.mcp.server.model_manager", mock_model_manager):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.run(server.call_tool("recommend_model", {"need": "controlnet"}))
            text = content[0].text
            assert "quality-ranked" in text
            # A FLUX control model should appear for a controlnet need.
            assert "controlnet" in text.lower() and "flux" in text.lower()

    def test_img2img_ranked_by_quality(self, mock_model_manager):
        with patch("ollamadiffuser.mcp.server.model_manager", mock_model_manager):
            from ollamadiffuser.mcp.server import create_mcp_server

            server = create_mcp_server()
            content, _ = asyncio.run(server.call_tool("recommend_model", {"need": "img2img"}))
            import re
            quals = [int(m.group(1)) for m in
                     re.finditer(r" q(\d+)/10 ", content[0].text)]
            assert quals  # some models matched
            # Quality is non-increasing down the ranked list.
            assert quals == sorted(quals, reverse=True)
