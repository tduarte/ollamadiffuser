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
            ):
                assert expected in tools
            # install_hf_model was removed: base-model checkpoints are CLI-only.
            assert "install_hf_model" not in tools
            assert len(tools) == 15


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
