"""Pruebas para la capa de presentacion de SolarGuard AI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from PIL import Image

from solarguard_ai.ingesta import LoadedImage
from solarguard_ai.view.pagina_principal import (
    render_header,
    render_image_metadata,
    render_image_preview,
    render_usage_guide,
    upload_image,
)


def _make_loaded_image(
    *,
    source: str = "panel.jpg",
    fmt: str = "JPEG",
    width: int = 640,
    height: int = 480,
    mode: str = "RGB",
    channels: int = 3,
) -> LoadedImage:
    image = Image.new(mode, (width, height), color=0)
    return LoadedImage(
        source=source,
        image=image,
        format=fmt,
        width=width,
        height=height,
        original_mode=mode,
        original_channels=channels,
    )


@patch("solarguard_ai.view.pagina_principal.st")
def test_render_image_metadata_muestra_campos_esperados(mock_st: MagicMock) -> None:
    mock_st.columns.return_value = (MagicMock(), MagicMock())
    loaded = _make_loaded_image()

    render_image_metadata(loaded)

    mock_st.subheader.assert_called_once_with("Informacion de la imagen")
    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert "**Formato:** JPEG" in markdown_calls
    assert "**Ancho:** 640 px" in markdown_calls
    assert "**Alto:** 480 px" in markdown_calls
    assert "**Modo original:** RGB" in markdown_calls
    assert "**Canales originales:** 3" in markdown_calls


@patch("solarguard_ai.view.pagina_principal.st")
def test_upload_image_retorna_none_cuando_no_archivo(
    mock_st: MagicMock,
) -> None:
    mock_st.file_uploader.return_value = None

    result = upload_image()

    assert result is None
    mock_st.file_uploader.assert_called_once()


@patch("solarguard_ai.view.pagina_principal.st")
def test_render_image_preview_muestra_la_imagen(mock_st: MagicMock) -> None:
    loaded = _make_loaded_image()

    render_image_preview(loaded)

    mock_st.image.assert_called_once_with(
        loaded.image, caption=loaded.source, use_container_width=True
    )


@patch("solarguard_ai.view.pagina_principal.st")
def test_render_header_muestra_titulo(mock_st: MagicMock) -> None:
    render_header()

    mock_st.title.assert_called_once_with("SolarGuard AI")
    mock_st.markdown.assert_called_once()
    texto = mock_st.markdown.call_args.args[0]
    assert "monitoreo" in texto.lower() or "panel" in texto.lower()


@patch("solarguard_ai.view.pagina_principal.st")
def test_render_usage_guide_muestra_tres_pasos(mock_st: MagicMock) -> None:
    render_usage_guide()

    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert len(markdown_calls) == 4
    assert any(call.startswith("1.") for call in markdown_calls)
    assert any(call.startswith("2.") for call in markdown_calls)
    assert any(call.startswith("3.") for call in markdown_calls)
    assert any("panel" in call.lower() for call in markdown_calls)
