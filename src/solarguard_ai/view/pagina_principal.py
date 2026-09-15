"""Componentes de la pagina principal de SolarGuard AI."""

from __future__ import annotations

import streamlit as st

from solarguard_ai.ingesta import LoadedImage

_SUPPORTED_TYPES = ["jpg", "jpeg", "png", "tif", "tiff"]


def render_header() -> None:
    """Muestra el titulo y la descripcion de SolarGuard AI."""
    st.title("SolarGuard AI")
    st.markdown(
        "Sistema de monitoreo inteligente de paneles solares. "
        "Suba una imagen de un panel para detectar posibles fallas "
        "y priorizar acciones de mantenimiento."
    )


def upload_image():
    """Muestra el widget de carga de imagen y retorna el archivo subido."""
    return st.file_uploader(
        "Seleccione una imagen del panel solar",
        type=_SUPPORTED_TYPES,
        accept_multiple_files=False,
        help="Formatos admitidos: JPEG, PNG y TIFF.",
    )


def render_image_preview(loaded: LoadedImage) -> None:
    """Muestra la imagen cargada con su nombre como referencia."""
    st.image(loaded.image, caption=loaded.source, use_container_width=True)


def render_image_metadata(loaded: LoadedImage) -> None:
    """Muestra la informacion basica de la imagen validada."""
    st.subheader("Informacion de la imagen")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Formato:** {loaded.format}")
        st.markdown(f"**Ancho:** {loaded.width} px")
        st.markdown(f"**Alto:** {loaded.height} px")
    with col2:
        st.markdown(f"**Modo original:** {loaded.original_mode}")
        st.markdown(f"**Canales originales:** {loaded.original_channels}")
