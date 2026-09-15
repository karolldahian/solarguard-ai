"""Entrypoint de la interfaz Streamlit de SolarGuard AI."""

import streamlit as st

from solarguard_ai.ingesta import ImageIngestionError, load_image
from solarguard_ai.view.pagina_principal import (
    render_header,
    render_image_metadata,
    render_image_preview,
    upload_image,
)

st.set_page_config(page_title="SolarGuard AI", page_icon="☀️", layout="wide")

render_header()

uploaded_file = upload_image()

if uploaded_file is not None:
    try:
        loaded = load_image(uploaded_file)
        render_image_preview(loaded)
        render_image_metadata(loaded)
    except ImageIngestionError as e:
        st.error(str(e))
else:
    st.info("Suba una imagen de un panel solar para comenzar el analisis.")
