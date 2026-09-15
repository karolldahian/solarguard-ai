"""Entrypoint de la interfaz Streamlit de SolarGuard AI."""

import streamlit as st

from solarguard_ai.inferencia import InferenceError
from solarguard_ai.ingesta import ImageIngestionError, load_image
from solarguard_ai.preprocesamiento import preprocess_for_solarscan
from solarguard_ai.priorizacion import PrioritizationError, prioritize_prediction
from solarguard_ai.view.alertas import render_ticket_section
from solarguard_ai.view.diagnostico import (
    render_configuration_error,
    render_diagnosis,
    render_model_unavailable,
)
from solarguard_ai.view.modelo import (
    get_inference_service,
    get_priority_config,
    resolve_model_path,
)
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
    except ImageIngestionError as e:
        st.error(str(e))
    else:
        render_image_preview(loaded)
        render_image_metadata(loaded)

        try:
            service = get_inference_service()
        except InferenceError:
            render_model_unavailable(resolve_model_path())
        else:
            try:
                tensor = preprocess_for_solarscan(loaded)
                prediction = service.predict(tensor)
            except (InferenceError, ValueError) as e:
                st.error(f"No se pudo completar el diagnostico visual: {e}")
            else:
                try:
                    config = get_priority_config()
                except PrioritizationError as e:
                    render_configuration_error(e)
                else:
                    try:
                        priority = prioritize_prediction(
                            prediction,
                            review_threshold=config.review_confidence,
                        )
                    except PrioritizationError as e:
                        st.error(
                            f"No se pudo calcular la prioridad de mantenimiento: {e}"
                        )
                    else:
                        render_diagnosis(prediction, priority)
                        render_ticket_section(
                            priority,
                            prediction.predicted_class,
                            prediction.confidence,
                        )
else:
    st.info("Suba una imagen de un panel solar para comenzar el analisis.")
