"""Presentacion del diagnostico visual y la prioridad de mantenimiento.

Este modulo solo muestra ``PredictionResult`` y ``PriorityResult`` ya
construidos con widgets de Streamlit. No ejecuta inferencia, no lee
configuracion y no resuelve rutas: esas responsabilidades viven en
``view/modelo.py`` y en el punto de orquestacion (``app.py``).

Los tipos de otros modulos solo se importan bajo ``TYPE_CHECKING`` para
mantener la vista desacoplada de ONNX Runtime en tiempo de ejecucion.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from solarguard_ai.inferencia import PredictionResult
    from solarguard_ai.priorizacion import PrioritizationError, PriorityResult

# Referencia local al nombre de clase no concluyente definido por el
# contrato de inferencia (``UNKNOWN_CLASS``). Se usa el literal para no
# importar el modulo de inferencia (que exige onnxruntime) en la vista.
_UNKNOWN_CLASS = "Unknown"

_DIAGNOSIS_DISCLAIMER = (
    "Hallazgo visual preliminar generado por IA: no constituye un diagnostico "
    "electrico definitivo y esta sujeto a validacion tecnica."
)


def render_diagnosis(prediction: PredictionResult, priority: PriorityResult) -> None:
    """Muestra el diagnostico completo: clasificacion y prioridad."""
    st.subheader("Diagnostico visual preliminar")
    st.markdown(f"_{_DIAGNOSIS_DISCLAIMER}_")
    render_prediction(prediction)
    render_probabilities(prediction.probabilities)
    render_priority(priority)


def render_prediction(prediction: PredictionResult) -> None:
    """Muestra la clase predicha y su nivel de confianza."""
    st.subheader("Resultado de la clasificacion")
    st.markdown(f"**Clase:** {prediction.predicted_class}")
    st.metric(label="Confianza", value=f"{prediction.confidence:.2%}")
    if prediction.predicted_class == _UNKNOWN_CLASS:
        st.warning(
            "Resultado no concluyente (Unknown): el modelo no alcanzo "
            "confianza suficiente. Se requiere revision humana o una "
            "nueva captura; Unknown no significa que el panel este sano."
        )


def render_probabilities(probabilities: dict[str, float]) -> None:
    """Muestra las probabilidades por clase, de mayor a menor."""
    st.subheader("Probabilidades por clase")
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for class_name, probability in ordered:
        st.markdown(f"**{class_name}:** {probability:.2%}")
        st.progress(max(0.0, min(1.0, float(probability))))


def render_priority(priority: PriorityResult) -> None:
    """Muestra la prioridad, la accion recomendada y su justificacion."""
    st.subheader("Prioridad de mantenimiento")
    st.markdown(f"**Prioridad:** {priority.priority}")
    st.markdown(f"**Accion recomendada:** {priority.recommended_action}")
    st.markdown(f"**Razon:** {priority.reason}")
    if priority.requires_human_review:
        st.warning("Se requiere revision humana antes de actuar sobre este resultado.")


def render_model_unavailable(model_path: Path) -> None:
    """Avisa que el modelo no esta disponible sin mostrar diagnostico ficticio."""
    st.warning(
        "Modelo de clasificacion no disponible: no se puede mostrar el "
        f"diagnostico. Ruta esperada: {model_path}. La carga y la vista "
        "previa de la imagen siguen disponibles."
    )


def render_configuration_error(error: PrioritizationError) -> None:
    """Muestra un error de configuracion sin fabricar priorizaciones."""
    st.error(
        "Error de configuracion de priorizacion: no se puede calcular "
        f"la prioridad de mantenimiento. Detalle: {error}"
    )


__all__ = [
    "render_configuration_error",
    "render_diagnosis",
    "render_model_unavailable",
    "render_prediction",
    "render_priority",
    "render_probabilities",
]
