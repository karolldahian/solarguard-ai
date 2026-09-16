"""Historial de analisis de la sesion actual de SolarGuard AI.

El historial vive unicamente en ``st.session_state`` y no persiste entre
sesiones de Streamlit. Cada registro representa un analisis completado con
exito durante la sesion actual.

La deduplicacion ante los reruns de Streamlit se controla con un
fingerprint SHA-256 del contenido de la imagen, no con el nombre del
archivo ni con el identificador del registro. Dos reruns sobre la misma
imagen producen el mismo fingerprint y, por tanto, no se registran dos
veces el mismo analisis.

Este modulo no ejecuta inferencia, no conoce ONNX Runtime y no aplica
reglas de priorizacion: solo guarda los resultados ya calculados.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import uuid4

import streamlit as st

if TYPE_CHECKING:
    from solarguard_ai.inferencia import PredictionResult
    from solarguard_ai.priorizacion import PriorityResult

_SESSION_HISTORY_KEY = "solarguard_analysis_history"


@dataclass(frozen=True)
class AnalysisRecord:
    """Registro inmutable de un analisis completado en la sesion."""

    analysis_id: str
    image_fingerprint: str
    image_name: str
    predicted_class: str
    confidence: float
    priority: str
    requires_human_review: bool


def compute_image_fingerprint(image_bytes: bytes) -> str:
    """Devuelve un identificador estable del contenido de la imagen (SHA-256).

    Es el criterio de identidad para evitar registros duplicados por los
    reruns de Streamlit: mismo contenido de archivo implica mismo fingerprint.
    """
    return sha256(image_bytes).hexdigest()


def normalize_priority(priority: object) -> str:
    """Normaliza un valor de prioridad a minusculas y sin espacios.

    ``"High"``, ``"high"`` o ``" HIGH "`` se tratan como la misma categoria.
    Un valor vacio se etiqueta como ``"unknown"`` en lugar de descartarlo.
    """
    value = str(priority).strip().lower()
    return value if value else "unknown"


def build_analysis_record(
    *,
    image_name: str,
    image_bytes: bytes,
    prediction: PredictionResult,
    priority: PriorityResult,
) -> AnalysisRecord:
    """Construye un registro a partir de los resultados ya calculados.

    Solo almacena el fingerprint de la imagen, nunca sus bytes completos.
    """
    return AnalysisRecord(
        analysis_id=uuid4().hex,
        image_fingerprint=compute_image_fingerprint(image_bytes),
        image_name=image_name,
        predicted_class=prediction.predicted_class,
        confidence=float(prediction.confidence),
        priority=normalize_priority(priority.priority),
        requires_human_review=bool(priority.requires_human_review),
    )


def register_analysis(
    history: list[AnalysisRecord],
    record: AnalysisRecord,
) -> bool:
    """Agrega un registro si la imagen no fue analizada antes en la sesion.

    Devuelve ``True`` solo cuando el registro se agrega. Si ya existe un
    registro con el mismo fingerprint de imagen, devuelve ``False`` y no
    modifica el historial.
    """
    if any(
        existing.image_fingerprint == record.image_fingerprint for existing in history
    ):
        return False
    history.append(record)
    return True


def get_history() -> list[AnalysisRecord]:
    """Devuelve el historial de la sesion, inicializandolo si no existe."""
    session_history = st.session_state.get(_SESSION_HISTORY_KEY)
    if session_history is None:
        session_history = []
        st.session_state[_SESSION_HISTORY_KEY] = session_history
    return session_history


def add_analysis(record: AnalysisRecord) -> bool:
    """Registra un analisis en el historial de la sesion sin duplicados."""
    return register_analysis(get_history(), record)


__all__ = [
    "AnalysisRecord",
    "add_analysis",
    "build_analysis_record",
    "compute_image_fingerprint",
    "get_history",
    "normalize_priority",
    "register_analysis",
]
