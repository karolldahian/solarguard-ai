"""Etiquetas de presentacion compartidas por la capa de vista.

Centraliza la traduccion a etiquetas legibles de valores de dominio
(p. ej. la prioridad de mantenimiento) para evitar duplicacion entre
diagnostico, batch, dashboard y mapa de riesgo. Este modulo es una funcion
pura de texto: no conoce Streamlit, no cambia contratos del dominio y
una prioridad inesperada nunca se reasigna silenciosamente.
"""

from __future__ import annotations

PRIORITY_LABELS: dict[str, str] = {
    "high": "Alta",
    "medium": "Media",
    "low": "Baja",
    "unknown": "Desconocida",
}


def format_priority_label(priority: object) -> str:
    """Devuelve la etiqueta en espanol de un valor de prioridad.

    ``high``/``medium``/``low`` se traducen a ``Alta``/``Media``/``Baja``.
    Una prioridad inesperada o el valor vacio se conservan tal cual para
    que la vista nunca oculte datos no reconocidos.

    Args:
        priority: valor de prioridad proveniente del dominio.

    Returns:
        Etiqueta legible para el usuario.
    """
    normalized = str(priority).strip().lower()
    return PRIORITY_LABELS.get(normalized, str(priority))


__all__ = ["PRIORITY_LABELS", "format_priority_label"]
