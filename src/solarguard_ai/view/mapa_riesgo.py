"""Mapa de calor analitico de riesgo de la sesion actual de SolarGuard AI.

No es un mapa geografico: SolarGuard AI no conoce coordenadas GPS ni la
disposicion fisica de los paneles. Este modulo responde, de forma
puramente analitica, en que combinaciones de condicion detectada y
prioridad se concentran los analisis registrados durante la sesion.

El modulo solo consume los ``AnalysisRecord`` ya construidos por el
historial. No ejecuta inferencia, no aplica reglas de priorizacion y no
fabricar informacion: las filas representan condiciones realmente
analizadas y los conteos provienen exclusivamente de la sesion actual.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure

from solarguard_ai.view.historial import AnalysisRecord, normalize_priority

_KNOWN_PRIORITIES = frozenset({"high", "medium", "low"})
_PRIORITY_ORDER = ("high", "medium", "low")
_PRIORITY_LABELS = {
    "high": "Alta",
    "medium": "Media",
    "low": "Baja",
    "unknown": "Desconocida",
}
_EMPTY_HISTORY_MESSAGE = (
    "Todavia no hay analisis registrados en esta sesion. "
    "Suba la imagen de un panel solar para que el mapa de calor de riesgo "
    "aparezca con datos reales."
)
_DISCLAIMER = (
    "Mapa de calor analitico: resume los analisis de la sesion actual y no "
    "representa la ubicacion geografica ni fisica de los paneles."
)


@dataclass(frozen=True)
class RiskMatrix:
    """Matriz de conteos de riesgo construida desde los registros reales.

    ``conditions`` contiene unicamente las condiciones presentes en la
    sesion. ``priorities`` siempre empieza con ``high``, ``medium`` y
    ``low``; las prioridades inesperadas se agregan al final como
    categorias independientes, nunca reasignadas a las conocidas.
    """

    conditions: tuple[str, ...]
    priorities: tuple[str, ...]
    counts: dict[tuple[str, str], int]
    unexpected_priorities: tuple[str, ...]

    def count(self, condition: str, priority: str) -> int:
        """Devuelve la cantidad de analisis para una combinacion dada."""
        return self.counts.get((condition, priority), 0)


def build_risk_matrix(records: Sequence[AnalysisRecord]) -> RiskMatrix:
    """Construye la matriz de riesgo a partir de los registros de la sesion.

    Cada celda cuenta cuantos analisis de la sesion coinciden con una
    combinacion de condicion detectada y prioridad. No modifica los
    registros recibidos y no fabrica condiciones que nunca fueron
    analizadas.
    """
    counts: dict[tuple[str, str], int] = {}
    conditions: dict[str, int] = {}
    unexpected_seen: set[str] = set()

    for record in records:
        condition = record.predicted_class or "unknown"
        priority = normalize_priority(record.priority)
        conditions[condition] = conditions.get(condition, 0) + 1
        cell_key = (condition, priority)
        counts[cell_key] = counts.get(cell_key, 0) + 1
        if priority not in _KNOWN_PRIORITIES:
            unexpected_seen.add(priority)

    sorted_conditions = tuple(
        sorted(conditions, key=lambda condition: (-conditions[condition], condition))
    )
    unexpected_priorities = tuple(sorted(unexpected_seen))
    priorities = _PRIORITY_ORDER + unexpected_priorities

    return RiskMatrix(
        conditions=sorted_conditions,
        priorities=priorities,
        counts=counts,
        unexpected_priorities=unexpected_priorities,
    )


def build_risk_heatmap(matrix: RiskMatrix) -> Figure:
    """Construye el heatmap Plotly con la matriz de riesgo de la sesion."""
    rows = [
        [matrix.count(condition, priority) for priority in matrix.priorities]
        for condition in matrix.conditions
    ]
    frame = pd.DataFrame(
        rows,
        index=list(matrix.conditions),
        columns=[
            _PRIORITY_LABELS.get(priority, priority) for priority in matrix.priorities
        ],
    )
    return px.imshow(
        frame,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={"x": "Prioridad", "y": "Condicion detectada", "color": "Analisis"},
        title="Mapa de calor analitico de riesgo",
    )


def render_risk_heatmap(records: Sequence[AnalysisRecord]) -> None:
    """Muestra el mapa de calor analitico de riesgo de la sesion.

    Si no hay analisis, muestra un estado vacio amigable sin fabricar
    ninguna celda. Si aparecen prioridades inesperadas, las conserva como
    categoria independiente y advierte de forma explicita al usuario.
    """
    st.subheader("Mapa de calor de riesgo")
    st.caption(_DISCLAIMER)
    if not records:
        st.info(_EMPTY_HISTORY_MESSAGE)
        return

    matrix = build_risk_matrix(records)
    _render_unexpected_priority_warning(matrix)
    st.plotly_chart(build_risk_heatmap(matrix), use_container_width=True)


def _render_unexpected_priority_warning(matrix: RiskMatrix) -> None:
    """Avisa explicitamente cuando el historial contiene prioridades inesperadas."""
    if not matrix.unexpected_priorities:
        return
    counts_by_priority: dict[str, int] = {}
    for (condition, priority), count in matrix.counts.items():
        if priority in matrix.unexpected_priorities:
            counts_by_priority[priority] = counts_by_priority.get(priority, 0) + count
    details = ", ".join(
        f"'{priority}' ({counts_by_priority[priority]})"
        for priority in matrix.unexpected_priorities
    )
    st.warning(
        "Se detectaron analisis con prioridad no reconocida, conservados "
        f"como categoria independiente fuera de Alta/Media/Baja: {details}."
    )


__all__ = [
    "RiskMatrix",
    "build_risk_heatmap",
    "build_risk_matrix",
    "render_risk_heatmap",
]
