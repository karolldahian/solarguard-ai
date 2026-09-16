"""Dashboard de estadisticas y graficos de la sesion actual de SolarGuard AI.

Este modulo solo recibe el historial de la sesion ya construido y calcula
estadisticas y graficos reales a partir de el. No ejecuta inferencia, no
lee modelos y no conoce ONNX Runtime: es agnostico respecto de la fuente
de inferencia que produzca los ``AnalysisRecord``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.graph_objects import Figure

from solarguard_ai.view.historial import AnalysisRecord, normalize_priority
from solarguard_ai.view.rotulos import PRIORITY_LABELS

_KNOWN_PRIORITIES = frozenset({"high", "medium", "low"})
_PRIORITY_ORDER = ("high", "medium", "low")
_EMPTY_HISTORY_MESSAGE = (
    "Todavia no hay analisis registrados en esta sesion. "
    "Suba la imagen de un panel solar para comenzar."
)


@dataclass(frozen=True)
class DashboardStats:
    """Estadisticas agregadas a partir del historial real de la sesion."""

    total: int
    requires_human_review: int
    by_priority: dict[str, int]
    by_class: dict[str, int]


def compute_dashboard_stats(records: Sequence[AnalysisRecord]) -> DashboardStats:
    """Agrega el historial en metricas y conteos por prioridad y clase.

    Las categorias ``high``, ``medium`` y ``low`` siempre estan presentes
    (aunque tengan cero registros). Una prioridad que no pertenece a ellas
    se mantiene como una categoria separada: nunca se convierte de forma
    silenciosa en una de las conocidas.
    """
    by_priority: dict[str, int] = {priority: 0 for priority in _PRIORITY_ORDER}
    by_class: dict[str, int] = {}
    requires_human_review = 0

    for record in records:
        priority = normalize_priority(record.priority)
        by_priority[priority] = by_priority.get(priority, 0) + 1
        class_name = record.predicted_class or "unknown"
        by_class[class_name] = by_class.get(class_name, 0) + 1
        if record.requires_human_review:
            requires_human_review += 1

    return DashboardStats(
        total=len(records),
        requires_human_review=requires_human_review,
        by_priority=by_priority,
        by_class=by_class,
    )


def build_priority_distribution_chart(by_priority: Mapping[str, int]) -> Figure:
    """Construye el grafico de distribucion por prioridad de la sesion."""
    known_items = [
        (priority, by_priority[priority])
        for priority in _PRIORITY_ORDER
        if priority in by_priority
    ]
    other_items = sorted(
        (priority, count)
        for priority, count in by_priority.items()
        if priority not in _PRIORITY_ORDER
    )
    items = known_items + other_items
    labels = [PRIORITY_LABELS.get(priority, priority) for priority, _ in items]
    counts = [count for _, count in items]
    frame = pd.DataFrame({"Prioridad": labels, "Cantidad": counts})
    return px.bar(
        frame,
        x="Prioridad",
        y="Cantidad",
        title="Distribucion por prioridad",
    )


def build_class_distribution_chart(by_class: Mapping[str, int]) -> Figure:
    """Construye el grafico de distribucion por condicion detectada."""
    items = sorted(by_class.items(), key=lambda item: (-item[1], item[0]))
    frame = pd.DataFrame(
        {
            "Condicion": [label for label, _ in items],
            "Cantidad": [count for _, count in items],
        }
    )
    return px.bar(
        frame,
        x="Condicion",
        y="Cantidad",
        title="Distribucion por condicion detectada",
    )


def render_dashboard(records: Sequence[AnalysisRecord]) -> None:
    """Muestra las estadisticas y graficos de la sesion.

    Si no hay registros, muestra un estado vacio amigable sin inventar
    ninguna metrica ni grafico.
    """
    st.subheader("Dashboard de la sesion")
    if not records:
        st.info(_EMPTY_HISTORY_MESSAGE)
        return

    stats = compute_dashboard_stats(records)
    _render_metrics(stats)
    _render_unexpected_priority_warning(stats.by_priority)
    st.plotly_chart(
        build_priority_distribution_chart(stats.by_priority),
        use_container_width=True,
    )
    st.plotly_chart(
        build_class_distribution_chart(stats.by_class),
        use_container_width=True,
    )


def _render_metrics(stats: DashboardStats) -> None:
    first_row = st.columns(3)
    first_row[0].metric("Total de analisis", stats.total)
    first_row[1].metric("Prioridad Alta", stats.by_priority.get("high", 0))
    first_row[2].metric("Prioridad Media", stats.by_priority.get("medium", 0))
    second_row = st.columns(2)
    second_row[0].metric("Prioridad Baja", stats.by_priority.get("low", 0))
    second_row[1].metric("Requieren revision humana", stats.requires_human_review)


def _render_unexpected_priority_warning(by_priority: Mapping[str, int]) -> None:
    """Avisa explicitamente cuando el historial contiene prioridades inesperadas."""
    unexpected = {
        priority: count
        for priority, count in by_priority.items()
        if priority not in _KNOWN_PRIORITIES
    }
    if unexpected:
        details = ", ".join(
            f"'{priority}' ({count})" for priority, count in unexpected.items()
        )
        st.warning(
            "Se detectaron analisis con prioridad no reconocida, "
            f"no incluidos en Alta/Media/Baja: {details}."
        )


__all__ = [
    "DashboardStats",
    "build_class_distribution_chart",
    "build_priority_distribution_chart",
    "compute_dashboard_stats",
    "render_dashboard",
]
