"""Pruebas del dashboard de sesion de SolarGuard AI (view/dashboard.py)."""

from unittest.mock import patch

from solarguard_ai.view.dashboard import (
    DashboardStats,
    build_class_distribution_chart,
    build_priority_distribution_chart,
    compute_dashboard_stats,
    render_dashboard,
)
from solarguard_ai.view.historial import AnalysisRecord


def _make_record(
    *,
    image_name: str,
    priority: str,
    predicted_class: str = "Dusty",
    requires_human_review: bool = False,
) -> AnalysisRecord:
    return AnalysisRecord(
        analysis_id=f"id-{image_name}-{priority}",
        image_fingerprint=f"fp-{image_name}",
        image_name=image_name,
        predicted_class=predicted_class,
        confidence=0.87,
        priority=priority,
        requires_human_review=requires_human_review,
    )


def test_compute_dashboard_stats_metricas_totales() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="high"),
        _make_record(image_name="b.jpg", priority="medium"),
        _make_record(image_name="c.jpg", priority="low", requires_human_review=True),
    ]

    # Act
    stats = compute_dashboard_stats(records)

    # Assert
    assert stats.total == 3
    assert stats.by_priority == {"high": 1, "medium": 1, "low": 1}
    assert stats.requires_human_review == 1


def test_compute_dashboard_stats_normaliza_prioridades() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="HIGH"),
        _make_record(image_name="b.jpg", priority="high"),
        _make_record(image_name="c.jpg", priority=" Low "),
    ]

    # Act
    stats = compute_dashboard_stats(records)

    # Assert
    assert stats.by_priority["high"] == 2
    assert stats.by_priority["low"] == 1
    assert stats.total == 3


def test_compute_dashboard_stats_categorias_ausentes_con_cero() -> None:
    # Arrange
    records = [_make_record(image_name="a.jpg", priority="low")]

    # Act
    stats = compute_dashboard_stats(records)

    # Assert
    assert stats.by_priority == {"high": 0, "medium": 0, "low": 1}


def test_compute_dashboard_stats_prioridad_inesperada_se_trata_aparte() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="high"),
        _make_record(image_name="b.jpg", priority="urgent"),
    ]

    # Act
    stats = compute_dashboard_stats(records)

    # Assert
    assert stats.by_priority == {"high": 1, "medium": 0, "low": 0, "urgent": 1}
    assert stats.by_priority["urgent"] == 1
    assert stats.requires_human_review == 0


def test_compute_dashboard_stats_distribucion_por_clase() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="low", predicted_class="Clean"),
        _make_record(image_name="b.jpg", priority="medium"),
        _make_record(image_name="c.jpg", priority="medium"),
        _make_record(image_name="d.jpg", priority="low", predicted_class="Bird-drop"),
    ]

    # Act
    stats = compute_dashboard_stats(records)

    # Assert
    assert stats.by_class["Dusty"] == 2
    assert stats.by_class["Clean"] == 1
    assert stats.by_class["Bird-drop"] == 1


def test_compute_dashboard_stats_historial_vacio() -> None:
    # Act
    stats = compute_dashboard_stats([])

    # Assert
    assert stats == DashboardStats(
        total=0,
        requires_human_review=0,
        by_priority={"high": 0, "medium": 0, "low": 0},
        by_class={},
    )


def test_render_dashboard_estado_vacio_no_inventa_metricas() -> None:
    # Arrange
    with patch("solarguard_ai.view.dashboard.st") as mock_st:
        # Act
        render_dashboard([])

    # Assert
    mock_st.subheader.assert_called_once()
    mock_st.info.assert_called_once()
    mock_st.metric.assert_not_called()
    mock_st.plotly_chart.assert_not_called()


def test_render_dashboard_muestra_metricas_y_graficos() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="high", requires_human_review=True),
        _make_record(image_name="b.jpg", priority="low"),
    ]
    with patch("solarguard_ai.view.dashboard.st") as mock_st:
        # Act
        render_dashboard(records)

    # Assert
    mock_st.info.assert_not_called()
    mock_st.columns.assert_called_once_with(5)
    assert mock_st.plotly_chart.call_count == 2
    assert mock_st.warning.call_count == 0


def test_render_dashboard_avisa_prioridad_inesperada() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="high"),
        _make_record(image_name="b.jpg", priority="urgent"),
    ]
    with patch("solarguard_ai.view.dashboard.st") as mock_st:
        # Act
        render_dashboard(records)

    # Assert
    mock_st.warning.assert_called_once()
    warning_message = mock_st.warning.call_args.args[0]
    assert "urgent" in warning_message


def test_build_priority_distribution_chart_incluye_ceros_y_reordena() -> None:
    # Arrange
    by_priority = {"high": 2, "medium": 0, "low": 1, "urgent": 1}

    # Act
    figure = build_priority_distribution_chart(by_priority)

    # Assert
    assert list(figure.data[0].x) == ["Alta", "Media", "Baja", "urgent"]
    assert list(figure.data[0].y) == [2, 0, 1, 1]


def test_build_class_distribution_chart_ordena_descendente() -> None:
    # Arrange
    by_class = {"Clean": 1, "Dusty": 3, "Bird-drop": 1}

    # Act
    figure = build_class_distribution_chart(by_class)

    # Assert
    assert list(figure.data[0].x) == ["Dusty", "Bird-drop", "Clean"]
    assert list(figure.data[0].y) == [3, 1, 1]
