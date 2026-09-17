"""Pruebas del mapa de calor analitico de riesgo (view/mapa_riesgo.py)."""

from unittest.mock import patch

from plotly.graph_objects import Figure

from solarguard_ai.view.historial import AnalysisRecord
from solarguard_ai.view.mapa_riesgo import (
    build_risk_heatmap,
    build_risk_matrix,
    render_risk_heatmap,
)


def _make_record(
    *,
    image_name: str,
    priority: str,
    predicted_class: str = "Dusty",
) -> AnalysisRecord:
    return AnalysisRecord(
        analysis_id=f"id-{image_name}-{priority}",
        image_fingerprint=f"fp-{image_name}",
        image_name=image_name,
        predicted_class=predicted_class,
        confidence=0.87,
        priority=priority,
        requires_human_review=False,
    )


def test_build_risk_matrix_historial_vacio() -> None:
    # Arrange/Act
    matrix = build_risk_matrix([])

    # Assert
    assert matrix.conditions == ()
    assert matrix.priorities == ("high", "medium", "low")
    assert matrix.counts == {}
    assert matrix.unexpected_priorities == ()
    assert matrix.count("Dusty", "high") == 0


def test_build_risk_matrix_un_unico_registro() -> None:
    # Arrange
    records = [_make_record(image_name="a.jpg", priority="medium")]

    # Act
    matrix = build_risk_matrix(records)

    # Assert
    assert matrix.conditions == ("Dusty",)
    assert matrix.priorities == ("high", "medium", "low")
    assert matrix.counts == {("Dusty", "medium"): 1}
    assert matrix.count("Dusty", "medium") == 1


def test_build_risk_matrix_conteo_correcto_por_condicion_y_prioridad() -> None:
    # Arrange
    records = [
        _make_record(
            image_name="e1.jpg",
            priority="high",
            predicted_class="Electrical-damage",
        ),
        _make_record(
            image_name="e2.jpg",
            priority="high",
            predicted_class="Electrical-damage",
        ),
        _make_record(
            image_name="p1.jpg",
            priority="high",
            predicted_class="Physical-Damage",
        ),
        _make_record(image_name="d1.jpg", priority="medium", predicted_class="Dusty"),
        _make_record(image_name="d2.jpg", priority="medium", predicted_class="Dusty"),
        _make_record(image_name="d3.jpg", priority="medium", predicted_class="Dusty"),
        _make_record(
            image_name="b1.jpg",
            priority="medium",
            predicted_class="Bird-drop",
        ),
        _make_record(
            image_name="b2.jpg",
            priority="medium",
            predicted_class="Bird-drop",
        ),
        _make_record(
            image_name="s1.jpg",
            priority="medium",
            predicted_class="Snow-Covered",
        ),
        _make_record(image_name="c1.jpg", priority="low", predicted_class="Clean"),
        _make_record(image_name="c2.jpg", priority="low", predicted_class="Clean"),
        _make_record(image_name="c3.jpg", priority="low", predicted_class="Clean"),
        _make_record(image_name="c4.jpg", priority="low", predicted_class="Clean"),
    ]

    # Act
    matrix = build_risk_matrix(records)

    # Assert
    assert matrix.count("Electrical-damage", "high") == 2
    assert matrix.count("Physical-Damage", "high") == 1
    assert matrix.count("Dusty", "medium") == 3
    assert matrix.count("Bird-drop", "medium") == 2
    assert matrix.count("Snow-Covered", "medium") == 1
    assert matrix.count("Clean", "low") == 4
    assert matrix.count("Clean", "high") == 0
    assert sum(matrix.counts.values()) == len(records)


def test_build_risk_matrix_varias_condiciones_misma_prioridad() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="medium", predicted_class="Dusty"),
        _make_record(
            image_name="b.jpg",
            priority="medium",
            predicted_class="Bird-drop",
        ),
        _make_record(
            image_name="c.jpg",
            priority="medium",
            predicted_class="Snow-Covered",
        ),
    ]

    # Act
    matrix = build_risk_matrix(records)

    # Assert
    assert matrix.count("Dusty", "medium") == 1
    assert matrix.count("Bird-drop", "medium") == 1
    assert matrix.count("Snow-Covered", "medium") == 1
    assert matrix.count("Dusty", "high") == 0
    assert matrix.count("Dusty", "low") == 0


def test_build_risk_matrix_una_condicion_distintas_prioridades() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="high"),
        _make_record(image_name="b.jpg", priority="low"),
        _make_record(image_name="c.jpg", priority="low"),
    ]

    # Act
    matrix = build_risk_matrix(records)

    # Assert
    assert matrix.count("Dusty", "high") == 1
    assert matrix.count("Dusty", "low") == 2
    assert matrix.count("Dusty", "medium") == 0
    assert matrix.priorities == ("high", "medium", "low")


def test_build_risk_matrix_prioridad_inesperada_se_conserva_aparte() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="high"),
        _make_record(image_name="b.jpg", priority="urgent"),
    ]

    # Act
    matrix = build_risk_matrix(records)

    # Assert
    assert matrix.count("Dusty", "urgent") == 1
    assert matrix.count("Dusty", "high") == 1
    assert matrix.unexpected_priorities == ("urgent",)
    assert matrix.priorities == ("high", "medium", "low", "urgent")


def test_build_risk_matrix_prioridades_inesperadas_ordenadas_y_sin_duplicar() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="critical"),
        _make_record(image_name="b.jpg", priority="urgent"),
        _make_record(image_name="c.jpg", priority="urgent"),
    ]

    # Act
    matrix = build_risk_matrix(records)

    # Assert
    assert matrix.unexpected_priorities == ("critical", "urgent")
    assert matrix.priorities == ("high", "medium", "low", "critical", "urgent")
    assert matrix.count("Dusty", "urgent") == 2
    assert matrix.count("Dusty", "critical") == 1


def test_build_risk_matrix_normaliza_prioridad_consistentemente() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority=" HIGH "),
        _make_record(image_name="b.jpg", priority="high"),
        _make_record(image_name="c.jpg", priority="Medium"),
    ]

    # Act
    matrix = build_risk_matrix(records)

    # Assert
    assert matrix.count("Dusty", "high") == 2
    assert matrix.count("Dusty", "medium") == 1
    assert matrix.unexpected_priorities == ()


def test_build_risk_matrix_solo_condiciones_realmente_analizadas() -> None:
    # Arrange
    records = [
        _make_record(image_name="c1.jpg", priority="low", predicted_class="Clean"),
        _make_record(image_name="c2.jpg", priority="low", predicted_class="Clean"),
        _make_record(image_name="d1.jpg", priority="medium", predicted_class="Dusty"),
    ]

    # Act
    matrix = build_risk_matrix(records)

    # Assert
    assert matrix.conditions == ("Clean", "Dusty")
    assert "Electrical-damage" not in matrix.conditions
    assert "Bird-drop" not in matrix.conditions


def test_build_risk_matrix_condicion_vacia_se_etiqueta_unknown() -> None:
    # Arrange
    records = [_make_record(image_name="a.jpg", priority="medium", predicted_class="")]

    # Act
    matrix = build_risk_matrix(records)

    # Assert
    assert matrix.conditions == ("unknown",)
    assert matrix.counts == {("unknown", "medium"): 1}


def test_build_risk_matrix_no_muta_los_registros_originales() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="high"),
        _make_record(image_name="b.jpg", priority="low"),
    ]
    snapshot = list(records)

    # Act
    build_risk_matrix(records)

    # Assert
    assert records == snapshot
    for record, original in zip(records, snapshot):
        assert record is original


def test_build_risk_heatmap_muestra_conteos_enteros_orientados() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="high"),
        _make_record(image_name="b.jpg", priority="low"),
        _make_record(image_name="c.jpg", priority="low"),
    ]
    matrix = build_risk_matrix(records)

    # Act
    figure = build_risk_heatmap(matrix)

    # Assert
    assert isinstance(figure, Figure)
    assert list(figure.data[0].x) == ["Alta", "Media", "Baja"]
    assert list(figure.data[0].y) == ["Dusty"]
    assert figure.data[0].z.tolist() == [[1, 0, 2]]


def test_build_risk_heatmap_prioridad_inesperada_columna_independiente() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="high"),
        _make_record(image_name="b.jpg", priority="urgent"),
    ]
    matrix = build_risk_matrix(records)

    # Act
    figure = build_risk_heatmap(matrix)

    # Assert
    assert list(figure.data[0].x) == ["Alta", "Media", "Baja", "urgent"]
    assert list(figure.data[0].y) == ["Dusty"]
    assert figure.data[0].z.tolist() == [[1, 0, 0, 1]]


def test_render_risk_heatmap_estado_vacio_no_inventa_datos() -> None:
    # Arrange
    with patch("solarguard_ai.view.mapa_riesgo.st") as mock_st:
        # Act
        render_risk_heatmap([])

    # Assert
    mock_st.subheader.assert_called_once()
    mock_st.caption.assert_called_once()
    mock_st.info.assert_called_once()
    mock_st.plotly_chart.assert_not_called()
    mock_st.warning.assert_not_called()


def test_render_risk_heatmap_muestra_heatmap_con_datos() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="high"),
        _make_record(image_name="b.jpg", priority="low"),
    ]
    with patch("solarguard_ai.view.mapa_riesgo.st") as mock_st:
        # Act
        render_risk_heatmap(records)

    # Assert
    mock_st.info.assert_not_called()
    mock_st.warning.assert_not_called()
    mock_st.plotly_chart.assert_called_once()
    figure = mock_st.plotly_chart.call_args.args[0]
    assert isinstance(figure, Figure)


def test_render_risk_heatmap_adviere_prioridad_inesperada() -> None:
    # Arrange
    records = [
        _make_record(image_name="a.jpg", priority="high"),
        _make_record(image_name="b.jpg", priority="urgent"),
    ]
    with patch("solarguard_ai.view.mapa_riesgo.st") as mock_st:
        # Act
        render_risk_heatmap(records)

    # Assert
    mock_st.warning.assert_called_once()
    warning_message = mock_st.warning.call_args.args[0]
    assert "urgent" in warning_message
    assert "independiente" in warning_message
    mock_st.plotly_chart.assert_called_once()
