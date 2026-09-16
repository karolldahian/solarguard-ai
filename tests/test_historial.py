"""Pruebas del historial de sesion de SolarGuard AI (view/historial.py)."""

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from solarguard_ai.view.historial import (
    AnalysisRecord,
    add_analysis,
    build_analysis_record,
    compute_image_fingerprint,
    get_history,
    normalize_priority,
    register_analysis,
)


class _FakePrediction:
    def __init__(
        self,
        *,
        predicted_class: str = "Clean",
        confidence: float = 0.95,
    ) -> None:
        self.predicted_class = predicted_class
        self.confidence = confidence


class _FakePriority:
    def __init__(
        self,
        *,
        priority: str = "high",
        requires_human_review: bool = False,
    ) -> None:
        self.priority = priority
        self.requires_human_review = requires_human_review


def _build_record(
    *,
    image_name: str = "panel-a.jpg",
    image_bytes: bytes = b"fake-image-bytes-a",
    predicted_class: str = "Clean",
    confidence: float = 0.95,
    priority: str = "low",
    requires_human_review: bool = False,
) -> AnalysisRecord:
    return build_analysis_record(
        image_name=image_name,
        image_bytes=image_bytes,
        prediction=_FakePrediction(
            predicted_class=predicted_class,
            confidence=confidence,
        ),
        priority=_FakePriority(
            priority=priority,
            requires_human_review=requires_human_review,
        ),
    )


def test_build_analysis_record_preserva_datos_requeridos() -> None:
    # Arrange
    prediction = _FakePrediction(predicted_class="Dusty", confidence=0.82)
    priority = _FakePriority(priority="medium", requires_human_review=True)

    # Act
    record = build_analysis_record(
        image_name="panel-1.jpg",
        image_bytes=b"payload",
        prediction=prediction,
        priority=priority,
    )

    # Assert
    assert record.image_name == "panel-1.jpg"
    assert record.image_fingerprint == compute_image_fingerprint(b"payload")
    assert record.predicted_class == "Dusty"
    assert record.confidence == 0.82
    assert record.priority == "medium"
    assert record.requires_human_review is True
    assert isinstance(record.analysis_id, str)
    assert record.analysis_id


def test_build_analysis_record_normaliza_prioridad() -> None:
    # Arrange/Act
    record = _build_record(priority=" HIGH ")

    # Assert
    assert record.priority == "high"


def test_build_analysis_record_normaliza_prioridad_vacia() -> None:
    # Arrange/Act
    record = _build_record(priority="  ")

    # Assert
    assert record.priority == "unknown"


def test_build_analysis_record_genera_ids_distintos() -> None:
    # Arrange/Act
    first = _build_record(image_bytes=b"bytes-a")
    second = _build_record(image_bytes=b"bytes-b")

    # Assert
    assert first.analysis_id != second.analysis_id


def test_analysis_record_es_inmutable() -> None:
    # Arrange
    record = _build_record()

    # Act/Assert
    with pytest.raises(FrozenInstanceError):
        record.image_name = "otra-imagen.jpg"


def test_register_analysis_agrega_varios_analisis() -> None:
    # Arrange
    history: list[AnalysisRecord] = []
    first = _build_record(image_bytes=b"bytes-a")
    second = _build_record(image_bytes=b"bytes-b")

    # Act
    first_added = register_analysis(history, first)
    second_added = register_analysis(history, second)

    # Assert
    assert first_added is True
    assert second_added is True
    assert history == [first, second]


def test_register_analysis_previene_mismo_nombre_mismos_bytes() -> None:
    # Arrange
    history: list[AnalysisRecord] = []
    first = _build_record(image_name="panel.jpg", image_bytes=b"bytes-a")
    second = _build_record(image_name="panel.jpg", image_bytes=b"bytes-a")

    # Act
    first_added = register_analysis(history, first)
    second_added = register_analysis(history, second)

    # Assert
    assert first_added is True
    assert second_added is False
    assert len(history) == 1


def test_register_analysis_mismo_nombre_bytes_diferentes_si_registra() -> None:
    # Arrange
    history: list[AnalysisRecord] = []
    first = _build_record(image_name="panel.jpg", image_bytes=b"bytes-a")
    second = _build_record(image_name="panel.jpg", image_bytes=b"bytes-b")

    # Act
    first_added = register_analysis(history, first)
    second_added = register_analysis(history, second)

    # Assert
    assert first_added is True
    assert second_added is True
    assert len(history) == 2


def test_register_analysis_nombres_diferentes_mismos_bytes_no_duplica() -> None:
    # Arrange
    history: list[AnalysisRecord] = []
    first = _build_record(image_name="panel-a.jpg", image_bytes=b"bytes-a")
    renamed = _build_record(image_name="panel-b.jpg", image_bytes=b"bytes-a")

    # Act
    first_added = register_analysis(history, first)
    renamed_added = register_analysis(history, renamed)

    # Assert
    assert first_added is True
    assert renamed_added is False
    assert len(history) == 1


def test_compute_image_fingerprint_es_estable_para_mismos_bytes() -> None:
    # Arrange
    payload_1: bytes = b"contenido de imagen"
    payload_2: bytes = b"contenido de imagen"

    # Act/Assert
    assert compute_image_fingerprint(payload_1) == compute_image_fingerprint(payload_2)


def test_compute_image_fingerprint_difiere_con_bytes_distintos() -> None:
    # Arrange/Act
    first = compute_image_fingerprint(b"imagen-a")
    second = compute_image_fingerprint(b"imagen-b")

    # Assert
    assert first != second


def test_normalize_priority_ignora_mayusculas_y_espacios() -> None:
    # Arrange/Act/Assert
    assert normalize_priority("High") == "high"
    assert normalize_priority("high") == "high"
    assert normalize_priority(" HIGH ") == "high"
    assert normalize_priority("urgent") == "urgent"


def test_get_history_devuelve_historial_vacio_y_lo_inicializa() -> None:
    # Arrange
    mock_session_state: dict[str, object] = {}
    with patch("solarguard_ai.view.historial.st") as mock_st:
        mock_st.session_state = mock_session_state

        # Act
        history = get_history()

    # Assert
    assert history == []
    assert mock_session_state["solarguard_analysis_history"] is history


def test_get_history_reutiliza_historial_existente() -> None:
    # Arrange
    existing = [_build_record()]
    mock_session_state: dict[str, object] = {"solarguard_analysis_history": existing}
    with patch("solarguard_ai.view.historial.st") as mock_st:
        mock_st.session_state = mock_session_state

        # Act
        history = get_history()

    # Assert
    assert history is existing


def test_add_analysis_registra_y_evita_duplicados_por_rerun() -> None:
    # Arrange
    mock_st = MagicMock()
    mock_st.session_state = {}
    with patch("solarguard_ai.view.historial.st", mock_st):
        # Act
        first_added = add_analysis(
            _build_record(image_name="panel.jpg", image_bytes=b"bytes-a")
        )
        rerun_same = add_analysis(
            _build_record(image_name="panel.jpg", image_bytes=b"bytes-a")
        )
        history = get_history()

    # Assert
    assert first_added is True
    assert rerun_same is False
    assert len(history) == 1
