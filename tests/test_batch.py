"""Pruebas del analisis por lote de imagenes (view/batch.py).

El nucleo puro (``run_batch_analysis`` y ``build_batch_results_table``) se
prueba sin Streamlit usando fuentes en memoria y un ``analyze_one`` fake. La
capa de render se prueba con ``patch`` de ``st`` siguiendo el patron de los
demas modulos de la vista.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from solarguard_ai.inferencia import InferenceError
from solarguard_ai.priorizacion import PrioritizationError
from solarguard_ai.view.batch import (
    BatchAlertSummary,
    BatchImageResult,
    BatchSummary,
    build_batch_alert_summary,
    build_batch_failure_details,
    build_batch_results_table,
    render_batch_alert_summary,
    render_batch_analysis,
    render_batch_results,
    run_batch_analysis,
)
from solarguard_ai.view.historial import (
    AnalysisRecord,
    build_analysis_record,
    compute_image_fingerprint,
)

_TABLE_COLUMNS = [
    "Imagen",
    "Condicion",
    "Confianza",
    "Prioridad",
    "Revision humana",
    "Accion recomendada",
    "Estado",
]


class _NamedBytesIO(BytesIO):
    def __init__(self, payload: bytes, name: str) -> None:
        super().__init__(payload)
        self.name = name


class _FakePrediction:
    def __init__(
        self, predicted_class: str = "Clean", confidence: float = 0.90
    ) -> None:
        self.predicted_class = predicted_class
        self.confidence = confidence


class _FakePriority:
    def __init__(
        self,
        priority: str = "low",
        requires_human_review: bool = False,
        recommended_action: str | None = None,
    ) -> None:
        self.priority = priority
        self.requires_human_review = requires_human_review
        self.recommended_action = recommended_action or "No requiere intervencion."


class _RecordingAnalyzer:
    """Fake de ``analyze_one`` que registra las fuentes analizadas."""

    def __init__(
        self,
        prediction: _FakePrediction | None = None,
        priority: _FakePriority | None = None,
    ) -> None:
        self.calls: list[str] = []
        self._prediction = prediction or _FakePrediction()
        self._priority = priority or _FakePriority()

    def __call__(self, loaded):
        self.calls.append(loaded.source)
        return self._prediction, self._priority


def _image_payload(
    image_format: str = "PNG",
    *,
    size: tuple[int, int] = (32, 24),
    color: int = 0,
) -> bytes:
    image = Image.new("RGB", size, color=color)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def _source(name: str, payload: bytes) -> _NamedBytesIO:
    return _NamedBytesIO(payload, name)


def _make_record(image_name: str, payload: bytes) -> AnalysisRecord:
    return build_analysis_record(
        image_name=image_name,
        image_bytes=payload,
        prediction=_FakePrediction(),
        priority=_FakePriority(),
    )


def _processed_result(
    image_name: str = "a.png",
    predicted_class: str = "Clean",
    confidence: float = 0.90,
    priority: str = "low",
    requires_human_review: bool | None = None,
    recommended_action: str | None = None,
) -> BatchImageResult:
    return BatchImageResult(
        image_name=image_name,
        status="processed",
        predicted_class=predicted_class,
        confidence=confidence,
        priority=priority,
        requires_human_review=requires_human_review,
        recommended_action=recommended_action,
    )


# ---------------------------------------------------------------------------
# Nucleo puro: run_batch_analysis
# ---------------------------------------------------------------------------


def test_run_batch_analysis_lote_vacio() -> None:
    # Arrange/Act
    summary = run_batch_analysis([], history=[], analyze_one=_RecordingAnalyzer())

    # Assert
    assert summary.selected == 0
    assert summary.processed == 0
    assert summary.failed == 0
    assert summary.duplicates == 0
    assert summary.results == ()


def test_run_batch_analysis_una_imagen() -> None:
    # Arrange
    payload = _image_payload()
    history: list[AnalysisRecord] = []
    analyzer = _RecordingAnalyzer(
        _FakePrediction("Dusty", 0.82), _FakePriority("medium")
    )

    # Act
    summary = run_batch_analysis(
        [_source("panel-a.png", payload)],
        history=history,
        analyze_one=analyzer,
    )

    # Assert
    assert summary.selected == 1
    assert summary.processed == 1
    assert summary.failed == 0
    assert summary.duplicates == 0
    result = summary.results[0]
    assert result.image_name == "panel-a.png"
    assert result.status == "processed"
    assert result.predicted_class == "Dusty"
    assert result.confidence == pytest.approx(0.82)
    assert result.priority == "medium"
    assert result.error_message is None
    assert analyzer.calls == ["panel-a.png"]
    assert len(history) == 1
    assert history[0].image_fingerprint == compute_image_fingerprint(payload)


def test_run_batch_analysis_procesada_conserva_mantenimiento() -> None:
    # Arrange: el PriorityResult ya calculado reutiliza sus campos.
    payload = _image_payload()
    history: list[AnalysisRecord] = []
    analyzer = _RecordingAnalyzer(
        _FakePrediction("Electrical-damage", 0.95),
        _FakePriority(
            priority="high",
            requires_human_review=False,
            recommended_action="Priorizar inspeccion tecnica.",
        ),
    )

    # Act
    summary = run_batch_analysis(
        [_source("panel-a.png", payload)],
        history=history,
        analyze_one=analyzer,
    )

    # Assert: no se re-ejecuta nada; se reutiliza el PriorityResult original.
    result = summary.results[0]
    assert result.status == "processed"
    assert result.requires_human_review is False
    assert result.recommended_action == "Priorizar inspeccion tecnica."
    assert analyzer.calls == ["panel-a.png"]


def test_run_batch_analysis_procesada_conserva_revision_humana() -> None:
    # Arrange: Unknown siempre exige revision humana en la priorizacion.
    payload = _image_payload()
    history: list[AnalysisRecord] = []
    analyzer = _RecordingAnalyzer(
        _FakePrediction("Unknown", 0.50),
        _FakePriority(
            priority="medium",
            requires_human_review=True,
            recommended_action="Requiere revision humana.",
        ),
    )

    # Act
    summary = run_batch_analysis(
        [_source("u.png", payload)],
        history=history,
        analyze_one=analyzer,
    )

    # Assert
    result = summary.results[0]
    assert result.status == "processed"
    assert result.requires_human_review is True
    assert result.recommended_action == "Requiere revision humana."


def test_run_batch_analysis_varias_imagenes_validas() -> None:
    # Arrange
    payload_a = _image_payload(size=(32, 24), color=200)
    payload_b = _image_payload(size=(40, 30), color=100)
    history: list[AnalysisRecord] = []
    analyzer = _RecordingAnalyzer()

    # Act
    summary = run_batch_analysis(
        [_source("a.png", payload_a), _source("b.png", payload_b)],
        history=history,
        analyze_one=analyzer,
    )

    # Assert
    assert summary.selected == 2
    assert summary.processed == 2
    assert summary.failed == 0
    assert summary.duplicates == 0
    by_name = {result.image_name: result for result in summary.results}
    assert set(by_name) == {"a.png", "b.png"}
    assert all(result.status == "processed" for result in by_name.values())
    assert analyzer.calls == ["a.png", "b.png"]
    assert len(history) == 2


def test_run_batch_analysis_archivo_invalido_entre_validos() -> None:
    # Arrange
    payload_a = _image_payload(size=(32, 24), color=200)
    payload_b = _image_payload(size=(40, 30), color=100)
    history: list[AnalysisRecord] = []
    analyzer = _RecordingAnalyzer()

    # Act
    summary = run_batch_analysis(
        [
            _source("v1.png", payload_a),
            _source("corrupt.png", b"not-an-image"),
            _source("v2.png", payload_b),
        ],
        history=history,
        analyze_one=analyzer,
    )

    # Assert: la imagen invalida no cancela el resto del lote.
    assert summary.selected == 3
    assert summary.processed == 2
    assert summary.failed == 1
    assert summary.duplicates == 0
    assert [result.status for result in summary.results] == [
        "processed",
        "failed",
        "processed",
    ]
    failed = summary.results[1]
    assert failed.image_name == "corrupt.png"
    assert failed.error_message is not None
    assert analyzer.calls == ["v1.png", "v2.png"]
    assert len(history) == 2


def test_run_batch_analysis_continua_despues_de_error() -> None:
    # Arrange
    payload_a = _image_payload(size=(32, 24), color=200)
    payload_b = _image_payload(size=(40, 30), color=100)
    history: list[AnalysisRecord] = []

    # Act: el primer archivo falla y los posteriores se procesan.
    summary = run_batch_analysis(
        [
            _source("corrupt.jpg", b"xx"),
            _source("ok1.png", payload_a),
            _source("ok2.png", payload_b),
        ],
        history=history,
        analyze_one=_RecordingAnalyzer(),
    )

    # Assert
    assert summary.failed == 1
    assert summary.processed == 2
    assert [result.status for result in summary.results] == [
        "failed",
        "processed",
        "processed",
    ]
    assert len(history) == 2


def test_run_batch_analysis_duplicado_contra_historial_existente() -> None:
    # Arrange
    payload = _image_payload()
    history = [_make_record("ya-analizada.png", payload)]
    analyzer = _RecordingAnalyzer()

    # Act
    summary = run_batch_analysis(
        [_source("copia.png", payload)],
        history=history,
        analyze_one=analyzer,
    )

    # Assert: se salta la inferencia y el historial no se duplica.
    assert summary.duplicates == 1
    assert summary.processed == 0
    assert summary.results[0].status == "duplicate"
    assert len(history) == 1
    assert analyzer.calls == []


def test_run_batch_analysis_duplicado_dentro_del_mismo_lote() -> None:
    # Arrange
    payload = _image_payload()
    history: list[AnalysisRecord] = []

    # Act: mismo contenido con nombres distintos dentro del mismo lote.
    summary = run_batch_analysis(
        [_source("a.png", payload), _source("b.png", payload)],
        history=history,
        analyze_one=_RecordingAnalyzer(),
    )

    # Assert
    assert summary.processed == 1
    assert summary.duplicates == 1
    assert [result.status for result in summary.results] == ["processed", "duplicate"]
    assert len(history) == 1


def test_run_batch_analysis_fallida_y_duplicada_sin_mantenimiento() -> None:
    # Arrange: fallida por archivo corrupto y duplicada contra el historial.
    payload = _image_payload()
    history = [_make_record("prev.png", payload)]

    # Act
    summary = run_batch_analysis(
        [_source("bad.png", b"corrupt"), _source("copia.png", payload)],
        history=history,
        analyze_one=_RecordingAnalyzer(),
    )

    # Assert: ni la fallida ni la duplicada exponen datos de mantenimiento.
    assert [result.status for result in summary.results] == ["failed", "duplicate"]
    for result in summary.results:
        assert result.requires_human_review is None
        assert result.recommended_action is None


def test_run_batch_analysis_invariante_del_resumen() -> None:
    # Arrange
    payload_a = _image_payload(size=(32, 24), color=200)
    payload_b = _image_payload(size=(40, 30), color=100)
    sources = [
        _source("a.png", payload_a),
        _source("bad.png", b"corrupt"),
        _source("b.png", payload_b),
        _source("dup.png", payload_a),
    ]

    # Act
    summary = run_batch_analysis(sources, history=[], analyze_one=_RecordingAnalyzer())

    # Assert
    assert summary.processed + summary.failed + summary.duplicates == summary.selected
    assert summary.selected == 4
    assert summary.processed == 2
    assert summary.failed == 1
    assert summary.duplicates == 1


def test_run_batch_analysis_resultado_por_archivo_y_registros() -> None:
    # Arrange
    payload = _image_payload()
    history: list[AnalysisRecord] = []
    analyzer = _RecordingAnalyzer(
        _FakePrediction("Snow-Covered", 0.71), _FakePriority("medium")
    )

    # Act
    summary = run_batch_analysis(
        [_source("s1.png", payload)],
        history=history,
        analyze_one=analyzer,
    )

    # Assert: el resultado por archivo y el registro del historial coinciden.
    result = summary.results[0]
    assert result.image_name == "s1.png"
    assert result.predicted_class == "Snow-Covered"
    assert result.confidence == pytest.approx(0.71)
    assert result.priority == "medium"

    record = history[0]
    assert record.image_name == "s1.png"
    assert record.image_fingerprint == compute_image_fingerprint(payload)
    assert record.predicted_class == "Snow-Covered"
    assert record.confidence == pytest.approx(0.71)
    assert record.priority == "medium"
    # No se inventan datos de panel ni ubicacion.
    assert not hasattr(record, "panel_id")
    assert not hasattr(record, "location")


def test_run_batch_analysis_fallos_no_mutan_el_historial() -> None:
    # Arrange
    existing_payload = _image_payload(size=(32, 24), color=10)
    history = [_make_record("prev.png", existing_payload)]
    snapshot = list(history)

    # Act
    summary = run_batch_analysis(
        [_source("bad.png", b"corrupt")],
        history=history,
        analyze_one=_RecordingAnalyzer(),
    )

    # Assert
    assert summary.failed == 1
    assert history == snapshot
    for record, original in zip(history, snapshot):
        assert record is original


def test_run_batch_analysis_resultados_y_resumen_son_inmutables() -> None:
    # Arrange
    result = BatchImageResult(image_name="a.png", status="processed")
    summary = BatchSummary(
        selected=1,
        processed=1,
        failed=0,
        duplicates=0,
        results=(result,),
    )

    # Act/Assert
    with pytest.raises(FrozenInstanceError):
        result.image_name = "b.png"
    with pytest.raises(FrozenInstanceError):
        summary.results = ()


def test_run_batch_analysis_persiste_solo_metadata() -> None:
    # Arrange
    result = _processed_result()
    summary = BatchSummary(
        selected=1,
        processed=1,
        failed=0,
        duplicates=0,
        results=(result,),
    )

    # Assert: ni el resumen ni los resultados exponen fingerprints ni bytes.
    assert not hasattr(summary, "image_fingerprint")
    assert not hasattr(result, "image_fingerprint")
    assert not hasattr(summary, "image_bytes")
    assert not hasattr(result, "image_bytes")


def test_run_batch_analysis_propaga_errores_inesperados_de_programacion() -> None:
    # Arrange: un error de programacion no debe convertirse en "failed".
    def explode(_loaded) -> tuple[_FakePrediction, _FakePriority]:
        raise TypeError("bug de programacion")

    # Act/Assert
    with pytest.raises(TypeError, match="bug de programacion"):
        run_batch_analysis(
            [_source("a.png", _image_payload())],
            history=[],
            analyze_one=explode,
        )


def test_run_batch_analysis_captura_inference_error_como_fallido() -> None:
    # Arrange
    def broken_inference(_loaded) -> tuple[_FakePrediction, _FakePriority]:
        raise InferenceError("el modelo no puede procesar la imagen")

    # Act
    summary = run_batch_analysis(
        [_source("a.png", _image_payload())],
        history=[],
        analyze_one=broken_inference,
    )

    # Assert
    assert summary.failed == 1
    assert summary.processed == 0
    assert summary.results[0].status == "failed"
    assert "el modelo no puede procesar" in summary.results[0].error_message


def test_run_batch_analysis_captura_prioritization_error_como_fallido() -> None:
    # Arrange
    def broken_prioritization(_loaded) -> tuple[_FakePrediction, _FakePriority]:
        raise PrioritizationError("clase no soportada")

    # Act
    summary = run_batch_analysis(
        [_source("a.png", _image_payload())],
        history=[],
        analyze_one=broken_prioritization,
    )

    # Assert
    assert summary.failed == 1
    assert "clase no soportada" in summary.results[0].error_message


def test_run_batch_analysis_captura_ingestion_error_como_fallido() -> None:
    # Arrange: la fuente no debe no exponer bytes como fallo de dominio.
    class _BytesLessSource:
        name = "misterioso.jpg"

    # Act
    summary = run_batch_analysis(
        [_BytesLessSource()],
        history=[],
        analyze_one=_RecordingAnalyzer(),
    )

    # Assert
    assert summary.failed == 1
    assert summary.results[0].image_name == "misterioso.jpg"
    assert "no expone los bytes" in summary.results[0].error_message


def test_run_batch_analysis_progreso_monotonico() -> None:
    # Arrange
    payload_a = _image_payload(size=(32, 24), color=200)
    payload_b = _image_payload(size=(40, 30), color=100)
    calls: list[tuple[int, int]] = []

    # Act
    run_batch_analysis(
        [_source("a.png", payload_a), _source("b.png", payload_b)],
        history=[],
        analyze_one=_RecordingAnalyzer(),
        progress=lambda current, total: calls.append((current, total)),
    )

    # Assert
    assert calls == [(1, 2), (2, 2)]


def test_run_batch_analysis_progreso_no_llama_en_lote_vacio() -> None:
    # Arrange
    calls: list[tuple[int, int]] = []

    # Act
    run_batch_analysis(
        [],
        history=[],
        analyze_one=_RecordingAnalyzer(),
        progress=lambda current, total: calls.append((current, total)),
    )

    # Assert
    assert calls == []


def test_run_batch_analysis_usa_fuentes_desde_ruta() -> None:
    # Arrange: las rutas validas del sistema de archivos tambien funcionan.
    import tempfile

    payload = _image_payload()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(payload)
        path = handle.name
    try:
        # Act
        summary = run_batch_analysis(
            [path],
            history=[],
            analyze_one=_RecordingAnalyzer(),
        )
    finally:
        from pathlib import Path

        Path(path).unlink(missing_ok=True)

    # Assert
    assert summary.processed == 1
    assert summary.results[0].status == "processed"


# ---------------------------------------------------------------------------
# Nucleo puro: build_batch_results_table
# ---------------------------------------------------------------------------


def test_build_batch_results_table_vacia() -> None:
    # Arrange
    summary = BatchSummary(selected=0, processed=0, failed=0, duplicates=0, results=())

    # Act
    frame = build_batch_results_table(summary)

    # Assert
    assert list(frame.columns) == _TABLE_COLUMNS
    assert frame.empty


def test_build_batch_results_table_con_resultados() -> None:
    # Arrange
    summary = BatchSummary(
        selected=3,
        processed=1,
        failed=1,
        duplicates=1,
        results=(
            _processed_result(
                predicted_class="Clean",
                priority="low",
                requires_human_review=False,
                recommended_action="No requiere intervencion inmediata.",
            ),
            BatchImageResult(
                image_name="bad.png",
                status="failed",
                error_message="No se pudo leer la imagen 'bad.png'.",
            ),
            BatchImageResult(image_name="dup.png", status="duplicate"),
        ),
    )

    # Act
    frame = build_batch_results_table(summary)

    # Assert
    assert list(frame.columns) == _TABLE_COLUMNS
    rows = frame.to_dict("records")
    assert rows[0]["Imagen"] == "a.png"
    assert rows[0]["Condicion"] == "Clean"
    assert rows[0]["Confianza"] == "90.00%"
    assert rows[0]["Prioridad"] == "Baja"
    assert rows[0]["Revision humana"] == "No"
    assert rows[0]["Accion recomendada"] == "No requiere intervencion inmediata."
    assert rows[0]["Estado"] == "Procesada"

    assert rows[1]["Imagen"] == "bad.png"
    assert rows[1]["Condicion"] == "-"
    assert rows[1]["Confianza"] == "-"
    assert rows[1]["Prioridad"] == "-"
    assert rows[1]["Revision humana"] == "-"
    assert rows[1]["Accion recomendada"] == "-"
    assert rows[1]["Estado"] == "Fallida"

    assert rows[2]["Imagen"] == "dup.png"
    assert rows[2]["Condicion"] == "-"
    assert rows[2]["Confianza"] == "-"
    assert rows[2]["Prioridad"] == "-"
    assert rows[2]["Revision humana"] == "-"
    assert rows[2]["Accion recomendada"] == "-"
    assert rows[2]["Estado"] == "Duplicada"


def test_build_batch_results_table_muestra_revision_humana_si_no() -> None:
    # Arrange: una procesada marca revision humana y otra no.
    summary = BatchSummary(
        selected=2,
        processed=2,
        failed=0,
        duplicates=0,
        results=(
            _processed_result(
                image_name="ok.png",
                requires_human_review=False,
                recommended_action="No requiere intervencion.",
            ),
            _processed_result(
                image_name="rev.png",
                requires_human_review=True,
                recommended_action="Requiere revision humana.",
            ),
        ),
    )

    # Act
    rows = build_batch_results_table(summary).to_dict("records")

    # Assert: "Si"/"No" y la accion recomendada conservada.
    assert rows[0]["Revision humana"] == "No"
    assert rows[1]["Revision humana"] == "Si"
    assert rows[0]["Accion recomendada"] == "No requiere intervencion."
    assert rows[1]["Accion recomendada"] == "Requiere revision humana."


def test_build_batch_failure_details_extrae_fallidas_con_mensaje() -> None:
    # Arrange: solo las fallidas con mensaje quedan en el detalle.
    summary = BatchSummary(
        selected=4,
        processed=1,
        failed=2,
        duplicates=1,
        results=(
            _processed_result(),
            BatchImageResult(
                image_name="bad1.png",
                status="failed",
                error_message="No se pudo leer 'bad1.png'.",
            ),
            BatchImageResult(image_name="bad2.png", status="failed"),
            BatchImageResult(image_name="dup.png", status="duplicate"),
        ),
    )

    # Act
    details = build_batch_failure_details(summary)

    # Assert
    assert details == (("bad1.png", "No se pudo leer 'bad1.png'."),)


def test_build_batch_failure_details_vacio() -> None:
    # Arrange
    summary = BatchSummary(
        selected=1,
        processed=1,
        failed=0,
        duplicates=0,
        results=(_processed_result(),),
    )

    # Act
    details = build_batch_failure_details(summary)

    # Assert
    assert details == ()


# ---------------------------------------------------------------------------
# Nucleo puro: build_batch_alert_summary
# ---------------------------------------------------------------------------


def test_build_batch_alert_summary_cuenta_prioridades_y_revision() -> None:
    # Arrange
    summary = BatchSummary(
        selected=6,
        processed=6,
        failed=0,
        duplicates=0,
        results=(
            _processed_result(
                image_name="a.png", priority="high", requires_human_review=True
            ),
            _processed_result(
                image_name="b.png", priority="high", requires_human_review=False
            ),
            _processed_result(
                image_name="c.png", priority="medium", requires_human_review=True
            ),
            _processed_result(
                image_name="d.png", priority="medium", requires_human_review=False
            ),
            _processed_result(
                image_name="e.png", priority="low", requires_human_review=False
            ),
            _processed_result(
                image_name="f.png", priority="low", requires_human_review=False
            ),
        ),
    )

    # Act
    alerts = build_batch_alert_summary(summary)

    # Assert
    assert alerts == BatchAlertSummary(
        high=2,
        medium=2,
        low=2,
        requires_human_review=2,
    )


def test_build_batch_alert_summary_ignora_fallidas_y_duplicadas() -> None:
    # Arrange: las fallidas/duplicadas no deben inflar ninguna metrica.
    summary = BatchSummary(
        selected=4,
        processed=1,
        failed=2,
        duplicates=1,
        results=(
            _processed_result(
                image_name="ok.png",
                priority="medium",
                requires_human_review=True,
            ),
            BatchImageResult(
                image_name="bad1.png",
                status="failed",
                error_message="No se pudo leer 'bad1.png'.",
            ),
            BatchImageResult(image_name="bad2.png", status="failed"),
            BatchImageResult(image_name="dup.png", status="duplicate"),
        ),
    )

    # Act
    alerts = build_batch_alert_summary(summary)

    # Assert: solo cuenta la imagen procesada.
    assert alerts == BatchAlertSummary(high=0, medium=1, low=0, requires_human_review=1)


def test_build_batch_alert_summary_sin_resultados() -> None:
    # Arrange
    summary = BatchSummary(selected=0, processed=0, failed=0, duplicates=0, results=())

    # Act
    alerts = build_batch_alert_summary(summary)

    # Assert
    assert alerts == BatchAlertSummary(
        high=0,
        medium=0,
        low=0,
        requires_human_review=0,
    )


# ---------------------------------------------------------------------------
# Capa de render
# ---------------------------------------------------------------------------


def test_render_batch_analysis_estado_vacio() -> None:
    # Arrange
    with (
        patch("solarguard_ai.view.batch.st") as mock_st,
        patch("solarguard_ai.view.batch.upload_images", return_value=None),
    ):
        mock_st.session_state = {}

        # Act
        render_batch_analysis()

    # Assert
    mock_st.info.assert_called_once()
    mock_st.button.assert_not_called()
    mock_st.dataframe.assert_not_called()
    mock_st.metric.assert_not_called()


def test_render_batch_results_muestra_metricas_y_tabla() -> None:
    # Arrange
    summary = BatchSummary(
        selected=2,
        processed=1,
        failed=1,
        duplicates=0,
        results=(
            _processed_result(
                requires_human_review=False,
                recommended_action="No requiere intervencion inmediata.",
            ),
            BatchImageResult(
                image_name="bad.png",
                status="failed",
                error_message="No se pudo leer la imagen.",
            ),
        ),
    )
    with patch("solarguard_ai.view.batch.st") as mock_st:
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        # Act
        render_batch_results(summary)

    # Assert: dos filas de 4 columnas (metricas del lote + resumen de alertas).
    assert mock_st.columns.call_args_list == [((4,),), ((4,),)]
    for column in mock_st.columns.return_value:
        assert column.metric.call_count == 2
    metric_labels = {
        call.args[0]
        for mock_column in mock_st.columns.return_value
        for call in mock_column.metric.call_args_list
    }
    assert metric_labels == {
        "Seleccionadas",
        "Procesadas",
        "Fallidas",
        "Duplicadas",
        "Prioridad Alta",
        "Prioridad Media",
        "Prioridad Baja",
        "Requieren revision humana",
    }
    mock_st.dataframe.assert_called_once()
    mock_st.expander.assert_called_once_with("Detalle de archivos fallidos")


def test_render_batch_results_muestra_resumen_de_alertas() -> None:
    # Arrange
    summary = BatchSummary(
        selected=1,
        processed=1,
        failed=0,
        duplicates=0,
        results=(
            _processed_result(
                priority="high",
                requires_human_review=True,
                recommended_action="Priorizar inspeccion tecnica.",
            ),
        ),
    )
    with patch("solarguard_ai.view.batch.st") as mock_st:
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        # Act
        render_batch_results(summary)

    # Assert: el bloque "Resumen de alertas del lote" se muestra una sola vez.
    subheaders = [call.args[0] for call in mock_st.subheader.call_args_list]
    assert subheaders.count("Resumen de alertas del lote") == 1
    metric_values = {
        call.args[1]
        for mock_column in mock_st.columns.return_value
        for call in mock_column.metric.call_args_list
    }
    assert metric_values == {1, 0}


def test_render_batch_alert_summary_muestra_metricas_calculadas() -> None:
    # Arrange
    summary = BatchSummary(
        selected=2,
        processed=2,
        failed=0,
        duplicates=0,
        results=(
            _processed_result(
                image_name="a.png", priority="high", requires_human_review=True
            ),
            _processed_result(
                image_name="b.png", priority="low", requires_human_review=False
            ),
        ),
    )
    with patch("solarguard_ai.view.batch.st") as mock_st:
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        # Act
        render_batch_alert_summary(summary)

    # Assert: valores exactos calculados por la funcion pura.
    labels = {
        call.args[0]
        for mock_column in mock_st.columns.return_value
        for call in mock_column.metric.call_args_list
    }
    assert labels == {
        "Prioridad Alta",
        "Prioridad Media",
        "Prioridad Baja",
        "Requieren revision humana",
    }
    mock_st.columns.return_value[0].metric.assert_called_once_with("Prioridad Alta", 1)


def test_render_batch_results_sin_fallidas_no_muestra_expander() -> None:
    # Arrange
    summary = BatchSummary(
        selected=1,
        processed=1,
        failed=0,
        duplicates=0,
        results=(_processed_result(),),
    )
    with patch("solarguard_ai.view.batch.st") as mock_st:
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        # Act
        render_batch_results(summary)

    # Assert
    mock_st.dataframe.assert_called_once()
    mock_st.expander.assert_not_called()


def test_render_batch_analysis_procesa_y_persiste_resumen_sin_bytes() -> None:
    # Arrange
    payload = _image_payload()
    summary = BatchSummary(
        selected=1,
        processed=1,
        failed=0,
        duplicates=0,
        results=(_processed_result(),),
    )
    session_state: dict[str, object] = {}
    with (
        patch("solarguard_ai.view.batch.st") as mock_st,
        patch(
            "solarguard_ai.view.batch.upload_images",
            return_value=[_source("a.png", payload)],
        ),
        patch(
            "solarguard_ai.view.batch.get_inference_service",
            return_value=MagicMock(),
        ),
        patch(
            "solarguard_ai.view.batch.get_priority_config",
            return_value=MagicMock(review_confidence=0.70),
        ),
        patch("solarguard_ai.view.batch.get_history", return_value=[]),
        patch(
            "solarguard_ai.view.batch.run_batch_analysis",
            return_value=summary,
        ) as mock_run,
    ):
        mock_st.session_state = session_state
        mock_st.button.return_value = True
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_st.progress.return_value = MagicMock()

        # Act
        render_batch_analysis()

    # Assert: el resumen se persiste con metadata, nunca bytes ni fingerprints.
    mock_run.assert_called_once()
    assert session_state["solarguard_batch_summary"] is summary
    assert session_state["solarguard_batch_names"] == ("a.png",)
    mock_st.dataframe.assert_called_once()


def test_render_batch_analysis_usa_resumen_almacenado_sin_reprocesar() -> None:
    # Arrange
    payload = _image_payload()
    summary = BatchSummary(
        selected=1,
        processed=1,
        failed=0,
        duplicates=0,
        results=(_processed_result(),),
    )
    session_state: dict[str, object] = {
        "solarguard_batch_summary": summary,
        "solarguard_batch_names": ("a.png",),
    }
    with (
        patch("solarguard_ai.view.batch.st") as mock_st,
        patch(
            "solarguard_ai.view.batch.upload_images",
            return_value=[_source("a.png", payload)],
        ),
        patch("solarguard_ai.view.batch.run_batch_analysis") as mock_run,
    ):
        mock_st.session_state = session_state
        mock_st.button.return_value = False
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        # Act
        render_batch_analysis()

    # Assert: no se reprocesa y se muestra el resumen guardado.
    mock_run.assert_not_called()
    mock_st.dataframe.assert_called_once()
    mock_st.info.assert_not_called()


def test_render_batch_analysis_relanzado_no_duplica_resultados() -> None:
    # Arrange: hay un resumen previo guardado y el usuario repulsa el boton.
    # Regresion Etapa 6: en ese rerun solo debe verse un unico bloque.
    payload = _image_payload()
    previous = BatchSummary(
        selected=1,
        processed=1,
        failed=0,
        duplicates=0,
        results=(_processed_result(image_name="anterior.png"),),
    )
    latest = BatchSummary(
        selected=1,
        processed=1,
        failed=0,
        duplicates=0,
        results=(_processed_result(image_name="nueva.png"),),
    )
    session_state: dict[str, object] = {
        "solarguard_batch_summary": previous,
        "solarguard_batch_names": ("a.png",),
    }
    with (
        patch("solarguard_ai.view.batch.st") as mock_st,
        patch(
            "solarguard_ai.view.batch.upload_images",
            return_value=[_source("a.png", payload)],
        ),
        patch(
            "solarguard_ai.view.batch.get_inference_service",
            return_value=MagicMock(),
        ),
        patch(
            "solarguard_ai.view.batch.get_priority_config",
            return_value=MagicMock(review_confidence=0.70),
        ),
        patch("solarguard_ai.view.batch.get_history", return_value=[]),
        patch(
            "solarguard_ai.view.batch.run_batch_analysis",
            return_value=latest,
        ) as mock_run,
    ):
        mock_st.session_state = session_state
        mock_st.button.return_value = True
        mock_st.progress.return_value = MagicMock()
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        # Act
        render_batch_analysis()

    # Assert: se muestra un unico bloque "Resultados del lote" con el resumen nuevo.
    mock_run.assert_called_once()
    assert session_state["solarguard_batch_summary"] is latest
    assert session_state["solarguard_batch_names"] == ("a.png",)
    subheaders = [c.args[0] for c in mock_st.subheader.call_args_list]
    assert subheaders.count("Resultados del lote") == 1
    mock_st.dataframe.assert_called_once()


def test_render_batch_analysis_no_muestra_resumen_de_otros_archivos() -> None:
    # Arrange: el resumen guardado pertenece a "a.png" pero se subio "b.png".
    payload = _image_payload()
    summary = BatchSummary(
        selected=1,
        processed=1,
        failed=0,
        duplicates=0,
        results=(_processed_result(),),
    )
    session_state: dict[str, object] = {
        "solarguard_batch_summary": summary,
        "solarguard_batch_names": ("a.png",),
    }
    with (
        patch("solarguard_ai.view.batch.st") as mock_st,
        patch(
            "solarguard_ai.view.batch.upload_images",
            return_value=[_source("b.png", payload)],
        ),
        patch("solarguard_ai.view.batch.run_batch_analysis") as mock_run,
    ):
        mock_st.session_state = session_state
        mock_st.button.return_value = False
        mock_st.columns.return_value = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )

        # Act
        render_batch_analysis()

    # Assert: no se reprocesa, no se muestra el resumen viejo ni el estado vacio.
    mock_run.assert_not_called()
    mock_st.dataframe.assert_not_called()
    mock_st.info.assert_not_called()
