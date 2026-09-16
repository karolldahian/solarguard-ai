"""Analisis por lote de imagenes para SolarGuard AI (capa Streamlit).

Este modulo permite analizar multiples imagenes de paneles solares en una
sola operacion secuencial, reutilizando el pipeline existente (ingesta,
inferencia, priorizacion y historial). No crea un segundo pipeline de IA:
``run_batch_analysis`` recibe por inyeccion ``analyze_one``, la misma
composicion de contratos que usa el analisis individual en ``app.py``.

El nucleo puro es testeable sin Streamlit. Una imagen invalida o corrupta no
cancela el procesamiento del resto del lote: cada archivo conserva sus
resultados y los errores de dominio explicitos (``ImageIngestionError``,
``InferenceError`` y ``PrioritizationError``) se registran como ``failed``
sin detener el lote. Los errores inesperados de programacion se propagan y
no se convierten silenciosamente en ``failed``.

La deduplicacion reutiliza el fingerprint SHA-256 de ``historial``. El
fingerprint se usa internamente para decidir si una imagen ya fue analizada
(en el historial o en el mismo lote) y nunca se persiste: el resumen
almacenado en ``st.session_state`` contiene unicamente la metadata necesaria
para representar los resultados (nombres, estados, clase, confianza y
prioridad). Nunca se almacenan bytes de imagen en ``st.session_state``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
import streamlit as st

from solarguard_ai.inferencia import InferenceError
from solarguard_ai.ingesta import (
    ImageIngestionError,
    ImageSource,
    LoadedImage,
    load_image,
)
from solarguard_ai.preprocesamiento import preprocess_for_solarscan
from solarguard_ai.priorizacion import PrioritizationError, prioritize_prediction
from solarguard_ai.view.diagnostico import (
    render_configuration_error,
    render_model_unavailable,
)
from solarguard_ai.view.historial import (
    AnalysisRecord,
    build_analysis_record,
    compute_image_fingerprint,
    get_history,
    register_analysis,
)
from solarguard_ai.view.modelo import (
    get_inference_service,
    get_priority_config,
    resolve_model_path,
)
from solarguard_ai.view.pagina_principal import upload_images

if TYPE_CHECKING:
    from solarguard_ai.inferencia import PredictionResult, SolarScanInference
    from solarguard_ai.priorizacion import PriorityResult

BatchImageStatus = Literal["processed", "failed", "duplicate"]

AnalyzeOne = Callable[[LoadedImage], tuple["PredictionResult", "PriorityResult"]]
ProgressCallback = Callable[[int, int], None]

_SESSION_SUMMARY_KEY = "solarguard_batch_summary"
_SESSION_NAMES_KEY = "solarguard_batch_names"

_BATCH_DISCLAIMER = (
    "Hallazgos visuales preliminares generados por IA sobre el lote: no "
    "constituyen un diagnostico electrico definitivo y estan sujetos a "
    "validacion tecnica."
)
_EMPTY_BATCH_MESSAGE = (
    "Seleccione varias imagenes de paneles solares para analizarlas de forma "
    "secuencial. Cada imagen usa el mismo pipeline del analisis individual."
)
_STATE_LABELS = {
    "processed": "Procesada",
    "failed": "Fallida",
    "duplicate": "Duplicada (ya registrada)",
}


@dataclass(frozen=True)
class BatchImageResult:
    """Resultado por archivo de un analisis en lote (solo metadata)."""

    image_name: str
    status: BatchImageStatus
    predicted_class: str | None = None
    confidence: float | None = None
    priority: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class BatchSummary:
    """Resumen agregado de un analisis en lote.

    Solo contiene metadata representativa de los resultados (conteos y
    resultados por archivo). No almacena bytes de imagen ni fingerprints.
    """

    selected: int
    processed: int
    failed: int
    duplicates: int
    results: tuple[BatchImageResult, ...]


def run_batch_analysis(
    sources: Sequence[ImageSource],
    *,
    history: list[AnalysisRecord],
    analyze_one: AnalyzeOne,
    progress: ProgressCallback | None = None,
) -> BatchSummary:
    """Procesa un lote de imagenes de forma secuencial.

    Reutiliza ``load_image``, ``compute_image_fingerprint``,
    ``build_analysis_record`` y ``register_analysis`` (los mismos contratos
    del analisis individual). El fingerprint se usa internamente para saltar
    imagenes ya analizadas y nunca se persiste en el resumen.

    Una imagen fallida no cancela el lote: los errores de dominio
    (``ImageIngestionError``, ``InferenceError``, ``PrioritizationError``) se
    registran como ``failed`` y el procesamiento continua. Cualquier otra
    excepcion se propaga sin convertirse en ``failed``.

    Args:
        sources: fuentes de imagen (rutas o archivos en memoria).
        history: historial de la sesion (la misma lista de ``get_history()``).
        analyze_one: funcion que analiza un ``LoadedImage`` y devuelve
            ``(prediction, priority)`` componiendo el pipeline existente.
        progress: callback opcional ``(completados, total)`` por archivo.

    Returns:
        ``BatchSummary`` con los conteos y los resultados por archivo.
    """
    items = list(sources)
    total = len(items)
    seen_fingerprints = {record.image_fingerprint for record in history}
    processed_fingerprints: set[str] = set()

    results: list[BatchImageResult] = []
    processed_count = 0
    failed_count = 0
    duplicate_count = 0

    for index, source in enumerate(items, start=1):
        try:
            payload = _raw_payload(source)
            loaded = load_image(source)
            fingerprint = compute_image_fingerprint(payload)
        except (ImageIngestionError, InferenceError, PrioritizationError) as error:
            failed_count += 1
            results.append(
                BatchImageResult(
                    image_name=_source_name(source),
                    status="failed",
                    error_message=str(error),
                )
            )
            _notify_progress(progress, index, total)
            continue

        if fingerprint in seen_fingerprints or fingerprint in processed_fingerprints:
            duplicate_count += 1
            results.append(
                BatchImageResult(image_name=loaded.source, status="duplicate")
            )
            _notify_progress(progress, index, total)
            continue

        try:
            prediction, priority = analyze_one(loaded)
        except (InferenceError, PrioritizationError) as error:
            failed_count += 1
            results.append(
                BatchImageResult(
                    image_name=loaded.source,
                    status="failed",
                    error_message=str(error),
                )
            )
            _notify_progress(progress, index, total)
            continue

        record = build_analysis_record(
            image_name=loaded.source,
            image_bytes=payload,
            prediction=prediction,
            priority=priority,
        )
        if register_analysis(history, record):
            processed_count += 1
            processed_fingerprints.add(fingerprint)
            results.append(
                BatchImageResult(
                    image_name=record.image_name,
                    status="processed",
                    predicted_class=record.predicted_class,
                    confidence=record.confidence,
                    priority=record.priority,
                )
            )
        else:
            duplicate_count += 1
            results.append(
                BatchImageResult(image_name=record.image_name, status="duplicate")
            )
        _notify_progress(progress, index, total)

    return BatchSummary(
        selected=total,
        processed=processed_count,
        failed=failed_count,
        duplicates=duplicate_count,
        results=tuple(results),
    )


def build_batch_results_table(summary: BatchSummary) -> pd.DataFrame:
    """Construye la tabla plana de resultados del lote (funcion pura)."""
    columns = ["Imagen", "Condicion", "Confianza", "Prioridad", "Estado"]
    rows = [
        {
            "Imagen": result.image_name,
            "Condicion": result.predicted_class or "-",
            "Confianza": _format_confidence(result.confidence),
            "Prioridad": result.priority or "-",
            "Estado": _status_label(result),
        }
        for result in summary.results
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def render_batch_analysis() -> None:
    """Muestra la seccion de analisis por lote: carga multiple, progreso y resumen."""
    st.subheader("Analisis por lote")
    st.markdown(f"_{_BATCH_DISCLAIMER}_")

    uploaded_files = upload_images()
    current_names = _names_of(uploaded_files)

    stored_summary = st.session_state.get(_SESSION_SUMMARY_KEY)
    stored_names = st.session_state.get(_SESSION_NAMES_KEY)
    if stored_summary is not None and stored_names == current_names and current_names:
        render_batch_results(stored_summary)

    if not uploaded_files:
        st.info(_EMPTY_BATCH_MESSAGE)
        return

    if st.button("Analizar lote"):
        service = _resolve_inference_service()
        if service is None:
            return
        try:
            config = get_priority_config()
        except PrioritizationError as error:
            render_configuration_error(error)
            return

        def analyze_one(
            loaded: LoadedImage,
        ) -> tuple[PredictionResult, PriorityResult]:
            tensor = preprocess_for_solarscan(loaded)
            prediction = service.predict(tensor)
            return prediction, prioritize_prediction(
                prediction,
                review_threshold=config.review_confidence,
            )

        progress_bar = st.progress(0, text="Preparando el lote...")

        def _on_progress(current: int, total: int) -> None:
            ratio = min(current / total, 1.0) if total > 0 else 0
            progress_bar.progress(ratio, text=f"Procesando {current} de {total}")

        history = get_history()
        summary = run_batch_analysis(
            uploaded_files,
            history=history,
            analyze_one=analyze_one,
            progress=_on_progress,
        )
        st.session_state[_SESSION_SUMMARY_KEY] = summary
        st.session_state[_SESSION_NAMES_KEY] = current_names
        render_batch_results(summary)


def render_batch_results(summary: BatchSummary) -> None:
    """Muestra el resumen agregado y la tabla de resultados de un lote."""
    st.subheader("Resultados del lote")
    columns = st.columns(4)
    columns[0].metric("Seleccionadas", summary.selected)
    columns[1].metric("Procesadas", summary.processed)
    columns[2].metric("Fallidas", summary.failed)
    columns[3].metric("Duplicadas", summary.duplicates)

    if not summary.results:
        st.info("No se obtuvieron resultados que mostrar.")
        return
    st.dataframe(
        build_batch_results_table(summary),
        use_container_width=True,
        hide_index=True,
    )


def _resolve_inference_service() -> SolarScanInference | None:
    try:
        return get_inference_service()
    except InferenceError:
        render_model_unavailable(resolve_model_path())
        return None


def _raw_payload(source: ImageSource) -> bytes:
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            return path.read_bytes()
        except OSError as error:
            raise ImageIngestionError(
                f"No se pudo abrir el archivo '{path}': {error}"
            ) from error
    payload_reader = getattr(source, "getvalue", None)
    if payload_reader is None:
        raise ImageIngestionError(
            f"La fuente '{_source_name(source)}' no expone los bytes de la imagen."
        )
    return payload_reader()


def _source_name(source: ImageSource) -> str:
    if isinstance(source, (str, Path)):
        return str(source)
    return str(getattr(source, "name", "<archivo>"))


def _names_of(files: object | None) -> tuple[str, ...]:
    if not files:
        return ()
    return tuple(sorted(str(getattr(file_, "name", "<archivo>")) for file_ in files))


def _status_label(result: BatchImageResult) -> str:
    label = _STATE_LABELS.get(result.status, result.status)
    if result.status == "failed" and result.error_message:
        return f"Fallida: {result.error_message}"
    return label


def _format_confidence(confidence: float | None) -> str:
    return f"{confidence:.2%}" if confidence is not None else "-"


def _notify_progress(
    progress: ProgressCallback | None, current: int, total: int
) -> None:
    if progress is not None:
        progress(current, total)


__all__ = [
    "AnalyzeOne",
    "BatchImageResult",
    "BatchImageStatus",
    "BatchSummary",
    "ProgressCallback",
    "build_batch_results_table",
    "render_batch_analysis",
    "render_batch_results",
    "run_batch_analysis",
]
