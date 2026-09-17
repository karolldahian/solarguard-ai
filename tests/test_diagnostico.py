"""Pruebas del diagnostico visual de la Etapa 2 (capa Streamlit).

Solo cubren comportamiento NUEVO de presentacion e integracion de
contratos. La matriz de severidad, la inferencia ONNX y el
preprocesamiento ya estan cubiertos en sus respectivos modulos.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from solarguard_ai.inferencia import PredictionResult, SolarScanInference
from solarguard_ai.ingesta import LoadedImage
from solarguard_ai.preprocesamiento import preprocess_for_solarscan
from solarguard_ai.priorizacion import (
    PrioritizationError,
    prioritize,
    prioritize_prediction,
)
from solarguard_ai.view.diagnostico import (
    render_configuration_error,
    render_diagnosis,
    render_model_unavailable,
    render_prediction,
    render_priority,
    render_probabilities,
)

_CLASS_NAMES = (
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered",
)


def _make_prediction(
    predicted_class: str = "Dusty",
    confidence: float = 0.82,
) -> PredictionResult:
    remaining = (1.0 - confidence) / (len(_CLASS_NAMES) - 1)
    probabilities = {
        name: (confidence if name == predicted_class else remaining)
        for name in _CLASS_NAMES
    }
    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
    )


class _FakeInput:
    name = "images"


class _FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]

    def run(self, _outputs: object, _inputs: object) -> list[np.ndarray]:
        return [self.output]


@patch("solarguard_ai.view.diagnostico.st")
def test_render_prediction_muestra_clase_y_confianza(
    mock_st: MagicMock,
) -> None:
    prediction = _make_prediction("Dusty", 0.82)

    render_prediction(prediction)

    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert any("Dusty" in call for call in markdown_calls)
    mock_st.metric.assert_called_once_with(label="Confianza", value="82.00%")
    mock_st.warning.assert_not_called()


@patch("solarguard_ai.view.diagnostico.st")
def test_render_prediction_unknown_avisa_revision(
    mock_st: MagicMock,
) -> None:
    prediction = _make_prediction("Unknown", 0.30)

    render_prediction(prediction)

    mock_st.warning.assert_called_once()
    texto = mock_st.warning.call_args.args[0].lower()
    assert "no concluyente" in texto
    assert "revision humana" in texto


@patch("solarguard_ai.view.diagnostico.st")
def test_render_probabilities_ordena_desc_con_porcentaje(
    mock_st: MagicMock,
) -> None:
    probabilities = {
        "Clean": 0.10,
        "Dusty": 0.60,
        "Bird-drop": 0.05,
        "Electrical-damage": 0.15,
        "Physical-Damage": 0.07,
        "Snow-Covered": 0.03,
    }

    render_probabilities(probabilities)

    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert len(markdown_calls) == 6
    assert markdown_calls[0].startswith("**Dusty:**")
    assert "60.00%" in markdown_calls[0]
    assert markdown_calls[1].startswith("**Electrical-damage:**")
    assert mock_st.progress.call_count == 6
    progress_values = [c.args[0] for c in mock_st.progress.call_args_list]
    assert progress_values == sorted(progress_values, reverse=True)


@patch("solarguard_ai.view.diagnostico.st")
def test_render_priority_sin_revision_no_emite_warning(
    mock_st: MagicMock,
) -> None:
    priority = prioritize("Electrical-damage", 0.90)

    render_priority(priority)

    mock_st.metric.assert_called_once_with(label="Prioridad", value="Alta")
    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert any("inspeccion tecnica" in call.lower() for call in markdown_calls)
    mock_st.warning.assert_not_called()


@patch("solarguard_ai.view.diagnostico.st")
def test_render_priority_con_revision_emite_warning(
    mock_st: MagicMock,
) -> None:
    priority = prioritize("Clean", 0.50)

    render_priority(priority)

    assert priority.priority == "low"
    assert priority.requires_human_review is True
    mock_st.warning.assert_called_once()
    texto = mock_st.warning.call_args.args[0].lower()
    assert "revision humana" in texto


@patch("solarguard_ai.view.diagnostico.st")
def test_render_priority_unknown_es_medium_con_revision(
    mock_st: MagicMock,
) -> None:
    priority = prioritize("Unknown", 0.90)

    render_priority(priority)

    assert priority.priority == "medium"
    assert priority.requires_human_review is True
    mock_st.warning.assert_called_once()


@patch("solarguard_ai.view.diagnostico.st")
def test_render_diagnosis_incluye_descargo_y_compone_secciones(
    mock_st: MagicMock,
) -> None:
    prediction = _make_prediction("Clean", 0.50)
    priority = prioritize_prediction(prediction)

    render_diagnosis(prediction, priority)

    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert any(
        "no constituye un diagnostico" in call.lower() for call in markdown_calls
    )
    assert any("validacion tecnica" in call.lower() for call in markdown_calls)
    subheaders = [c.args[0] for c in mock_st.subheader.call_args_list]
    assert "Diagnostico visual preliminar" in subheaders
    mock_st.expander.assert_called_once_with("Probabilidades por clase")
    mock_st.warning.assert_called()


@patch("solarguard_ai.view.diagnostico.st")
def test_render_model_unavailable_no_muestra_diagnostico(
    mock_st: MagicMock,
) -> None:
    render_model_unavailable(Path("models/best.onnx"))

    mock_st.warning.assert_called_once()
    texto = mock_st.warning.call_args.args[0].lower()
    assert "no disponible" in texto
    mock_st.metric.assert_not_called()
    mock_st.progress.assert_not_called()


@patch("solarguard_ai.view.diagnostico.st")
def test_render_configuration_error_muestra_error(
    mock_st: MagicMock,
) -> None:
    error = PrioritizationError("Falta la seccion [thresholds].")

    render_configuration_error(error)

    mock_st.error.assert_called_once()
    texto = mock_st.error.call_args.args[0].lower()
    assert "configuracion" in texto
    mock_st.progress.assert_not_called()


def test_encadenamiento_preprocess_predict_prioritize(tmp_path) -> None:
    # Arrange: imagen RGB pequena + sesion ONNX simulada.
    image = Image.new("RGB", (64, 48), color=(200, 100, 50))
    loaded = LoadedImage(
        source="panel.jpg",
        image=image,
        format="JPEG",
        width=64,
        height=48,
        original_mode="RGB",
        original_channels=3,
    )
    tensor = preprocess_for_solarscan(loaded)
    model_path = tmp_path / "best.onnx"
    model_path.write_bytes(b"fake-model")
    probabilities = np.array([[0.05, 0.05, 0.75, 0.05, 0.05, 0.05]], dtype=np.float32)

    def factory(_path: str) -> _FakeSession:
        return _FakeSession(probabilities)

    service = SolarScanInference(model_path, session_factory=factory)  # type: ignore[arg-type]

    # Act: solo se verifica el encadenamiento de contratos existentes.
    prediction = service.predict(tensor)
    priority = prioritize_prediction(prediction, review_threshold=0.70)

    # Assert
    assert prediction.predicted_class == "Dusty"
    assert set(prediction.probabilities) == set(_CLASS_NAMES)
    assert priority.priority == "medium"
    assert priority.requires_human_review is False
