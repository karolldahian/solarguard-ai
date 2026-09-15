"""Pruebas de ``view/modelo.py``: rutas y contrato de errores.

Cubren comportamiento NUEVO: resolucion determinista de rutas y el
contrato de ``get_inference_service``/``get_priority_config`` (usar la
ruta resuelta por el wrapper, devolver el objeto cuando funciona y
propagar el error tipado, nunca ``None`` ni fallbacks).

Para aislar ``get_inference_service`` de ``@st.cache_resource`` se extrae
la funcion original mediante ``__wrapped__`` y se sustituye en el espacio
del modulo; no se prueban ni dependen de los detalles de la cache de
Streamlit. La propagacion de errores de ``SolarScanInference`` y
``load_priority_config`` ya esta cubierta en sus propios modulos.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from solarguard_ai.inferencia import InferenceError
from solarguard_ai.priorizacion import PrioritizationError, PriorityConfig
from solarguard_ai.view import modelo
from solarguard_ai.view.modelo import (
    get_priority_config,
    resolve_model_path,
    resolve_priority_config_path,
)


@pytest.fixture
def plain_get_inference_service(monkeypatch) -> None:
    """Expone la funcion original sin el envoltorio de @st.cache_resource."""
    unwrapped = getattr(
        modelo.get_inference_service, "__wrapped__", modelo.get_inference_service
    )
    monkeypatch.setattr(modelo, "get_inference_service", unwrapped)


def test_resolve_model_path_es_determinista_y_absoluta(tmp_path, monkeypatch) -> None:
    # Arrange: cambiar el cwd no debe afectar la resolucion.
    monkeypatch.chdir(tmp_path)

    # Act
    first = resolve_model_path()
    second = resolve_model_path()

    # Assert
    assert first == second
    assert first.is_absolute()
    assert first.parts[-2:] == ("models", "best.onnx")


def test_resolve_priority_config_path_es_determinista(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    config_path = resolve_priority_config_path()

    assert config_path.is_absolute()
    assert config_path.parts[-2:] == ("config", "prioritization.toml")


def test_resolve_paths_comparten_raiz_del_proyecto() -> None:
    model_path = resolve_model_path()
    config_path = resolve_priority_config_path()

    assert model_path.parents[2] == config_path.parents[2]
    assert isinstance(model_path, Path)


# ---------------------------------------------------------------------------
# get_inference_service
# ---------------------------------------------------------------------------


def test_get_inference_service_usa_ruta_resuelta_y_devuelve_servicio(
    plain_get_inference_service, monkeypatch
) -> None:
    # Arrange
    fake_path = Path("resuelto") / "models" / "best.onnx"
    expected_service = MagicMock()
    monkeypatch.setattr(modelo, "resolve_model_path", lambda: fake_path)
    fake_constructor = MagicMock(return_value=expected_service)
    monkeypatch.setattr(modelo, "SolarScanInference", fake_constructor)

    # Act
    service = modelo.get_inference_service()

    # Assert
    assert service is expected_service
    fake_constructor.assert_called_once_with(fake_path)


def test_get_inference_service_propaga_inference_error_sin_none(
    plain_get_inference_service, monkeypatch
) -> None:
    # Arrange
    fake_path = Path("resuelto") / "models" / "best.onnx"
    monkeypatch.setattr(modelo, "resolve_model_path", lambda: fake_path)
    fake_constructor = MagicMock(
        side_effect=InferenceError("No existe el modelo ONNX.")
    )
    monkeypatch.setattr(modelo, "SolarScanInference", fake_constructor)

    # Act / Assert: el error tipado se propaga; nunca se transforma en None.
    with pytest.raises(InferenceError, match="No existe el modelo"):
        modelo.get_inference_service()
    fake_constructor.assert_called_once_with(fake_path)


# ---------------------------------------------------------------------------
# get_priority_config
# ---------------------------------------------------------------------------


def test_get_priority_config_usa_ruta_resuelta_y_devuelve_config(monkeypatch) -> None:
    # Arrange
    fake_path = Path("resuelto") / "config" / "prioritization.toml"
    expected_config = PriorityConfig(review_confidence=0.80)
    monkeypatch.setattr(modelo, "resolve_priority_config_path", lambda: fake_path)
    fake_loader = MagicMock(return_value=expected_config)
    monkeypatch.setattr(modelo, "load_priority_config", fake_loader)

    # Act
    config = get_priority_config()

    # Assert
    assert config is expected_config
    assert config.review_confidence == pytest.approx(0.80)
    fake_loader.assert_called_once_with(fake_path)


def test_get_priority_config_propaga_error_sin_fallback_a_070(monkeypatch) -> None:
    # Arrange: la configuracion es invalida y la carga falla.
    fake_path = Path("resuelto") / "config" / "prioritization.toml"
    monkeypatch.setattr(modelo, "resolve_priority_config_path", lambda: fake_path)
    fake_loader = MagicMock(side_effect=PrioritizationError("Falta la seccion."))
    monkeypatch.setattr(modelo, "load_priority_config", fake_loader)

    # Act / Assert: se propaga el error; no se fabrica un umbral 0.70.
    with pytest.raises(PrioritizationError, match="Falta la seccion"):
        get_priority_config()
    fake_loader.assert_called_once_with(fake_path)
