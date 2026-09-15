"""Pruebas unitarias para el módulo de MLflow tracking."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from solarguard_ai.mlflow_tracking import (
    MlflowConfig,
    _load_config,
    is_enabled,
    log_error,
    log_pipeline_request,
    log_prediction,
    log_streamlit_request,
    start_run,
)


def test_load_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que la configuración usa valores por defecto."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT", raising=False)
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)

    config = _load_config()

    assert config.tracking_uri == "http://localhost:5000"
    assert config.experiment_name == "solarguard-inference"
    assert config.enabled is True


def test_load_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que la configuración se lee de variables de entorno."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://custom:5000")
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "custom-exp")
    monkeypatch.setenv("MLFLOW_ENABLED", "false")

    config = _load_config()

    assert config.tracking_uri == "http://custom:5000"
    assert config.experiment_name == "custom-exp"
    assert config.enabled is False


def test_is_enabled_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica is_enabled cuando MLflow está activado."""
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    # Reset cached config
    import solarguard_ai.mlflow_tracking as mt

    mt._config = None

    assert is_enabled() is True


def test_is_enabled_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica is_enabled cuando MLflow está desactivado."""
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    import solarguard_ai.mlflow_tracking as mt

    mt._config = None

    assert is_enabled() is False


def test_start_run_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifica que start_run hace no-op cuando MLflow está deshabilitado."""
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    import solarguard_ai.mlflow_tracking as mt

    mt._config = None

    with start_run(run_name="test") as run:
        assert run is None


@patch("solarguard_ai.mlflow_tracking.mlflow")
def test_log_prediction_disabled(
    mock_mlflow: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica que log_prediction no hace nada cuando MLflow está deshabilitado."""
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    import solarguard_ai.mlflow_tracking as mt

    mt._config = None
    mt._mlflow_client = None

    log_prediction(
        predicted_class="Clean",
        confidence=0.95,
        latency_ms=10.0,
        cache_hit=False,
        probabilities={"Clean": 0.9, "Dusty": 0.1},
        model_path="models/best.onnx",
        confidence_threshold=0.5,
        panel_id="panel-01",
    )

    # No debe llamar a mlflow.log_metric ni log_param
    mock_mlflow.log_metric.assert_not_called()
    mock_mlflow.log_param.assert_not_called()


@patch("solarguard_ai.mlflow_tracking.mlflow")
@patch("solarguard_ai.mlflow_tracking._get_client")
def test_log_prediction_enabled(
    mock_get_client: MagicMock, mock_mlflow: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica que log_prediction registra métricas cuando MLflow está habilitado."""
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    import solarguard_ai.mlflow_tracking as mt

    mt._config = None
    mt._mlflow_client = None

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = None

    log_prediction(
        predicted_class="Electrical-damage",
        confidence=0.92,
        latency_ms=45.5,
        cache_hit=False,
        probabilities={"Electrical-damage": 0.92, "Physical-Damage": 0.08},
        model_path="models/best.onnx",
        confidence_threshold=0.7,
        panel_id="panel-001",
    )

    # Verificar que se llamaron las funciones de logging
    assert (
        mock_mlflow.log_metric.call_count >= 4
    )  # confidence, latency_ms, cache_hit, is_unknown + probs
    assert (
        mock_mlflow.log_param.call_count >= 3
    )  # model_path, confidence_threshold, panel_id
    assert mock_mlflow.set_tag.call_count >= 2  # predicted_class, event_type


@patch("solarguard_ai.mlflow_tracking.mlflow")
@patch("solarguard_ai.mlflow_tracking._get_client")
def test_log_pipeline_request_enabled(
    mock_get_client: MagicMock, mock_mlflow: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica que log_pipeline_request registra métricas del pipeline."""
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    import solarguard_ai.mlflow_tracking as mt

    mt._config = None
    mt._mlflow_client = None

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = None

    log_pipeline_request(
        panel_id="panel-001",
        image_size=(640, 480),
        image_format="JPEG",
        total_latency_ms=150.0,
        ingestion_latency_ms=5.0,
        preprocessing_latency_ms=10.0,
        inference_latency_ms=45.0,
        ticket_latency_ms=20.0,
        predicted_class="Dusty",
        confidence=0.88,
        priority="medium",
        requires_human_review=False,
        ticket_status="created",
    )

    assert mock_mlflow.log_metric.call_count >= 7
    assert mock_mlflow.log_param.call_count >= 2
    assert mock_mlflow.set_tag.call_count >= 4


@patch("solarguard_ai.mlflow_tracking.mlflow")
@patch("solarguard_ai.mlflow_tracking._get_client")
def test_log_error_enabled(
    mock_get_client: MagicMock, mock_mlflow: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica que log_error registra errores."""
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    import solarguard_ai.mlflow_tracking as mt

    mt._config = None
    mt._mlflow_client = None

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = None

    log_error(
        error_type="InferenceError",
        error_message="Model not found",
        context={"panel_id": "panel-001", "model_path": "models/best.onnx"},
    )

    mock_mlflow.set_tags.assert_called()
    mock_mlflow.log_metric.assert_called_with("error_count", 1)


@patch("solarguard_ai.mlflow_tracking.mlflow")
@patch("solarguard_ai.mlflow_tracking._get_client")
def test_log_streamlit_request_enabled(
    mock_get_client: MagicMock, mock_mlflow: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifica que log_streamlit_request registra métricas desde Streamlit."""
    monkeypatch.setenv("MLFLOW_ENABLED", "true")
    import solarguard_ai.mlflow_tracking as mt

    mt._config = None
    mt._mlflow_client = None

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = None

    log_streamlit_request(
        panel_id="panel-002",
        image_size=(800, 600),
        image_format="PNG",
        total_latency_ms=200.0,
        network_latency_ms=50.0,
        predicted_class="Bird-drop",
        confidence=0.75,
        priority="medium",
    )

    assert mock_mlflow.log_metric.call_count >= 4
    assert mock_mlflow.log_param.call_count >= 3
    assert mock_mlflow.set_tag.call_count >= 3


def test_mlflow_config_dataclass() -> None:
    """Verifica que MlflowConfig es inmutable."""
    config = MlflowConfig(
        tracking_uri="http://test:5000",
        experiment_name="test-exp",
        enabled=True,
    )

    assert config.tracking_uri == "http://test:5000"
    assert config.experiment_name == "test-exp"
    assert config.enabled is True

    # frozen=True previene modificación
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.tracking_uri = "http://other:5000"
