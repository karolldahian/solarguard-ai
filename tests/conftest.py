"""Fixtures compartidas para la suite de pruebas."""

from __future__ import annotations

import pytest

from solarguard_ai import mlflow_tracking


@pytest.fixture(autouse=True)
def disable_mlflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Desactiva MLflow y limpia su caché antes y después de cada test."""
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    mlflow_tracking._config = None
    mlflow_tracking._mlflow_client = None
    yield
    mlflow_tracking._config = None
    mlflow_tracking._mlflow_client = None
