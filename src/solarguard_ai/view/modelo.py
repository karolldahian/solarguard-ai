"""Acceso y configuracion del servicio de inferencia para la vista.

Este modulo resuelve las rutas deterministas del proyecto y construye el
servicio ONNX de forma cacheada. No contiene logica de presentacion,
no simula resultados y no aplica fallbacks: los errores tipados se
propagan hasta el punto de orquestacion (``app.py``).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from solarguard_ai.inferencia import SolarScanInference
from solarguard_ai.priorizacion import PriorityConfig, load_priority_config


def resolve_model_path() -> Path:
    """Resuelve la ruta determinista ``models/best.onnx`` del proyecto.

    La ruta se ancla a la ubicacion de este modulo con ``pathlib`` y no
    depende del directorio desde el que se ejecute Streamlit.
    """
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "models" / "best.onnx"


def resolve_priority_config_path() -> Path:
    """Resuelve la ruta determinista ``config/prioritization.toml``."""
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "config" / "prioritization.toml"


@st.cache_resource
def get_inference_service() -> SolarScanInference:
    """Construye (una sola vez por sesion) el servicio SolarScan.

    Utiliza ``@st.cache_resource`` para no reconstruir la sesion ONNX en
    cada rerun de Streamlit. Si el modelo no existe o no puede cargarse,
    propaga ``InferenceError``; nunca retorna ``None`` ni simula resultados.
    """
    return SolarScanInference(resolve_model_path())


def get_priority_config() -> PriorityConfig:
    """Carga la configuracion explicita del umbral operativo.

    La configuracion TOML es la unica fuente del umbral: si falta, esta
    corrupta o es invalida, propaga ``PrioritizationError`` sin sustituir
    por ningun valor por defecto.
    """
    return load_priority_config(resolve_priority_config_path())


__all__ = [
    "get_inference_service",
    "get_priority_config",
    "resolve_model_path",
    "resolve_priority_config_path",
]
