"""Priorizacion operativa para SolarGuard AI.

Convierte el resultado del servicio de inferencia (clase y confianza)
en una prioridad operativa, una accion recomendada y una senal de
revision humana. No depende de ONNX Runtime, no procesa imagenes y
no contiene logica de interfaz.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

PriorityLevel = Literal["high", "medium", "low"]

DEFAULT_HUMAN_REVIEW_THRESHOLD: float = 0.70

# Conjunto explícito de clases soportadas: 6 del modelo + Unknown.
# Se define localmente para evitar importar inferencia (que exige
# onnxruntime) y mantener la priorizacion independiente.
SUPPORTED_CLASSES: frozenset[str] = frozenset(
    {
        "Bird-drop",
        "Clean",
        "Dusty",
        "Electrical-damage",
        "Physical-Damage",
        "Snow-Covered",
        "Unknown",
    }
)


class PrioritizationError(ValueError):
    """Indica que la entrada de priorizacion no puede procesarse."""


@dataclass(frozen=True)
class PriorityResult:
    """Resultado estructurado de la priorizacion operativa.

    Attributes:
        priority: prioridad operativa normalizada (high/medium/low).
        recommended_action: accion sugerida sujeta a revision tecnica.
        requires_human_review: indica si se necesita revision humana.
        reason: justificacion textual de la decision.
    """

    priority: PriorityLevel
    recommended_action: str
    requires_human_review: bool
    reason: str


@dataclass(frozen=True)
class PriorityConfig:
    """Configuracion operativa de priorizacion cargada desde un archivo TOML."""

    review_confidence: float


# Reglas base centralizadas: unica fuente de verdad clase -> prioridad/accion/razon.
# Evita numeros magicos y condiciones dispersas; cualquier cambio de texto
# o prioridad se hace aqui.
class _BaseRule(TypedDict):
    priority: PriorityLevel
    recommended_action: str
    reason: str


_BASE_RULES: dict[str, _BaseRule] = {
    "Electrical-damage": {
        "priority": "high",
        "recommended_action": (
            "Priorizar inspeccion tecnica por posible dano electrico visible; "
            "no constituye un diagnostico electrico definitivo y requiere "
            "validacion humana."
        ),
        "reason": (
            "Posible dano electrico visible: priorizacion alta sujeta a "
            "revision tecnica; no es un diagnostico definitivo."
        ),
    },
    "Physical-Damage": {
        "priority": "high",
        "recommended_action": (
            "Priorizar inspeccion tecnica y evaluar reparacion o sustitucion; "
            "no sustituir automaticamente sin validacion."
        ),
        "reason": (
            "Dano fisico visible: priorizacion alta; la sustitucion requiere "
            "evaluacion tecnica."
        ),
    },
    "Dusty": {
        "priority": "medium",
        "recommended_action": "Programar limpieza del panel.",
        "reason": "Suciedad visible: programar limpieza.",
    },
    "Bird-drop": {
        "priority": "medium",
        "recommended_action": "Programar limpieza del panel.",
        "reason": "Excremento de aves: programar limpieza.",
    },
    "Snow-Covered": {
        "priority": "medium",
        "recommended_action": (
            "Programar limpieza o retiro seguro de nieve; la cobertura reduce "
            "la exposicion solar y puede disminuir la produccion del panel."
        ),
        "reason": (
            "Cobertura de nieve: reduce la exposicion solar y puede disminuir "
            "la produccion; programar retiro seguro."
        ),
    },
    "Clean": {
        "priority": "low",
        "recommended_action": "No requiere intervencion inmediata.",
        "reason": "Panel visualmente limpio.",
    },
    "Unknown": {
        "priority": "medium",
        "recommended_action": (
            "Requiere revision humana o nueva captura; Unknown no significa "
            "que el panel este sano."
        ),
        "reason": (
            "Prediccion no concluyente (Unknown): requiere revision humana o "
            "nueva captura."
        ),
    },
}


def prioritize(
    predicted_class: str,
    confidence: float,
    review_threshold: float = DEFAULT_HUMAN_REVIEW_THRESHOLD,
) -> PriorityResult:
    """Asigna prioridad operativa a partir de clase y confianza.

    La prioridad refleja la gravedad base de la clase; la confianza solo
    determina `requires_human_review` mediante el umbral operativo.

    Args:
        predicted_class: clase predicha (case-sensitive, 7 valores soportados).
        confidence: confianza entre 0.0 y 1.0 inclusive.
        review_threshold: umbral operativo de revision humana (0.0..1.0).
            confidence < threshold implica revision. Es un parametro operativo
            configurable, no una metrica cientifica de precision del modelo.

    Returns:
        PriorityResult con prioridad, accion, revision y justificacion.

    Raises:
        PrioritizationError: si la clase no es soportada o si confidence/
            review_threshold son invalidos (fuera de rango, no finitos, bool,
            tipo incorrecto).
    """
    _validate_predicted_class(predicted_class)
    _validate_confidence(confidence, "confidence")
    _validate_confidence(review_threshold, "review_threshold")

    # Unknown siempre requiere revision humana independientemente de confidence.
    # Para el resto, solo confidence < threshold dispara revision; el caso
    # confidence == threshold NO requiere revision por umbral.
    requires_review = predicted_class == "Unknown" or confidence < review_threshold

    rule = _BASE_RULES[predicted_class]
    priority = rule["priority"]
    recommended_action = rule["recommended_action"]
    base_reason = rule["reason"]

    # Si la revision se debe al umbral, se anade sufijo explicativo.
    # Unknown no necesita sufijo porque su reason ya indica revision.
    if (
        requires_review
        and predicted_class != "Unknown"
        and confidence < review_threshold
    ):
        reason = (
            f"{base_reason} Confianza {confidence:.4f} por debajo del umbral "
            f"operativo {review_threshold:.2f}: se marca para revision humana."
        )
    else:
        reason = base_reason

    return PriorityResult(
        priority=priority,
        recommended_action=recommended_action,
        requires_human_review=requires_review,
        reason=reason,
    )


def prioritize_prediction(
    prediction: object,
    review_threshold: float = DEFAULT_HUMAN_REVIEW_THRESHOLD,
) -> PriorityResult:
    """Adaptador para objetos tipo PredictionResult sin acoplamiento en runtime.

    No importa `inferencia.py`; usa duck typing sobre atributos
    `predicted_class` y `confidence`.

    Args:
        prediction: objeto con atributos predicted_class y confidence.
        review_threshold: umbral operativo configurable.

    Raises:
        PrioritizationError: si el objeto no expone los atributos requeridos.
    """
    try:
        predicted_class = prediction.predicted_class
        confidence = prediction.confidence
    except AttributeError as error:
        raise PrioritizationError(
            "prediction debe exponer 'predicted_class' y 'confidence'."
        ) from error

    # Validar que los atributos existan (getattr con default None detecta falta)
    if predicted_class is None or confidence is None:
        raise PrioritizationError(
            "prediction debe exponer 'predicted_class' y 'confidence'."
        )

    return prioritize(predicted_class, confidence, review_threshold)


def load_priority_config(path: str | Path) -> PriorityConfig:
    """Carga y valida la configuracion de priorizacion desde un archivo TOML.

    El archivo debe contener una seccion ``[thresholds]`` con la clave
    ``review_confidence``, validada con las mismas reglas del umbral
    operativo. La lectura siempre es explicita: ``prioritize()`` es una
    funcion pura y no carga configuraciones automaticamente.

    Args:
        path: ruta al archivo TOML de configuracion.

    Returns:
        PriorityConfig inmutable con el umbral ``review_confidence``.

    Raises:
        PrioritizationError: si el archivo no existe, el TOML esta
            malformado o la configuracion no es valida.
    """
    raw = _read_toml(path)
    thresholds = _extract_thresholds(raw)
    review_confidence = _extract_review_confidence(thresholds)
    return PriorityConfig(review_confidence=review_confidence)


def _read_toml(path: str | Path) -> dict[str, object]:
    config_path = Path(path)
    try:
        with config_path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as error:
        raise PrioritizationError(
            f"No existe el archivo de configuracion '{config_path}'."
        ) from error
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PrioritizationError(
            f"No se pudo leer el archivo de configuracion '{config_path}': {error}"
        ) from error


def _extract_thresholds(raw: dict[str, object]) -> dict[str, object]:
    thresholds = raw.get("thresholds")
    if not isinstance(thresholds, dict):
        raise PrioritizationError(
            "Falta la seccion [thresholds] en el archivo de configuracion."
        )
    return thresholds


def _extract_review_confidence(thresholds: dict[str, object]) -> float:
    if "review_confidence" not in thresholds:
        raise PrioritizationError(
            "Falta la clave 'review_confidence' en la seccion [thresholds]."
        )
    return _validate_confidence(thresholds["review_confidence"], "review_confidence")


def _validate_predicted_class(predicted_class: object) -> None:
    if not isinstance(predicted_class, str):
        raise PrioritizationError("predicted_class debe ser una cadena.")
    if predicted_class not in SUPPORTED_CLASSES:
        supported = ", ".join(sorted(SUPPORTED_CLASSES))
        raise PrioritizationError(
            f"Clase no soportada '{predicted_class}'. Clases soportadas: {supported}."
        )


def _validate_confidence(value: object, name: str) -> float:
    # bool es subclase de int; debe rechazarse explicitamente.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PrioritizationError(f"{name} debe ser un numero entre 0 y 1.")
    float_value = float(value)
    if not math.isfinite(float_value):
        raise PrioritizationError(f"{name} debe ser un valor finito entre 0 y 1.")
    if not 0.0 <= float_value <= 1.0:
        raise PrioritizationError(f"{name} debe estar entre 0 y 1 inclusive.")
    return float_value


__all__ = [
    "DEFAULT_HUMAN_REVIEW_THRESHOLD",
    "SUPPORTED_CLASSES",
    "PrioritizationError",
    "PriorityConfig",
    "PriorityLevel",
    "PriorityResult",
    "load_priority_config",
    "prioritize",
    "prioritize_prediction",
]
