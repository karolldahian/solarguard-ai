"""Servicio de inferencia ONNX para el modelo SolarScan."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from .mlflow_tracking import is_enabled, log_prediction

FloatTensor = NDArray[np.float32]

CLASS_NAMES = (
    "Bird-drop",
    "Clean",
    "Dusty",
    "Electrical-damage",
    "Physical-Damage",
    "Snow-Covered",
)
UNKNOWN_CLASS = "Unknown"
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
EXPECTED_INPUT_SHAPE = (1, 3, 224, 224)


class InferenceError(ValueError):
    """Indica que el modelo o la entrada no pueden procesarse."""


@dataclass(frozen=True)
class PredictionResult:
    """Resultado estructurado de una prediccion SolarScan."""

    predicted_class: str
    confidence: float
    probabilities: dict[str, float]


SessionFactory = Callable[[str], Any]


class SolarScanInference:
    """Carga SolarScan y ejecuta predicciones individuales o por lotes."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        session_factory: SessionFactory = ort.InferenceSession,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise InferenceError(f"No existe el modelo ONNX: '{model_path}'.")
        if not 0 <= confidence_threshold <= 1:
            raise InferenceError("confidence_threshold debe estar entre 0 y 1.")

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        try:
            # La sesion se crea una sola vez y se reutiliza para todas las inferencias.
            self._session = session_factory(str(model_path))
            self._input_name = self._session.get_inputs()[0].name
        except (IndexError, OSError, RuntimeError, ValueError) as error:
            raise InferenceError(
                f"No se pudo cargar el modelo '{model_path}': {error}"
            ) from error
        self._cache: dict[bytes, PredictionResult] = {}

    def predict(self, tensor: FloatTensor) -> PredictionResult:
        """Predice una imagen preprocesada con forma `(1, 3, 224, 224)`."""
        batch = _validate_tensor(tensor)
        cache_key = _tensor_cache_key(batch)
        cached_result = self._cache.get(cache_key)
        if cached_result is not None:
            if is_enabled():
                log_prediction(
                    predicted_class=cached_result.predicted_class,
                    confidence=cached_result.confidence,
                    latency_ms=0.0,  # Cache hit = latencia ~0
                    cache_hit=True,
                    probabilities=cached_result.probabilities,
                    model_path=str(self.model_path),
                    confidence_threshold=self.confidence_threshold,
                )
            return cached_result

        start_time = time.perf_counter()
        probabilities = self._run_model(batch)[0]
        latency_ms = (time.perf_counter() - start_time) * 1000
        result = _build_result(probabilities, self.confidence_threshold)
        # La clave usa el contenido y la forma: imágenes idénticas evitan inferencias repetidas.
        self._cache[cache_key] = result

        if is_enabled():
            log_prediction(
                predicted_class=result.predicted_class,
                confidence=result.confidence,
                latency_ms=latency_ms,
                cache_hit=False,
                probabilities=result.probabilities,
                model_path=str(self.model_path),
                confidence_threshold=self.confidence_threshold,
            )
        return result

    def predict_batch(self, tensors: Iterable[FloatTensor]) -> list[PredictionResult]:
        """Predice un lote de tensores y reutiliza la caché por imagen."""
        return [self.predict(tensor) for tensor in tensors]

    def clear_cache(self) -> None:
        """Elimina resultados cacheados sin recargar la sesión ONNX."""
        self._cache.clear()

    def _run_model(self, batch: FloatTensor) -> NDArray[np.float32]:
        try:
            outputs = self._session.run(None, {self._input_name: batch})
            probabilities = np.asarray(outputs[0], dtype=np.float32)
        except (IndexError, OSError, RuntimeError, ValueError) as error:
            raise InferenceError(
                f"Error durante la inferencia ONNX: {error}"
            ) from error

        if probabilities.ndim != 2 or probabilities.shape[0] != 1:
            raise InferenceError(
                f"El modelo devolvio una salida invalida: forma {probabilities.shape}."
            )
        if probabilities.shape[1] != len(CLASS_NAMES):
            raise InferenceError(
                f"Se esperaban {len(CLASS_NAMES)} clases y se recibieron "
                f"{probabilities.shape[1]}."
            )
        if not np.isfinite(probabilities).all():
            raise InferenceError("El modelo devolvio probabilidades no finitas.")

        # Algunos exportadores devuelven logits; softmax los convierte en probabilidades.
        if np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0):
            probabilities = _softmax(probabilities)
        return probabilities


def _validate_tensor(tensor: NDArray[np.generic]) -> FloatTensor:
    values = np.asarray(tensor, dtype=np.float32)
    if values.shape != EXPECTED_INPUT_SHAPE:
        raise InferenceError(
            f"forma de entrada invalida: se esperaba {EXPECTED_INPUT_SHAPE} "
            f"y se recibio {values.shape}."
        )
    if not np.isfinite(values).all():
        raise InferenceError("La entrada contiene valores no finitos.")
    if values.min() < 0 or values.max() > 1:
        raise InferenceError(
            "La entrada debe contener valores normalizados entre 0 y 1."
        )
    return np.ascontiguousarray(values, dtype=np.float32)


def _tensor_cache_key(tensor: FloatTensor) -> bytes:
    return tensor.shape.__repr__().encode() + tensor.tobytes()


def _build_result(
    probabilities: NDArray[np.float32], threshold: float
) -> PredictionResult:
    # _run_model entrega la fila de probabilidades correspondiente a una imagen.
    scores = probabilities
    top_index = int(np.argmax(scores))
    confidence = float(scores[top_index])
    predicted_class = (
        CLASS_NAMES[top_index] if confidence >= threshold else UNKNOWN_CLASS
    )
    return PredictionResult(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities={
            class_name: float(score) for class_name, score in zip(CLASS_NAMES, scores)
        },
    )


def _softmax(values: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


__all__ = [
    "CLASS_NAMES",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "EXPECTED_INPUT_SHAPE",
    "InferenceError",
    "PredictionResult",
    "SolarScanInference",
]
