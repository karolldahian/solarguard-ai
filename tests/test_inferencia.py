from __future__ import annotations

import time

import numpy as np
import pytest

from solarguard_ai.inferencia import (
    CLASS_NAMES,
    InferenceError,
    SolarScanInference,
)


class FakeInput:
    name = "images"


class FakeSession:
    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.run_count = 0

    def get_inputs(self) -> list[FakeInput]:
        return [FakeInput()]

    def run(self, _outputs, _inputs) -> list[np.ndarray]:
        self.run_count += 1
        return [self.output]


def tensor(value: float = 0.25) -> np.ndarray:
    return np.full((1, 3, 224, 224), value, dtype=np.float32)


def create_service(tmp_path, output: np.ndarray, holder: dict) -> SolarScanInference:
    model_path = tmp_path / "best.onnx"
    model_path.write_bytes(b"fake-model")

    def factory(_path: str) -> FakeSession:
        holder["session"] = FakeSession(output)
        return holder["session"]

    return SolarScanInference(model_path, session_factory=factory)


def test_service_loads_model_and_returns_structured_prediction(tmp_path) -> None:
    holder: dict[str, FakeSession] = {}
    probabilities = np.array([[0.05, 0.1, 0.7, 0.05, 0.05, 0.05]], dtype=np.float32)
    service = create_service(tmp_path, probabilities, holder)

    result = service.predict(tensor())

    assert result.predicted_class == "Dusty"
    assert result.confidence == pytest.approx(0.7)
    assert result.probabilities["Dusty"] == pytest.approx(0.7)
    assert set(result.probabilities) == set(CLASS_NAMES)
    assert holder["session"].run_count == 1


def test_predict_marks_low_confidence_as_unknown(tmp_path) -> None:
    holder: dict[str, FakeSession] = {}
    probabilities = np.full((1, 6), 1 / 6, dtype=np.float32)
    service = create_service(tmp_path, probabilities, holder)

    result = service.predict(tensor())

    assert result.predicted_class == "Unknown"
    assert result.confidence == pytest.approx(1 / 6)


def test_predict_applies_softmax_to_logits(tmp_path) -> None:
    holder: dict[str, FakeSession] = {}
    logits = np.array([[0, 0, 5, 0, 0, 0]], dtype=np.float32)
    service = create_service(tmp_path, logits, holder)

    result = service.predict(tensor())

    assert result.predicted_class == "Dusty"
    assert sum(result.probabilities.values()) == pytest.approx(1)


def test_predict_caches_identical_tensors(tmp_path) -> None:
    holder: dict[str, FakeSession] = {}
    service = create_service(
        tmp_path,
        np.array([[0.8, 0.04, 0.04, 0.04, 0.04, 0.04]], dtype=np.float32),
        holder,
    )

    first = service.predict(tensor(0.5))
    second = service.predict(tensor(0.5))

    assert first == second
    assert holder["session"].run_count == 1

    service.clear_cache()
    service.predict(tensor(0.5))
    assert holder["session"].run_count == 2


def test_predict_batch_handles_individual_images_and_cache(tmp_path) -> None:
    holder: dict[str, FakeSession] = {}
    service = create_service(
        tmp_path,
        np.array([[0.8, 0.04, 0.04, 0.04, 0.04, 0.04]], dtype=np.float32),
        holder,
    )

    results = service.predict_batch([tensor(0.1), tensor(0.2), tensor(0.1)])

    assert len(results) == 3
    assert all(result.predicted_class == "Bird-drop" for result in results)
    assert holder["session"].run_count == 2


def test_predict_rejects_invalid_shape_and_values(tmp_path) -> None:
    holder: dict[str, FakeSession] = {}
    service = create_service(
        tmp_path,
        np.array([[0.8, 0.04, 0.04, 0.04, 0.04, 0.04]], dtype=np.float32),
        holder,
    )

    with pytest.raises(InferenceError, match="forma"):
        service.predict(np.zeros((3, 224, 224), dtype=np.float32))
    with pytest.raises(InferenceError, match="normalizados"):
        service.predict(np.full((1, 3, 224, 224), 2, dtype=np.float32))
    with pytest.raises(InferenceError, match="no finitos"):
        invalid = tensor()
        invalid[0, 0, 0, 0] = np.nan
        service.predict(invalid)


def test_service_rejects_missing_model(tmp_path) -> None:
    with pytest.raises(InferenceError, match="No existe el modelo"):
        SolarScanInference(tmp_path / "missing.onnx")


def test_inference_is_under_two_seconds_for_one_image(tmp_path) -> None:
    holder: dict[str, FakeSession] = {}
    service = create_service(
        tmp_path,
        np.array([[0.8, 0.04, 0.04, 0.04, 0.04, 0.04]], dtype=np.float32),
        holder,
    )

    started = time.perf_counter()
    service.predict(tensor())
    elapsed = time.perf_counter() - started

    assert elapsed < 2
