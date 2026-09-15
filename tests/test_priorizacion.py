"""Tests para el componente de priorizacion operativa."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from solarguard_ai.priorizacion import (
    DEFAULT_HUMAN_REVIEW_THRESHOLD,
    SUPPORTED_CLASSES,
    PrioritizationError,
    PriorityConfig,
    PriorityResult,
    load_priority_config,
    prioritize,
    prioritize_prediction,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakePrediction:
    def __init__(self, predicted_class: str, confidence: float) -> None:
        self.predicted_class = predicted_class
        self.confidence = confidence


# ---------------------------------------------------------------------------
# Contrato basico por clase (7 clases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("predicted_class", "confidence", "expected_priority", "expected_review"),
    [
        ("Clean", 0.95, "low", False),
        ("Dusty", 0.90, "medium", False),
        ("Bird-drop", 0.90, "medium", False),
        ("Snow-Covered", 0.90, "medium", False),
        ("Electrical-damage", 0.90, "high", False),
        ("Physical-Damage", 0.90, "high", False),
        ("Unknown", 0.90, "medium", True),
    ],
)
def test_prioritize_asigna_prioridad_y_review_por_clase(
    predicted_class: str,
    confidence: float,
    expected_priority: str,
    expected_review: bool,
) -> None:
    # Arrange / Act
    result = prioritize(predicted_class, confidence)

    # Assert
    assert result.priority == expected_priority
    assert result.requires_human_review is expected_review
    assert result.recommended_action != ""
    assert result.reason != ""


def test_priorizacion_textos_respetan_reglas_de_negocio() -> None:
    electrical = prioritize("Electrical-damage", 0.95)
    physical = prioritize("Physical-Damage", 0.95)
    unknown = prioritize("Unknown", 0.95)
    snow = prioritize("Snow-Covered", 0.95)

    # Electrical: la recomendacion niega explicitamente diagnostico definitivo
    assert (
        "no constituye un diagnostico electrico definitivo"
        in electrical.recommended_action.lower()
    )
    # Electrical: el reason tambien niega diagnostico definitivo
    assert "no es un diagnostico definitivo" in electrical.reason.lower()

    # Physical no ordena sustitucion automatica
    assert "no sustituir automaticamente" in physical.recommended_action.lower()

    # Unknown indica revision/nueva captura y que no significa sano
    assert "revision humana" in unknown.recommended_action.lower()
    assert "no significa" in unknown.recommended_action.lower()

    # Snow menciona exposicion solar / produccion
    assert "exposicion solar" in snow.recommended_action.lower()
    assert "produccion" in snow.recommended_action.lower()


# ---------------------------------------------------------------------------
# Umbral operativo
# ---------------------------------------------------------------------------


def test_confidence_por_debajo_del_umbral_requiere_revision() -> None:
    result = prioritize("Clean", 0.6999)
    assert result.requires_human_review is True
    assert result.priority == "low"


def test_confidence_exactamente_en_umbral_no_requiere_revision_por_umbral() -> None:
    result = prioritize("Clean", 0.70)
    assert result.requires_human_review is False
    assert result.priority == "low"


def test_confidence_por_encima_del_umbral_no_requiere_revision() -> None:
    result = prioritize("Clean", 0.7001)
    assert result.requires_human_review is False


def test_confidence_cero_requiere_revision() -> None:
    result = prioritize("Dusty", 0.0)
    assert result.requires_human_review is True
    assert result.priority == "medium"


def test_confidence_uno_no_requiere_revision_salvo_unknown() -> None:
    assert prioritize("Dusty", 1.0).requires_human_review is False
    assert prioritize("Unknown", 1.0).requires_human_review is True


def test_umbral_configurable_gobierna_la_revision() -> None:
    # Umbral custom 0.80: 0.75 debe requerir revision, 0.85 no
    assert (
        prioritize("Dusty", 0.75, review_threshold=0.80).requires_human_review is True
    )
    assert (
        prioritize("Dusty", 0.85, review_threshold=0.80).requires_human_review is False
    )
    # Umbral custom 0.50: 0.55 no requiere revision
    assert (
        prioritize("Dusty", 0.55, review_threshold=0.50).requires_human_review is False
    )


def test_unknown_siempre_requiere_revision_independiente_de_confianza() -> None:
    for conf in (0.0, 0.40, 0.70, 0.95, 1.0):
        assert prioritize("Unknown", conf).requires_human_review is True


def test_clean_con_confianza_baja_conserva_low_pero_marca_revision() -> None:
    result = prioritize("Clean", 0.40)
    assert result.priority == "low"
    assert result.requires_human_review is True


def test_electrical_con_confianza_baja_conserva_high_pero_marca_revision() -> None:
    result = prioritize("Electrical-damage", 0.65)
    assert result.priority == "high"
    assert result.requires_human_review is True


def test_physical_con_confianza_baja_conserva_high() -> None:
    result = prioritize("Physical-Damage", 0.10)
    assert result.priority == "high"
    assert result.requires_human_review is True


# ---------------------------------------------------------------------------
# Entradas invalidas
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid", [-0.01, -1, -0.0001])
def test_confidence_negativo_lanza_error(invalid: float) -> None:
    with pytest.raises(PrioritizationError, match="entre 0 y 1"):
        prioritize("Clean", invalid)


@pytest.mark.parametrize("invalid", [1.01, 1.5, 2.0])
def test_confidence_mayor_que_uno_lanza_error(invalid: float) -> None:
    with pytest.raises(PrioritizationError, match="entre 0 y 1"):
        prioritize("Clean", invalid)


def test_confidence_nan_lanza_error() -> None:
    with pytest.raises(PrioritizationError, match="finito"):
        prioritize("Clean", float("nan"))


def test_confidence_inf_lanza_error() -> None:
    with pytest.raises(PrioritizationError, match="finito"):
        prioritize("Clean", float("inf"))
    with pytest.raises(PrioritizationError, match="finito"):
        prioritize("Clean", float("-inf"))


@pytest.mark.parametrize("invalid", [True, False])
def test_confidence_bool_lanza_error(invalid: bool) -> None:
    with pytest.raises(PrioritizationError, match="debe ser un numero"):
        prioritize("Clean", invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid", ["0.9", None, [], {}])
def test_confidence_tipo_invalido_lanza_error(invalid: object) -> None:
    with pytest.raises(PrioritizationError):
        prioritize("Clean", invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_class", ["Rusty", "dusty", "clean", "", " Clean", "Unknown "]
)
def test_clase_inexistente_lanza_error(invalid_class: str) -> None:
    with pytest.raises(PrioritizationError, match="Clase no soportada"):
        prioritize(invalid_class, 0.90)


def test_clase_no_string_lanza_error() -> None:
    with pytest.raises(PrioritizationError, match="debe ser una cadena"):
        prioritize(123, 0.9)  # type: ignore[arg-type]
    with pytest.raises(PrioritizationError):
        prioritize(None, 0.9)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_threshold", [-0.1, 1.1, float("nan"), float("inf"), True]
)
def test_threshold_invalido_lanza_error(invalid_threshold: object) -> None:
    with pytest.raises(PrioritizationError):
        prioritize("Clean", 0.9, review_threshold=invalid_threshold)  # type: ignore[arg-type]


def test_threshold_no_numerico_lanza_error() -> None:
    with pytest.raises(PrioritizationError):
        prioritize("Clean", 0.9, review_threshold="0.7")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Contrato de salida
# ---------------------------------------------------------------------------


def test_priority_result_es_inmutable() -> None:
    result = prioritize("Clean", 0.9)
    with pytest.raises(FrozenInstanceError):
        result.priority = "high"  # type: ignore[misc]


def test_priority_result_tipos_correctos() -> None:
    result = prioritize("Dusty", 0.9)
    assert isinstance(result, PriorityResult)
    assert result.priority in ("high", "medium", "low")
    assert isinstance(result.recommended_action, str)
    assert isinstance(result.requires_human_review, bool)
    assert isinstance(result.reason, str)


def test_supported_classes_contiene_siete_valores() -> None:
    assert len(SUPPORTED_CLASSES) == 7
    assert "Unknown" in SUPPORTED_CLASSES


def test_default_threshold_es_0_70() -> None:
    assert DEFAULT_HUMAN_REVIEW_THRESHOLD == pytest.approx(0.70)


def test_prioritize_prediction_adaptador() -> None:
    pred = _FakePrediction("Dusty", 0.90)
    result = prioritize_prediction(pred)
    assert result.priority == "medium"
    assert result.requires_human_review is False


def test_prioritize_prediction_con_baja_confianza() -> None:
    pred = _FakePrediction("Clean", 0.50)
    result = prioritize_prediction(pred)
    assert result.priority == "low"
    assert result.requires_human_review is True


def test_prioritize_prediction_sin_atributos_lanza_error() -> None:
    with pytest.raises(PrioritizationError, match="predicted_class"):
        prioritize_prediction(object())

    class Incomplete:
        predicted_class = "Clean"

    with pytest.raises(PrioritizationError):
        prioritize_prediction(Incomplete())


def test_reason_incluye_sufijo_cuando_revision_por_umbral() -> None:
    result_low = prioritize("Clean", 0.50)
    assert "umbral operativo" in result_low.reason.lower()

    result_ok = prioritize("Clean", 0.80)
    assert "umbral operativo" not in result_ok.reason.lower()


def test_unknown_no_duplica_sufijo_de_umbral() -> None:
    # Unknown con confianza baja no debe anadir sufijo extra de umbral
    result = prioritize("Unknown", 0.30)
    # Debe contener el reason base pero no el sufijo generico de umbral
    assert "umbral operativo" not in result.reason.lower()


@pytest.mark.parametrize(
    "klass",
    [
        "Bird-drop",
        "Clean",
        "Dusty",
        "Electrical-damage",
        "Physical-Damage",
        "Snow-Covered",
        "Unknown",
    ],
)
def test_todas_las_clases_devuelven_accion_y_razon_no_vacias(klass: str) -> None:
    result = prioritize(klass, 0.85)
    assert len(result.recommended_action.strip()) > 0
    assert len(result.reason.strip()) > 0


# ---------------------------------------------------------------------------
# Configuracion desde archivo TOML
# ---------------------------------------------------------------------------


def _write_toml_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "prioritization.toml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_load_priority_config_valido_con_0_70(tmp_path: Path) -> None:
    config_path = _write_toml_config(
        tmp_path, "[thresholds]\nreview_confidence = 0.70\n"
    )

    config = load_priority_config(config_path)

    assert isinstance(config, PriorityConfig)
    assert config.review_confidence == pytest.approx(0.70)


def test_load_priority_config_valido_con_0_80(tmp_path: Path) -> None:
    config_path = _write_toml_config(
        tmp_path, "[thresholds]\nreview_confidence = 0.80\n"
    )

    config = load_priority_config(config_path)

    assert config.review_confidence == pytest.approx(0.80)


def test_umbral_desde_archivo_cambia_el_comportamiento(tmp_path: Path) -> None:
    config_path = _write_toml_config(
        tmp_path, "[thresholds]\nreview_confidence = 0.80\n"
    )
    config = load_priority_config(config_path)

    with_default = prioritize("Dusty", 0.75)
    with_config = prioritize("Dusty", 0.75, review_threshold=config.review_confidence)

    assert with_default.requires_human_review is False
    assert with_config.requires_human_review is True


def test_config_sin_seccion_thresholds_es_invalida(tmp_path: Path) -> None:
    config_path = _write_toml_config(tmp_path, "review_confidence = 0.70\n")

    with pytest.raises(PrioritizationError, match="thresholds"):
        load_priority_config(config_path)


def test_config_sin_clave_review_confidence_es_invalida(tmp_path: Path) -> None:
    config_path = _write_toml_config(tmp_path, "[thresholds]\nother = 0.70\n")

    with pytest.raises(PrioritizationError, match="review_confidence"):
        load_priority_config(config_path)


@pytest.mark.parametrize("toml_value", ["1.1", "1.5"])
def test_config_con_review_confidence_mayor_a_uno_es_invalida(
    tmp_path: Path, toml_value: str
) -> None:
    config_path = _write_toml_config(
        tmp_path, f"[thresholds]\nreview_confidence = {toml_value}\n"
    )

    with pytest.raises(PrioritizationError, match="entre 0 y 1"):
        load_priority_config(config_path)


@pytest.mark.parametrize("toml_value", ["-0.1", "-1"])
def test_config_con_review_confidence_negativa_es_invalida(
    tmp_path: Path, toml_value: str
) -> None:
    config_path = _write_toml_config(
        tmp_path, f"[thresholds]\nreview_confidence = {toml_value}\n"
    )

    with pytest.raises(PrioritizationError, match="entre 0 y 1"):
        load_priority_config(config_path)


@pytest.mark.parametrize("toml_value", ['"0.70"', "true", "[]"])
def test_config_con_tipo_invalido_es_invalida(tmp_path: Path, toml_value: str) -> None:
    config_path = _write_toml_config(
        tmp_path, f"[thresholds]\nreview_confidence = {toml_value}\n"
    )

    with pytest.raises(PrioritizationError, match="debe ser un numero"):
        load_priority_config(config_path)


@pytest.mark.parametrize("toml_value", ["nan", "inf", "-inf"])
def test_config_con_review_confidence_no_finita_es_invalida(
    tmp_path: Path, toml_value: str
) -> None:
    config_path = _write_toml_config(
        tmp_path, f"[thresholds]\nreview_confidence = {toml_value}\n"
    )

    with pytest.raises(PrioritizationError, match="finito"):
        load_priority_config(config_path)


def test_config_toml_malformado_es_invalido(tmp_path: Path) -> None:
    config_path = _write_toml_config(
        tmp_path, "[thresholds\nreview_confidence = 0.70\n"
    )

    with pytest.raises(PrioritizationError):
        load_priority_config(config_path)


def test_config_inexistente_lanza_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"

    with pytest.raises(PrioritizationError, match="No existe"):
        load_priority_config(missing)


def test_priority_config_es_inmutable(tmp_path: Path) -> None:
    config_path = _write_toml_config(
        tmp_path, "[thresholds]\nreview_confidence = 0.70\n"
    )
    config = load_priority_config(config_path)

    with pytest.raises(FrozenInstanceError):
        config.review_confidence = 0.90  # type: ignore[misc]
