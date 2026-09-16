"""Pruebas de las etiquetas de presentacion compartidas (view/rotulos.py)."""

from __future__ import annotations

from solarguard_ai.view.rotulos import PRIORITY_LABELS, format_priority_label


def test_prioridades_conocidas_devuelven_etiqueta_en_espanol() -> None:
    # Act/Assert
    assert format_priority_label("high") == "Alta"
    assert format_priority_label("medium") == "Media"
    assert format_priority_label("low") == "Baja"


def test_prioridad_normaliza_mayusculas_y_espacios() -> None:
    # Act/Assert
    assert format_priority_label(" HIGH ") == "Alta"
    assert format_priority_label("Medium") == "Media"
    assert format_priority_label("low") == "Baja"


def test_prioridad_vacia_devuelve_etiqueta_desconocida() -> None:
    # Act/Assert
    assert format_priority_label("unknown") == "Desconocida"


def test_prioridad_inesperada_se_conserva_sin_traducir() -> None:
    # Act/Assert
    assert format_priority_label("urgent") == "urgent"
    assert format_priority_label(None) == "None"


def test_prioridad_vacia_o_en_blanco_evita_valores_falsos() -> None:
    # Act/Assert: el valor original se conserva, nunca se convierte en "Alta".
    assert format_priority_label("  ") == "  "


def test_preserva_literales_existentes() -> None:
    # Act/Assert: literales usados por dashboard y mapa de riesgo.
    assert PRIORITY_LABELS == {
        "high": "Alta",
        "medium": "Media",
        "low": "Baja",
        "unknown": "Desconocida",
    }
