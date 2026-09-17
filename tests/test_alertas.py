"""Pruebas de la visualizacion de alertas/tickets (Etapa 3).

Cubren la presentacion de ``TicketCreationResult`` y la generacion segura de
tickets sin cliente GitHub real. No se realizan llamadas de red; Streamlit se
simula con ``@patch`` siguiendo el estilo de ``test_diagnostico.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from solarguard_ai.priorizacion import prioritize
from solarguard_ai.tickets import (
    DEFAULT_LOCATION,
    MaintenanceTicket,
    TicketCreationResult,
    build_ticket,
)
from solarguard_ai.view.alertas import (
    render_ticket_result,
    render_ticket_section,
    validate_panel_id,
)


def _make_ticket(
    *,
    predicted_class: str = "Dusty",
    confidence: float = 0.82,
    panel_id: str = "PV-1024",
    location: str = "Parque Solar - Bloque General",
) -> MaintenanceTicket:
    priority = prioritize(predicted_class, confidence)
    return build_ticket(
        priority_result=priority,
        panel_id=panel_id,
        predicted_class=predicted_class,
        confidence=confidence,
        location=location,
    )


def _make_result(
    status: str,
    *,
    ticket: MaintenanceTicket | None = None,
    issue_number: int | None = None,
    message: str = "resultado",
) -> TicketCreationResult:
    return TicketCreationResult(
        status=status,
        message=message,
        ticket=ticket,
        issue_number=issue_number,
    )


@patch("solarguard_ai.view.alertas.st")
def test_render_simulado_avisa_simulacion_sin_issue_github(
    mock_st: MagicMock,
) -> None:
    ticket = _make_ticket()
    result = _make_result("simulated", ticket=ticket)

    render_ticket_result(result)

    warning_texts = [c.args[0] for c in mock_st.warning.call_args_list]
    assert any("simulacion" in texto.lower() for texto in warning_texts)
    assert any("github" in texto.lower() for texto in warning_texts)
    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert any(ticket.title in call for call in markdown_calls)


@patch("solarguard_ai.view.alertas.st")
def test_render_skipped_no_fabrica_ticket_visual(mock_st: MagicMock) -> None:
    result = _make_result("skipped")

    render_ticket_result(result)

    mock_st.info.assert_called_once()
    texto = mock_st.info.call_args.args[0].lower()
    assert "no requiere" in texto
    assert "ticket" in texto
    mock_st.subheader.assert_not_called()


@patch("solarguard_ai.view.alertas.st")
def test_render_muestra_severidad_prioridad(mock_st: MagicMock) -> None:
    ticket = _make_ticket(predicted_class="Electrical-damage", confidence=0.90)
    result = _make_result("simulated", ticket=ticket)

    render_ticket_result(result)

    assert ticket.severity == "high"
    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert any(
        "severidad" in call.lower() and "high" in call.lower()
        for call in markdown_calls
    )


@patch("solarguard_ai.view.alertas.st")
def test_render_muestra_condicion_y_confianza(mock_st: MagicMock) -> None:
    ticket = _make_ticket(predicted_class="Dusty", confidence=0.82)
    result = _make_result("simulated", ticket=ticket)

    render_ticket_result(result)

    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert any("dusty" in call.lower() for call in markdown_calls)
    mock_st.metric.assert_called_once_with(label="Confianza", value="82.00%")


@patch("solarguard_ai.view.alertas.st")
def test_render_muestra_panel_y_ubicacion(mock_st: MagicMock) -> None:
    ticket = _make_ticket(
        panel_id="PV-1024",
        location="Parque Solar - Bloque Norte",
    )
    result = _make_result("simulated", ticket=ticket)

    render_ticket_result(result)

    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert any("pv-1024" in call.lower() for call in markdown_calls)
    assert any("bloque norte" in call.lower() for call in markdown_calls)


@patch("solarguard_ai.view.alertas.st")
def test_render_requiere_revision_humana_muestra_aviso(mock_st: MagicMock) -> None:
    ticket = _make_ticket(predicted_class="Unknown", confidence=0.90)
    result = _make_result("simulated", ticket=ticket)

    render_ticket_result(result)

    assert ticket.requires_human_review is True
    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert any("revision humana" in call.lower() for call in markdown_calls)
    warning_texts = [c.args[0] for c in mock_st.warning.call_args_list]
    assert any("revision humana" in texto.lower() for texto in warning_texts)


@patch("solarguard_ai.view.alertas.st")
def test_render_sin_revision_humana_indica_no(mock_st: MagicMock) -> None:
    ticket = _make_ticket(predicted_class="Dusty", confidence=0.82)
    result = _make_result("simulated", ticket=ticket)

    render_ticket_result(result)

    assert ticket.requires_human_review is False
    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert "**Revision humana requerida:** No" in markdown_calls
    warning_texts = [c.args[0] for c in mock_st.warning.call_args_list]
    assert not any("revision humana" in texto.lower() for texto in warning_texts)


@patch("solarguard_ai.view.alertas.st")
def test_render_muestra_assignees_y_labels(mock_st: MagicMock) -> None:
    ticket = _make_ticket(predicted_class="Electrical-damage", confidence=0.90)
    result = _make_result("simulated", ticket=ticket)

    render_ticket_result(result)

    assert ticket.assignees == ["ing-electrico"]
    assert "maintenance" in ticket.labels
    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert any("ing-electrico" in call for call in markdown_calls)
    assert any(
        "etiquetas" in call.lower() and "maintenance" in call for call in markdown_calls
    )


@patch("solarguard_ai.view.alertas.st")
def test_render_assignees_vacios_muestra_mensaje(mock_st: MagicMock) -> None:
    ticket = _make_ticket(predicted_class="Clean", confidence=0.90)
    result = _make_result("simulated", ticket=ticket)

    render_ticket_result(result)

    assert ticket.assignees == []
    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert any("sin responsables" in call.lower() for call in markdown_calls)


@patch("solarguard_ai.view.alertas.generate_maintenance_ticket")
@patch("solarguard_ai.view.alertas.st")
def test_generacion_sin_cliente_github_y_validacion(
    mock_st: MagicMock,
    mock_generate: MagicMock,
) -> None:
    mock_generate.return_value = _make_result("simulated")
    mock_st.button.return_value = True
    mock_st.text_input.side_effect = ["  PV-2048  ", DEFAULT_LOCATION]
    priority = prioritize("Dusty", 0.82)

    render_ticket_section(priority, "Dusty", 0.82)

    mock_generate.assert_called_once()
    _, kwargs = mock_generate.call_args
    assert kwargs["priority_result"] is priority
    assert kwargs["panel_id"] == "PV-2048"
    assert kwargs["predicted_class"] == "Dusty"
    assert kwargs["confidence"] == 0.82
    assert kwargs["location"] == DEFAULT_LOCATION
    assert "client" not in kwargs
    assert "force" not in kwargs


@patch("solarguard_ai.view.alertas.generate_maintenance_ticket")
@patch("solarguard_ai.view.alertas.st")
def test_panel_id_vacio_muestra_error_sin_generar(
    mock_st: MagicMock,
    mock_generate: MagicMock,
) -> None:
    mock_st.button.return_value = True
    mock_st.text_input.side_effect = ["   ", DEFAULT_LOCATION]
    priority = prioritize("Dusty", 0.82)

    render_ticket_section(priority, "Dusty", 0.82)

    mock_generate.assert_not_called()
    mock_st.error.assert_called_once()
    assert "identificador del panel" in mock_st.error.call_args.args[0].lower()


@patch("solarguard_ai.view.alertas.st")
def test_render_failed_muestra_error(mock_st: MagicMock) -> None:
    result = _make_result(
        "failed",
        message="Error de red al contactar GitHub API",
    )

    render_ticket_result(result)

    mock_st.error.assert_called_once()
    texto = mock_st.error.call_args.args[0].lower()
    assert "no se pudo generar" in texto
    assert "error de red" in texto
    mock_st.success.assert_not_called()


@patch("solarguard_ai.view.alertas.st")
def test_render_created_muestra_exito_sin_error(mock_st: MagicMock) -> None:
    ticket = _make_ticket()
    result = _make_result(
        "created",
        ticket=ticket,
        issue_number=42,
        message="Issue #42 creado exitosamente.",
    )

    render_ticket_result(result)

    mock_st.error.assert_not_called()
    mock_st.success.assert_called_once()
    markdown_calls = [c.args[0] for c in mock_st.markdown.call_args_list]
    assert any("#42" in call for call in markdown_calls)
    assert any(ticket.title in call for call in markdown_calls)


def test_validate_panel_id_normaliza_con_strip() -> None:
    assert validate_panel_id("  PV-1024  ") == "PV-1024"


@pytest.mark.parametrize(
    "panel_id",
    ["", "   ", None, 123, 0.5, ["PV-1"]],
)
def test_validate_panel_id_rechaza_entradas_invalidas(panel_id: object) -> None:
    with pytest.raises(ValueError):
        validate_panel_id(panel_id)
