"""Pruebas unitarias para el módulo de generación de tickets de mantenimiento."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest

from solarguard_ai.priorizacion import PriorityResult, prioritize
from solarguard_ai.tickets import (
    GitHubTicketClient,
    MaintenanceTicket,
    TicketCreationResult,
    TicketGenerationError,
    build_ticket,
    format_ticket_body,
    format_ticket_title,
    generate_maintenance_ticket,
    resolve_assignees,
    resolve_labels,
    should_generate_ticket,
)

# ---------------------------------------------------------------------------
# 1. Reglas de filtrado de tickets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("priority_level", "force", "expected"),
    [
        ("high", False, True),
        ("medium", False, True),
        ("low", False, False),
        ("low", True, True),
        ("high", True, True),
        ("medium", True, True),
    ],
)
def test_should_generate_ticket_evalua_severidad_y_forzado(
    priority_level: str,
    force: bool,
    expected: bool,
) -> None:
    # Arrange & Act
    resultado = should_generate_ticket(priority_level, force=force)

    # Assert
    assert resultado is expected


# ---------------------------------------------------------------------------
# 2. Asignación automática de responsables por área
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("predicted_class", "expected_assignees"),
    [
        ("Electrical-damage", ["ing-electrico"]),
        ("Physical-Damage", ["tecnico-campo"]),
        ("Dusty", ["mantenimiento-limpieza"]),
        ("Bird-drop", ["mantenimiento-limpieza"]),
        ("Snow-Covered", ["mantenimiento-limpieza"]),
        ("Unknown", ["supervisor-triaje"]),
        ("Clean", []),
    ],
)
def test_resolve_assignees_mapea_por_area_tecnica(
    predicted_class: str,
    expected_assignees: list[str],
) -> None:
    # Arrange & Act
    assignees = resolve_assignees(predicted_class)

    # Assert
    assert assignees == expected_assignees


def test_resolve_assignees_permite_mapa_personalizado() -> None:
    # Arrange
    custom_map = {"Electrical-damage": ["experto-alta-tension"]}

    # Act
    assignees = resolve_assignees("Electrical-damage", custom_map=custom_map)

    # Assert
    assert assignees == ["experto-alta-tension"]


def test_resolve_assignees_clase_desconocida_asigna_triaje() -> None:
    # Arrange & Act
    assignees = resolve_assignees("ClaseInexistente")

    # Assert
    assert assignees == ["supervisor-triaje"]


# ---------------------------------------------------------------------------
# 3. Generación y resolución de etiquetas
# ---------------------------------------------------------------------------


def test_resolve_labels_alerta_alta_incluye_urgente_y_etiquetas_clase() -> None:
    # Arrange / Act
    labels = resolve_labels(
        severity="high",
        predicted_class="Electrical-damage",
        requires_human_review=False,
    )

    # Assert
    assert "maintenance" in labels
    assert "urgent" in labels
    assert "severity:high" in labels
    assert "electrical" in labels
    assert "risk" in labels
    assert "requires-human-review" not in labels


def test_resolve_labels_revision_humana_agrega_etiqueta_correspondiente() -> None:
    # Arrange / Act
    labels = resolve_labels(
        severity="medium",
        predicted_class="Unknown",
        requires_human_review=True,
    )

    # Assert
    assert "maintenance" in labels
    assert "severity:medium" in labels
    assert "requires-human-review" in labels
    assert "urgent" not in labels


# ---------------------------------------------------------------------------
# 4. Formato de títulos y cuerpo en Markdown
# ---------------------------------------------------------------------------


def test_format_ticket_title_sigue_convencion() -> None:
    # Arrange / Act
    title = format_ticket_title(
        severity="high",
        predicted_class="Physical-Damage",
        panel_id="PANEL-B12-04",
    )

    # Assert
    assert title == "[SolarGuard - HIGH] Physical-Damage: Panel PANEL-B12-04"


def test_format_ticket_body_incluye_secciones_y_descargo() -> None:
    # Arrange
    priority_result = PriorityResult(
        priority="high",
        recommended_action="Aislar string y revisar cableado.",
        requires_human_review=False,
        reason="Posible falla en caja de conexiones.",
    )

    # Act
    body = format_ticket_body(
        priority_result=priority_result,
        panel_id="INV-01-M05",
        location="Sector Norte - Bloque 3",
        predicted_class="Electrical-damage",
        confidence=0.945,
    )

    # Assert
    assert "## Reporte Automático de Mantenimiento Fotovoltaico" in body
    assert "`INV-01-M05`" in body
    assert "Sector Norte - Bloque 3" in body
    assert "`HIGH`" in body
    assert "94.50%" in body
    assert "Aislar string y revisar cableado." in body
    assert "Aviso de Responsabilidad (SolarGuard AI)" in body


def test_format_ticket_body_incluye_alerta_cuando_requiere_revision() -> None:
    # Arrange
    priority_result = PriorityResult(
        priority="medium",
        recommended_action="Revisión presencial.",
        requires_human_review=True,
        reason="Confianza baja.",
    )

    # Act
    body = format_ticket_body(
        priority_result=priority_result,
        panel_id="PANEL-01",
        location="Bloque 1",
        predicted_class="Bird-drop",
        confidence=0.55,
    )

    # Assert
    assert "> [!WARNING]" in body
    assert "Requiere Revisión Humana Obligatoria" in body


# ---------------------------------------------------------------------------
# 5. Compilación del ticket (build_ticket)
# ---------------------------------------------------------------------------


def test_build_ticket_construye_objeto_inmutable_completo() -> None:
    # Arrange
    p_result = prioritize("Electrical-damage", 0.92)

    # Act
    ticket = build_ticket(
        priority_result=p_result,
        panel_id="PANEL-E-01",
        predicted_class="Electrical-damage",
        confidence=0.92,
        location="Granja Este",
    )

    # Assert
    assert isinstance(ticket, MaintenanceTicket)
    assert ticket.panel_id == "PANEL-E-01"
    assert ticket.severity == "high"
    assert ticket.assignees == ["ing-electrico"]
    assert "urgent" in ticket.labels
    assert "maintenance" in ticket.labels


# ---------------------------------------------------------------------------
# 6. Cliente GitHub y creación de Issues (Mocks y Simulación)
# ---------------------------------------------------------------------------


def test_github_ticket_client_dry_run_retorna_simulacion() -> None:
    # Arrange
    client = GitHubTicketClient(token="fake-token", dry_run=True)
    p_result = prioritize("Dusty", 0.88)
    ticket = build_ticket(p_result, "P-100", "Dusty", 0.88)

    # Act
    result = client.create_issue(ticket)

    # Assert
    assert isinstance(result, TicketCreationResult)
    assert result.status == "simulated"
    assert result.issue_number == 999
    assert "issues/999" in (result.issue_url or "")


def test_github_ticket_client_sin_token_retorna_simulacion() -> None:
    # Arrange (sin token y dry_run=False)
    client = GitHubTicketClient(token=None, dry_run=False)
    p_result = prioritize("Dusty", 0.88)
    ticket = build_ticket(p_result, "P-101", "Dusty", 0.88)

    # Act
    result = client.create_issue(ticket)

    # Assert
    assert result.status == "simulated"
    assert result.issue_number == 999


def test_github_ticket_client_con_requester_mock_crea_issue() -> None:
    # Arrange
    captured_requests: list[urllib.request.Request] = []

    def mock_requester(req: urllib.request.Request) -> dict[str, Any]:
        captured_requests.append(req)
        return {
            "number": 45,
            "html_url": "https://github.com/test-org/solar-repo/issues/45",
        }

    client = GitHubTicketClient(
        token="ghp_test1234567890",
        repository="test-org/solar-repo",
        dry_run=False,
        requester=mock_requester,
    )
    p_result = prioritize("Physical-Damage", 0.96)
    ticket = build_ticket(p_result, "PANEL-F-99", "Physical-Damage", 0.96)

    # Act
    result = client.create_issue(ticket)

    # Assert
    assert result.status == "created"
    assert result.issue_number == 45
    assert result.issue_url == "https://github.com/test-org/solar-repo/issues/45"
    assert len(captured_requests) == 1

    # Validar la petición HTTP enviada
    sent_req = captured_requests[0]
    assert (
        sent_req.full_url == "https://api.github.com/repos/test-org/solar-repo/issues"
    )
    assert sent_req.get_header("Authorization") == "Bearer ghp_test1234567890"
    assert sent_req.get_header("Accept") == "application/vnd.github+json"

    sent_body = json.loads(sent_req.data.decode("utf-8"))
    assert sent_body["title"] == ticket.title
    assert sent_body["labels"] == ticket.labels
    assert sent_body["assignees"] == ["tecnico-campo"]


def test_github_ticket_client_error_en_peticion_lanza_excepcion() -> None:
    # Arrange
    def failing_requester(req: urllib.request.Request) -> dict[str, Any]:
        raise TicketGenerationError("Error 403 Forbidden")

    client = GitHubTicketClient(
        token="ghp_invalid_token",
        dry_run=False,
        requester=failing_requester,
    )
    p_result = prioritize("Electrical-damage", 0.95)
    ticket = build_ticket(p_result, "P-102", "Electrical-damage", 0.95)

    # Act & Assert
    with pytest.raises(TicketGenerationError, match="Error 403 Forbidden"):
        client.create_issue(ticket)


# ---------------------------------------------------------------------------
# 7. Orquestador: generate_maintenance_ticket
# ---------------------------------------------------------------------------


def test_generate_maintenance_ticket_omite_clase_clean_por_defecto() -> None:
    # Arrange
    clean_result = prioritize("Clean", 0.98)

    # Act
    result = generate_maintenance_ticket(
        priority_result=clean_result,
        panel_id="P-CLEAN-01",
        predicted_class="Clean",
        confidence=0.98,
    )

    # Assert
    assert result.status == "skipped"
    assert "no amerita ticket" in result.message
    assert result.ticket is None


def test_generate_maintenance_ticket_fuerza_emision_para_clean() -> None:
    # Arrange
    clean_result = prioritize("Clean", 0.98)

    # Act
    result = generate_maintenance_ticket(
        priority_result=clean_result,
        panel_id="P-CLEAN-02",
        predicted_class="Clean",
        confidence=0.98,
        force=True,
    )

    # Assert
    assert result.status == "simulated"
    assert result.ticket is not None
    assert result.ticket.panel_id == "P-CLEAN-02"


def test_generate_maintenance_ticket_genera_alerta_alta_satisfactoriamente() -> None:
    # Arrange
    p_result = prioritize("Electrical-damage", 0.89)

    # Act
    result = generate_maintenance_ticket(
        priority_result=p_result,
        panel_id="P-ELEC-01",
        predicted_class="Electrical-damage",
        confidence=0.89,
        location="Subestación Este",
    )

    # Assert
    assert result.status == "simulated"
    assert result.ticket is not None
    assert result.ticket.severity == "high"
    assert result.ticket.location == "Subestación Este"
    assert "urgent" in result.ticket.labels
