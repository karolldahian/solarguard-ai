"""Generación automática de tickets de mantenimiento para SolarGuard AI."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from solarguard_ai.priorizacion import PriorityResult

DEFAULT_GITHUB_REPO = "karolldahian/solarguard-ai"
DEFAULT_LOCATION = "Parque Solar - Bloque General"

DEFAULT_ASSIGNEE_MAP: dict[str, list[str]] = {
    "Electrical-damage": ["ing-electrico"],
    "Physical-Damage": ["tecnico-campo"],
    "Dusty": ["mantenimiento-limpieza"],
    "Bird-drop": ["mantenimiento-limpieza"],
    "Snow-Covered": ["mantenimiento-limpieza"],
    "Unknown": ["supervisor-triaje"],
    "Clean": [],
}

LABEL_SEVERITY_MAP: dict[str, str] = {
    "high": "severity:high",
    "medium": "severity:medium",
    "low": "severity:low",
}

LABEL_CLASS_MAP: dict[str, list[str]] = {
    "Electrical-damage": ["electrical", "risk"],
    "Physical-Damage": ["structural", "hardware"],
    "Dusty": ["cleaning", "preventive"],
    "Bird-drop": ["cleaning", "biological"],
    "Snow-Covered": ["cleaning", "weather"],
    "Unknown": ["triaje", "review-needed"],
    "Clean": ["routine"],
}

SAFETY_RECOMMENDATIONS: dict[str, list[str]] = {
    "Electrical-damage": [
        "Aislar de inmediato la sección del string o inversor antes de intervenir.",
        "Verificar ausencia de tensión con voltímetro calibrado y usar EPP dieléctrico clase 0.",
        "Inspeccionar caja de conexiones (junction box) y cableado por sobrecalentamiento.",
    ],
    "Physical-Damage": [
        "Inspeccionar integridad mecánica del vidrio templado y marco de aluminio.",
        "Evaluar presencia de microfisuras o delaminación mediante electroluminiscencia o inspección visual.",
        "Señalizar el panel para evitar pisadas o cargas mecánicas adicionales.",
    ],
    "Dusty": [
        "Programar limpieza en horarios de baja irradiancia (madrugada o atardecer) para evitar choque térmico.",
        "Utilizar agua desmineralizada y cepillos de cerdas suaves no abrasivas.",
    ],
    "Bird-drop": [
        "Remover incrustaciones orgánicas para mitigar puntos calientes (hotspots).",
        "No raspar con herramientas metálicas ni químicos corrosivos.",
    ],
    "Snow-Covered": [
        "Evaluar sobrecarga de peso en la estructura de soporte.",
        "Retirar la nieve con palas de goma suave sin golpear la superficie de vidrio.",
    ],
    "Unknown": [
        "Realizar nueva captura fotográfica con mejor iluminación y ángulo perpendicular.",
        "Verificar presencialmente el estado del módulo fotovoltaico.",
    ],
    "Clean": [
        "Mantener registro en bitácora de inspección rutinaria.",
    ],
}


class TicketGenerationError(Exception):
    """Indica un fallo en la generación o envío de un ticket de mantenimiento."""


@dataclass(frozen=True)
class MaintenanceTicket:
    """Representa un ticket de mantenimiento generado por el sistema."""

    title: str
    body: str
    labels: list[str]
    assignees: list[str]
    severity: str
    panel_id: str
    location: str
    predicted_class: str
    confidence: float
    requires_human_review: bool
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass(frozen=True)
class TicketCreationResult:
    """Resultado de la operación de creación o simulación de ticket."""

    status: str  # "created", "skipped", "simulated", "failed"
    message: str
    ticket: MaintenanceTicket | None = None
    issue_number: int | None = None
    issue_url: str | None = None


def should_generate_ticket(priority_level: str, force: bool = False) -> bool:
    """Determina si una severidad amerita generación automática de ticket.

    Por regla de negocio, únicamente las alertas de prioridad 'high' y 'medium'
    generan tickets de mantenimiento correctivo/preventivo. Prioridad 'low' (Clean)
    se omite a menos que se indique `force=True`.

    Args:
        priority_level: Nivel de prioridad ('high', 'medium', 'low').
        force: Si True, fuerza la generación independientemente del nivel.

    Returns:
        True si debe generarse el ticket, False en caso contrario.
    """
    if force:
        return True
    return priority_level in {"high", "medium"}


def resolve_assignees(
    predicted_class: str,
    custom_map: dict[str, list[str]] | None = None,
) -> list[str]:
    """Asigna responsables automáticos según el área técnica de la anomalía.

    Args:
        predicted_class: Clase predicha por el modelo.
        custom_map: Diccionario opcional para personalizar asignaciones.

    Returns:
        Lista de usuarios responsables (usernames de GitHub sin '@').
    """
    assignee_map = custom_map or DEFAULT_ASSIGNEE_MAP
    return list(assignee_map.get(predicted_class, ["supervisor-triaje"]))


def resolve_labels(
    severity: str,
    predicted_class: str,
    requires_human_review: bool,
) -> list[str]:
    """Genera las etiquetas descriptivas automáticas para el ticket.

    Args:
        severity: Nivel de severidad ('high', 'medium', 'low').
        predicted_class: Clase detectada en el panel.
        requires_human_review: Si requiere verificación de un técnico humano.

    Returns:
        Lista de etiquetas únicas ordenadas.
    """
    labels: set[str] = {"maintenance"}

    if severity == "high":
        labels.add("urgent")

    severity_label = LABEL_SEVERITY_MAP.get(severity)
    if severity_label:
        labels.add(severity_label)

    class_labels = LABEL_CLASS_MAP.get(predicted_class, [])
    labels.update(class_labels)

    if requires_human_review:
        labels.add("requires-human-review")

    return sorted(labels)


def format_ticket_title(
    severity: str,
    predicted_class: str,
    panel_id: str,
) -> str:
    """Genera el título estructurado y formal para el ticket de soporte.

    Args:
        severity: Nivel de severidad ('high', 'medium', 'low').
        predicted_class: Clase de anomalía detectada.
        panel_id: Identificador único del panel solar.

    Returns:
        Cadena con formato estandarizado.
    """
    return f"[SolarGuard - {severity.upper()}] {predicted_class}: Panel {panel_id}"


def format_ticket_body(
    priority_result: PriorityResult,
    panel_id: str,
    location: str,
    predicted_class: str,
    confidence: float,
) -> str:
    """Construye el cuerpo detallado del ticket en formato Markdown.

    Args:
        priority_result: Objeto con la prioridad, acción y justificación.
        panel_id: Identificador del panel solar.
        location: Ubicación geográfica o bloque dentro del parque solar.
        predicted_class: Clase predicha por el clasificador.
        confidence: Nivel de confianza numérico (0.0 a 1.0).

    Returns:
        Texto en Markdown formateado con tablas y recomendaciones de seguridad.
    """
    recs = SAFETY_RECOMMENDATIONS.get(
        predicted_class, SAFETY_RECOMMENDATIONS["Unknown"]
    )
    recommendations_list = "\n".join(f"- {rec}" for rec in recs)

    review_alert = ""
    if priority_result.requires_human_review:
        review_alert = (
            "> [!WARNING]\n"
            "> **Requiere Revisión Humana Obligatoria:** La predicción presenta baja "
            f"confianza ({confidence:.2%}) o clasificación incierta ('{predicted_class}'). "
            "No ejecutar reemplazos sin inspección visual directa.\n\n"
        )

    return f"""## Reporte Automático de Mantenimiento Fotovoltaico

{review_alert}### 1. Resumen de la Alerta
| Campo | Detalle |
| :--- | :--- |
| **Identificador de Panel** | `{panel_id}` |
| **Ubicación** | {location} |
| **Severidad Asignada** | `{priority_result.priority.upper()}` |
| **Diagnóstico Visual** | `{predicted_class}` |
| **Nivel de Confianza** | `{confidence:.2%}` |
| **Revisión Humana Requerida** | `{"Sí" if priority_result.requires_human_review else "No"}` |

---

### 2. Acción Requerida
**{priority_result.recommended_action}**

*Justificación Técnica:* {priority_result.reason}

---

### 3. Procedimiento y Seguridad Operativa
{recommendations_list}

---

> [!NOTE]
> **Aviso de Responsabilidad (SolarGuard AI):**
> Este ticket fue generado automáticamente a partir de clasificación visual realizada
> por el modelo `solarscan-yolov8n-cls`. Los resultados constituyen una alerta preliminar
> y no representan un diagnóstico eléctrico definitivo ni sustituyen la verificación de un técnico calificado.
"""


def build_ticket(
    priority_result: PriorityResult,
    panel_id: str,
    predicted_class: str,
    confidence: float,
    location: str = DEFAULT_LOCATION,
    custom_assignees: dict[str, list[str]] | None = None,
) -> MaintenanceTicket:
    """Compila un objeto inmutable MaintenanceTicket listo para emisión.

    Args:
        priority_result: Resultado del análisis de priorización.
        panel_id: Identificador del panel.
        predicted_class: Clase predicha.
        confidence: Confianza del modelo.
        location: Ubicación física del panel.
        custom_assignees: Mapa opcional de personal técnico.

    Returns:
        Instancia de MaintenanceTicket.
    """
    title = format_ticket_title(
        severity=priority_result.priority,
        predicted_class=predicted_class,
        panel_id=panel_id,
    )
    body = format_ticket_body(
        priority_result=priority_result,
        panel_id=panel_id,
        location=location,
        predicted_class=predicted_class,
        confidence=confidence,
    )
    labels = resolve_labels(
        severity=priority_result.priority,
        predicted_class=predicted_class,
        requires_human_review=priority_result.requires_human_review,
    )
    assignees = resolve_assignees(
        predicted_class=predicted_class,
        custom_map=custom_assignees,
    )

    return MaintenanceTicket(
        title=title,
        body=body,
        labels=labels,
        assignees=assignees,
        severity=priority_result.priority,
        panel_id=panel_id,
        location=location,
        predicted_class=predicted_class,
        confidence=confidence,
        requires_human_review=priority_result.requires_human_review,
    )


HttpRequester = Callable[[urllib.request.Request], dict[str, Any]]


def _default_http_requester(request: urllib.request.Request) -> dict[str, Any]:
    """Ejecuta una petición HTTP a la API de GitHub."""
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            data = response.read().decode("utf-8")
            return json.loads(data) if data else {}
    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        raise TicketGenerationError(
            f"Error HTTP {error.code} al contactar GitHub API: {error_body}"
        ) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise TicketGenerationError(
            f"Error de red al contactar GitHub API: {error}"
        ) from error


class GitHubTicketClient:
    """Cliente para la creación de tickets mediante la API REST de GitHub."""

    def __init__(
        self,
        token: str | None = None,
        repository: str = DEFAULT_GITHUB_REPO,
        dry_run: bool = False,
        requester: HttpRequester = _default_http_requester,
    ) -> None:
        """Inicializa el cliente.

        Args:
            token: Token de acceso personal de GitHub (o variable GITHUB_TOKEN).
            repository: Repositorio en formato 'owner/repo'.
            dry_run: Si True, no envía peticiones y simula la respuesta.
            requester: Función inyectable para peticiones HTTP (útil en testing).
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.repository = repository
        self.dry_run = dry_run
        self._requester = requester

    def create_issue(self, ticket: MaintenanceTicket) -> TicketCreationResult:
        """Crea un GitHub Issue a partir del ticket suministrado.

        Args:
            ticket: Instancia de MaintenanceTicket.

        Returns:
            TicketCreationResult con número de issue y URL.

        Raises:
            TicketGenerationError: Si la petición falla o no hay autenticación.
        """
        if self.dry_run or not self.token:
            # Modo simulación seguro para desarrollo local sin tokens
            simulated_number = 999
            simulated_url = (
                f"https://github.com/{self.repository}/issues/{simulated_number}"
            )
            return TicketCreationResult(
                status="simulated",
                message=(
                    "Ticket simulado exitosamente (modo dry-run o sin GITHUB_TOKEN)."
                ),
                ticket=ticket,
                issue_number=simulated_number,
                issue_url=simulated_url,
            )

        payload: dict[str, Any] = {
            "title": ticket.title,
            "body": ticket.body,
            "labels": ticket.labels,
        }
        if ticket.assignees:
            payload["assignees"] = ticket.assignees

        url = f"https://api.github.com/repos/{self.repository}/issues"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "SolarGuard-AI-TicketService",
            "Content-Type": "application/json",
        }

        req = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        response_data = self._requester(req)
        issue_number = response_data.get("number")
        issue_url = response_data.get("html_url")

        return TicketCreationResult(
            status="created",
            message=f"Issue #{issue_number} creado exitosamente.",
            ticket=ticket,
            issue_number=issue_number,
            issue_url=issue_url,
        )


def generate_maintenance_ticket(
    priority_result: PriorityResult,
    panel_id: str,
    predicted_class: str,
    confidence: float,
    location: str = DEFAULT_LOCATION,
    client: GitHubTicketClient | None = None,
    force: bool = False,
) -> TicketCreationResult:
    """Orquesta la evaluación y generación automática de tickets de mantenimiento.

    Args:
        priority_result: Resultado emitido por el módulo de priorización.
        panel_id: Identificador del módulo o string fotovoltaico.
        predicted_class: Condición detectada.
        confidence: Confianza del modelo.
        location: Ubicación del panel.
        client: Cliente GitHub configurado (usa simulación por defecto si es None).
        force: Si True, emite el ticket aunque sea de baja prioridad (Clean).

    Returns:
        TicketCreationResult indicando si fue creado, simulado u omitido.
    """
    if not should_generate_ticket(priority_result.priority, force=force):
        return TicketCreationResult(
            status="skipped",
            message=(
                f"Alerta de prioridad '{priority_result.priority}' no amerita "
                "ticket automático de mantenimiento correctivo."
            ),
        )

    ticket = build_ticket(
        priority_result=priority_result,
        panel_id=panel_id,
        predicted_class=predicted_class,
        confidence=confidence,
        location=location,
    )

    ticket_client = client or GitHubTicketClient(dry_run=True)
    return ticket_client.create_issue(ticket)


__all__ = [
    "DEFAULT_ASSIGNEE_MAP",
    "DEFAULT_GITHUB_REPO",
    "DEFAULT_LOCATION",
    "LABEL_CLASS_MAP",
    "LABEL_SEVERITY_MAP",
    "GitHubTicketClient",
    "MaintenanceTicket",
    "TicketCreationResult",
    "TicketGenerationError",
    "build_ticket",
    "format_ticket_body",
    "format_ticket_title",
    "generate_maintenance_ticket",
    "resolve_assignees",
    "resolve_labels",
    "should_generate_ticket",
]
