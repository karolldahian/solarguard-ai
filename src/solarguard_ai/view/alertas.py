"""Visualizacion de alertas y tickets de mantenimiento en Streamlit.

Este modulo encapsula unicamente la presentacion de ``TicketCreationResult``
generado por ``generate_maintenance_ticket``. No reimplementa las reglas de
tickets (viven en ``solarguard_ai.tickets``) ni ejecuta llamadas de red: en
esta etapa siempre se trabaja en modo dry-run sin cliente GitHub.

La capa visual permanece desacoplada de la fuente de datos: recibe valores
simples (clase, confianza) y un ``PriorityResult`` ya calculado, y no depende
de gRPC ni de ONNX Runtime en tiempo de ejecucion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from solarguard_ai.tickets import DEFAULT_LOCATION, generate_maintenance_ticket

if TYPE_CHECKING:
    from solarguard_ai.priorizacion import PriorityResult
    from solarguard_ai.tickets import MaintenanceTicket, TicketCreationResult

_TICKET_DISCLAIMER = (
    "Alerta preliminar generada por IA: no constituye un diagnostico electrico "
    "definitivo; los danos fisicos o electricos requieren validacion por "
    "personal tecnico calificado."
)

_SIMULATION_BANNER = (
    "SIMULACION: este ticket no fue creado en GitHub. No se emitio ningun Issue real."
)


def validate_panel_id(panel_id: object) -> str:
    """Valida y normaliza el identificador del panel.

    El identificador es obligatorio y no debe asumirse desde el nombre del
    archivo de imagen.

    Args:
        panel_id: identificador ingresado por el usuario.

    Returns:
        Identificador normalizado (sin espacios al inicio ni al final).

    Raises:
        ValueError: si no es una cadena o queda vacio tras normalizar.
    """
    if not isinstance(panel_id, str):
        # Contrato UI confirmado: ValueError unico para entradas invalidas
        # (no se distingue TypeError) para simplificar el manejo en Streamlit.
        raise ValueError("El identificador del panel es obligatorio.")  # noqa: TRY004
    normalized = panel_id.strip()
    if not normalized:
        raise ValueError("El identificador del panel no puede estar vacio.")
    return normalized


def render_ticket_section(
    priority: PriorityResult,
    predicted_class: str,
    confidence: float,
) -> None:
    """Solicita los datos del panel y genera el ticket en modo seguro.

    No crea ni recibe un cliente GitHub: ``generate_maintenance_ticket`` usa
    su comportamiento dry-run por defecto (``client=None``), por lo que nunca
    se emite un Issue real.

    Args:
        priority: resultado de priorizacion ya calculado.
        predicted_class: clase predicha por el modelo.
        confidence: confianza de la prediccion (0.0 a 1.0).
    """
    st.subheader("Alerta / ticket de mantenimiento")
    st.markdown(f"_{_TICKET_DISCLAIMER}_")

    panel_id_input = st.text_input(
        "Identificador del panel",
        help="Identificador fisico real del panel; no se deduce del archivo.",
    )
    location_input = st.text_input("Ubicacion", value=DEFAULT_LOCATION)

    if st.button("Generar ticket (simulacion)"):
        try:
            panel_id = validate_panel_id(panel_id_input)
        except ValueError as error:
            st.error(str(error))
            return

        result = generate_maintenance_ticket(
            priority_result=priority,
            panel_id=panel_id,
            predicted_class=predicted_class,
            confidence=confidence,
            location=location_input,
        )
        render_ticket_result(result)


def render_ticket_result(result: TicketCreationResult) -> None:
    """Presenta el resultado de la generacion de ticket segun su status.

    ``simulated`` y ``created`` comparten la vista del ticket: solo cambia el
    encabezado. ``skipped`` no fabrica un ticket visual falso y ``failed`` se
    muestra como error.

    Args:
        result: resultado emitido por ``generate_maintenance_ticket``.
    """
    if result.status == "simulated":
        st.warning(_SIMULATION_BANNER)
        st.markdown(
            "Se genero una simulacion del ticket: validar su contenido antes "
            "de emitir un Issue real en GitHub."
        )
        st.markdown(f"**Estado:** {result.status}")
        if result.ticket is not None:
            _render_ticket_details(result.ticket)
        return

    if result.status == "created":
        st.success("Ticket creado exitosamente.")
        st.markdown(f"**Mensaje:** {result.message}")
        if result.issue_number is not None:
            st.markdown(f"**Issue:** #{result.issue_number}")
        if result.issue_url:
            st.markdown(f"[Ver Issue]({result.issue_url})")
        if result.ticket is not None:
            _render_ticket_details(result.ticket)
        return

    if result.status == "skipped":
        st.info(
            "De acuerdo con las reglas actuales de priorizacion, este "
            "resultado no requiere ticket automatico de mantenimiento."
        )
        return

    st.error(f"No se pudo generar el ticket: {result.message}")


def _render_ticket_details(ticket: MaintenanceTicket) -> None:
    """Muestra los campos reales de un ``MaintenanceTicket`` de forma amigable."""
    st.subheader("Detalle del ticket")
    st.markdown(f"**Titulo:** {ticket.title}")
    st.markdown(f"**Identificador del panel:** `{ticket.panel_id}`")
    st.markdown(f"**Ubicacion:** {ticket.location}")
    st.markdown(f"**Condicion detectada:** {ticket.predicted_class}")
    st.metric(label="Confianza", value=f"{ticket.confidence:.2%}")
    st.markdown(f"**Severidad / prioridad:** {ticket.severity}")

    review_text = "Si" if ticket.requires_human_review else "No"
    st.markdown(f"**Revision humana requerida:** {review_text}")
    if ticket.requires_human_review:
        st.warning("Se requiere revision humana antes de actuar sobre este resultado.")

    if ticket.assignees:
        st.markdown(f"**Responsables:** {', '.join(ticket.assignees)}")
    else:
        st.markdown("**Responsables:** Sin responsables asignados")

    if ticket.labels:
        st.markdown(f"**Etiquetas:** {', '.join(ticket.labels)}")

    st.markdown("**Accion / informacion relevante:**")
    st.markdown(ticket.body)


__all__ = [
    "render_ticket_result",
    "render_ticket_section",
    "validate_panel_id",
]
