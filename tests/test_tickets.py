"""
Pruebas para el modulo de tickets de mantenimiento.

Aqui verifico que:
  - Los tickets se generan correctamente para cada condicion
  - Las prioridades son las correctas segun el README
  - Los errores se manejan bien cuando llegan datos invalidos
  - Las funciones de filtrado y ordenamiento funcionan

Para correr estas pruebas:
    uv run python -m pytest tests/test_tickets.py -v
"""

from datetime import datetime

import pytest

from solarguard_ai.tickets import (
    TicketMantenimiento,
    filtrar_por_prioridad,
    generar_ticket,
    generar_tickets_lote,
    ordenar_por_prioridad,
)


# --- Pruebas de generar_ticket ---

def test_panel_limpio_tiene_prioridad_baja() -> None:
    """
    Un panel limpio no necesita atencion urgente, su prioridad debe ser Baja.
    """
    ticket = generar_ticket(
        fuente="panel_01.jpg",
        condicion="Clean",
        confianza=0.95,
    )

    assert ticket.prioridad == "Baja"
    assert ticket.condicion == "Clean"
    assert ticket.confianza == 0.95


def test_panel_polvoriento_tiene_prioridad_media() -> None:
    """
    Un panel con polvo debe programarse para limpieza: prioridad Media.
    """
    ticket = generar_ticket(
        fuente="panel_02.jpg",
        condicion="Dusty",
        confianza=0.80,
    )

    assert ticket.prioridad == "Media"


def test_panel_con_excremento_tiene_prioridad_media() -> None:
    """
    El excremento de aves tambien requiere limpieza: prioridad Media.
    """
    ticket = generar_ticket(
        fuente="panel_03.jpg",
        condicion="Bird-drop",
        confianza=0.75,
    )

    assert ticket.prioridad == "Media"


def test_panel_con_dano_fisico_tiene_prioridad_alta() -> None:
    """
    El dano fisico visible es urgente: prioridad Alta.
    """
    ticket = generar_ticket(
        fuente="panel_04.jpg",
        condicion="Physical-Damage",
        confianza=0.88,
    )

    assert ticket.prioridad == "Alta"


def test_panel_con_dano_electrico_tiene_prioridad_critica() -> None:
    """
    El dano electrico es lo mas urgente: prioridad Critica.
    Este tipo de panel puede ser peligroso para el personal.
    """
    ticket = generar_ticket(
        fuente="panel_05.jpg",
        condicion="Electrical-damage",
        confianza=0.91,
    )

    assert ticket.prioridad == "Critica"


def test_panel_con_nieve_tiene_prioridad_baja() -> None:
    """
    La nieve es de baja relevancia en contexto local: prioridad Baja.
    """
    ticket = generar_ticket(
        fuente="panel_06.jpg",
        condicion="Snow-Covered",
        confianza=0.70,
    )

    assert ticket.prioridad == "Baja"


def test_ticket_tiene_fecha_hora() -> None:
    """
    El ticket debe registrar cuando fue generado.
    Verifico que la fecha exista y sea de tipo datetime.
    """
    ticket = generar_ticket("panel.jpg", "Clean", 0.90)

    assert isinstance(ticket.fecha_hora, datetime)


def test_ticket_guarda_la_fuente_correctamente() -> None:
    """
    El ticket debe recordar de que imagen vino el resultado.
    """
    ticket = generar_ticket("mi_panel_solar_01.png", "Dusty", 0.82)

    assert ticket.fuente == "mi_panel_solar_01.png"


def test_ticket_tiene_accion_recomendada() -> None:
    """
    El ticket siempre debe tener una accion que no este vacia.
    """
    ticket = generar_ticket("panel.jpg", "Physical-Damage", 0.77)

    # La accion no debe ser vacia ni None
    assert ticket.accion
    assert len(ticket.accion) > 0


def test_ticket_texto_imprimible() -> None:
    """
    El metodo __str__ debe devolver algo legible con la info principal.
    """
    ticket = generar_ticket("panel.jpg", "Dusty", 0.85)
    texto = str(ticket)

    # Verifico que el texto contenga los campos importantes
    assert "Dusty" in texto
    assert "Media" in texto
    assert "panel.jpg" in texto


# --- Pruebas de validaciones ---

def test_condicion_desconocida_lanza_error() -> None:
    """
    Si el modelo devuelve una clase que no conocemos, debe lanzar un error
    en lugar de generar un ticket con datos incorrectos.
    """
    with pytest.raises(ValueError, match="Condicion desconocida"):
        generar_ticket("panel.jpg", "Condition-Inventada", 0.90)


def test_confianza_mayor_a_uno_lanza_error() -> None:
    """
    Una confianza mayor a 1.0 no tiene sentido (el maximo es 100%).
    El codigo debe rechazarla.
    """
    with pytest.raises(ValueError, match="entre 0.0 y 1.0"):
        generar_ticket("panel.jpg", "Clean", 1.5)


def test_confianza_negativa_lanza_error() -> None:
    """
    Una confianza negativa tampoco tiene sentido.
    """
    with pytest.raises(ValueError, match="entre 0.0 y 1.0"):
        generar_ticket("panel.jpg", "Clean", -0.1)


# --- Pruebas de generar_tickets_lote ---

def test_lote_genera_varios_tickets() -> None:
    """
    Con tres imagenes debo obtener tres tickets.
    """
    resultados = [
        ("panel_01.jpg", "Clean", 0.95),
        ("panel_02.jpg", "Dusty", 0.80),
        ("panel_03.jpg", "Electrical-damage", 0.91),
    ]

    tickets = generar_tickets_lote(resultados)

    assert len(tickets) == 3
    # Verifico que el orden se mantenga
    assert tickets[0].condicion == "Clean"
    assert tickets[1].condicion == "Dusty"
    assert tickets[2].condicion == "Electrical-damage"


def test_lote_vacio_devuelve_lista_vacia() -> None:
    """
    Si no hay imagenes, el resultado debe ser una lista vacia (no un error).
    """
    tickets = generar_tickets_lote([])

    assert tickets == []


# --- Pruebas de filtrar_por_prioridad ---

def test_filtrar_solo_tickets_criticos() -> None:
    """
    Si hay tickets de distintas prioridades, filtrar por Critica
    debe devolver solo los criticos.
    """
    resultados = [
        ("panel_01.jpg", "Clean", 0.95),
        ("panel_02.jpg", "Electrical-damage", 0.88),
        ("panel_03.jpg", "Physical-Damage", 0.77),
        ("panel_04.jpg", "Electrical-damage", 0.92),
    ]
    tickets = generar_tickets_lote(resultados)

    criticos = filtrar_por_prioridad(tickets, "Critica")

    assert len(criticos) == 2
    assert all(t.prioridad == "Critica" for t in criticos)


def test_filtrar_sin_coincidencias_devuelve_lista_vacia() -> None:
    """
    Si ninguno tiene la prioridad buscada, devuelve lista vacia.
    """
    tickets = generar_tickets_lote([("p.jpg", "Clean", 0.9)])

    criticos = filtrar_por_prioridad(tickets, "Critica")

    assert criticos == []


# --- Pruebas de ordenar_por_prioridad ---

def test_ordenar_pone_criticos_primero() -> None:
    """
    Al ordenar, los tickets criticos deben aparecer primero
    y los de baja prioridad al final.
    """
    resultados = [
        ("p1.jpg", "Clean", 0.9),           # Baja
        ("p2.jpg", "Dusty", 0.8),            # Media
        ("p3.jpg", "Electrical-damage", 0.9), # Critica
        ("p4.jpg", "Physical-Damage", 0.85), # Alta
    ]
    tickets = generar_tickets_lote(resultados)

    ordenados = ordenar_por_prioridad(tickets)

    # El primero debe ser Critica y el ultimo Baja
    assert ordenados[0].prioridad == "Critica"
    assert ordenados[-1].prioridad == "Baja"
