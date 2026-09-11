"""
Generacion de tickets de mantenimiento para SolarGuard AI.

Cuando el modelo clasifica una imagen de panel solar, hay que
convertir ese resultado en algo util para el equipo de mantenimiento.
Eso es lo que hace este modulo: toma la clase detectada y la confianza
del modelo, y genera un "ticket" con:
  - La condicion visual del panel
  - La prioridad de atencion
  - La accion recomendada
  - La fecha y hora en que se genero el ticket

Aprendi que esto se conoce como "capa de negocio" porque traduce
los resultados tecnicos del modelo en decisiones practicas.
"""

from __future__ import annotations

# dataclass es una forma comoda de crear clases que solo guardan datos
# frozen=True significa que una vez creado el ticket, no se puede modificar
from dataclasses import dataclass

# Para registrar la fecha y hora del ticket
from datetime import datetime

# Esta es una forma de crear "constantes de texto" en Python
# Se llama Literal y sirve para decirle al codigo que valores son validos
from typing import Literal


# --- Tipos de condicion que puede detectar el modelo ---
# Estos nombres vienen del README y deben coincidir exactamente
# con lo que devuelve el modelo solarscan-yolov8n-cls

CondicionPanel = Literal[
    "Clean",
    "Dusty",
    "Bird-drop",
    "Physical-Damage",
    "Electrical-damage",
    "Snow-Covered",
]

# --- Niveles de prioridad que usaremos en los tickets ---
Prioridad = Literal["Critica", "Alta", "Media", "Baja"]


# --- Reglas de prioridad y accion segun la condicion detectada ---
# Esto viene directamente de la tabla del README.
# Lo guardo en un diccionario para no tener que escribir muchos if/elif.
# Clave: nombre de la clase del modelo
# Valor: (prioridad, descripcion de la condicion, accion recomendada)

REGLAS_POR_CONDICION: dict[str, tuple[Prioridad, str, str]] = {
    "Clean": (
        "Baja",
        "Panel visualmente limpio",
        "Sin intervencion inmediata. Continuar con monitoreo rutinario.",
    ),
    "Dusty": (
        "Media",
        "Acumulacion visible de polvo",
        "Programar limpieza en el proximo ciclo de mantenimiento.",
    ),
    "Bird-drop": (
        "Media",
        "Presencia de excremento de aves",
        "Programar limpieza. El excremento puede reducir la eficiencia del panel.",
    ),
    "Physical-Damage": (
        "Alta",
        "Dano fisico visible en el panel",
        "Priorizar inspeccion presencial. Evaluar si requiere reparacion o reemplazo.",
    ),
    "Electrical-damage": (
        "Critica",
        "Posible dano electrico visible",
        "Priorizar revision tecnica urgente. No operar hasta confirmar seguridad.",
    ),
    "Snow-Covered": (
        "Baja",
        "Panel cubierto por nieve",
        "Registrar la condicion. Baja relevancia para el contexto local.",
    ),
}


@dataclass(frozen=True)
class TicketMantenimiento:
    """
    Representa un ticket de mantenimiento generado a partir
    de la clasificacion del modelo.

    Contiene toda la informacion necesaria para que el equipo
    de mantenimiento sepa que hacer con un panel solar.
    """

    # De donde viene la imagen analizada (nombre del archivo o fuente)
    fuente: str

    # La condicion que detecto el modelo (ej: "Dusty", "Clean")
    condicion: str

    # Descripcion en texto de lo que significa esa condicion
    descripcion: str

    # Que tan seguro esta el modelo (entre 0.0 y 1.0)
    confianza: float

    # Que tan urgente es atender este panel
    prioridad: Prioridad

    # Que debe hacer el equipo de mantenimiento
    accion: str

    # Cuando se genero este ticket
    fecha_hora: datetime

    def __str__(self) -> str:
        """
        Representacion legible del ticket para mostrarlo en pantalla.
        Esto me ayuda cuando quiero imprimirlo rapidamente durante pruebas.
        """
        # Formateo la fecha de forma amigable: "10/09/2026 18:30"
        fecha_formateada = self.fecha_hora.strftime("%d/%m/%Y %H:%M")
        confianza_porcentaje = self.confianza * 100

        return (
            f"--- TICKET DE MANTENIMIENTO ---\n"
            f"Fuente      : {self.fuente}\n"
            f"Condicion   : {self.condicion} ({self.descripcion})\n"
            f"Confianza   : {confianza_porcentaje:.1f}%\n"
            f"Prioridad   : {self.prioridad}\n"
            f"Accion      : {self.accion}\n"
            f"Fecha/Hora  : {fecha_formateada}\n"
            f"-------------------------------"
        )


def generar_ticket(
    fuente: str,
    condicion: str,
    confianza: float,
) -> TicketMantenimiento:
    """
    Genera un ticket de mantenimiento a partir del resultado del modelo.

    Recibe:
      - fuente: nombre del archivo o imagen analizada
      - condicion: clase que predijo el modelo (ej: "Dusty")
      - confianza: que tan seguro esta el modelo (entre 0.0 y 1.0)

    Devuelve un TicketMantenimiento con la prioridad y accion correspondiente.

    Lanza un ValueError si la condicion no es una de las 6 clases conocidas.
    """
    # Valido que la confianza este en el rango correcto
    if not (0.0 <= confianza <= 1.0):
        raise ValueError(
            f"La confianza debe estar entre 0.0 y 1.0, "
            f"pero se recibio: {confianza}"
        )

    # Busco las reglas para esta condicion
    # Si la condicion no esta en el diccionario, el modelo dio algo inesperado
    if condicion not in REGLAS_POR_CONDICION:
        clases_validas = ", ".join(REGLAS_POR_CONDICION.keys())
        raise ValueError(
            f"Condicion desconocida: '{condicion}'. "
            f"Las condiciones validas son: {clases_validas}."
        )

    # Saco la prioridad, descripcion y accion del diccionario
    prioridad, descripcion, accion = REGLAS_POR_CONDICION[condicion]

    # Creo y devuelvo el ticket con la fecha y hora actual
    return TicketMantenimiento(
        fuente=fuente,
        condicion=condicion,
        descripcion=descripcion,
        confianza=confianza,
        prioridad=prioridad,
        accion=accion,
        fecha_hora=datetime.now(),
    )


def generar_tickets_lote(
    resultados: list[tuple[str, str, float]],
) -> list[TicketMantenimiento]:
    """
    Genera varios tickets de una vez cuando se analizan varias imagenes.

    Recibe una lista de tuplas donde cada una tiene:
      (fuente, condicion, confianza)

    Devuelve una lista de tickets en el mismo orden.

    Esto es util cuando se suben varias fotos de paneles al mismo tiempo.
    """
    tickets = []

    for fuente, condicion, confianza in resultados:
        ticket = generar_ticket(fuente, condicion, confianza)
        tickets.append(ticket)

    return tickets


def filtrar_por_prioridad(
    tickets: list[TicketMantenimiento],
    prioridad: Prioridad,
) -> list[TicketMantenimiento]:
    """
    Filtra una lista de tickets y devuelve solo los de una prioridad especifica.

    Por ejemplo, si quiero ver solo los paneles con dano critico:
        urgentes = filtrar_por_prioridad(tickets, "Critica")

    Esto ayuda al lider de mantenimiento a saber por donde empezar.
    """
    # List comprehension: manera corta de filtrar una lista en Python
    return [t for t in tickets if t.prioridad == prioridad]


def ordenar_por_prioridad(
    tickets: list[TicketMantenimiento],
) -> list[TicketMantenimiento]:
    """
    Ordena una lista de tickets de mas urgente a menos urgente.

    El orden es: Critica → Alta → Media → Baja

    Aprendi que esto se puede hacer con sorted() y un diccionario
    que asigna un numero a cada prioridad.
    """
    # Le asigno un numero a cada nivel: menor numero = mas urgente
    orden = {"Critica": 0, "Alta": 1, "Media": 2, "Baja": 3}

    return sorted(tickets, key=lambda ticket: orden[ticket.prioridad])


# Lo que este modulo exporta para que otros archivos lo puedan usar
__all__ = [
    "CondicionPanel",
    "Prioridad",
    "TicketMantenimiento",
    "REGLAS_POR_CONDICION",
    "generar_ticket",
    "generar_tickets_lote",
    "filtrar_por_prioridad",
    "ordenar_por_prioridad",
]
