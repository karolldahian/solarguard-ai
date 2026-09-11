"""
Cliente gRPC de SolarGuard AI (la parte que usa la interfaz).

Este modulo contiene las funciones que la interfaz (Streamlit o una
consola) usa para hablar con el backend. Aprendi que en gRPC el cliente
es un "stub": una copia local del servicio que sabe como traducir
nuestras funciones en peticiones de red.

Las funciones de aqui devuelven diccionarios normales de Python para
que la interfaz no tenga que conocer los detalles de los mensajes del
proto. La idea es que Streamlit se concentre en dibujar y este modulo
en comunicarse.
"""

from __future__ import annotations

from typing import TypeAlias

import grpc

from solarguard_ai.grpc_interface import solarguard_pb2, solarguard_pb2_grpc

# Direccion donde corre el backend por defecto.
# En la aplicacion puede cambiarse desde la barra lateral.
DIRECCION_POR_DEFECTO = "localhost:50051"

# Tipo util para las opciones de canal: asi no hay que repetir la firma.
Channel: TypeAlias = grpc.Channel

# Tiempo maximo (en segundos) que esperamos una respuesta del backend
# antes de decirle al usuario que tardamos demasiado.
TIEMPO_ESPERA = 30.0


def crear_canal(direccion: str = DIRECCION_POR_DEFECTO) -> Channel:
    """
    Abre un canal de comunicacion hacia el backend.

    El canal no se conecta todavia: eso pasa con la primera peticion.
    Esto permite crear el canal sin que explote si el backend esta apagado.
    """
    return grpc.insecure_channel(direccion)


def clasificar_imagen(
    canal: Channel,
    bytes_imagen: bytes,
    nombre: str,
    tiempo_espera: float = TIEMPO_ESPERA,
) -> dict:
    """
    Envia una imagen al backend y devuelve el ticket como diccionario.

    Recibe:
      - canal: creado con crear_canal().
      - bytes_imagen: la foto ya leida en bytes (jpg, png o tif).
      - nombre: nombre del archivo para mostrarlo en el ticket.

    Devuelve un diccionario con las llaves:
      condicion, confianza, descripcion, prioridad, accion, fecha_hora

    Si el backend responde con un error (imagen invalida, backend caido),
    el error grpc.RpcError se propaga para que la interfaz lo muestre.
    """
    # El stub es nuestra vista del servicio remoto
    stub = solarguard_pb2_grpc.SolarGuardServicioStub(canal)

    # Armo la peticion tal como la define el .proto
    peticion = solarguard_pb2.ImagenSolicitud(
        imagen=bytes_imagen,
        nombre=nombre,
    )

    # Hago la llamada remota (esto si viaja por la red)
    respuesta = stub.ClasificarImagen(peticion, timeout=tiempo_espera)

    # Convierto la respuesta del proto en un diccionario simple
    return {
        "condicion": respuesta.condicion,
        "confianza": respuesta.confianza,
        "descripcion": respuesta.descripcion,
        "prioridad": respuesta.prioridad,
        "accion": respuesta.accion,
        "fecha_hora": respuesta.fecha_hora,
    }


def verificar_servidor(
    canal: Channel,
    tiempo_espera: float = 10.0,
) -> str:
    """
    Pregunta al backend si esta vivo.

    Devuelve el mensaje de estado del servidor. Con esto la interfaz
    puede avisar "no encontramos el backend" antes de intentar clasificar.
    """
    stub = solarguard_pb2_grpc.SolarGuardServicioStub(canal)
    peticion = solarguard_pb2.Vacio()

    respuesta = stub.VerificarServicio(peticion, timeout=tiempo_espera)
    return respuesta.mensaje


# Lo que este modulo exporta para que otros archivos lo puedan usar
__all__ = [
    "DIRECCION_POR_DEFECTO",
    "TIEMPO_ESPERA",
    "crear_canal",
    "clasificar_imagen",
    "verificar_servidor",
]