"""
Servidor gRPC de SolarGuard AI (el "backend").

Este modulo es el corazon de la integracion: recibe la foto desde la
interfaz (Streamlit) y la procesa con todo el pipeline del proyecto:

  1. ingesta.load_image        -> valida la imagen y la deja en RGB
  2. preprocesamiento.preprocesar -> la convierte en tensor (1, 3, 224, 224)
  3. inferencia.predecir       -> "simula" al modelo y da condicion + confianza
  4. tickets.generar_ticket    -> convierte el resultado en un ticket util

Aprendi que en gRPC el servidor implementa una clase que "hereda" del
servidor generado (add_SolarGuardServicioServicer_to_server) y cada
metodo de la clase corresponde a un rpc del .proto.

Para iniciarlo desde consola:
    uv run python -m solarguard_ai.servidor_grpc
"""

from __future__ import annotations

import argparse
import zlib
from concurrent import futures
from io import BytesIO

import grpc
from PIL import UnidentifiedImageError

# Importaciones del proyecto
from solarguard_ai.ingesta import ImageIngestionError, load_image
from solarguard_ai import preprocesamiento
from solarguard_ai import inferencia
from solarguard_ai.tickets import generar_ticket

from solarguard_ai.grpc_interface import solarguard_pb2, solarguard_pb2_grpc

# Puerto por defecto. 50051 es el puerto clasico de gRPC,
# asi como el 8000 es el clasico de los servidores web.
PUERTO_POR_DEFECTO = 50051


def _calcular_semilla(bytes_imagen: bytes) -> int:
    """
    Calcula la semilla de la simulacion a partir de los bytes de la imagen.

    La idea es que la misma foto siempre produzca la misma clasificacion
    (determinismo). Se usa crc32, que es una funcion rapida de huella
    digital: para un mismo archivo devuelve siempre el mismo numero.
    """
    return zlib.crc32(bytes_imagen)


class SolarGuardServicio(solarguard_pb2_grpc.SolarGuardServicioServicer):
    """
    Implementacion concreta del servicio definido en el .proto.

    Cada metodo de esta clase es la "funcion remota" que llama la interfaz.
    Si el metodo arroja una excepcion grpc.RpcError, el cliente la recibe
    con el codigo y el mensaje para mostrarlos en la interfaz.
    """

    def ClasificarImagen(self, request, context):
        """
        Pipeline completo: recibe la imagen, la clasifica y devuelve el ticket.

        Los errores de validacion (imagen corrupta, formato raro, etc.)
        se convierten en errores de gRPC con el codigo INVALID_ARGUMENT
        para que la interfaz pueda mostrarlos en pantalla.
        """
        nombre = request.nombre or "imagen_subida.jpg"

        # --- Paso 0: que la peticion traiga datos utiles ---------------------------------
        if not request.imagen:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "La peticion no incluye bytes de imagen.",
            )

        try:
            # --- Paso 1: cargar y validar la imagen (ingesta) ------------------------------
            imagen_cargada = load_image(BytesIO(request.imagen))

            # --- Paso 2: preprocesar la imagen a tensor del modelo --------------------------
            tensor = preprocesamiento.preprocesar(imagen_cargada)

            # --- Paso 3: inferir la condicion y confianza (por ahora simulado) --------------
            semilla = _calcular_semilla(request.imagen)
            prediccion = inferencia.predecir(tensor, semilla=semilla)

            # --- Paso 4: generar el ticket de mantenimiento ------------------------------------
            ticket = generar_ticket(
                fuente=nombre,
                condicion=prediccion.condicion,
                confianza=prediccion.confianza,
            )

        except (ImageIngestionError, UnidentifiedImageError, OSError) as error:
            contexto = grpc.StatusCode.INVALID_ARGUMENT
            context.abort(contexto, f"Error al procesar '{nombre}': {error}")
        except ValueError as error:
            # Errores de reglas de negocio (ej: condicion desconocida)
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))

        # --- Paso 5: armar la respuesta del .proto -----------------------------------------
        return solarguard_pb2.ResultadoClasificacion(
            condicion=ticket.condicion,
            confianza=ticket.confianza,
            descripcion=ticket.descripcion,
            prioridad=ticket.prioridad,
            accion=ticket.accion,
            fecha_hora=ticket.fecha_hora.strftime("%d/%m/%Y %H:%M"),
        )

    def VerificarServicio(self, request, context):
        """
        Responde un mensaje simple para que la interfaz sepa que el backend esta vivo.
        """
        return solarguard_pb2.EstadoServicio(
            mensaje="SolarGuard AI backend activo",
        )


def _construir_servidor() -> grpc.Server:
    """
    Crea el servidor gRPC con el servicio registrado y listo para escuchar.

    Se separo de iniciar_servidor() para poder probarlo en los tests
    sin tener que abrir una conexion de red real.
    """
    # El ThreadPoolExecutor decide cuantas peticiones atiende al mismo tiempo.
    ejecutador = futures.ThreadPoolExecutor(max_workers=10)
    servidor = grpc.server(ejecutador)
    solarguard_pb2_grpc.add_SolarGuardServicioServicer_to_server(
        SolarGuardServicio(), servidor
    )
    return servidor


def iniciar_servidor(puerto: int = PUERTO_POR_DEFECTO) -> grpc.Server:
    """
    Pone el servidor a escuchar en el puerto indicado (ya registrado).
    """
    servidor = _construir_servidor()
    # puerto 0 significa "deja que el sistema escoja uno" (util en pruebas)
    direccion = f"[::]:{puerto}"
    servidor.add_insecure_port(direccion)
    servidor.start()
    return servidor


def main() -> None:
    """
    Punto de entrada para correr el backend desde la terminal.

    Recibe el puerto opcional:
        uv run python -m solarguard_ai.servidor_grpc --puerto 50051
    """
    analizador = argparse.ArgumentParser(
        description="Backend gRPC de SolarGuard AI"
    )
    analizador.add_argument(
        "--puerto",
        type=int,
        default=PUERTO_POR_DEFECTO,
        help="Puerto donde escucha el backend",
    )
    argumentos = analizador.parse_args()

    servidor = iniciar_servidor(puerto=argumentos.puerto)
    print(f"Backend SolarGuard AI escuchando en el puerto {argumentos.puerto}...")
    print("Para detenerlo: Ctrl+C")
    servidor.wait_for_termination()


# Lo que este modulo exporta para que otros archivos lo puedan usar
__all__ = [
    "PUERTO_POR_DEFECTO",
    "SolarGuardServicio",
    "iniciar_servidor",
    "_construir_servidor",
]