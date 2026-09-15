"""
Servidor gRPC de SolarGuard AI (el "backend").

Este modulo es el corazon de la integracion: recibe la foto desde la
interfaz (Streamlit) y la procesa con todo el pipeline del proyecto:

  1. ingesta.load_image               -> valida la imagen y la deja en RGB
  2. preprocesamiento.preprocess_for_solarscan -> la convierte en tensor (1, 3, 224, 224)
  3. inferencia.SolarScanInference    -> ejecuta el modelo ONNX y da prediccion
  4. tickets.generate_maintenance_ticket -> convierte el resultado en un ticket

Para iniciarlo desde consola:
    uv run python -m solarguard_ai.servidor_grpc
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent import futures
from io import BytesIO

import grpc
from PIL import UnidentifiedImageError

from solarguard_ai import inferencia, preprocesamiento, tickets
from solarguard_ai.grpc_interface import solarguard_pb2, solarguard_pb2_grpc

# Importaciones del proyecto
from solarguard_ai.ingesta import ImageIngestionError, load_image
from solarguard_ai.mlflow_tracking import is_enabled, log_error, log_pipeline_request
from solarguard_ai.priorizacion import load_priority_config, prioritize

# Puerto por defecto. 50051 es el puerto clasico de gRPC,
# asi como el 8000 es el clasico de los servidores web.
PUERTO_POR_DEFECTO = 50051

# Configuración de priorización
PRIORITY_CONFIG_PATH = "config/prioritization.toml"
MODEL_PATH = "models/best.onnx"

# Cliente de inferencia (se inicializa perezosamente)
_inference_client: inferencia.SolarScanInference | None = None
_priority_config = None


def _get_inference_client() -> inferencia.SolarScanInference:
    global _inference_client
    if _inference_client is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No se encuentra el modelo ONNX en '{MODEL_PATH}'. "
                "Descargue 'best.onnx' desde Hugging Face y colóquelo en la carpeta models/."
            )
        _inference_client = inferencia.SolarScanInference(MODEL_PATH)
    return _inference_client


def _get_priority_config():
    global _priority_config
    if _priority_config is None:
        _priority_config = load_priority_config(PRIORITY_CONFIG_PATH)
    return _priority_config


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
        total_start = time.perf_counter()

        # Variables para métricas de cada paso
        ingestion_latency_ms = 0.0
        preprocessing_latency_ms = 0.0
        inference_latency_ms = 0.0
        ticket_latency_ms = 0.0

        # --- Paso 0: que la peticion traiga datos utiles ---------------------------------
        if not request.imagen:
            if is_enabled():
                log_pipeline_request(
                    panel_id=nombre,
                    image_size=(0, 0),
                    image_format="unknown",
                    total_latency_ms=(time.perf_counter() - total_start) * 1000,
                    error="empty_image",
                )
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "La peticion no incluye bytes de imagen.",
            )

        try:
            # --- Paso 1: cargar y validar la imagen (ingesta) ------------------------------
            ingestion_start = time.perf_counter()
            imagen_cargada = load_image(BytesIO(request.imagen))
            ingestion_latency_ms = (time.perf_counter() - ingestion_start) * 1000

            image_size = (imagen_cargada.width, imagen_cargada.height)
            image_format = imagen_cargada.format or "unknown"

            # --- Paso 2: preprocesar la imagen a tensor del modelo --------------------------
            preprocessing_start = time.perf_counter()
            tensor = preprocesamiento.preprocess_for_solarscan(imagen_cargada.image)
            preprocessing_latency_ms = (
                time.perf_counter() - preprocessing_start
            ) * 1000

            # --- Paso 3: inferir la condicion y confianza (ONNX Runtime) -------------------
            inference_client = _get_inference_client()
            prediction = inference_client.predict(tensor)
            # La inferencia ya loguea su propia latencia interna, pero medimos la latencia vista desde aquí
            inference_latency_ms = 0.0  # Se logea dentro de predict()

            # --- Paso 4: priorizacion con reglas de negocio ----------------------------------
            config = _get_priority_config()
            priority_result = prioritize(
                predicted_class=prediction.predicted_class,
                confidence=prediction.confidence,
                review_threshold=config.review_confidence,
            )

            # --- Paso 5: generar el ticket de mantenimiento -----------------------------------
            ticket_start = time.perf_counter()
            ticket_result = tickets.generate_maintenance_ticket(
                priority_result=priority_result,
                panel_id=nombre,
                predicted_class=prediction.predicted_class,
                confidence=prediction.confidence,
                location="Parque Solar - Bloque General",
                force=True,  # En gRPC siempre generamos ticket para mostrar resultado
            )
            ticket_latency_ms = (time.perf_counter() - ticket_start) * 1000

            if ticket_result.status == "skipped":
                # Si no se genera ticket (ej. Clean sin force), creamos uno simulado para la respuesta
                ticket = tickets.build_ticket(
                    priority_result=priority_result,
                    panel_id=nombre,
                    predicted_class=prediction.predicted_class,
                    confidence=prediction.confidence,
                    location="Parque Solar - Bloque General",
                )
            else:
                ticket = ticket_result.ticket

            # Log exitoso del pipeline completo
            if is_enabled():
                log_pipeline_request(
                    panel_id=nombre,
                    image_size=image_size,
                    image_format=image_format,
                    total_latency_ms=(time.perf_counter() - total_start) * 1000,
                    ingestion_latency_ms=ingestion_latency_ms,
                    preprocessing_latency_ms=preprocessing_latency_ms,
                    inference_latency_ms=inference_latency_ms,
                    ticket_latency_ms=ticket_latency_ms,
                    predicted_class=prediction.predicted_class,
                    confidence=prediction.confidence,
                    priority=priority_result.priority,
                    requires_human_review=priority_result.requires_human_review,
                    ticket_status=ticket_result.status,
                )

        except (ImageIngestionError, UnidentifiedImageError, OSError) as error:
            if is_enabled():
                log_pipeline_request(
                    panel_id=nombre,
                    image_size=(0, 0),
                    image_format="unknown",
                    total_latency_ms=(time.perf_counter() - total_start) * 1000,
                    ingestion_latency_ms=ingestion_latency_ms,
                    error=str(error),
                )
                log_error("IngestionError", str(error), {"panel_id": nombre})
            contexto = grpc.StatusCode.INVALID_ARGUMENT
            context.abort(contexto, f"Error al procesar '{nombre}': {error}")
        except ValueError as error:
            # Errores de reglas de negocio (ej: condicion desconocida)
            if is_enabled():
                log_pipeline_request(
                    panel_id=nombre,
                    image_size=(0, 0),
                    image_format="unknown",
                    total_latency_ms=(time.perf_counter() - total_start) * 1000,
                    error=str(error),
                )
                log_error("PrioritizationError", str(error), {"panel_id": nombre})
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        except grpc.RpcError:
            # Re-lanzar errores de gRPC sin envolver
            raise
        except RuntimeError as error:
            # Errores de inferencia ONNX, configuración, etc.
            if is_enabled():
                log_pipeline_request(
                    panel_id=nombre,
                    image_size=(0, 0),
                    image_format="unknown",
                    total_latency_ms=(time.perf_counter() - total_start) * 1000,
                    error=str(error),
                )
                log_error("InferenceError", str(error), {"panel_id": nombre})
            context.abort(
                grpc.StatusCode.INTERNAL, f"Error interno del servidor: {error}"
            )

        # --- Paso 6: armar la respuesta del .proto -----------------------------------------
        return solarguard_pb2.ResultadoClasificacion(
            condicion=ticket.predicted_class,
            confianza=ticket.confidence,
            descripcion=ticket.body.split("### 2. Acción Requerida")[0]
            .split("**Diagnóstico Visual**")[1]
            .split("|")[1]
            .replace("`", "")
            .strip()
            if "**Diagnóstico Visual**" in ticket.body
            else ticket.predicted_class,
            prioridad=ticket.severity,
            accion=ticket.body.split("### 2. Acción Requerida")[1]
            .split("\n")[1]
            .strip()
            if "### 2. Acción Requerida" in ticket.body
            else "Revisar panel",
            fecha_hora=ticket.created_at[:19].replace("T", " "),
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
    analizador = argparse.ArgumentParser(description="Backend gRPC de SolarGuard AI")
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
    "_construir_servidor",
    "iniciar_servidor",
]
