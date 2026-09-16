"""
Pruebas para la integracion gRPC de SolarGuard AI.

Aqui verifico que la integracion entre la interfaz (cliente) y el
backend (servidor) funcione de verdad, de punta a punta:

  - Que el servicio responde cuando el backend esta corriendo.
  - Que la clasificacion llega con todos los campos del ticket.
  - Que los errores (imagen vacia, archivo invalido) llegan bien al cliente.
  - Que la simulacion de inferencia es determinista (misma semilla, mismo resultado).

Para correr:
    uv run pytest tests/test_grpc.py -v
"""

from __future__ import annotations

from io import BytesIO

import grpc
import pytest
from PIL import Image

# Funciones del servidor y del cliente
from solarguard_ai.cliente_grpc import clasificar_imagen, verificar_servidor
from solarguard_ai.grpc_interface import solarguard_pb2, solarguard_pb2_grpc
from solarguard_ai.inferencia import CLASS_NAMES
from solarguard_ai.servidor_grpc import _construir_servidor

# Prioridades validas que puede devolver un ticket (mismo set que tickets.py)
PRIORIDADES_VALIDAS = {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------


def crear_imagen_bytes(ancho: int = 640, alto: int = 480) -> bytes:
    """Crea una imagen simple en memoria y la devuelve como bytes PNG."""
    imagen = Image.new("RGB", (ancho, alto), color=(100, 150, 200))
    buffer = BytesIO()
    imagen.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def backend_grpc(monkeypatch, tmp_path):
    """
    Levanta un backend gRPC 'de juguete' para las pruebas.
    """
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    import numpy as np

    from solarguard_ai.inferencia import SolarScanInference

    class FakeInput:
        name = "images"

    class FakeSession:
        def get_inputs(self):
            return [FakeInput()]

        def run(self, _outputs, _inputs):
            # Simulamos que predice la primera clase con alta confianza
            return [np.array([[0.8, 0.04, 0.04, 0.04, 0.04, 0.04]], dtype=np.float32)]

    def factory(_path):
        return FakeSession()

    model_path = tmp_path / "best.onnx"
    model_path.write_bytes(b"fake")
    fake_service = SolarScanInference(model_path, session_factory=factory)

    monkeypatch.setattr(
        "solarguard_ai.servidor_grpc._get_inference_client", lambda: fake_service
    )

    servidor = _construir_servidor()

    # add_insecure_port devuelve el numero de puerto que toco
    puerto = servidor.add_insecure_port("127.0.0.1:0")
    servidor.start()

    canal = grpc.insecure_channel(f"127.0.0.1:{puerto}")
    yield canal

    # Cierre limpio al terminar cada prueba
    canal.close()
    servidor.stop(grace=0.5)


# ---------------------------------------------------------------------------
# Pruebas del servicio (servidor <-> cliente de verdad)
# ---------------------------------------------------------------------------


def test_verificar_servidor_responde_activo(backend_grpc) -> None:
    """
    El boton "Verificar conexion" de la interfaz usa este rpc:
    debe responder un mensaje de que el backend esta vivo.
    """
    mensaje = verificar_servidor(backend_grpc)

    assert "activo" in mensaje


def test_clasificar_imagen_devuelve_ticket_completo(backend_grpc) -> None:
    """
    Con una imagen valida, el backend debe devolver un ticket con
    todos los campos que necesita la interfaz.
    """
    bytes_imagen = crear_imagen_bytes()

    resultado = clasificar_imagen(
        backend_grpc,
        bytes_imagen=bytes_imagen,
        nombre="panel_prueba.png",
    )

    # Las llaves que la interfaz usa para dibujar el ticket
    assert set(resultado) == {
        "condicion",
        "confianza",
        "descripcion",
        "prioridad",
        "accion",
        "fecha_hora",
    }

    # La condicion debe ser una de las 6 clases del modelo
    assert resultado["condicion"] in CLASS_NAMES

    # La confianza debe estar entre 0.0 y 1.0
    assert 0.0 <= resultado["confianza"] <= 1.0

    # La prioridad debe ser una de las 4 definidas en tickets.py
    assert resultado["prioridad"] in PRIORIDADES_VALIDAS

    # La accion y la descripcion no pueden llegar vacias
    assert resultado["accion"]
    assert resultado["descripcion"]

    # La fecha debe venir como texto listo para mostrar
    assert isinstance(resultado["fecha_hora"], str)
    assert resultado["fecha_hora"]


def test_imagen_vacia_devuelve_error_grpc(backend_grpc) -> None:
    """
    Si la peticion no trae bytes de imagen, el servidor debe responder
    un error de gRPC (INVALID_ARGUMENT) en lugar de devolver un ticket raro.
    """
    stub = solarguard_pb2_grpc.SolarGuardServicioStub(backend_grpc)
    peticion = solarguard_pb2.ImagenSolicitud(imagen=b"", nombre="vacia.png")

    with pytest.raises(grpc.RpcError) as error:
        stub.ClasificarImagen(peticion)

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_archivo_invalido_devuelve_error_grpc(backend_grpc) -> None:
    """
    Si los bytes no son una imagen reconocible, la ingesta del backend
    debe fallar y el error debe llegar como error de gRPC (no como un
    ticket con basura).
    """
    stub = solarguard_pb2_grpc.SolarGuardServicioStub(backend_grpc)
    # "no soy una imagen" no es JPEG, PNG ni TIFF
    peticion = solarguard_pb2.ImagenSolicitud(
        imagen=b"no soy una imagen",
        nombre="falso.png",
    )

    with pytest.raises(grpc.RpcError) as error:
        stub.ClasificarImagen(peticion)

    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_misma_imagen_siempre_da_misma_clasificacion(backend_grpc) -> None:
    """
    Como la simulacion es determinista, la misma foto debe dar el mismo
    resultado aunque se pregunte dos veces seguidas. Esto es deseable
    para que el historial no cambie de un momento a otro.
    """
    bytes_imagen = crear_imagen_bytes()

    primer_resultado = clasificar_imagen(
        backend_grpc, bytes_imagen=bytes_imagen, nombre="panel_1.png"
    )
    segundo_resultado = clasificar_imagen(
        backend_grpc, bytes_imagen=bytes_imagen, nombre="panel_1.png"
    )

    assert primer_resultado["condicion"] == segundo_resultado["condicion"]
    assert primer_resultado["confianza"] == segundo_resultado["confianza"]
