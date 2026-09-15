"""
Genera los archivos de gRPC a partir de proto/solarguard.proto.

Aprendi que este paso es necesario cada vez que se modifica el .proto:
sin los archivos _pb2, tanto el servidor como los clientes no saben
"traducir" los mensajes. La ventaja es que como dejamos generados los
archivos en el repositorio, los companeros no tienen que correr esto
a menos que cambiemos el contrato.

Aqui suceden dos cosas:
  1. grpc_tools.protoc genera solarguard_pb2.py y solarguard_pb2_grpc.py
     dentro de solarguard_ai/grpc_interface.
  2. Parcheamos el import del archivo generado: por defecto genera
     "import solarguard_pb2" (literal), que falla cuando el paquete
     se importa como solarguard_ai.grpc_interface. Lo dejamos en
     "from . import solarguard_pb2" para que sea un import relativo.

Para ejecutarlo:
    uv run python scripts/grpc_gen.py
"""

from __future__ import annotations

from pathlib import Path

# Ubicaciones de los archivos del proyecto, relativas a este script
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIRECCION_PROTO = RAIZ_PROYECTO / "proto"
DIRECCION_GENERADO = RAIZ_PROYECTO / "src" / "solarguard_ai" / "grpc_interface"
NOMBRE_PROTO = "solarguard.proto"


def _generar_codigo() -> None:
    """Ejecuta el compilador de protos dentro de Python."""
    from grpc_tools import protoc

    # gramatica del comando normalmente seria:
    #   python -m grpc_tools.protoc -I proto \
    #     --python_out=... --grpc_python_out=... proto/solarguard.proto
    # Aqui lo mismo pero pasandole la lista de argumentos a protoc.main()
    argumentos = [
        "grpc_tools.protoc",
        f"-I{DIRECCION_PROTO}",
        f"--python_out={DIRECCION_GENERADO}",
        f"--grpc_python_out={DIRECCION_GENERADO}",
        str(DIRECCION_PROTO / NOMBRE_PROTO),
    ]
    resultado = protoc.main(argumentos)
    if resultado != 0:
        raise SystemExit(f"protoc fallo con codigo {resultado}")


def _parchear_import_relativo() -> None:
    """
    Arregla el import del archivo generado para que funcione dentro
    del paquete solarguard_ai (import relativo en lugar de absoluto).
    """
    archivo_grpc = DIRECCION_GENERADO / "solarguard_pb2_grpc.py"
    contenido = archivo_grpc.read_text(encoding="utf-8")
    # Texto que genera grpc_tools por defecto
    import_original = "import solarguard_pb2 as solarguard__pb2"
    # Texto que necesitamos para usarlo como paquete interno
    import_relativo = "from . import solarguard_pb2 as solarguard__pb2"
    if import_original in contenido and import_relativo not in contenido:
        contenido = contenido.replace(import_original, import_relativo)
        archivo_grpc.write_text(contenido, encoding="utf-8")


def main() -> None:
    _generar_codigo()
    _parchear_import_relativo()
    print("Codigo de gRPC generado en:", DIRECCION_GENERADO)
    print("Recuerda incluir los archivos _pb2 en el commit (se versionan).")


if __name__ == "__main__":
    main()
