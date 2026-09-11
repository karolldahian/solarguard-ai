"""
SolarGuard AI - Paquete principal.

Modulos disponibles:
  - ingesta: carga y validacion de imagenes
  - preprocesamiento: preparar las imagenes para el modelo
  - inferencia: clasificacion de la condicion del panel (simulada por ahora)
  - tickets: generacion del ticket de mantenimiento
  - grpc_interface: mensajes y servidor/cliente generados de gRPC
  - servidor_grpc: el backend que clasifica imagenes por gRPC
  - cliente_grpc: funciones que usa la interfaz para hablar con el backend
"""


def main() -> None:
    print("SolarGuard AI - paquete listo")
    print(
        "Modulos: ingesta, preprocesamiento, inferencia, tickets, "
        "grpc_interface, servidor_grpc, cliente_grpc"
    )
