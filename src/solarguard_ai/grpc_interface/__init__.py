"""
Modulo de la interfaz gRPC de SolarGuard AI.

Aqui viven los archivos generados a partir de proto/solarguard.proto:

  - solarguard_pb2.py:        clases de Python de los mensajes (proto).
  - solarguard_pb2_grpc.py:   clases del servidor y del stub (cliente).

El servidor y el cliente se implementan en los modulos
solarguard_ai.servidor_grpc y solarguard_ai.cliente_grpc.

Estos archivos _pb2 se generan con scripts/grpc_gen.py y se dejan
versionados en el repositorio para que nadie tenga que regenerarlos.
"""
