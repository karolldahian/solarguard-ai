# Declara objetivos que no representan archivos y siempre deben ejecutarse.
.PHONY: help install run streamlit lock-check compile check grpc-gen servidor

# Muestra una guía rápida de los comandos disponibles.
help:
	@echo "SolarGuard AI - comandos disponibles"
	@echo "  make install    Sincroniza el entorno y las dependencias"
	@echo "  make run        Ejecuta el entry point actual del paquete"
	@echo "  make streamlit  Inicia la aplicación Streamlit"
	@echo "  make grpc-gen   Regenera el código gRPC desde proto/solarguard.proto"
	@echo "  make servidor   Inicia el backend gRPC"
	@echo "  make lock-check Comprueba que uv.lock está actualizado"
	@echo "  make compile    Comprueba la sintaxis de los módulos Python"
	@echo "  make check      Ejecuta las validaciones disponibles"

# Crea o actualiza el entorno virtual e instala las dependencias del proyecto.
install:
	uv sync

# Ejecuta el punto de entrada actual del paquete SolarGuard AI.
run:
	uv run solarguard-ai

# Inicia la aplicación web de Streamlit cuando app.py esté disponible.
streamlit:
	uv run streamlit run app.py

# Regenera los archivos _pb2 a partir de proto/solarguard.proto.
# Se ejecuta solo cuando cambia el contrato gRPC.
grpc-gen:
	uv run python scripts/grpc_gen.py

# Inicia el backend gRPC que atiende las peticiones de la interfaz.
servidor:
	uv run python -m solarguard_ai.servidor_grpc

# Comprueba que el archivo de bloqueo coincide con pyproject.toml.
lock-check:
	uv lock --check

# Recorre los módulos fuente y verifica que su sintaxis Python sea válida.
compile:
	uv run python -m compileall src

# Ejecuta todas las validaciones disponibles para el estado actual del proyecto.
check: lock-check compile
