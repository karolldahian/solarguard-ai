# ==============================================================================
# SolarGuard AI — Makefile de Automatización y Calidad (UV)
# Especialización en Inteligencia Artificial — Universidad Autónoma de Occidente
# ==============================================================================

.PHONY: help status run streamlit backend install sync test check format clean lock-check compile gga gga-pr pre-commit pre-commit-install grpc-gen servidor mlflow-ui mlflow-clean

help:
	@echo ====================================================================
	@echo   SolarGuard AI • Comandos Make para Automatizacion (UV)
	@echo ====================================================================
	@echo [ESTADO Y PIPELINE]
	@echo   make status       - Diagnostico de entorno, dependencias y componentes
	@echo   make run          - Ejecuta el entry point actual del paquete
	@echo   make streamlit    - Inicia la aplicacion interactiva de Streamlit
	@echo   make backend      - Inicia el servidor backend gRPC (cuando este disponible)
	@echo   make grpc-gen     - Regenera el codigo gRPC desde proto/solarguard.proto
	@echo   make servidor     - Inicia el backend gRPC (alias de backend)
	@echo --------------------------------------------------------------------
	@echo [MLFLOW TRACKING]
	@echo   make mlflow-ui    - Inicia la interfaz web de MLflow (puerto 5000)
	@echo   make mlflow-clean - Elimina runs locales de MLflow (mlruns/)
	@echo --------------------------------------------------------------------
	@echo [CALIDAD Y HERRAMIENTAS]
	@echo   make test         - Ejecuta suite completa de pruebas unitarias (Pytest)
	@echo   make check        - Audita linters, formato y compatibilidad
	@echo   make format       - Formatea y corrige estilos con Ruff
	@echo   make gga          - Ejecuta auditoria local con Gentleman Guardian Angel
	@echo   make gga-pr       - Audita el Pull Request actual contra main con GGA
	@echo   make pre-commit   - Ejecuta todos los hooks de pre-commit
	@echo   make compile      - Verifica la sintaxis de todos los modulos Python
	@echo   make lock-check   - Comprueba sincronizacion de dependencias en uv.lock
	@echo   make clean        - Limpia caches y artefactos temporales
	@echo   make install      - Sincroniza el entorno virtual con UV (uv sync)
	@echo ====================================================================

# ------------------------------------------------------------------------------
# Estado y Pipeline
# ------------------------------------------------------------------------------

status:
	uv run python scripts/status.py

run:
	uv run solarguard-ai

streamlit:
	@python -c "import pathlib, subprocess; subprocess.run(['uv', 'run', 'streamlit', 'run', 'app.py']) if pathlib.Path('app.py').is_file() else print('\n[AVISO] app.py aun no esta disponible en la rama actual.\n        La interfaz Streamlit esta siendo desarrollada por el equipo (Tarea 9 - Karoll).\n')"

backend:
	@python -c "import pathlib, subprocess; subprocess.run(['uv', 'run', 'python', '-m', 'solarguard_ai.servidor_grpc']) if pathlib.Path('src/solarguard_ai/servidor_grpc.py').is_file() else print('\n[AVISO] servidor_grpc.py aun no esta disponible en la rama actual.\n        El servidor gRPC esta siendo desarrollado por el equipo (Tarea 11 - Jarvin).\n')"

# ------------------------------------------------------------------------------
# Entorno y Dependencias
# ------------------------------------------------------------------------------

install: sync

sync:
	uv sync

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

compile:
	uv run python -m compileall src scripts

# ------------------------------------------------------------------------------
# Calidad y Pruebas
# ------------------------------------------------------------------------------

test:
	uv run pytest -v

format:
	uv run ruff format .
	uv run ruff check --fix .

check: lock-check compile
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -q --cov --cov-report=term-missing

gga:
	@echo "Ejecutando Gentleman Guardian Angel sobre cambios locales..."
	@gga run

gga-pr:
	@echo "Ejecutando Gentleman Guardian Angel para revision de PR..."
	@gga run --pr-mode --diff-only

pre-commit:
	uv run pre-commit run --all-files

pre-commit-install:
	uv run pre-commit install
	gga install
	gga install --commit-msg

# ------------------------------------------------------------------------------
# Limpieza
# ------------------------------------------------------------------------------

clean:
	@echo Limpiando caches y archivos temporales...
	@python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
	@python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
	@python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.ruff_cache')]"
	@python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('htmlcov')]"
	@echo Limpieza completada exitosamente.

# ------------------------------------------------------------------------------
# MLflow Tracking
# ------------------------------------------------------------------------------

mlflow-ui:
	@echo Iniciando MLflow UI en http://localhost:5000 ...
	@uv run mlflow ui --host 0.0.0.0 --port 5000

mlflow-clean:
	@echo Limpiando runs locales de MLflow...
	@python -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('mlruns')]"
	@echo Limpieza de MLflow completada.
