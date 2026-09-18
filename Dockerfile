# Usa la versión requerida por las reglas: Python 3.13
FROM python:3.13-slim

# Variables de entorno recomendadas
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    DISPLAY=:0

# Instala dependencias del sistema necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instala uv (Obligatorio por reglas)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Define el directorio de trabajo
WORKDIR /app

# Copia los archivos de gestión de dependencias
COPY pyproject.toml uv.lock ./

# Instala las dependencias del proyecto usando uv
RUN uv sync --frozen --no-dev

# Copia el resto del código
COPY . .

# Expone los puertos (Streamlit: 8501, gRPC: 50051, MLflow: 5000)
EXPOSE 8501 50051 5000

# Comando por defecto (inicia el frontend por ahora)
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
