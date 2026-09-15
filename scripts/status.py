"""Script de diagnóstico y estado general de SolarGuard AI."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

HEADER = """
===================================================================
      SolarGuard AI • Diagnóstico de Estado y Componentes
===================================================================
"""


def check_module(name: str) -> bool:
    return find_spec(name) is not None


def main() -> None:
    print(HEADER)
    root = Path(__file__).resolve().parent.parent

    # 1. Información del Entorno
    print("[1. ENTORNO Y RUNTIME]")
    print(
        f"  • Versión Python:    {sys.version.split()[0]} ({'OK' if sys.version_info >= (3, 13) else 'ADVERTENCIA: requiere >= 3.13'})"
    )
    print(f"  • Directorio Raíz:   {root}")

    # 2. Dependencias Principales
    deps = [
        ("onnxruntime", "onnxruntime"),
        ("streamlit", "streamlit"),
        ("PIL", "Pillow (PIL)"),
        ("pandas", "pandas"),
        ("plotly", "plotly"),
        ("pytest", "pytest"),
        ("ruff", "ruff"),
    ]
    print("\n[2. DEPENDENCIAS PRINCIPALES]")
    for mod_name, label in deps:
        installed = check_module(mod_name)
        badge = "✓ Instalado" if installed else "✗ No encontrado"
        print(f"  • {label:<18} {badge}")

    # 3. Módulos Internos de Arquitectura
    print("\n[3. COMPONENTES DEL PIPELINE (src/solarguard_ai)]")
    components = [
        ("ingesta", "Validación y carga RGB de imágenes"),
        ("preprocesamiento", "Redimensionamiento 224x224 y tensores NCHW"),
        ("inferencia", "Servicio ONNX Runtime SolarScan"),
        ("priorizacion", "Reglas de severidad y umbral operativo"),
        ("tickets", "Generador de tickets en GitHub Issues"),
    ]
    for mod, desc in components:
        mod_path = root / "src" / "solarguard_ai" / f"{mod}.py"
        status = "✓ Operativo" if mod_path.is_file() else "✗ Pendiente"
        print(f"  • {mod:<18} {status:<15} ({desc})")

    # 4. Servicios de Interfaz y Backend
    print("\n[4. INTERFAZ Y SERVICIOS]")
    app_path = root / "app.py"
    app_status = (
        "✓ Disponible" if app_path.is_file() else "⏳ En desarrollo (Tarea 9 - Karoll)"
    )
    print(f"  • Streamlit UI:      {app_status}")

    grpc_path = root / "src" / "solarguard_ai" / "servidor_grpc.py"
    grpc_status = (
        "✓ Disponible"
        if grpc_path.is_file()
        else "⏳ En desarrollo (Tarea 11 - Jarvin)"
    )
    print(f"  • Backend gRPC:      {grpc_status}")

    # 5. Artefactos del Modelo
    print("\n[5. ARTEFACTO DE MACHINE LEARNING]")
    onnx_path = root / "models" / "best.onnx"
    onnx_status = (
        "✓ Presente en models/best.onnx"
        if onnx_path.is_file()
        else "ℹ Requiere descarga desde Hugging Face Hub"
    )
    print(f"  • SolarScan ONNX:    {onnx_status}")

    # 6. Documentación
    print("\n[6. DOCUMENTACIÓN Y CONTRATOS]")
    arch_doc = root / "docs" / "arquitectura.md"
    arch_status = "✓ Generado (Issue #4)" if arch_doc.is_file() else "✗ No encontrado"
    print(f"  • docs/arquitectura: {arch_status}")

    print("\n===================================================================")
    print("  Para ejecutar pruebas de calidad: make test")
    print("  Para verificar linters y formato: make check")
    print("===================================================================\n")


if __name__ == "__main__":
    main()
