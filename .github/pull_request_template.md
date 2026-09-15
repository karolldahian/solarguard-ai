### 📝 Convención del Título de la PR
Aplica un [prefijo convencional](https://www.conventionalcommits.org/) en el título para clasificar el trabajo:

| Prefijo | Cuándo usar | Ejemplo |
|--------|-------------|---------|
| `feat:` | Nueva funcionalidad o módulo | `feat: agregar preprocesamiento CLAHE en preprocesamiento.py` |
| `fix:` | Corrección de errores o eliminación de warnings | `fix: corregir advertencia en tensores ONNX` |
| `test:` | Adición o refactorización de pruebas unitarias | `test: agregar pruebas unitarias para tickets.py` |
| `chore:` | Tareas de mantenimiento, CI/CD, docs o config | `chore: actualizar Makefile y README.md con badges` |

---

### 🧱 Módulo o Componente Afectado
Selecciona las áreas principales en las que trabaja esta PR:
- [ ] **Ingesta & Validación de Imágenes** (`src/solarguard_ai/ingesta.py`)
- [ ] **Preprocesamiento & Tensores** (`src/solarguard_ai/preprocesamiento.py`)
- [ ] **Modelo e Inferencia ONNX** (`src/solarguard_ai/inferencia.py`)
- [ ] **Motor de Priorización & Reglas** (`src/solarguard_ai/priorizacion.py`, `config/prioritization.toml`)
- [ ] **Generación de Tickets de Mantenimiento** (`src/solarguard_ai/tickets.py`)
- [ ] **Pruebas Unitarias (Pytest AAA)** (`tests/`)
- [ ] **Infraestructura, Calidad & CI/CD** (`Makefile`, `.github/`, `.pre-commit-config.yaml`, `.gga`)
- [ ] **Documentación & Arquitectura** (`docs/`, `README.md`)

---

### 📚 Descripción de los Cambios
<!---
Describe qué logra esta PR, la motivación y los detalles técnicos relevantes.
-->

**Módulo / Historia / Tarea asociada:** <!-- Ej. Issue #10: Generación de tickets o Issue #4: Arquitectura -->

#### Resumen de Cambios:
-

#### Justificación del Diseño (Clean Code / Principios SOLID):
<!---
Explica brevemente cómo se mantuvo la Alta Cohesión y Bajo Acoplamiento (Clean Code) en las clases/funciones modificadas.
-->
-

---

### ✅ Lista de Chequeo Pre-PR (Estándares del Curso UAO)
Antes de solicitar revisión, verifica que tu código cumpla con los lineamientos imperativos del curso:

- [ ] **Entorno de Ejecución:** El código se ejecutó y probó exitosamente usando exclusivamente **`uv`** (Python 3.13).
- [ ] **Sin Warnings:** La ejecución de `uv run pytest` o `make check` corre limpia **sin advertencias (warnings)** ni deprecaciones.
- [ ] **Control de Exclusiones (`.gitignore`):** Los modelos pesados (`models/*.onnx`, `*.h5`) y `.venv` **NO** están rastreados por Git.
- [ ] **Estructura del Código:** El código fuente reside en `src/solarguard_ai/` y las pruebas en `tests/`.
- [ ] **Pruebas Unitarias (`pytest`):** Se ejecutó `make test` y todas las pruebas pasaron correctamente con patrón AAA.
- [ ] **Linter & Formato (`ruff`):** Cumple con `ruff check .` y `ruff format --check .` al 100%.
- [ ] **Auditoría GGA:** Se verificó cumplimiento de reglas arquitectónicas de `AGENTS.md` con Gentleman Guardian Angel.
- [ ] **Clean Code & Tipado:** Tipado estricto (Type Hints) y Docstrings completos (PEP 257) en todas las funciones y clases.

---

### 🧪 Evidencia de Pruebas Ejecutadas
<!---
Adjunta logs o salidas de los comandos make test / make check demostrando que el código funciona y aprueba.
-->

```bash
# Salida de validaciones y pruebas ejecutadas con UV / Make:
make check
```
