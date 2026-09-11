# SolarGuard AI - Reglas de desarrollo

## Principios generales

- Mantener alta cohesión y bajo acoplamiento.
- Cada módulo debe tener una responsabilidad clara y limitada.
- No mezclar lógica de interfaz, preprocesamiento, inferencia y reglas de negocio en un mismo módulo.
- No modificar código fuera del alcance de la tarea actual, salvo que sea estrictamente necesario.
- Evitar dependencias innecesarias.
- Priorizar código explícito, legible y comprobable sobre soluciones innecesariamente complejas.

## Reglas de arquitectura

- El módulo de ingesta de imágenes debe encargarse únicamente de validar y cargar imágenes.
- El módulo de preprocesamiento debe encargarse únicamente de transformar los datos al formato requerido por el modelo.
- El módulo de inferencia debe encargarse de ejecutar el modelo y devolver resultados de predicción estructurados.
- La lógica de priorización del mantenimiento no debe depender directamente de ONNX Runtime.
- La interfaz desarrollada con Streamlit no debe contener detalles internos de implementación de la inferencia.
- Las reglas de negocio y recomendaciones deben permanecer separadas de la interfaz de usuario.
- Evitar módulos genéricos como `utils.py` que acumulen responsabilidades diferentes, salvo que exista una justificación clara.

## Reglas de Machine Learning e inferencia

- La entrada del modelo SolarScan debe conservar compatibilidad con el pipeline de preprocesamiento esperado por el modelo.
- No incorporar cálculos de NDVI o NDWI dentro del pipeline estándar de inferencia RGB de SolarScan.
- El aumento de datos (data augmentation) no debe utilizarse durante la inferencia normal en producción.
- Las predicciones deben incluir, como mínimo, la clase predicha y su nivel de confianza.
- Las predicciones con baja confianza no deben presentarse como diagnósticos ciertos.
- La salida del modelo no debe describirse como un diagnóstico eléctrico definitivo.
- Las recomendaciones de mantenimiento deben estar sujetas a la revisión técnica o humana correspondiente.
- No afirmar niveles de precisión en condiciones reales de campo sin evidencia obtenida mediante una evaluación independiente.

## Reglas de pruebas

- Todo comportamiento nuevo debe incluir pruebas automatizadas apropiadas cuando corresponda.
- Las pruebas existentes deben continuar pasando después de realizar cambios.
- Utilizar Pytest para las pruebas de Python.
- Las pruebas deben seguir una estructura clara Arrange-Act-Assert cuando sea apropiado.
- Las pruebas unitarias de software no sustituyen la evaluación estadística del rendimiento del modelo.

## Reglas de calidad de código

- El código Python debe superar las verificaciones de Ruff.
- El formato del código Python debe cumplir las reglas establecidas por Ruff.
- Utilizar nombres descriptivos para funciones, variables, clases y módulos.
- Evitar lógica duplicada.
- Eliminar imports sin utilizar y código muerto.
- Mantener interfaces públicas explícitas y estables cuando sea posible.

## Reglas de seguridad

Nunca realizar commit ni exponer:

- archivos `.env`
- API keys
- contraseñas
- tokens de acceso
- claves privadas
- credenciales de servicios cloud
- credenciales de bases de datos
- cookies de sesión
- datos personales
- certificados privados

Los secretos deben proporcionarse mediante mecanismos seguros de configuración o variables de entorno.

## Reglas de Git

- El trabajo debe realizarse en ramas específicas para cada tarea.
- No realizar commits directamente sobre `main`.
- No modificar las ramas de otros integrantes del equipo.
- Revisar los cambios antes de realizar un commit.
- Utilizar mensajes descriptivos siguiendo Conventional Commits cuando se realicen commits.
- No subir datasets generados, modelos pesados ni archivos temporales sin una justificación explícita.

## Reglas de dependencias

- Utilizar UV como gestor de dependencias del proyecto.
- `pyproject.toml` y `uv.lock` son las fuentes de verdad para las dependencias.
- Las herramientas utilizadas únicamente durante el desarrollo deben registrarse como dependencias de desarrollo.
- No introducir `requirements.txt` como una fuente de dependencias paralela, salvo que exista un requisito explícito que lo justifique.

## Reglas de documentación

- Los cambios importantes de arquitectura o comportamiento deben documentarse.
- Las instrucciones del `README.md` deben corresponder con el comportamiento real del proyecto.
- Diferenciar claramente la licencia del código de SolarGuard de las licencias o condiciones de uso de modelos y datasets externos.
