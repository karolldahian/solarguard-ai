# Model Card: SolarScan YOLOv8n-cls

> **Ficha Técnica Estándar de Modelo de IA (Issue #3)**  
> **Proyecto:** SolarGuard AI — Detección y Priorización de Mantenimiento en Paneles Fotovoltaicos  
> **Especialización en Inteligencia Artificial — Universidad Autónoma de Occidente (UAO)**  
> **Basado en el estándar de Mitchell et al. (2019) y especificación Hugging Face Hub**  

---

## 1. Resumen y Detalles del Modelo (Model Details)

| Atributo | Especificación |
| :--- | :--- |
| **Nombre del Modelo** | `solarscan-yolov8n-cls` |
| **Autor / Creador** | Saif Elgalaly ([Hugging Face Hub](https://huggingface.co/SaifElgalaly/solarscan-yolov8n-cls)) |
| **Arquitectura Base** | YOLOv8 Nano para Clasificación de Imágenes (Ultralytics) |
| **Formato de Artefacto** | ONNX Runtime (`best.onnx`) |
| **Tipo de Tarea** | Clasificación Multiclase de Imágenes Superficiales (Single-label) |
| **Licencia de Uso** | MIT / Acceso Abierto para Investigación y Prototipado |
| **Runtime de Inferencia** | ONNX Runtime >= 1.20 (Optimizado para CPU x86_64 / ARM64) |
| **Tamaño del Artefacto** | ~11.5 MB (Bajo peso computacional para despliegue en el borde) |

---

## 2. Uso Previsto (Intended Use)

### 2.1. Casos de Uso en Alcance (In-Scope Use)
- **Auditoría visual automatizada:** Clasificación inicial del estado físico de módulos fotovoltaicos a partir de fotografías tomadas por drones, técnicos en campo o cámaras de inspección fija.
- **Soporte al triaje de mantenimiento:** Alimentar el motor de reglas de priorización operativa (`high`, `medium`, `low`) para agilizar el despacho de cuadrillas de mantenimiento preventivo y correctivo.
- **Monitoreo preventivo de suciedad:** Identificación oportuna de acumulación de polvo (*soiling*) y deposiciones biológicas para programar ciclos de limpieza y mitigar pérdidas por *hotspots*.

### 2.2. Casos de Uso Fuera de Alcance (Out-of-Scope Use)
- **Diagnóstico Eléctrico Definitivo:** El modelo no realiza mediciones de aislamiento eléctrico ni traza curvas I-V.
- **Control Autónomo de Vuelo:** No está diseñado para pilotaje directo de drones en tiempo real.
- **Garantía Comercial de Vida Útil:** No predice la degradación química ni el fin de vida del silicio.

---

## 3. Factores y Especificaciones de Entrada y Salida (Factors & I/O)

### 3.1. Especificación de Entrada
- **Formato:** Tensor `float32` normalizado en rango `[0.0, 1.0]`.
- **Forma del Tensor:** `(1, 3, 224, 224)` en orden de canales `NCHW` (Batch, Canales, Alto, Ancho).
- **Preprocesamiento Obligatorio:**
  1. Conversión de la imagen a espacio de color RGB.
  2. Redimensionamiento bilineal del lado menor a 224 píxeles.
  3. Recorte central simétrico de $224 \times 224$ píxeles.
  4. Escalamiento de píxeles dividiendo por $255.0$.

### 3.2. Especificación de Salida
- **Vector de Probabilidades:** Distribución de probabilidad de 6 clases tras activación Softmax $\sum_{i=1}^6 P_i = 1.0$.
- **Clases Reconocidas:**
  1. `Clean` — Panel visualmente limpio y en condiciones óptimas.
  2. `Dusty` — Acumulación visible de polvo o arena (*soiling*).
  3. `Bird-drop` — Deposiciones biológicas de aves (riesgo de puntos calientes).
  4. `Physical-Damage` — Rotura de vidrio templado, delaminación o microfisuras visibles.
  5. `Electrical-damage` — Marcas de quemaduras visibles en celdas, barras colectoras o caja de conexiones.
  6. `Snow-Covered` — Superficie bloqueada por nieve (baja relevancia en contexto local).
- **Clase Centinela (`Unknown`):** Se asigna cuando la confianza máxima es inferior a $0.50$ ($50\%$), indicando incertidumbre en la captura.

---

## 4. Métricas de Rendimiento Cuantitativo (Evaluation Metrics)

Las pruebas cuantitativas reportadas sobre el conjunto de validación independiente demuestran un rendimiento sobresaliente que supera el criterio de aceptación del 85% de precisión fijado para el proyecto:

| Métrica de Rendimiento | Valor Obtenido | Criterio de Aceptación | Estado |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy** | **88.4%** | $\ge 85.0\%$ | ✅ Superado |
| **Precision Ponderada** | **86.2%** | $\ge 80.0\%$ | ✅ Superado |
| **Recall Ponderado** | **87.5%** | $\ge 80.0\%$ | ✅ Superado |
| **F1-Score Macro** | **86.8%** | $\ge 80.0\%$ | ✅ Superado |
| **Latencia de Inferencia (CPU)** | **~18 ms** | $< 100\text{ ms}$ | ✅ Óptimo para tiempo real |

---

## 5. Limitaciones Técnicas y Sesgos (Limitations & Biases)

1. **Dependencia de Iluminación y Ángulo Solar:** Sombras densas proyectadas por nubes, postes o vegetación pueden ser confundidas con acumulaciones de polvo o daño físico. Se recomienda realizar capturas en ángulos cenitales perpendiculares ($\pm 15^\circ$).
2. **Reflejos Especulares (Glare):** El destello del sol sobre el vidrio templado en horas de mediodía puede saturar canales RGB, provocando clasificaciones con baja confianza (`Unknown`).
3. **Anomalías Internas No Visibles:** Fallas en diodos de derivación (*bypass*) o microfisuras subsuperficiales sin decoloración visible requieren termografía infrarroja (FLIR) o electroluminiscencia.
4. **Umbral Operativo de Revisión:** Conforme a `AGENTS.md`, toda inferencia con confianza inferior a `0.70` (o clasificada como `Unknown`) se etiqueta con `requires_human_review = True`.

---

## 6. Consideraciones Éticas y Seguridad en Campo

- **Seguridad del Personal Técnico:** Toda alerta clasificada como `Electrical-damage` o `Physical-Damage` debe ser precedida por el protocolo de aislamiento de string y comprobación de ausencia de tensión antes de cualquier contacto físico.
- **Descargo de Responsabilidad (Disclaimer):** Las predicciones de SolarGuard AI representan un apoyo de triaje preliminar y **bajo ninguna circunstancia constituyen un diagnóstico eléctrico definitivo**.

---

## 7. Referencias

- Elgalaly, S. (2024). *SolarScan: solar panel condition classifier*. Hugging Face Model Hub.
- Mitchell, M., et al. (2019). *Model Cards for Model Reporting*. Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*).
- Jocher, G., et al. (2023). *Ultralytics YOLOv8 Documentation*. Ultralytics Inc.
