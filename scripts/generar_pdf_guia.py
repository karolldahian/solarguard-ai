"""
Script para generar el PDF con el Guion Ejecutivo de Exposición:
Desacoplamiento gRPC y Monitoreo MLOps - SolarGuard AI.
"""

import sys
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Linea superior de pie de pagina
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(40, 35, letter[0] - 40, 35)
        
        # Texto pie de pagina
        texto_izq = "SolarGuard AI • Especialización en Inteligencia Artificial — UAO"
        texto_der = f"Página {self._pageNumber} de {page_count}"
        self.drawString(40, 24, texto_izq)
        self.drawRightString(letter[0] - 40, 24, texto_der)
        self.restoreState()

def generar_pdf():
    carpeta_salida = Path(r"c:\Users\jalvarado\Documents\ESPECIALIZACION IA\DESARROLLO DE PROYECTOS DE INTELIGENCIA ARTIFICIAL\SOLAR")
    ruta_pdf = carpeta_salida / "Guia_Exposicion_SolarGuard_AI.pdf"
    
    doc = SimpleDocTemplate(
        str(ruta_pdf),
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=45
    )
    
    styles = getSampleStyleSheet()
    
    # Colores tematicos
    AZUL_PRIMARIO = colors.HexColor("#1E3A8A")
    AZUL_SECUNDARIO = colors.HexColor("#2563EB")
    GRIS_TEXTO = colors.HexColor("#1E293B")
    GRIS_FONDO = colors.HexColor("#F8FAFC")
    GRIS_BORDE = colors.HexColor("#E2E8F0")
    VERDE_EXITO = colors.HexColor("#0D9488")
    
    # Estilos personalizados
    titulo_style = ParagraphStyle(
        'TituloDoc',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=AZUL_PRIMARIO,
        spaceAfter=4
    )
    
    subtitulo_style = ParagraphStyle(
        'SubtituloDoc',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#475569"),
        spaceAfter=12
    )
    
    h1_style = ParagraphStyle(
        'Header1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=AZUL_PRIMARIO,
        spaceBefore=10,
        spaceAfter=6
    )
    
    h2_style = ParagraphStyle(
        'Header2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=AZUL_SECUNDARIO,
        spaceBefore=6,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=GRIS_TEXTO,
        spaceAfter=4
    )
    
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0F172A")
    )
    
    quote_style = ParagraphStyle(
        'CierreQuote',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=15,
        textColor=AZUL_PRIMARIO,
        alignment=1
    )

    story = []
    
    # --- ENCABEZADO ---
    story.append(Paragraph("SolarGuard AI — Guion Ejecutivo de Exposición", titulo_style))
    story.append(Paragraph("<b>Expositor:</b> Jarvin David Alvarado Garcés &nbsp;|&nbsp; <b>Especialidad:</b> Desacoplamiento gRPC y Monitoreo MLOps<br/><b>Especialización en IA</b> — Universidad Autónoma de Occidente (UAO)", subtitulo_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=AZUL_SECUNDARIO, spaceAfter=10))

    # --- INTRODUCCION GENERAL ---
    story.append(Paragraph("1. Mensaje de Apertura (1 Minuto)", h1_style))
    texto_intro = (
        "<i>«Profesor, en SolarGuard AI mi contribución se centró en llevar el modelo de Machine Learning a un "
        "entorno de producción desacoplado y observable. Desarrollé la arquitectura de microservicio gRPC sobre HTTP/2 "
        "para aislar la interfaz de la carga computacional, e implementé el sistema de monitoreo MLOps con MLflow Tracking "
        "para auditar latencias milisegundo a milisegundo y confianzas de inferencia en tiempo real con tolerancia a fallos.»</i>"
    )
    story.append(Paragraph(texto_intro, body_style))
    story.append(Spacer(1, 6))

    # --- PILAR 1: DESACOPLAMIENTO GRPC ---
    story.append(Paragraph("2. Pilar 1: Desacoplamiento con Microservicio gRPC", h1_style))
    
    pilar1_data = [
        [Paragraph("<b>Componente / Archivo</b>", body_bold), Paragraph("<b>Función Técnica</b>", body_bold), Paragraph("<b>Qué Explicar al Profesor</b>", body_bold)],
        [
            Paragraph("<b>Contrato Proto3</b><br/><code>proto/solarguard.proto</code>", body_style),
            Paragraph("Define los RPCs <code>ClasificarImagen</code> y <code>VerificarServicio</code> con payloads binarios tipados.", body_style),
            Paragraph("Se transmiten bytes crudos sobre HTTP/2. Ahorra 33% de sobrecarga vs Base64/JSON y asegura contrato estricto frontend-backend.", body_style)
        ],
        [
            Paragraph("<b>Servidor gRPC</b><br/><code>servidor_grpc.py</code>", body_style),
            Paragraph("Servidor multihilo con <code>ThreadPoolExecutor(max_workers=10)</code> escuchando en puerto <code>50051</code>.", body_style),
            Paragraph("Orquesta el flujo: validación &rarr; ingesta &rarr; preprocesamiento NCHW &rarr; ONNX Runtime &rarr; telemetría &rarr; ticket.", body_style)
        ],
        [
            Paragraph("<b>Cliente gRPC</b><br/><code>cliente_grpc.py</code>", body_style),
            Paragraph("Stub cliente reutilizable con <code>timeout=30s</code> y verificación de salud de red.", body_style),
            Paragraph("Desacopla Streamlit. Si el backend cae, la UI no se bloquea; propaga error controlado en lugar de congelar la pantalla.", body_style)
        ],
        [
            Paragraph("<b>Compilador & Fix</b><br/><code>scripts/grpc_gen.py</code>", body_style),
            Paragraph("Invoca <code>grpc_tools.protoc</code> y parchea imports relativos automáticamente.", body_style),
            Paragraph("Resuelve el fallo de importación absoluta de protoc para empaquetar gRPC como módulo estándar de Python.", body_style)
        ],
    ]
    t_pilar1 = Table(pilar1_data, colWidths=[130, 180, 220])
    t_pilar1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, GRIS_BORDE),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_pilar1)
    story.append(Spacer(1, 10))

    # --- PILAR 2: MONITOREO MLOPS ---
    story.append(Paragraph("3. Pilar 2: Monitoreo MLOps con MLflow Tracking", h1_style))
    
    pilar2_data = [
        [Paragraph("<b>Aspecto Técnico</b>", body_bold), Paragraph("<b>Implementación (mlflow_tracking.py)</b>", body_bold), Paragraph("<b>Valor Operativo en Producción</b>", body_bold)],
        [
            Paragraph("<b>Concurrencia Thread-Safe</b>", body_style),
            Paragraph("Patrón Singleton con <code>threading.Lock()</code> en <code>_get_client()</code>.", body_style),
            Paragraph("Evita condiciones de carrera cuando múltiples hilos gRPC registran inferencias simultáneamente.", body_style)
        ],
        [
            Paragraph("<b>Latencias Granulares</b>", body_style),
            Paragraph("Métricas: <code>total_latency_ms</code>, <code>ingestion_latency_ms</code>, <code>preprocessing_latency_ms</code>, <code>inference_latency_ms</code>.", body_style),
            Paragraph("Permite aislar cuellos de botella: diagnostica si la demora ocurre en la red, decodificación de imagen o cómputo ONNX.", body_style)
        ],
        [
            Paragraph("<b>Corridas Anidadas</b><br/>(Nested Runs)", body_style),
            Paragraph("Ejecución con <code>mlflow.start_run(nested=True)</code> por cada solicitud gRPC.", body_style),
            Paragraph("Aísla cada panel analizado bajo el experimento <code>solarguard-inference</code> evitando sobreescritura de parámetros.", body_style)
        ],
        [
            Paragraph("<b>Modo Degradado</b><br/>(Graceful Degradation)", body_style),
            Paragraph("Función <code>is_enabled()</code> con no-op seguro ante caída del servidor MLflow.", body_style),
            Paragraph("Alta disponibilidad: si MLflow se apaga, la planta solar sigue clasificando paneles y emitiendo tickets sin interrupciones.", body_style)
        ],
    ]
    t_pilar2 = Table(pilar2_data, colWidths=[130, 180, 220])
    t_pilar2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, GRIS_BORDE),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_pilar2)
    story.append(Spacer(1, 10))

    # --- GUIA DE PANTALLAS EN VIVO ---
    story.append(Paragraph("4. Guía de Clics y Qué Mostrar en Pantalla", h1_style))
    
    pasos_data = [
        [Paragraph("<b>Paso</b>", body_bold), Paragraph("<b>Pantalla / URL</b>", body_bold), Paragraph("<b>Acción a Ejecutar</b>", body_bold), Paragraph("<b>Qué Decir al Profesor</b>", body_bold)],
        [
            Paragraph("<b>1. Repositorio</b>", body_style),
            Paragraph("GitHub PRs & Actions", body_style),
            Paragraph("Mostrar PRs cerrados con Assignee y Reviewer par. Mostrar CI en verde en Ubuntu y Windows.", body_style),
            Paragraph("«Aplicamos Gitflow y Conventional Commits (feat, fix, chore). Ningún cambio entró sin revisión y CI verde.»", body_style)
        ],
        [
            Paragraph("<b>2. Pruebas</b>", body_style),
            Paragraph("Terminal PowerShell", body_style),
            Paragraph("Ejecutar <code>make test</code>.<br/>Ver pasar <b>284 pruebas al 100%</b> en ~15s.", body_style),
            Paragraph("«284 pruebas bajo patrón AAA con 89.90% de cobertura. Sin dependencias externas ni tokens de red.»", body_style)
        ],
        [
            Paragraph("<b>3. gRPC UI</b>", body_style),
            Paragraph("Streamlit<br/><code>localhost:8501</code>", body_style),
            Paragraph("Click en <b>'Verificar conexion gRPC'</b> (barra lateral). Subir <code>panel_polvoriento.jpg</code>.", body_style),
            Paragraph("«La interfaz consulta al microservicio remoto en puerto 50051 y devuelve el diagnóstico y ticket estructurado.»", body_style)
        ],
        [
            Paragraph("<b>4. MLOps UI</b>", body_style),
            Paragraph("MLflow UI<br/><code>127.0.0.1:5000</code>", body_style),
            Paragraph("En <b>Model training</b>: ver los runs generados, columnas de latencia, marcar 2 y pulsar icono 📈 (Chart).", body_style),
            Paragraph("«Cada inferencia queda auditada con su latencia por etapa y confianza para análisis de degradación en producción.»", body_style)
        ],
    ]
    t_pasos = Table(pasos_data, colWidths=[70, 110, 170, 180])
    t_pasos.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
        ('GRID', (0,0), (-1,-1), 0.5, GRIS_BORDE),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_pasos)
    story.append(Spacer(1, 14))

    # --- FRASE DE CIERRE ---
    caja_cierre_data = [
        [Paragraph("<b>FRASE CLAVE DE CIERRE (Memorizar o Leer Directamente):</b>", body_bold)],
        [Paragraph(
            "«La arquitectura del sistema sigue un patrón desacoplado estricto auditado por herramientas automatizadas. "
            "Toda la orquestación local se ejecuta mediante nuestro Makefile con Astral uv (cero pip), y en la nube "
            "nuestro CI de GitHub Actions valida automáticamente en Linux y Windows que cada commit cumpla con Ruff, "
            "cero warnings y 89.9% de cobertura en 284 pruebas antes de permitir cualquier merge.»",
            quote_style
        )]
    ]
    t_cierre = Table(caja_cierre_data, colWidths=[530])
    t_cierre.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1, AZUL_SECUNDARIO),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_cierre)

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generado exitosamente en: {ruta_pdf}")

if __name__ == "__main__":
    generar_pdf()
