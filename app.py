"""Entrypoint de la interfaz Streamlit de SolarGuard AI."""

from __future__ import annotations

import grpc
import streamlit as st

from solarguard_ai.cliente_grpc import (
    DIRECCION_POR_DEFECTO,
    TIEMPO_ESPERA,
    clasificar_imagen,
    crear_canal,
    verificar_servidor,
)
from solarguard_ai.inferencia import InferenceError, PredictionResult
from solarguard_ai.ingesta import ImageIngestionError, load_image
from solarguard_ai.preprocesamiento import preprocess_for_solarscan
from solarguard_ai.priorizacion import (
    PrioritizationError,
    PriorityResult,
    prioritize_prediction,
)
from solarguard_ai.view.alertas import render_ticket_section
from solarguard_ai.view.batch import render_batch_analysis
from solarguard_ai.view.dashboard import render_dashboard
from solarguard_ai.view.diagnostico import (
    render_configuration_error,
    render_diagnosis,
    render_model_unavailable,
)
from solarguard_ai.view.historial import (
    add_analysis,
    build_analysis_record,
    get_history,
)
from solarguard_ai.view.mapa_riesgo import render_risk_heatmap
from solarguard_ai.view.modelo import (
    get_inference_service,
    get_priority_config,
    resolve_model_path,
)
from solarguard_ai.view.pagina_principal import (
    render_header,
    render_image_metadata,
    render_image_preview,
    render_usage_guide,
    upload_image,
)

st.set_page_config(page_title="SolarGuard AI", page_icon="☀️", layout="wide")

# ---------------------------------------------------------------------------
# Barra lateral: Arquitectura y Modo de Inferencia (gRPC / Local)
# ---------------------------------------------------------------------------
st.sidebar.header("Arquitectura de Servicios")
modo_backend = st.sidebar.radio(
    "Modo de Inferencia",
    ["Microservicio gRPC (Desacoplado)", "In-Process Directo (Local)"],
    index=0,
    help=(
        "gRPC desacopla la interfaz comunicandose con el puerto 50051. "
        "In-Process ejecuta ONNX directamente en Streamlit."
    ),
)

direccion_grpc = DIRECCION_POR_DEFECTO
if modo_backend == "Microservicio gRPC (Desacoplado)":
    direccion_grpc = st.sidebar.text_input(
        "Direccion del Backend gRPC",
        value=DIRECCION_POR_DEFECTO,
        help="host:puerto del servidor gRPC (ej: localhost:50051)",
    )
    if st.sidebar.button("Verificar conexion gRPC"):
        with st.sidebar.spinner("Comprobando backend..."):
            try:
                canal_test = crear_canal(direccion_grpc)
                mensaje = verificar_servidor(canal_test, tiempo_espera=3.0)
                st.sidebar.success(f"Conectado: {mensaje}")
            except grpc.RpcError as error:
                st.sidebar.error(f"Fallo gRPC: {error.details() or error.code()}")
            except (OSError, TimeoutError) as error:
                st.sidebar.error(f"No se pudo conectar: {error}")
    st.sidebar.caption("Backend requerido: `make servidor` o `make docker-up`")

st.sidebar.divider()
st.sidebar.markdown(
    "**SolarGuard AI v1.0**  \n"
    "Especializacion en IA — UAO  \n"
    "[Documentacion Docker](docs/despliegue_docker.md)"
)

render_header()
render_usage_guide()

individual_tab, batch_tab, summary_tab = st.tabs(
    ["Analisis individual", "Analisis por lote", "Resumen de la sesion"]
)

with individual_tab:
    uploaded_file = upload_image()

    if uploaded_file is not None:
        try:
            loaded = load_image(uploaded_file)
        except ImageIngestionError as e:
            st.error(str(e))
        else:
            render_image_preview(loaded)
            render_image_metadata(loaded)

            prediction: PredictionResult | None = None
            priority: PriorityResult | None = None

            if modo_backend == "Microservicio gRPC (Desacoplado)":
                try:
                    with st.spinner("Consultando servicio remoto gRPC..."):
                        canal = crear_canal(direccion_grpc)
                        resultado_grpc = clasificar_imagen(
                            canal,
                            bytes_imagen=uploaded_file.getvalue(),
                            nombre=loaded.source,
                            tiempo_espera=TIEMPO_ESPERA,
                        )
                        clase = resultado_grpc["condicion"]
                        confianza = float(resultado_grpc["confianza"])
                        prioridad_str = resultado_grpc["prioridad"].lower()
                        accion = resultado_grpc["accion"]
                        descripcion = resultado_grpc["descripcion"]

                        prediction = PredictionResult(
                            predicted_class=clase,
                            confidence=confianza,
                            probabilities={clase: confianza},
                        )
                        priority = PriorityResult(
                            priority=prioridad_str,
                            recommended_action=accion,
                            requires_human_review=(
                                prioridad_str == "medium" or clase == "Unknown"
                            ),
                            reason=descripcion,
                        )
                except grpc.RpcError as error:
                    st.error(
                        f"Error de comunicacion gRPC ({direccion_grpc}): "
                        f"{error.details() or error.code()}. "
                        "Verifique que el backend gRPC este activo (`make servidor` o `make docker-up`)."
                    )
                except (OSError, ValueError) as error:
                    st.error(f"Error inesperado en llamada gRPC: {error}")
            else:
                # In-Process Directo (Local)
                try:
                    service = get_inference_service()
                except InferenceError:
                    render_model_unavailable(resolve_model_path())
                else:
                    try:
                        tensor = preprocess_for_solarscan(loaded)
                        prediction = service.predict(tensor)
                    except (InferenceError, ValueError) as e:
                        st.error(f"No se pudo completar el diagnostico visual: {e}")
                    else:
                        try:
                            config = get_priority_config()
                        except PrioritizationError as e:
                            render_configuration_error(e)
                        else:
                            try:
                                priority = prioritize_prediction(
                                    prediction,
                                    review_threshold=config.review_confidence,
                                )
                            except PrioritizationError as e:
                                st.error(
                                    f"No se pudo calcular la prioridad de mantenimiento: {e}"
                                )

            if prediction is not None and priority is not None:
                render_diagnosis(prediction, priority)
                render_ticket_section(
                    priority,
                    prediction.predicted_class,
                    prediction.confidence,
                )
                analysis_record = build_analysis_record(
                    image_name=loaded.source,
                    image_bytes=uploaded_file.getvalue(),
                    prediction=prediction,
                    priority=priority,
                )
                add_analysis(analysis_record)
    else:
        st.info("Suba una imagen de un panel solar para comenzar el analisis.")

with batch_tab:
    render_batch_analysis()

with summary_tab:
    render_dashboard(get_history())
    render_risk_heatmap(get_history())
