"""
SolarGuard AI - Interfaz web (Streamlit).

Esta es la pantalla que ve el usuario. Su unica mision es:
  1. Pedirle una foto de un panel solar.
  2. Enviarla al backend por gRPC.
  3. Mostrar el ticket de mantenimiento que devuelve.

Aprendi que es buena practica que la interfaz NO haga el trabajo pesado:
toda la clasificacion se hace en el backend (solarguard_ai.servidor_grpc)
y aqui solo dibujamos el resultado. El modulo cliente_grpc es el puente.

Para iniciarla:
    uv run streamlit run app.py
Y no olvides tener corriendo el backend en otra terminal:
    uv run python -m solarguard_ai.servidor_grpc
"""

from __future__ import annotations

from io import BytesIO

import grpc
import streamlit as st

# El cliente gRPC que habla con el backend
from solarguard_ai.cliente_grpc import (
    DIRECCION_POR_DEFECTO,
    TIEMPO_ESPERA,
    clasificar_imagen,
    crear_canal,
    verificar_servidor,
)

# Validacion rapida de la imagen antes de enviarla (para un primer filtro)
from solarguard_ai.ingesta import ImageIngestionError, load_image

# Personalizacion de la pagina
st.set_page_config(
    page_title="SolarGuard AI",
    page_icon=":sunny:",
    layout="wide",
)

# Titulo principal de la aplicacion
st.title("SolarGuard AI - Inspeccion de paneles solares")
st.caption(
    "Sube una foto de un panel y el backend la clasifica para "
    "priorizar su mantenimiento. La respuesta llega por gRPC."
)

# ---------------------------------------------------------------------------
# Barra lateral: conexion al backend
# ---------------------------------------------------------------------------
st.sidebar.header("Conexion al backend")

# Aqui el usuario puede poner la direccion si no usa la de por defecto
direccion = st.sidebar.text_input(
    "Direccion del backend",
    value=DIRECCION_POR_DEFECTO,
    help="Formato: host:puerto (ej: localhost:50051)",
)

# Boton para comprobar que el backend este vivo
if st.sidebar.button("Verificar conexion"):
    with st.sidebar.spinner("Preguntando al backend..."):
        try:
            mensaje = verificar_servidor(crear_canal(direccion))
            st.sidebar.success(mensaje)
        except grpc.RpcError as error:
            st.sidebar.error(f"No se pudo conectar: {error.code() and error.details()}")

st.sidebar.divider()
st.sidebar.caption(
    "Recordatorio: el backend se inicia en otra terminal con "
    "`uv run python -m solarguard_ai.servidor_grpc`"
)

# ---------------------------------------------------------------------------
# Carga de la imagen
# ---------------------------------------------------------------------------
archivo = st.file_uploader(
    "Sube la foto del panel solar",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

if archivo is not None:
    # Muestro una miniatura para que el usuario vea lo que subio
    st.image(archivo, caption=archivo.name, width=360)

    # Valido localmente antes de enviar: mejor fallar temprano que enviar basura
    try:
        # load_image ademas nos obtiene la imagen en RGB para la vista previa
        cargada = load_image(BytesIO(archivo.getvalue()))
        st.success(f"Imagen valida: {cargada.width}x{cargada.height} px")
    except ImageIngestionError as error:
        st.error(f"La imagen no es valida: {error}")
        archivo = None  # No dejamos continuar con un archivo malo

# ---------------------------------------------------------------------------
# Boton de clasificacion
# ---------------------------------------------------------------------------
if archivo is not None and st.button("Clasificar panel", type="primary"):
    # El boton de Streamlit solo llama al servidor cuando se pulsa
    with st.spinner("Enviando imagen al backend y clasificando..."):
        try:
            # Creo el canal gRPC y hago la llamada remota
            canal = crear_canal(direccion)
            resultado = clasificar_imagen(
                canal,
                bytes_imagen=archivo.getvalue(),
                nombre=archivo.name,
                tiempo_espera=TIEMPO_ESPERA,
            )
            # Guardo el ticket junto con el nombre del archivo que lo origino,
            # asi no mostramos un resultado de una foto anterior.
            st.session_state["resultado_ticket"] = resultado
            st.session_state["resultado_de"] = archivo.name
        except grpc.RpcError as error:
            st.error(
                "El backend devolvio un error. Verifica que este corriendo "
                f"(detalle: {error.details()})."
            )

# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
# Solo muestro el ticket si corresponde al archivo que esta cargado ahora
resultado = st.session_state.get("resultado_ticket")
resultado_coincide = (
    archivo is not None and st.session_state.get("resultado_de") == archivo.name
)
if resultado is not None and resultado_coincide:
    st.header("Ticket de mantenimiento")

    # Distribucion del ticket: izquierda la info general, derecha detalle
    columna_resumen, columna_detalle = st.columns([1, 2])

    with columna_resumen:
        st.metric("Condicion", resultado["condicion"])
        # Confianza como porcentaje (ej: 0.85 -> 85%)
        st.metric("Confianza", f"{resultado['confianza'] * 100:.1f}%")
        st.metric("Prioridad", resultado["prioridad"])

    with columna_detalle:
        # Barra de progreso visual de la confianza
        st.progress(resultado["confianza"], text="Nivel de confianza del modelo")

        st.markdown(f"**Que significa:** {resultado['descripcion']}")
        st.markdown(f"**Que hacer:** {resultado['accion']}")
        st.caption(f"Generado el {resultado['fecha_hora']}")

    # Pie de advertencia, igual que lo dice el README
    st.warning(
        "Esta es una clasificacion visual inicial. Cualquier indicio de dano "
        "fisico o electrico debe confirmarse con personal tecnico calificado."
    )
