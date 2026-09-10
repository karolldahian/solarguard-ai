"""
Preprocesamiento de imagenes para SolarGuard AI.

Antes de pasarle una imagen al modelo, hay que prepararla.
El modelo solarscan-yolov8n-cls espera que la imagen tenga:
  - Tamano exacto de 224 x 224 pixeles
  - Colores en rango 0.0 a 1.0 (no de 0 a 255 como vienen normalmente)
  - Un formato especifico de array que entienda ONNX Runtime

Aprendi que esto se llama "pipeline de preprocesamiento" y es muy comun
en proyectos de vision por computador. Cada paso transforma la imagen
un poco hasta dejarla lista para el modelo.
"""

# numpy es la libreria que usamos para manejar los arrays numericos
# PIL (Pillow) es la que ya usabamos en ingesta.py para abrir imagenes
import numpy as np
from PIL import Image

# Importamos el tipo LoadedImage que definimos en ingesta.py
# asi podemos recibir imagenes ya validadas
from solarguard_ai.ingesta import LoadedImage

# Este es el tamano que pide el modelo. Lo pongo como constante
# para no escribir 224 en varios lugares y confundirme.
TAMANO_MODELO = 224


def resize_lado_corto(imagen: Image.Image, tamano: int = TAMANO_MODELO) -> Image.Image:
    """
    Cambia el tamano de la imagen para que el lado mas corto mida 'tamano' pixeles.
    El lado mas largo se ajusta automaticamente para no deformar la imagen.

    Por ejemplo: si la imagen es 800 x 600, despues de aplicar esta funcion
    quedaria de 299 x 224 aproximadamente (porque 600 es el lado corto).

    Aprendi que esto se llama "resize proporcional" y es importante para
    no aplastar ni estirar las imagenes antes del recorte.
    """
    ancho_original, alto_original = imagen.size

    # Averiguo cual es el lado mas corto
    if ancho_original < alto_original:
        # El ancho es el lado corto, lo fijo en 'tamano'
        nuevo_ancho = tamano
        # Calculo el alto proporcional usando regla de tres
        nuevo_alto = int(alto_original * tamano / ancho_original)
    else:
        # El alto es el lado corto (o son iguales), lo fijo en 'tamano'
        nuevo_alto = tamano
        # Calculo el ancho proporcional
        nuevo_ancho = int(ancho_original * tamano / alto_original)

    # LANCZOS es un metodo de remuestreo de alta calidad
    # lo recomiendan cuando se achica una imagen porque conserva mejor los detalles
    imagen_redimensionada = imagen.resize((nuevo_ancho, nuevo_alto), Image.LANCZOS)

    return imagen_redimensionada


def recorte_central(imagen: Image.Image, tamano: int = TAMANO_MODELO) -> Image.Image:
    """
    Recorta un cuadrado del centro de la imagen.

    Despues del resize, la imagen puede ser algo como 299 x 224.
    Esta funcion toma solo los 224 x 224 pixeles del centro.

    El centro es la parte mas importante de una foto de panel solar
    (normalmente ahi esta el panel principal).
    """
    ancho, alto = imagen.size

    # Calculo desde donde empieza el recorte en X (horizontal)
    # Ejemplo: si ancho=299 y tamano=224 → inicio_x = (299-224)/2 = 37
    inicio_x = (ancho - tamano) // 2

    # Lo mismo para Y (vertical)
    inicio_y = (alto - tamano) // 2

    # El recorte termina 'tamano' pixeles despues del inicio
    fin_x = inicio_x + tamano
    fin_y = inicio_y + tamano

    # crop() de Pillow recibe (izquierda, arriba, derecha, abajo)
    imagen_recortada = imagen.crop((inicio_x, inicio_y, fin_x, fin_y))

    return imagen_recortada


def convertir_a_array(imagen: Image.Image) -> np.ndarray:
    """
    Convierte la imagen de Pillow a un array de numpy con valores entre 0.0 y 1.0.

    Las imagenes normalmente guardan los colores como numeros enteros de 0 a 255.
    El modelo espera numeros decimales de 0.0 a 1.0, asi que hay que dividir entre 255.

    Ademas, el array tiene que estar en formato CHW (Canales, Alto, Ancho)
    pero numpy lo entrega en HWC (Alto, Ancho, Canales), asi que hay que reorganizarlo.
    Esto se hace con transpose().
    """
    # Convierto la imagen a array de numpy
    # El resultado tiene forma (alto, ancho, 3) porque son 3 canales RGB
    array_hwc = np.array(imagen, dtype=np.float32)

    # Divido entre 255 para que los valores queden entre 0.0 y 1.0
    # Aprendi que esto se llama "normalizacion"
    array_normalizado = array_hwc / 255.0

    # Reorganizo de (alto, ancho, canales) a (canales, alto, ancho)
    # transpose((2, 0, 1)) mueve el eje 2 al frente
    array_chw = array_normalizado.transpose((2, 0, 1))

    return array_chw


def agregar_dimension_batch(array: np.ndarray) -> np.ndarray:
    """
    Agrega una dimension extra al inicio del array.

    ONNX Runtime espera que le mandes varios imagenes a la vez (un "batch").
    Como solo mandamos una, igual tenemos que poner esa dimension.
    El array pasa de forma (3, 224, 224) a forma (1, 3, 224, 224).

    np.expand_dims con axis=0 inserta una dimension nueva en la posicion 0.
    """
    array_con_batch = np.expand_dims(array, axis=0)
    return array_con_batch


def preprocesar(imagen_cargada: LoadedImage) -> np.ndarray:
    """
    Funcion principal que aplica todos los pasos de preprocesamiento.

    Recibe una imagen ya validada (LoadedImage de ingesta.py) y
    devuelve el array listo para darselo al modelo.

    Los pasos son:
      1. Resize del lado corto a 224 px
      2. Recorte central de 224 x 224 px
      3. Convertir a array float32 normalizado
      4. Agregar dimension de batch

    Al final el array tiene forma (1, 3, 224, 224).
    """
    # Paso 1: redimensionar
    imagen_resize = resize_lado_corto(imagen_cargada.image, TAMANO_MODELO)

    # Paso 2: recortar el centro
    imagen_crop = recorte_central(imagen_resize, TAMANO_MODELO)

    # Paso 3: pasar a array normalizado en formato CHW
    array = convertir_a_array(imagen_crop)

    # Paso 4: agregar la dimension de batch
    tensor_final = agregar_dimension_batch(array)

    return tensor_final


# Lista de cosas que este modulo exporta para que otros archivos puedan importarlas
__all__ = [
    "TAMANO_MODELO",
    "resize_lado_corto",
    "recorte_central",
    "convertir_a_array",
    "agregar_dimension_batch",
    "preprocesar",
]
