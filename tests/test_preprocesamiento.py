"""
Pruebas para el modulo de preprocesamiento.

Aqui verifico que cada funcion hace exactamente lo que se supone.
Aprendi que las pruebas son importantes para saber si el codigo funciona
sin tener que probarlo manualmente cada vez que hago un cambio.

Para correr las pruebas se usa:
    uv run pytest tests/test_preprocesamiento.py -v
"""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

# Importo las funciones que quiero probar
from solarguard_ai.ingesta import load_image
from solarguard_ai.preprocesamiento import (
    TAMANO_MODELO,
    agregar_dimension_batch,
    convertir_a_array,
    preprocesar,
    recorte_central,
    resize_lado_corto,
)


# --- Funcion auxiliar para crear imagenes de prueba ---

def crear_imagen_prueba(ancho: int, alto: int, modo: str = "RGB") -> Image.Image:
    """Crea una imagen simple de color solido para usar en las pruebas."""
    # El color (100, 150, 200) es un azul medio, solo para tener algo
    return Image.new(modo, (ancho, alto), color=(100, 150, 200))


def crear_imagen_bytes(ancho: int, alto: int, formato: str = "PNG") -> bytes:
    """Crea una imagen en memoria y la devuelve como bytes."""
    imagen = crear_imagen_prueba(ancho, alto)
    buffer = BytesIO()
    imagen.save(buffer, format=formato)
    return buffer.getvalue()


# --- Pruebas de resize_lado_corto ---

def test_resize_con_imagen_mas_ancha_que_alta() -> None:
    """
    Si la imagen es mas ancha que alta, el alto debe quedar en 224
    y el ancho debe ser mayor a 224 (proporcional).
    """
    # Creo una imagen horizontal de 800 x 400
    imagen = crear_imagen_prueba(ancho=800, alto=400)

    resultado = resize_lado_corto(imagen, tamano=224)

    # El lado corto (alto) debe ser exactamente 224
    assert resultado.height == 224
    # El ancho debe ser mayor a 224 porque la imagen era mas ancha
    assert resultado.width > 224


def test_resize_con_imagen_mas_alta_que_ancha() -> None:
    """
    Si la imagen es mas alta que ancha, el ancho debe quedar en 224.
    """
    # Creo una imagen vertical de 300 x 600
    imagen = crear_imagen_prueba(ancho=300, alto=600)

    resultado = resize_lado_corto(imagen, tamano=224)

    # El lado corto (ancho) debe ser exactamente 224
    assert resultado.width == 224
    # El alto debe ser mayor
    assert resultado.height > 224


def test_resize_con_imagen_cuadrada() -> None:
    """
    Si la imagen es cuadrada, ambos lados deben quedar en 224.
    """
    imagen = crear_imagen_prueba(ancho=500, alto=500)

    resultado = resize_lado_corto(imagen, tamano=224)

    assert resultado.width == 224
    assert resultado.height == 224


# --- Pruebas de recorte_central ---

def test_recorte_central_entrega_tamano_exacto() -> None:
    """
    El recorte siempre debe dar una imagen de exactamente 224 x 224.
    """
    # Creo una imagen un poco mas grande que 224 x 224
    imagen = crear_imagen_prueba(ancho=300, alto=250)

    resultado = recorte_central(imagen, tamano=224)

    assert resultado.width == 224
    assert resultado.height == 224


def test_recorte_central_con_imagen_exacta() -> None:
    """
    Si la imagen ya mide 224 x 224, el recorte no la cambia.
    """
    imagen = crear_imagen_prueba(ancho=224, alto=224)

    resultado = recorte_central(imagen, tamano=224)

    assert resultado.width == 224
    assert resultado.height == 224


# --- Pruebas de convertir_a_array ---

def test_convertir_a_array_tiene_forma_correcta() -> None:
    """
    El array debe tener forma (3, 224, 224): canales, alto, ancho.
    """
    imagen = crear_imagen_prueba(ancho=224, alto=224)

    resultado = convertir_a_array(imagen)

    # La forma debe ser (3, 224, 224)
    assert resultado.shape == (3, 224, 224)


def test_convertir_a_array_tipo_float32() -> None:
    """
    El tipo de datos del array debe ser float32 (decimales de 32 bits).
    """
    imagen = crear_imagen_prueba(ancho=224, alto=224)

    resultado = convertir_a_array(imagen)

    assert resultado.dtype == np.float32


def test_convertir_a_array_valores_entre_cero_y_uno() -> None:
    """
    Todos los valores del array deben estar entre 0.0 y 1.0.
    """
    imagen = crear_imagen_prueba(ancho=224, alto=224)

    resultado = convertir_a_array(imagen)

    assert resultado.min() >= 0.0
    assert resultado.max() <= 1.0


# --- Pruebas de agregar_dimension_batch ---

def test_agregar_dimension_batch_forma_correcta() -> None:
    """
    Despues de agregar el batch, la forma debe pasar de
    (3, 224, 224) a (1, 3, 224, 224).
    """
    # Creo un array de prueba con la forma que tendria despues de convertir_a_array
    array_ejemplo = np.zeros((3, 224, 224), dtype=np.float32)

    resultado = agregar_dimension_batch(array_ejemplo)

    assert resultado.shape == (1, 3, 224, 224)


# --- Prueba del pipeline completo ---

def test_preprocesar_entrega_tensor_listo_para_modelo() -> None:
    """
    La funcion preprocesar() debe tomar una imagen cualquiera y dejarla
    lista para el modelo: forma (1, 3, 224, 224) y tipo float32.

    Esta es la prueba mas importante porque verifica que todo el pipeline
    funciona junto correctamente.
    """
    # Creo una imagen de prueba y la cargo usando load_image (que ya valida)
    imagen_bytes = crear_imagen_bytes(ancho=640, alto=480)
    imagen_cargada = load_image(BytesIO(imagen_bytes))

    # Aplico el preprocesamiento completo
    tensor = preprocesar(imagen_cargada)

    # Verifico la forma del tensor
    assert tensor.shape == (1, 3, 224, 224), (
        f"Se esperaba forma (1, 3, 224, 224) pero se obtuvo {tensor.shape}"
    )
    # Verifico el tipo de datos
    assert tensor.dtype == np.float32

    # Verifico que los valores esten en el rango correcto
    assert tensor.min() >= 0.0
    assert tensor.max() <= 1.0


def test_preprocesar_funciona_con_imagen_pequena() -> None:
    """
    Probar con una imagen mas pequena que 224 x 224 para ver que
    el resize la agranda correctamente antes del recorte.
    """
    imagen_bytes = crear_imagen_bytes(ancho=100, alto=80)
    imagen_cargada = load_image(BytesIO(imagen_bytes))

    tensor = preprocesar(imagen_cargada)

    # Aunque la imagen era pequena, el resultado siempre debe ser (1, 3, 224, 224)
    assert tensor.shape == (1, 3, 224, 224)
