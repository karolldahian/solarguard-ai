"""
Inferencia del modelo solarscan-yolov8n-cls para SolarGuard AI.

Este modulo es el que "conecta" el tensor preprocesado con una prediccion.

IMPORTANTE: Aprendi que en el proyecto todavia no estan listos el archivo
del modelo (best.onnx) ni el modulo de inferencia real con ONNX Runtime.
Por eso este modulo es una SIMULACION: en vez de llamar al modelo, genera
un resultado "falso pero seria" usando una semilla.

Lo importante es que el resto del sistema (el servidor gRPC, la interfaz)
ya se puede probar de punta a punta, y cuando llegue el modelo real,
solo habra que cambiar la funcion predecir() por la que use ONNX Runtime.
El resto del codigo no se tiene que tocar.

Decidi que la simulacion sea DETERMINISTA: la misma semilla siempre da
el mismo resultado. Asi, la misma imagen siempre produce la misma
clasificacion y las pruebas no dependen del azar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Estas son las 6 condiciones que reconoce el modelo segun el README.
# Las dejo como constante para que este modulo y tickets.py hablen el mismo idioma.
CLASES = [
    "Clean",
    "Dusty",
    "Bird-drop",
    "Physical-Damage",
    "Electrical-damage",
    "Snow-Covered",
]

# Probabilidad "tentativa" de cada clase en la simulacion.
# La deje mas cargada hacia lo comun (paneles normales o con polvo),
# como suele pasar en un parque solar real.
PROBABILIDADES = np.array([0.35, 0.30, 0.10, 0.12, 0.05, 0.08], dtype=np.float64)


@dataclass(frozen=True)
class PrediccionModelo:
    """
    Resultado de la inferencia sobre una sola imagen.

    Contiene exactamente lo que el resto del sistema necesita:
      - condicion: cual de las 6 clases se detecto
      - confianza: cuanto "cree" el modelo (entre 0.0 y 1.0)
    """

    condicion: str
    confianza: float


def predecir(tensor: np.ndarray, semilla: int = 0) -> PrediccionModelo:
    """
    Simula la clasificacion del modelo sobre un tensor preprocesado.

    Recibe:
      - tensor: array con forma (1, 3, 224, 224) como lo deja preprocesar().
      - semilla: numero entero. La misma semilla siempre da el mismo resultado.

    Devuelve una PrediccionModelo con la condicion y la confianza.

    Cuando el equipo de inferencia entregue el modelo real, esta funcion
    se reemplaza por algo como:
        sesion = onnxruntime.InferenceSession("modelos/best.onnx")
        salida = sesion.run(None, {nombre_entrada: tensor})[0]
    y el resto del codigo seguira funcionando igual.
    """
    # Valido la forma del tensor para que cualquier error sea claro desde el inicio
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(
            "El tensor debe tener forma (1, 3, 224, 224), "
            f"pero se recibio: {tensor.shape}"
        )

    # np.random.default_rng crea un generador de azar, pero "plantado"
    # con la semilla: cada vez que se llame con la misma semilla,
    # genera exactamente los mismos numeros (por eso es determinista).
    generador = np.random.default_rng(semilla)

    # Elijo una clase de forma aleatoria pero usando las probabilidades
    # de arriba. choice() con p=... pesa la seleccion.
    condicion = generador.choice(CLASES, p=PROBABILIDADES)

    # La confianza tambien sale del generador "plantado":
    # arranca entre 0.70 y 1.0 para que sea un valor creible.
    confianza = float(generador.uniform(0.70, 1.0))

    return PrediccionModelo(condicion=condicion, confianza=confianza)


# Lo que este modulo exporta para que otros archivos lo puedan usar
__all__ = [
    "CLASES",
    "PROBABILIDADES",
    "PrediccionModelo",
    "predecir",
]