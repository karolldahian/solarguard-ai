"""Preprocesamiento de imagenes para SolarGuard AI y SolarScan."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

from .ingesta import LoadedImage

TARGET_SIZE = 224
NormalizationMethod = Literal["minmax", "zscore"]
FloatArray = NDArray[np.float32]


def normalize_pixels(
    pixels: NDArray[np.generic], method: NormalizationMethod = "minmax"
) -> FloatArray:
    """Normaliza pixeles con escala 0..1 o estandarizacion z-score."""
    values = np.asarray(pixels, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Los pixeles contienen valores no finitos.")
    if values.size == 0:
        raise ValueError("No se puede normalizar un arreglo vacio.")

    if method == "minmax":
        # Convierte imagenes uint8 y rangos arbitrarios al intervalo requerido.
        minimum = values.min()
        maximum = values.max()
        if maximum == minimum:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - minimum) / (maximum - minimum)).astype(np.float32)

    if method == "zscore":
        # La desviacion cero se representa con ceros para evitar divisiones invalidas.
        standard_deviation = values.std()
        if standard_deviation == 0:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - values.mean()) / standard_deviation).astype(np.float32)

    raise ValueError(f"Metodo de normalizacion no soportado: {method}")


def preprocess_for_solarscan(
    image: Image.Image | LoadedImage,
    target_size: int = TARGET_SIZE,
) -> FloatArray:
    """Prepara una imagen para SolarScan como tensor float32 con forma NCHW.

    Sigue el preprocesamiento publicado por el modelo: RGB, redimensionamiento
    del lado menor a 224, recorte central de 224x224 y escala de 0 a 1.
    """
    if target_size < 1:
        raise ValueError("target_size debe ser mayor que cero.")

    source_image = image.image if isinstance(image, LoadedImage) else image
    rgb_image = source_image.convert("RGB")
    width, height = rgb_image.size
    if width < 1 or height < 1:
        raise ValueError("La imagen debe tener dimensiones positivas.")

    # Se conserva la proporcion antes del recorte para no deformar el panel.
    scale = target_size / min(width, height)
    resized_size = (round(width * scale), round(height * scale))
    resized_image = rgb_image.resize(resized_size, Image.Resampling.BILINEAR)
    cropped_image = ImageOps.fit(
        resized_image,
        (target_size, target_size),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )

    # ONNX Runtime recibe el orden de ejes NCHW y valores RGB en 0..1.
    channels_last = np.asarray(cropped_image, dtype=np.float32) / 255.0
    channels_first = np.transpose(channels_last, (2, 0, 1))
    return channels_first[np.newaxis, ...].astype(np.float32)


def augment_image(
    image: Image.Image, include_original: bool = True
) -> list[Image.Image]:
    """Genera variantes geometricas simples para ampliar un conjunto de imagenes."""
    augmented: list[Image.Image] = []
    if include_original:
        augmented.append(image.copy())

    # Estas transformaciones no cambian las clases visuales del panel.
    augmented.extend(
        [
            ImageOps.mirror(image),
            ImageOps.flip(image),
            image.rotate(90, expand=True),
        ]
    )
    return augmented


def split_rgb_channels(
    image: Image.Image | NDArray[np.generic],
) -> dict[str, FloatArray]:
    """Separa una imagen RGB en canales normalizados R, G y B."""
    channels_last = _as_channels_last(image)
    if channels_last.shape[2] != 3:
        raise ValueError("Se esperaba una imagen RGB con exactamente 3 canales.")

    normalized = normalize_pixels(channels_last, method="minmax")
    return {
        "red": normalized[:, :, 0],
        "green": normalized[:, :, 1],
        "blue": normalized[:, :, 2],
    }


def calculate_ndvi(
    image: NDArray[np.generic], nir_band: int, red_band: int, epsilon: float = 1e-8
) -> FloatArray:
    """Calcula NDVI para un arreglo multiespectral en formato HWC."""
    nir, red = _select_bands(image, nir_band, red_band)
    return _normalized_difference(nir, red, epsilon)


def calculate_ndwi(
    image: NDArray[np.generic], nir_band: int, green_band: int, epsilon: float = 1e-8
) -> FloatArray:
    """Calcula NDWI para un arreglo multiespectral en formato HWC."""
    nir, green = _select_bands(image, nir_band, green_band)
    return _normalized_difference(nir, green, epsilon)


def extract_spectral_indices(
    image: NDArray[np.generic],
    nir_band: int,
    red_band: int,
    green_band: int,
) -> dict[str, FloatArray]:
    """Extrae NDVI y NDWI sin incorporarlos al tensor RGB de SolarScan."""
    return {
        "ndvi": calculate_ndvi(image, nir_band, red_band),
        "ndwi": calculate_ndwi(image, nir_band, green_band),
    }


def _as_channels_last(image: Image.Image | NDArray[np.generic]) -> NDArray[np.float32]:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.float32)
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("La imagen debe tener forma alto, ancho, canales (HWC).")
    return values


def _select_bands(
    image: NDArray[np.generic], first_band: int, second_band: int
) -> tuple[FloatArray, FloatArray]:
    values = _as_channels_last(image)
    channel_count = values.shape[2]
    if not 0 <= first_band < channel_count or not 0 <= second_band < channel_count:
        raise ValueError(
            f"Los indices de banda deben estar entre 0 y {channel_count - 1}."
        )
    return values[:, :, first_band], values[:, :, second_band]


def _normalized_difference(
    first: FloatArray, second: FloatArray, epsilon: float
) -> FloatArray:
    if epsilon <= 0:
        raise ValueError("epsilon debe ser mayor que cero.")

    denominator = first + second
    # Los pixeles con suma cero no aportan informacion y se fijan en cero.
    result = np.divide(
        first - second,
        denominator,
        out=np.zeros_like(first, dtype=np.float32),
        where=np.abs(denominator) > epsilon,
    )
    return np.clip(result, -1.0, 1.0).astype(np.float32)


__all__ = [
    "TARGET_SIZE",
    "augment_image",
    "calculate_ndvi",
    "calculate_ndwi",
    "extract_spectral_indices",
    "normalize_pixels",
    "preprocess_for_solarscan",
    "split_rgb_channels",
]
