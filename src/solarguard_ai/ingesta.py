"""Carga y validacion de imagenes para SolarGuard AI."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable, TypeAlias

from PIL import Image, UnidentifiedImageError

ImageSource: TypeAlias = str | Path | BinaryIO

_SUPPORTED_EXTENSIONS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}
_SUPPORTED_CHANNEL_COUNTS = frozenset({1, 2, 3, 4})
_MAX_PIXELS = 25_000_000


class ImageIngestionError(ValueError):
    """Indica que una imagen no puede cargarse o validarse."""


@dataclass(frozen=True)
class LoadedImage:
    """Imagen validada y lista para el preprocesamiento del modelo."""

    source: str
    image: Image.Image
    format: str
    width: int
    height: int
    original_mode: str
    original_channels: int


def load_image(source: ImageSource) -> LoadedImage:
    """Carga y valida una imagen JPEG, PNG o TIFF/GeoTIFF.

    La imagen devuelta siempre esta en RGB, que es el formato de entrada
    utilizado por solarscan-yolov8n-cls. La imagen original puede ser
    monocromatica o tener mas canales siempre que Pillow pueda decodificarla.
    """
    payload, source_name = _read_source(source)
    expected_format = _expected_format(source_name)

    try:
        # Verifica la estructura completa antes de cargar los pixeles en memoria.
        with Image.open(BytesIO(payload)) as opened_image:
            detected_format = opened_image.format
            if expected_format and detected_format != expected_format:
                raise ImageIngestionError(
                    f"El archivo '{source_name}' tiene extension incompatible: "
                    f"se esperaba {expected_format} y se detecto {detected_format}."
                )
            opened_image.verify()

        # Pillow requiere abrir el archivo de nuevo despues de llamar a verify().
        with Image.open(BytesIO(payload)) as opened_image:
            width, height = opened_image.size
            _validate_dimensions(source_name, width, height)
            original_mode = opened_image.mode
            original_channels = len(opened_image.getbands())
            _validate_channels(source_name, original_channels)
            # El modelo SolarScan recibe imagenes RGB, independientemente del modo original.
            rgb_image = opened_image.convert("RGB")
            rgb_image.load()

    except ImageIngestionError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise ImageIngestionError(
            f"No se pudo leer la imagen '{source_name}': {error}"
        ) from error

    return LoadedImage(
        source=source_name,
        image=rgb_image,
        format=detected_format or "UNKNOWN",
        width=width,
        height=height,
        original_mode=original_mode,
        original_channels=original_channels,
    )


def load_images(sources: Iterable[ImageSource]) -> list[LoadedImage]:
    """Carga y valida un lote de imagenes, fallando con el elemento afectado."""
    loaded_images: list[LoadedImage] = []
    for index, source in enumerate(sources, start=1):
        try:
            # Conserva el indice para identificar rapidamente un archivo invalido del lote.
            loaded_images.append(load_image(source))
        except ImageIngestionError as error:
            raise ImageIngestionError(
                f"Error en la imagen {index} del lote: {error}"
            ) from error
    return loaded_images


def _read_source(source: ImageSource) -> tuple[bytes, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            return path.read_bytes(), str(path)
        except OSError as error:
            raise ImageIngestionError(
                f"No se pudo abrir el archivo '{path}': {error}"
            ) from error

    source_name = str(getattr(source, "name", "<archivo>"))
    try:
        # Las fuentes de Streamlit son archivos en memoria; se leen desde el inicio
        # y se devuelve el cursor a su posicion original para no sorprender al llamador.
        current_position = source.tell()
        source.seek(0)
        payload = source.read()
        source.seek(current_position)
    except (AttributeError, OSError) as error:
        raise ImageIngestionError(
            f"No se pudo leer la fuente '{source_name}': {error}"
        ) from error

    if not isinstance(payload, bytes):
        raise ImageIngestionError(
            f"La fuente '{source_name}' no devolvio bytes de imagen validos."
        )
    return payload, source_name


def _expected_format(source_name: str) -> str | None:
    suffix = Path(source_name).suffix.lower()
    if not suffix:
        return None
    expected_format = _SUPPORTED_EXTENSIONS.get(suffix)
    if expected_format is None:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise ImageIngestionError(
            f"Formato no soportado para '{source_name}'. "
            f"Formatos permitidos: {supported}."
        )
    return expected_format


def _validate_dimensions(source_name: str, width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise ImageIngestionError(
            f"La imagen '{source_name}' tiene dimensiones invalidas: {width}x{height}."
        )
    if width * height > _MAX_PIXELS:
        raise ImageIngestionError(
            f"La imagen '{source_name}' supera el limite de {_MAX_PIXELS:,} pixeles."
        )


def _validate_channels(source_name: str, channels: int) -> None:
    """Rechaza bandas que no pueden convertirse de forma segura a la entrada RGB."""
    if channels not in _SUPPORTED_CHANNEL_COUNTS:
        supported = ", ".join(str(value) for value in sorted(_SUPPORTED_CHANNEL_COUNTS))
        raise ImageIngestionError(
            f"La imagen '{source_name}' tiene {channels} canales; "
            f"se admiten {supported} antes de convertir a RGB."
        )


__all__ = ["ImageIngestionError", "LoadedImage", "ImageSource", "load_image", "load_images"]
