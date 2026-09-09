from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from solarguard_ai.ingesta import (
    ImageIngestionError,
    _validate_channels,
    load_image,
    load_images,
)


def image_bytes(image_format: str, *, size: tuple[int, int] = (32, 24), mode: str = "RGB") -> bytes:
    image = Image.new(mode, size, color=0)
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("extension", "image_format"),
    [(".jpg", "JPEG"), (".jpeg", "JPEG"), (".png", "PNG"), (".tif", "TIFF")],
)
def test_load_image_supports_declared_formats(
    tmp_path, extension: str, image_format: str
) -> None:
    image_path = tmp_path / f"panel{extension}"
    image_path.write_bytes(image_bytes(image_format))

    loaded = load_image(image_path)

    assert loaded.format == image_format
    assert loaded.image.mode == "RGB"
    assert (loaded.width, loaded.height) == (32, 24)
    assert loaded.original_channels == 3


def test_load_image_supports_file_like_sources_without_extension() -> None:
    source = BytesIO(image_bytes("PNG"))

    loaded = load_image(source)

    assert loaded.source == "<archivo>"
    assert loaded.format == "PNG"
    assert loaded.image.mode == "RGB"


def test_load_image_rejects_corrupt_content(tmp_path) -> None:
    image_path = tmp_path / "corrupt.png"
    image_path.write_bytes(b"not-an-image")

    with pytest.raises(ImageIngestionError, match="No se pudo leer la imagen"):
        load_image(image_path)


def test_load_image_rejects_unsupported_extension(tmp_path) -> None:
    image_path = tmp_path / "panel.bmp"
    image_path.write_bytes(image_bytes("PNG"))

    with pytest.raises(ImageIngestionError, match="Formato no soportado"):
        load_image(image_path)


def test_load_image_rejects_extension_format_mismatch(tmp_path) -> None:
    image_path = tmp_path / "panel.jpg"
    image_path.write_bytes(image_bytes("PNG"))

    with pytest.raises(ImageIngestionError, match="extension incompatible"):
        load_image(image_path)


def test_load_image_rejects_images_above_pixel_limit(tmp_path) -> None:
    image_path = tmp_path / "large.png"
    image_path.write_bytes(image_bytes("PNG", size=(5001, 5001)))

    with pytest.raises(ImageIngestionError, match="supera el limite"):
        load_image(image_path)


def test_validate_channels_rejects_multiband_images_for_rgb_model() -> None:
    with pytest.raises(ImageIngestionError, match="se admiten 1, 2, 3, 4"):
        _validate_channels("multiband.tif", 5)


def test_load_images_processes_a_batch(tmp_path) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.jpg"
    first_path.write_bytes(image_bytes("PNG"))
    second_path.write_bytes(image_bytes("JPEG"))

    loaded_images = load_images([first_path, second_path])

    assert len(loaded_images) == 2
    assert [image.format for image in loaded_images] == ["PNG", "JPEG"]


def test_load_images_reports_the_invalid_batch_item(tmp_path) -> None:
    valid_path = tmp_path / "valid.png"
    invalid_path = tmp_path / "invalid.gif"
    valid_path.write_bytes(image_bytes("PNG"))
    invalid_path.write_bytes(image_bytes("PNG"))

    with pytest.raises(ImageIngestionError, match="imagen 2 del lote"):
        load_images([valid_path, invalid_path])
