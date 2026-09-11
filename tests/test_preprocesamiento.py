from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from solarguard_ai.preprocesamiento import (
    TARGET_SIZE,
    augment_image,
    calculate_ndvi,
    calculate_ndwi,
    extract_spectral_indices,
    normalize_pixels,
    preprocess_for_solarscan,
    split_rgb_channels,
)


def test_normalize_pixels_minmax_returns_values_between_zero_and_one() -> None:
    pixels = np.array([[0, 127], [255, 64]], dtype=np.uint8)

    normalized = normalize_pixels(pixels)

    assert normalized.dtype == np.float32
    assert normalized.min() == 0
    assert normalized.max() == 1
    assert normalized[0, 1] == pytest.approx(127 / 255)


def test_normalize_pixels_zscore_has_zero_mean_and_unit_deviation() -> None:
    pixels = np.array([1, 2, 3, 4], dtype=np.float32)

    normalized = normalize_pixels(pixels, method="zscore")

    assert normalized.mean() == pytest.approx(0)
    assert normalized.std() == pytest.approx(1)


def test_normalize_pixels_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Metodo de normalizacion"):
        normalize_pixels(np.array([1]), method="unknown")


def test_preprocess_for_solarscan_resizes_crops_and_returns_nchw() -> None:
    image = Image.new("RGB", (448, 224), color=(255, 128, 0))

    tensor = preprocess_for_solarscan(image)

    assert tensor.shape == (1, 3, TARGET_SIZE, TARGET_SIZE)
    assert tensor.dtype == np.float32
    assert tensor.min() >= 0
    assert tensor.max() <= 1
    assert tensor[0, 0, 0, 0] == pytest.approx(1)
    assert tensor[0, 1, 0, 0] == pytest.approx(128 / 255)


def test_preprocess_for_solarscan_accepts_custom_target_size() -> None:
    image = Image.new("L", (80, 40), color=128)

    tensor = preprocess_for_solarscan(image, target_size=32)

    assert tensor.shape == (1, 3, 32, 32)
    assert tensor[0, :, 0, 0].tolist() == pytest.approx([128 / 255] * 3)


def test_augment_image_returns_original_and_three_variants() -> None:
    image = Image.new("RGB", (8, 4), color="red")

    augmented = augment_image(image)

    assert len(augmented) == 4
    assert all(variant.mode == "RGB" for variant in augmented)
    assert all(variant is not image for variant in augmented)


def test_split_rgb_channels_returns_normalized_named_channels() -> None:
    image = Image.new("RGB", (2, 1), color=(255, 128, 0))

    channels = split_rgb_channels(image)

    assert set(channels) == {"red", "green", "blue"}
    assert channels["red"][0, 0] == pytest.approx(1)
    assert channels["green"][0, 0] == pytest.approx(128 / 255)
    assert channels["blue"][0, 0] == pytest.approx(0)


def test_calculate_ndvi_uses_nir_and_red_bands() -> None:
    image = np.zeros((2, 2, 4), dtype=np.float32)
    image[:, :, 0] = 0.8
    image[:, :, 1] = 0.2

    ndvi = calculate_ndvi(image, nir_band=0, red_band=1)

    assert ndvi.shape == (2, 2)
    assert ndvi[0, 0] == pytest.approx(0.6 / 1.0)


def test_calculate_ndwi_uses_nir_and_green_bands() -> None:
    image = np.zeros((2, 2, 4), dtype=np.float32)
    image[:, :, 2] = 0.7
    image[:, :, 3] = 0.3

    ndwi = calculate_ndwi(image, nir_band=2, green_band=3)

    assert ndwi[0, 0] == pytest.approx(0.4)


def test_extract_spectral_indices_returns_both_indices() -> None:
    image = np.zeros((1, 1, 3), dtype=np.float32)
    image[0, 0] = (0.8, 0.2, 0.4)

    indices = extract_spectral_indices(image, nir_band=0, red_band=1, green_band=2)

    assert set(indices) == {"ndvi", "ndwi"}
    assert indices["ndvi"][0, 0] == pytest.approx(0.6)
    assert indices["ndwi"][0, 0] == pytest.approx(1 / 3)


def test_spectral_indices_reject_invalid_band_index() -> None:
    image = np.zeros((2, 2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="indices de banda"):
        calculate_ndvi(image, nir_band=3, red_band=0)


def test_spectral_indices_set_zero_when_bands_sum_to_zero() -> None:
    image = np.zeros((1, 1, 2), dtype=np.float32)

    ndvi = calculate_ndvi(image, nir_band=0, red_band=1)

    assert ndvi[0, 0] == 0
