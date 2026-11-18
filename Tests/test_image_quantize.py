from __future__ import annotations

from pathlib import Path

import pytest

from PIL import Image

from .helper import (
    assert_image_similar,
    has_feature_version,
    hopper,
    is_ppc64le,
    skip_unless_feature,
)


def test_sanity() -> None:
    image = hopper()
    converted = image.quantize()
    assert converted.mode == "P"
    assert_image_similar(converted.convert("RGB"), image, 10)

    image = hopper()
    converted = image.quantize(palette=hopper("P"))
    assert converted.mode == "P"
    assert_image_similar(converted.convert("RGB"), image, 60)


@skip_unless_feature("libimagequant")
def test_libimagequant_quantize() -> None:
    image = hopper()
    if is_ppc64le() and not has_feature_version("libimagequant", "4"):
        pytest.skip("Fails with libimagequant earlier than 4.0.0 on ppc64le")
    converted = image.quantize(100, Image.Quantize.LIBIMAGEQUANT)
    assert converted.mode == "P"
    assert_image_similar(converted.convert("RGB"), image, 15)
    colors = converted.getcolors()
    assert colors is not None
    assert len(colors) == 100


def test_octree_quantize() -> None:
    image = hopper()
    converted = image.quantize(100, Image.Quantize.FASTOCTREE)
    assert converted.mode == "P"
    assert_image_similar(converted.convert("RGB"), image, 20)
    colors = converted.getcolors()
    assert colors is not None
    assert len(colors) == 100


def test_rgba_quantize() -> None:
    image = hopper("RGBA")
    with pytest.raises(ValueError):
        image.quantize(method=0)

    assert image.quantize().convert().mode == "RGBA"


def test_quantize() -> None:
    with Image.open("Tests/images/caption_6_33_22.png") as image:
        converted = image.convert("RGB")
    converted = converted.quantize()
    assert converted.mode == "P"
    assert_image_similar(converted.convert("RGB"), image, 1)


def test_quantize_no_dither() -> None:
    image = hopper()
    with Image.open("Tests/images/caption_6_33_22.png") as palette:
        palette_p = palette.convert("P")

    converted = image.quantize(dither=Image.Dither.NONE, palette=palette_p)
    assert converted.mode == "P"
    assert converted.palette is not None
    assert palette_p.palette is not None
    assert converted.palette.palette == palette_p.palette.palette


def test_quantize_no_dither2() -> None:
    im = Image.new("RGB", (9, 1))
    im.putdata([(p,) * 3 for p in range(0, 36, 4)])

    palette = Image.new("P", (1, 1))
    data = (0, 0, 0, 32, 32, 32)
    palette.putpalette(data)
    quantized = im.quantize(dither=Image.Dither.NONE, palette=palette)

    assert quantized.palette is not None
    assert tuple(quantized.palette.palette) == data

    px = quantized.load()
    assert px is not None
    for x in range(9):
        assert px[x, 0] == (0 if x < 5 else 1)


def test_quantize_dither_diff() -> None:
    image = hopper()
    with Image.open("Tests/images/caption_6_33_22.png") as palette:
        palette_p = palette.convert("P")

    dither = image.quantize(dither=Image.Dither.FLOYDSTEINBERG, palette=palette_p)
    nodither = image.quantize(dither=Image.Dither.NONE, palette=palette_p)

    assert dither.tobytes() != nodither.tobytes()


@pytest.mark.parametrize(
    "method", (Image.Quantize.MEDIANCUT, Image.Quantize.MAXCOVERAGE)
)
def test_quantize_kmeans(method: Image.Quantize) -> None:
    im = hopper()
    no_kmeans = im.quantize(kmeans=0, method=method)
    kmeans = im.quantize(kmeans=1, method=method)
    assert kmeans.tobytes() != no_kmeans.tobytes()

    with pytest.raises(ValueError):
        im.quantize(kmeans=-1, method=method)


@skip_unless_feature("libimagequant")
def test_resize() -> None:
    im = hopper().resize((100, 100))
    converted = im.quantize(100, Image.Quantize.LIBIMAGEQUANT)
    colors = converted.getcolors()
    assert colors is not None
    assert len(colors) == 100


def test_colors() -> None:
    im = hopper()
    colors = 2
    converted = im.quantize(colors)
    assert converted.palette is not None
    assert len(converted.palette.palette) == colors * len("RGB")


def test_transparent_colors_equal() -> None:
    im = Image.new("RGBA", (1, 2), (0, 0, 0, 0))
    px = im.load()
    assert px is not None
    px[0, 1] = (255, 255, 255, 0)

    converted = im.quantize()
    converted_px = converted.load()
    assert converted_px is not None
    assert converted_px[0, 0] == converted_px[0, 1]


@pytest.mark.parametrize(
    "method, color",
    (
        (Image.Quantize.MEDIANCUT, (0, 0, 0)),
        (Image.Quantize.MAXCOVERAGE, (0, 0, 0)),
        (Image.Quantize.FASTOCTREE, (0, 0, 0)),
        (Image.Quantize.FASTOCTREE, (0, 0, 0, 0)),
    ),
)
def test_palette(method: Image.Quantize, color: tuple[int, ...]) -> None:
    im = Image.new("RGBA" if len(color) == 4 else "RGB", (1, 1), color)

    converted = im.quantize(method=method)
    assert converted.palette is not None
    assert converted.getpixel((0, 0)) == converted.palette.colors[color]


def test_small_palette() -> None:
    # Arrange
    im = hopper()

    colors = (255, 0, 0, 0, 0, 255)
    p = Image.new("P", (1, 1))
    p.putpalette(colors)

    # Act
    im = im.quantize(palette=p)

    # Assert
    quantized_colors = im.getcolors()
    assert quantized_colors is not None
    assert len(quantized_colors) == 2


def test_quantize_rgba_palette(tmp_path: Path) -> None:
    palette_data = [
        10,
        20,
        30,
        255,
        200,
        0,
        0,
        128,
        200,
        0,
        0,
        255,
        50,
        60,
        70,
        0,
        90,
        80,
        70,
        0,
    ]
    palette = Image.new("P", (1, 1))
    palette.putpalette(palette_data, "RGBA")
    palette_copy = palette.copy()

    image = Image.new("RGBA", (6, 1))
    image.putdata(
        [
            (0, 0, 0, 0),
            (200, 0, 0, 128),
            (200, 0, 0, 255),
            (10, 20, 30, 255),
            (255, 255, 255, 0),
            (5, 0, 0, 255),
        ]
    )
    image_copy = image.copy()

    quantized = image.quantize(dither=Image.Dither.NONE, palette=palette)

    assert list(quantized.get_flattened_data()) == [3, 1, 2, 0, 3, 0]
    assert quantized.getpalette("RGBA") == palette_data
    assert list(quantized.convert("RGBA").get_flattened_data()) == [
        (50, 60, 70, 0),
        (200, 0, 0, 128),
        (200, 0, 0, 255),
        (10, 20, 30, 255),
        (50, 60, 70, 0),
        (10, 20, 30, 255),
    ]

    assert image.tobytes() == image_copy.tobytes()
    assert palette.getpalette("RGBA") == palette_copy.getpalette("RGBA")
    assert palette.info == palette_copy.info

    out = tmp_path / "rgba_palette.png"
    quantized.save(out)
    with Image.open(out) as reloaded:
        assert reloaded.mode == "P"
        assert reloaded.info["transparency"] == bytes([255, 128, 255, 0, 0])
        assert list(reloaded.convert("RGBA").get_flattened_data()) == list(
            quantized.convert("RGBA").get_flattened_data()
        )


def test_quantize_rgba_palette_distance_tie() -> None:
    palette = Image.new("P", (1, 1))
    palette.putpalette([0, 0, 0, 255, 10, 0, 0, 255], "RGBA")
    image = Image.new("RGBA", (1, 1), (5, 0, 0, 255))

    quantized = image.quantize(dither=Image.Dither.NONE, palette=palette)

    assert quantized.getpixel((0, 0)) == 0


@pytest.mark.parametrize(
    "transparency,expected_indices,expected_palette",
    (
        (2, [2, 1], [0, 0, 0, 255, 255, 0, 0, 255, 128, 128, 128, 0]),
        (
            bytes([255, 64, 0]),
            [2, 1, 2],
            [0, 0, 0, 255, 255, 0, 0, 64, 128, 128, 128, 0],
        ),
    ),
)
def test_quantize_rgba_palette_transparency(
    tmp_path: Path,
    transparency: int | bytes,
    expected_indices: list[int],
    expected_palette: list[int],
) -> None:
    palette = Image.new("P", (1, 1))
    palette.putpalette([0, 0, 0, 255, 0, 0, 128, 128, 128])
    palette.info["transparency"] = transparency
    palette_copy = palette.copy()

    image = Image.new("RGBA", (3 if isinstance(transparency, bytes) else 2, 1))
    image.putdata(
        [(0, 0, 0, 0), (255, 0, 0, 64), (255, 255, 255, 0)]
        if isinstance(transparency, bytes)
        else [(0, 0, 0, 0), (255, 0, 0, 255)]
    )

    quantized = image.quantize(dither=Image.Dither.NONE, palette=palette)

    assert list(quantized.get_flattened_data()) == expected_indices
    assert quantized.getpalette("RGBA") == expected_palette
    assert quantized.info["transparency"] == transparency
    assert palette.getpalette("RGBA") == palette_copy.getpalette("RGBA")
    assert palette.info == palette_copy.info

    out = tmp_path / "transparency.png"
    quantized.save(out)
    with Image.open(out) as reloaded:
        assert reloaded.mode == "P"
        assert reloaded.info["transparency"] == transparency
        assert list(reloaded.convert("RGBA").get_flattened_data()) == list(
            quantized.convert("RGBA").get_flattened_data()
        )


def test_quantize_rgba_palette_dither() -> None:
    palette = Image.new("P", (1, 1))
    palette.putpalette(
        [0, 0, 0, 255, 255, 255, 255, 255, 1, 2, 3, 0],
        "RGBA",
    )
    image = Image.new("RGBA", (3, 1))
    image.putdata(
        [(180, 180, 180, 255), (255, 255, 255, 0), (127, 127, 127, 255)]
    )

    quantized = image.quantize(dither=Image.Dither.FLOYDSTEINBERG, palette=palette)

    assert list(quantized.get_flattened_data()) == [1, 2, 0]
    assert quantized.convert("RGBA").getpixel((1, 0)) == (1, 2, 3, 0)


def test_quantize_rgba_invalid_palette() -> None:
    image = Image.new("RGBA", (1, 1))

    with pytest.raises(ValueError, match="bad mode for palette image"):
        image.quantize(palette=hopper("RGB"))

    palette = Image.new("P", (1, 1))
    palette.putpalette([0, 0, 0, 0], "CMYK")
    with pytest.raises(ValueError, match="only RGB or RGBA palettes"):
        image.quantize(palette=palette)
