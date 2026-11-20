from __future__ import annotations

import ctypes
import math
import os
import time

import pytest

from PIL import Image, ImageFilter

# Small, hand-checkable 4x3 height map covering zero, tiny steps, a high
# pulse and the full unsigned 16-bit range.
SAMPLE_DATA = [
    0,
    100,
    65535,
    300,
    1,
    200,
    60000,
    400,
    2,
    300,
    500,
    500,
]
WIDTH, HEIGHT = 4, 3
I16_MODES = ("I;16", "I;16L", "I;16B")


def make_image(mode: str, data: list[int] | None = None) -> Image.Image:
    image = Image.new(mode, (WIDTH, HEIGHT))
    image.putdata(SAMPLE_DATA if data is None else data)
    return image


def reference_line_blur(values: list[int], float_radius: float) -> list[int]:
    """Model of the C extended box blur line pass for 16-bit samples.

    The filter uses a ``1 << 24`` fixed-point weight together with clamp
    (edge replicate) extension. The 16-bit path accumulates in 64 bits and
    applies the same ``>> 24`` normalization as the 8-bit path.
    """
    scale = 1 << 24
    radius = int(float_radius)
    # Emulate the C ``(UINT32)(1 << 24) / (floatRadius * 2 + 1)``. The
    # division is performed in 32-bit float precision before truncation.
    float_weight = ctypes.c_float(
        ctypes.c_float(scale).value
        / ctypes.c_float(float_radius * 2 + 1).value
    ).value
    weight = ctypes.c_uint32(int(float_weight)).value
    fractional_weight = (scale - (radius * 2 + 1) * weight) // 2

    def pixel(index: int) -> int:
        if index < 0:
            return values[0]
        if index > len(values) - 1:
            return values[-1]
        return values[index]

    result = []
    for x in range(len(values)):
        accumulator = sum(
            pixel(index) for index in range(x - radius, x + radius + 1)
        )
        bulk = accumulator * weight
        bulk += (pixel(x - radius - 1) + pixel(x + radius + 1)) * fractional_weight
        result.append((bulk + (1 << 23)) >> 24)
    return result


def reference_box_blur(
    data: list[int], width: int, height: int, xradius: float, yradius: float
) -> list[int]:
    current = list(data)
    if xradius:
        current = [
            value
            for y in range(height)
            for value in reference_line_blur(
                current[y * width : (y + 1) * width], xradius
            )
        ]
    if yradius:
        columns = [
            reference_line_blur(
                [current[y * width + x] for y in range(height)], yradius
            )
            for x in range(width)
        ]
        current = [columns[x][y] for y in range(height) for x in range(width)]
    return current


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize("radius", (0, 0.25, 0.5, 1, 1.5, 2, 10, 600))
def test_box_blur_matches_reference(mode: str, radius: float) -> None:
    image = make_image(mode)
    output = image.filter(ImageFilter.BoxBlur(radius))
    expected = reference_box_blur(SAMPLE_DATA, WIDTH, HEIGHT, radius, radius)
    assert list(output.get_flattened_data()) == expected
    assert output.mode == mode
    assert output.size == image.size


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize(
    "radii", ((1, 0), (0, 1), (0.5, 0), (0, 0.5), (3, 1), (10, 0), (0, 10))
)
def test_box_blur_single_axis(mode: str, radii: tuple[float, float]) -> None:
    xradius, yradius = radii
    image = make_image(mode)
    output = image.filter(ImageFilter.BoxBlur(radii))
    expected = reference_box_blur(
        SAMPLE_DATA, WIDTH, HEIGHT, xradius, yradius
    )
    assert list(output.get_flattened_data()) == expected


def test_integer_radius_hardcoded_values() -> None:
    # Radius 1 horizontal pass over [0, 100, 65535, 300] with edge clamp.
    image = Image.new("I;16", (4, 1))
    image.putdata([0, 100, 65535, 300])
    output = list(
        image.filter(ImageFilter.BoxBlur((1, 0))).get_flattened_data()
    )
    assert output == [33, 21878, 21978, 22045]


@pytest.mark.parametrize("mode", I16_MODES)
def test_endian_modes_produce_identical_pixels(mode: str) -> None:
    outputs = {
        candidate: list(
            make_image(candidate)
            .filter(ImageFilter.BoxBlur(1.25))
            .get_flattened_data()
        )
        for candidate in I16_MODES
    }
    assert outputs[mode] == outputs["I;16"]
    assert outputs["I;16"] == outputs["I;16L"] == outputs["I;16B"]

    little_bytes = (
        make_image("I;16L").filter(ImageFilter.BoxBlur(1.25)).tobytes()
    )
    big = make_image("I;16B").filter(ImageFilter.BoxBlur(1.25))
    swapped = bytes(
        value
        for index in range(0, len(little_bytes), 2)
        for value in (little_bytes[index + 1], little_bytes[index])
    )
    assert big.tobytes() == swapped


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize(
    "blur_filter",
    (
        ImageFilter.BoxBlur(0),
        ImageFilter.BoxBlur((0, 0)),
        ImageFilter.GaussianBlur(0),
        ImageFilter.GaussianBlur((0, 0)),
    ),
)
def test_zero_radius_returns_identical_image(
    mode: str, blur_filter: ImageFilter.Filter
) -> None:
    image = make_image(mode)
    output = image.filter(blur_filter)
    assert list(output.get_flattened_data()) == SAMPLE_DATA
    assert output.mode == mode
    assert output.size == image.size


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize("value", (0, 1, 256, 32768, 65535))
def test_constant_image_is_preserved(mode: str, value: int) -> None:
    image = Image.new(mode, (5, 4))
    image.putdata([value] * 20)
    for radius in (0, 0.3, 1, 5, 100):
        output = image.filter(ImageFilter.BoxBlur(radius))
        assert list(output.get_flattened_data()) == [value] * 20
    assert list(
        image.filter(ImageFilter.GaussianBlur(2)).get_flattened_data()
    ) == [value] * 20


@pytest.mark.parametrize("mode", I16_MODES)
def test_pulse_and_large_radius_do_not_overflow(mode: str) -> None:
    pulse = [0] * 9 + [65535] + [0] * 9
    image = Image.new(mode, (19, 1))
    image.putdata(pulse)

    blurred = list(image.filter(ImageFilter.BoxBlur(3)).get_flattened_data())
    assert min(blurred) >= 0
    assert max(blurred) <= 65535
    assert blurred == blurred[::-1]
    assert abs(blurred[9] - round(65535 / 7)) <= 1

    # Radius far larger than the image dimension must still stay in range.
    for radius in (100, 100000):
        blurred = list(
            image.filter(ImageFilter.BoxBlur(radius)).get_flattened_data()
        )
        assert min(blurred) >= 0
        assert max(blurred) <= 65535

    gaussian = list(
        image.filter(ImageFilter.GaussianBlur(50)).get_flattened_data()
    )
    assert min(gaussian) >= 0
    assert max(gaussian) <= 65535


def test_single_row_column_and_pixel() -> None:
    row = Image.new("I;16B", (6, 1))
    row.putdata([10, 20, 30, 40, 50, 60])
    row_output = row.filter(ImageFilter.BoxBlur(50))
    assert all(0 <= value <= 65535 for value in row_output.get_flattened_data())

    column = Image.new("I;16", (1, 6))
    column.putdata([10, 20, 30, 40, 50, 60])
    column_output = column.filter(ImageFilter.BoxBlur(50))
    assert all(
        0 <= value <= 65535 for value in column_output.get_flattened_data()
    )

    pixel = Image.new("I;16L", (1, 1))
    pixel.putdata([65535])
    assert list(pixel.filter(ImageFilter.BoxBlur(5)).get_flattened_data()) == [
        65535
    ]
    assert list(
        pixel.filter(ImageFilter.GaussianBlur(5)).get_flattened_data()
    ) == [65535]


def test_low_brightness_changes_survive() -> None:
    image = Image.new("I;16", (7, 1))
    image.putdata([1000, 1001, 1000, 1001, 1000, 1001, 1000])
    output = list(image.filter(ImageFilter.BoxBlur(0.25)).get_flattened_data())
    assert all(999 <= value <= 1002 for value in output)
    assert len(set(output)) > 1


@pytest.mark.parametrize("mode", I16_MODES)
def test_source_image_is_not_modified(mode: str) -> None:
    image = make_image(mode)
    before = list(image.get_flattened_data())
    image.filter(ImageFilter.BoxBlur(2))
    image.filter(ImageFilter.GaussianBlur(2))
    assert list(image.get_flattened_data()) == before


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize(
    "blur_filter", (ImageFilter.BoxBlur(1.25), ImageFilter.GaussianBlur(1.25))
)
def test_tiff_roundtrip_keeps_samples_and_byte_order(
    mode: str, blur_filter: ImageFilter.Filter, tmp_path: str
) -> None:
    filtered = make_image(mode).filter(blur_filter)
    path = os.path.join(
        tmp_path, f"roundtrip_{mode}_{type(blur_filter).__name__}.tif"
    )
    filtered.save(path, "TIFF")

    reopened = Image.open(path)
    reopened.load()
    assert reopened.size == filtered.size
    if mode == "I;16B":
        assert reopened.mode == "I;16B"
    else:
        # Pillow may normalize the little-endian aliases on reload.
        assert reopened.mode in ("I;16", "I;16L")
    assert list(reopened.get_flattened_data()) == list(
        filtered.get_flattened_data()
    )

    with open(path, "rb") as handle:
        byte_order_mark = handle.read(2)
    assert byte_order_mark == (b"MM" if mode == "I;16B" else b"II")


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize("radius", (float("nan"), float("inf")))
def test_non_finite_radius_raises(mode: str, radius: float) -> None:
    image = make_image(mode)
    with pytest.raises(ValueError):
        image.filter(ImageFilter.BoxBlur(radius))
    with pytest.raises(ValueError):
        image.filter(ImageFilter.GaussianBlur(radius))


def test_negative_radius_raises() -> None:
    with pytest.raises(ValueError):
        ImageFilter.BoxBlur(-1)
    with pytest.raises(ValueError):
        ImageFilter.BoxBlur((1, -1))


@pytest.mark.parametrize("mode", ("I", "F", "I;16N", "1", "P"))
def test_unsupported_modes_are_rejected(mode: str) -> None:
    image = Image.new(mode, (4, 4))
    if mode == "P":
        image.putpalette([0] * 768)
    with pytest.raises(ValueError):
        image.filter(ImageFilter.BoxBlur(1))


@pytest.mark.parametrize("mode", I16_MODES)
def test_unsharp_mask_still_rejects_16_bit(mode: str) -> None:
    # UnsharpMask's sharpening stage is 8-bit only and must keep its
    # historical mode behavior.
    image = make_image(mode)
    with pytest.raises(ValueError):
        image.filter(ImageFilter.UnsharpMask(2))


def _gaussian_box_radius(radius: float, passes: int = 3) -> float:
    sigma2 = radius * radius / passes
    box_length = math.sqrt(12.0 * sigma2 + 1.0)
    integer_radius = math.floor((box_length - 1.0) / 2.0)
    fraction = (2 * integer_radius + 1) * (
        integer_radius * (integer_radius + 1) - 3 * sigma2
    )
    fraction /= 6 * (sigma2 - (integer_radius + 1) * (integer_radius + 1))
    return integer_radius + fraction


@pytest.mark.parametrize("mode", I16_MODES)
@pytest.mark.parametrize("radius", (0.5, 1, 2, 3.5, 10))
def test_gaussian_blur_matches_three_pass_reference(
    mode: str, radius: float
) -> None:
    box_radius = _gaussian_box_radius(radius)
    expected = list(SAMPLE_DATA)
    for _ in range(3):
        expected = reference_box_blur(
            expected, WIDTH, HEIGHT, box_radius, box_radius
        )
    output = list(
        make_image(mode)
        .filter(ImageFilter.GaussianBlur(radius))
        .get_flattened_data()
    )
    # The C (UINT32) float cast of the fixed-point weight can differ by a
    # least significant bit per extended box pass.
    assert len(output) == len(expected)
    assert all(abs(got - want) <= 2 for got, want in zip(output, expected))
    # Endian variants must be bit-for-bit identical at the sample level.
    other = "I;16B" if mode != "I;16B" else "I;16L"
    other_output = list(
        make_image(other)
        .filter(ImageFilter.GaussianBlur(radius))
        .get_flattened_data()
    )
    assert other_output == output


def test_runtime_does_not_grow_with_window_area() -> None:
    # A window-area algorithm would slow down quadratically with the radius.
    # The running-sum implementation must take roughly constant time.
    size = 1024
    image = Image.new("I;16", (size, size))
    image.putdata(
        [(x * 257 + y * 131) & 0xFFFF for y in range(size) for x in range(size)]
    )

    timings = {}
    for radius in (1, 64, 256):
        start = time.perf_counter()
        image.filter(ImageFilter.BoxBlur(radius))
        timings[radius] = time.perf_counter() - start
    assert max(timings.values()) / min(timings.values()) < 2.5
