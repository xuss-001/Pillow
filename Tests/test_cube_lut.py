from __future__ import annotations

import io
from itertools import permutations
from pathlib import Path

import pytest

from PIL import Image, ImageFilter


def identity_table(size: int) -> list[float]:
    return [
        value
        for b in range(size)
        for g in range(size)
        for r in range(size)
        for value in (r / (size - 1), g / (size - 1), b / (size - 1))
    ]


def corner_table(
    corners: dict[tuple[int, int, int], tuple[float, float, float]],
) -> list[float]:
    return [
        value
        for b in range(2)
        for g in range(2)
        for r in range(2)
        for value in corners[(r, g, b)]
    ]


def cube_text(
    table: list[float],
    size: int = 2,
    title: str | None = None,
    domain_min: tuple[float, float, float] | None = None,
    domain_max: tuple[float, float, float] | None = None,
) -> str:
    lines = []
    if title is not None:
        lines.append(f'TITLE "{title}"')
    lines.append(f"LUT_3D_SIZE {size}")
    if domain_min is not None:
        lines.append("DOMAIN_MIN " + " ".join(str(value) for value in domain_min))
    if domain_max is not None:
        lines.append("DOMAIN_MAX " + " ".join(str(value) for value in domain_max))
    for index in range(0, len(table), 3):
        lines.append(" ".join(str(value) for value in table[index : index + 3]))
    return "\n".join(lines) + "\n"


def to_8bit(value: float) -> int:
    return min(255, max(0, int(value * 255 + 0.5)))


def tetrahedral_expected(
    corners: dict[tuple[int, int, int], tuple[float, float, float]],
    fraction_r: float,
    fraction_g: float,
    fraction_b: float,
) -> tuple[int, int, int]:
    if fraction_r >= fraction_g:
        if fraction_g >= fraction_b:
            sequence = (
                ((0, 0, 0), 1 - fraction_r),
                ((1, 0, 0), fraction_r - fraction_g),
                ((1, 1, 0), fraction_g - fraction_b),
                ((1, 1, 1), fraction_b),
            )
        elif fraction_r >= fraction_b:
            sequence = (
                ((0, 0, 0), 1 - fraction_r),
                ((1, 0, 0), fraction_r - fraction_b),
                ((1, 0, 1), fraction_b - fraction_g),
                ((1, 1, 1), fraction_g),
            )
        else:
            sequence = (
                ((0, 0, 0), 1 - fraction_b),
                ((0, 0, 1), fraction_b - fraction_r),
                ((1, 0, 1), fraction_r - fraction_g),
                ((1, 1, 1), fraction_g),
            )
    elif fraction_b >= fraction_g:
        sequence = (
            ((0, 0, 0), 1 - fraction_b),
            ((0, 0, 1), fraction_b - fraction_g),
            ((0, 1, 1), fraction_g - fraction_r),
            ((1, 1, 1), fraction_r),
        )
    elif fraction_b >= fraction_r:
        sequence = (
            ((0, 0, 0), 1 - fraction_g),
            ((0, 1, 0), fraction_g - fraction_b),
            ((0, 1, 1), fraction_b - fraction_r),
            ((1, 1, 1), fraction_r),
        )
    else:
        sequence = (
            ((0, 0, 0), 1 - fraction_g),
            ((0, 1, 0), fraction_g - fraction_r),
            ((1, 1, 0), fraction_r - fraction_b),
            ((1, 1, 1), fraction_b),
        )

    values = [0.0, 0.0, 0.0]
    for corner, weight in sequence:
        for channel, value in enumerate(corners[corner]):
            values[channel] += value * weight
    return to_8bit(values[0]), to_8bit(values[1]), to_8bit(values[2])


class TestCubeLUTParsing:
    def test_path_stream_title_comments_and_scientific_notation(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "night.cube"
        path.write_text(
            "\n".join(
                [
                    "# exported by a color grading application",
                    'TITLE "Night & Day #1" # an inline comment',
                    "LUT_3D_SIZE 2",
                    "DOMAIN_MIN -1e-1 0.0 0",
                    "DOMAIN_MAX 1.1 1e0 1",
                    "",
                    "-0e0 0.0 0 # red changes fastest",
                    "1.0 0 0",
                    "0 1 0",
                    "1 1 0",
                    "0 0 1",
                    "1 0 1",
                    "0 1 1",
                    "1e0 1.0 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        from_path = ImageFilter.CubeLUT(path)
        stream = io.StringIO(path.read_text(encoding="utf-8"))
        from_stream = ImageFilter.CubeLUT(stream)

        assert not stream.closed
        assert from_path.title == "Night & Day #1"
        assert from_path.size == (2, 2, 2)
        assert from_path.domain_min == (-0.1, 0.0, 0.0)
        assert from_path.domain_max == (1.1, 1.0, 1.0)
        assert from_path.table == identity_table(2)
        assert from_stream.table == from_path.table

        image = Image.new("RGB", (1, 1), (51, 102, 153))
        assert image.filter(from_path).getpixel((0, 0)) == image.filter(
            from_stream
        ).getpixel((0, 0))

    def test_color3dlut_from_cube(self) -> None:
        lut = ImageFilter.Color3DLUT.from_cube(
            io.StringIO(cube_text(identity_table(2)))
        )
        assert isinstance(lut, ImageFilter.CubeLUT)

    def test_table_order_is_red_fastest(self) -> None:
        table = [
            value
            for index in range(8)
            for value in (index / 10, index + 1, index + 2)
        ]
        lut = ImageFilter.CubeLUT(io.StringIO(cube_text(table)))
        assert lut.table == table

    def test_domain_defaults(self) -> None:
        lut = ImageFilter.CubeLUT(io.StringIO(cube_text(identity_table(2))))
        assert lut.domain_min == (0.0, 0.0, 0.0)
        assert lut.domain_max == (1.0, 1.0, 1.0)

    def test_maximum_size(self) -> None:
        table = [0.0] * (65**3 * 3)
        lut = ImageFilter.CubeLUT(io.StringIO(cube_text(table, size=65)))
        assert lut.size == (65, 65, 65)


class TestCubeLUTFilter:
    def test_identity(self) -> None:
        image = Image.new("RGB", (256, 1))
        image.putdata([(value, value, value) for value in range(256)])
        result = image.filter(
            ImageFilter.CubeLUT(io.StringIO(cube_text(identity_table(17), size=17)))
        )
        assert result.tobytes() == image.tobytes()

    def test_non_default_domains_and_clamping(self) -> None:
        lut = ImageFilter.CubeLUT(
            io.StringIO(
                cube_text(
                    identity_table(2),
                    domain_min=(0.2, 0.4, 0.0),
                    domain_max=(0.8, 0.6, 0.5),
                )
            )
        )
        image = Image.new("RGB", (5, 1))
        image.putdata(
            [
                (0, 0, 0),
                (51, 102, 0),
                (128, 128, 128),
                (204, 153, 128),
                (255, 255, 255),
            ]
        )

        assert (
            image.filter(lut).getpixel((0, 0)),
            image.filter(lut).getpixel((1, 0)),
            image.filter(lut).getpixel((2, 0)),
            image.filter(lut).getpixel((3, 0)),
            image.filter(lut).getpixel((4, 0)),
        ) == ((0, 0, 0), (0, 0, 0), (128, 130, 255), (255, 255, 255), (255, 255, 255))

    def test_tetrahedral_not_trilinear(self) -> None:
        corners = {
            (0, 0, 0): (0.0, 0.0, 0.0),
            (1, 0, 0): (0.0, 0.0, 0.0),
            (0, 1, 0): (0.0, 0.0, 0.0),
            (1, 1, 0): (0.0, 0.0, 0.0),
            (0, 0, 1): (0.0, 0.0, 0.0),
            (1, 0, 1): (0.0, 0.0, 0.0),
            (0, 1, 1): (0.0, 0.0, 0.0),
            (1, 1, 1): (1.0, 0.0, 0.0),
        }
        image = Image.new("RGB", (1, 1), (51, 102, 153))
        result = image.filter(
            ImageFilter.CubeLUT(io.StringIO(cube_text(corner_table(corners))))
        )
        # Tetrahedral interpolation uses the nearest 0.2 corner. Trilinear
        # interpolation would produce 0.2 * 0.4 * 0.6 = 0.048 instead.
        assert result.getpixel((0, 0)) == (51, 0, 0)

    @pytest.mark.parametrize("fractions", list(permutations((0.2, 0.4, 0.6))))
    def test_all_tetrahedral_cases(
        self, fractions: tuple[float, float, float]
    ) -> None:
        corners = {
            (0, 0, 0): (0.0, 0.1, 0.2),
            (1, 0, 0): (0.9, 0.3, 0.4),
            (0, 1, 0): (0.3, 0.5, 0.6),
            (1, 1, 0): (1.2, 0.7, 0.8),
            (0, 0, 1): (-0.5, 0.9, 1.0),
            (1, 0, 1): (1.5, 1.1, 1.2),
            (0, 1, 1): (0.6, 1.3, 1.4),
            (1, 1, 1): (2.0, 1.5, 1.6),
        }
        pixel = tuple(int(round(value * 255)) for value in fractions)
        expected = tetrahedral_expected(
            corners,
            *fractions,
        )
        image = Image.new("RGB", (1, 1), pixel)
        result = image.filter(
            ImageFilter.CubeLUT(io.StringIO(cube_text(corner_table(corners))))
        )
        assert result.getpixel((0, 0)) == expected

    def test_equal_coordinates_and_endpoints(self) -> None:
        lut = ImageFilter.CubeLUT(
            io.StringIO(cube_text(identity_table(3), size=3))
        )
        image = Image.new("RGB", (5, 1))
        image.putdata([(value, value, value) for value in (0, 63, 128, 192, 255)])
        result = image.filter(lut)
        assert result.tobytes() == image.tobytes()

    def test_interpolates_before_clamping_table_values(self) -> None:
        corners = {
            (0, 0, 0): (-1.0, -1.0, -1.0),
            (1, 0, 0): (0.0, 0.0, 0.0),
            (0, 1, 0): (0.0, 0.0, 0.0),
            (1, 1, 0): (0.0, 0.0, 0.0),
            (0, 0, 1): (0.0, 0.0, 0.0),
            (1, 0, 1): (0.0, 0.0, 0.0),
            (0, 1, 1): (0.0, 0.0, 0.0),
            (1, 1, 1): (2.0, 2.0, 2.0),
        }
        image = Image.new("RGB", (1, 1), (51, 102, 153))
        result = image.filter(
            ImageFilter.CubeLUT(io.StringIO(cube_text(corner_table(corners))))
        )
        # -1 * (1 - 0.6) + 2 * 0.2 == 0. Clipping the table first would give 51.
        assert result.getpixel((0, 0)) == (0, 0, 0)

    def test_rgba_alpha_is_preserved_and_input_is_not_modified(self) -> None:
        corners = {
            (0, 0, 0): (1.0, 1.0, 1.0),
            (1, 0, 0): (0.0, 1.0, 1.0),
            (0, 1, 0): (1.0, 0.0, 1.0),
            (1, 1, 0): (0.0, 0.0, 1.0),
            (0, 0, 1): (1.0, 1.0, 0.0),
            (1, 0, 1): (0.0, 1.0, 0.0),
            (0, 1, 1): (1.0, 0.0, 0.0),
            (1, 1, 1): (0.0, 0.0, 0.0),
        }
        image = Image.new("RGBA", (2, 1))
        image.putdata([(0, 0, 0, 0), (255, 255, 255, 231)])
        original = image.tobytes()
        lut = ImageFilter.CubeLUT(io.StringIO(cube_text(corner_table(corners))))
        table_before = lut.table.copy()

        result = image.filter(lut)

        assert result.getpixel((0, 0)) == (255, 255, 255, 0)
        assert result.getpixel((1, 0)) == (0, 0, 0, 231)
        assert image.tobytes() == original
        assert lut.table == table_before


class TestCubeLUTErrors:
    @pytest.mark.parametrize(
        "text, match",
        [
            ("TITLE \"missing size\"\n", "LUT_3D_SIZE is required"),
            (
                "LUT_3D_SIZE 2\nLUT_3D_SIZE 2\n" + cube_text(identity_table(2)),
                "Duplicate LUT_3D_SIZE",
            ),
            (
                cube_text(identity_table(2)).replace(
                    "LUT_3D_SIZE 2", "LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MIN 0 0 0"
                ),
                "Duplicate DOMAIN_MIN",
            ),
            (
                cube_text(identity_table(2)).replace(
                    'LUT_3D_SIZE 2', 'LUT_3D_SIZE 2\nTITLE "a"\nTITLE "b"'
                ),
                "Duplicate TITLE",
            ),
            (
                "LUT_3D_SIZE 2\n" + "\n".join(["0 0 0"] * 7),
                "expected 8 rows, found 7",
            ),
            (
                "LUT_3D_SIZE 2\n" + "\n".join(["0 0 0"] * 9),
                "expected 8 rows, found 9",
            ),
            (
                "LUT_3D_SIZE 2\n" + "\n".join(["0 0 nan"] + ["0 0 0"] * 7),
                "must be finite",
            ),
            (
                "LUT_3D_SIZE 2\n" + "\n".join(["0 0 inf"] + ["0 0 0"] * 7),
                "must be finite",
            ),
            (
                "LUT_3D_SIZE 2\nDOMAIN_MIN nan 0 0\nDOMAIN_MAX 1 1 1\n",
                "must be finite",
            ),
            (
                "LUT_3D_SIZE 2\nDOMAIN_MIN 1 0 0\nDOMAIN_MAX 1 1 1\n",
                "DOMAIN_MIN must be less than DOMAIN_MAX",
            ),
            ("LUT_1D_SIZE 2\n", "1D LUTs and combined 1D/3D LUTs"),
            (
                "LUT_1D_SIZE 2\nLUT_3D_SIZE 2\n",
                "1D LUTs and combined 1D/3D LUTs",
            ),
            (
                "LUT_3D_SIZE 2\nDOMAIN_MIN 0 0\n",
                "exactly three numeric values",
            ),
            ("LUT_3D_SIZE 2\n0 0\n", "Expected three table values"),
            ("LUT_3D_SIZE 2.0\n", "Invalid LUT_3D_SIZE"),
            ("LUT_3D_SIZE 1\n", r"\[2, 65\]"),
            ("LUT_3D_SIZE 66\n", r"\[2, 65\]"),
            ("LUT_UNKNOWN 2\n", "Unsupported directive"),
        ],
    )
    def test_invalid_files(self, text: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            ImageFilter.CubeLUT(io.StringIO(text))

    def test_invalid_source_types(self) -> None:
        with pytest.raises(TypeError, match="path or an open text stream"):
            ImageFilter.CubeLUT(123)  # type: ignore[arg-type]
        stream = io.BytesIO(b"LUT_3D_SIZE 2\n")
        with pytest.raises(TypeError, match="should return str"):
            ImageFilter.CubeLUT(stream)  # type: ignore[arg-type]

    def test_caller_stream_is_not_closed(self) -> None:
        stream = io.StringIO(cube_text(identity_table(2)))
        ImageFilter.CubeLUT(stream)
        assert not stream.closed

    def test_wrong_image_mode(self) -> None:
        lut = ImageFilter.CubeLUT(io.StringIO(cube_text(identity_table(2))))
        with pytest.raises(ValueError, match="RGB and RGBA"):
            Image.new("L", (1, 1)).filter(lut)
