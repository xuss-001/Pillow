from __future__ import annotations

import struct
import zlib
from io import BytesIO
from pathlib import Path
from typing import TypedDict

import pytest

from PIL import Image, ImageSequence, PngImagePlugin


class _APNGFrame(TypedDict, total=False):
    image: Image.Image
    bbox: tuple[int, int, int, int]
    duration: int
    disposal: int
    blend: int


def _chunk(cid: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + cid
        + data
        + struct.pack(">I", zlib.crc32(cid + data) & 0xFFFFFFFF)
    )


def _make_apng(
    frames: list[_APNGFrame],
    size: tuple[int, int],
    default_image: Image.Image | None = None,
) -> bytes:
    data = b"\x89PNG\r\n\x1a\n"
    data += _chunk(b"IHDR", struct.pack(">IIBBBBB", size[0], size[1], 8, 6, 0, 0, 0))
    data += _chunk(b"acTL", struct.pack(">II", len(frames), 0))

    seq_num = 0
    if default_image:
        width, height = default_image.size
        default_raw = b"".join(
            b"\0" + default_image.tobytes()[row * width * 4 : (row + 1) * width * 4]
            for row in range(height)
        )
        data += _chunk(b"IDAT", zlib.compress(default_raw))

    for frame_index, frame in enumerate(frames):
        image = frame["image"]
        bbox = frame.get("bbox", (0, 0) + image.size)
        x, y, right, bottom = bbox
        width, height = right - x, bottom - y
        duration = frame.get("duration", 100)
        disposal = frame.get("disposal", PngImagePlugin.Disposal.OP_NONE)
        blend = frame.get("blend", PngImagePlugin.Blend.OP_SOURCE)

        data += _chunk(
            b"fcTL",
            struct.pack(
                ">IIIIIHHBB",
                seq_num,
                width,
                height,
                x,
                y,
                duration,
                1000,
                disposal,
                blend,
            ),
        )
        seq_num += 1

        raw = b"".join(
            b"\0" + image.tobytes()[row * width * 4 : (row + 1) * width * 4]
            for row in range(height)
        )
        compressed = zlib.compress(raw)
        if frame_index == 0 and default_image is None:
            data += _chunk(b"IDAT", compressed)
        else:
            data += _chunk(b"fdAT", struct.pack(">I", seq_num) + compressed)
            seq_num += 1

    return data + _chunk(b"IEND", b"")


def _get_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    chunks: list[tuple[bytes, bytes]] = []
    pos = 8
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        cid = data[pos + 4 : pos + 8]
        chunks.append((cid, data[pos + 8 : pos + 8 + length]))
        pos += length + 12
    return chunks


def _get_fctl(data: bytes) -> list[tuple[int, ...]]:
    return [
        struct.unpack(">IIIIIHHBB", chunk)
        for cid, chunk in _get_chunks(data)
        if cid == b"fcTL"
    ]


# APNG browser support tests and fixtures via:
# https://philip.html5.org/tests/apng/tests.html
# (referenced from https://wiki.mozilla.org/APNG_Specification)
def test_apng_basic() -> None:
    with Image.open("Tests/images/apng/single_frame.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert not im.is_animated
        assert im.n_frames == 1
        assert im.get_format_mimetype() == "image/apng"
        assert im.info.get("default_image") is None
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    with Image.open("Tests/images/apng/single_frame_default.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.is_animated
        assert im.n_frames == 2
        assert im.get_format_mimetype() == "image/apng"
        assert im.info.get("default_image")
        assert im.getpixel((0, 0)) == (255, 0, 0, 255)
        assert im.getpixel((64, 32)) == (255, 0, 0, 255)
        im.seek(1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

        # test out of bounds seek
        with pytest.raises(EOFError):
            im.seek(2)

        im.seek(0)
        with pytest.raises(ValueError, match="cannot seek to frame 2"):
            im._seek(2)

        # test rewind support
        assert im.getpixel((0, 0)) == (255, 0, 0, 255)
        assert im.getpixel((64, 32)) == (255, 0, 0, 255)
        im.seek(1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)


@pytest.mark.parametrize(
    "filename",
    ("Tests/images/apng/split_fdat.png", "Tests/images/apng/split_fdat_zero_chunk.png"),
)
def test_apng_fdat(filename: str) -> None:
    with Image.open(filename) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)


def test_apng_dispose() -> None:
    with Image.open("Tests/images/apng/dispose_op_none.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    with Image.open("Tests/images/apng/dispose_op_background.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 0, 0, 0)
        assert im.getpixel((64, 32)) == (0, 0, 0, 0)

    with Image.open("Tests/images/apng/dispose_op_background_final.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    with Image.open("Tests/images/apng/dispose_op_previous.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    with Image.open("Tests/images/apng/dispose_op_previous_final.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    with Image.open("Tests/images/apng/dispose_op_previous_first.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 0, 0, 0)
        assert im.getpixel((64, 32)) == (0, 0, 0, 0)


def test_apng_dispose_region() -> None:
    with Image.open("Tests/images/apng/dispose_op_none_region.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    with Image.open("Tests/images/apng/dispose_op_background_before_region.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 0, 0, 0)
        assert im.getpixel((64, 32)) == (0, 0, 0, 0)

    with Image.open("Tests/images/apng/dispose_op_background_region.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 0, 255, 255)
        assert im.getpixel((64, 32)) == (0, 0, 0, 0)

    with Image.open("Tests/images/apng/dispose_op_previous_region.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)


def test_apng_dispose_op_previous_frame() -> None:
    # Test that the dispose settings being used are from the previous frame
    #
    # Image created with:
    # red = Image.new("RGBA", (128, 64), (255, 0, 0, 255))
    # green = red.copy()
    # green.paste(Image.new("RGBA", (64, 32), (0, 255, 0, 255)))
    # blue = red.copy()
    # blue.paste(Image.new("RGBA", (64, 32), (0, 255, 0, 255)), (64, 32))
    #
    # red.save(
    #     "Tests/images/apng/dispose_op_previous_frame.png",
    #     save_all=True,
    #     append_images=[green, blue],
    #     disposal=[
    #         PngImagePlugin.Disposal.OP_NONE,
    #         PngImagePlugin.Disposal.OP_PREVIOUS,
    #         PngImagePlugin.Disposal.OP_PREVIOUS
    #     ],
    # )
    with Image.open("Tests/images/apng/dispose_op_previous_frame.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (255, 0, 0, 255)


def test_apng_dispose_op_background_p_mode() -> None:
    with Image.open("Tests/images/apng/dispose_op_background_p_mode.png") as im:
        im.seek(1)
        im.load()
        assert im.size == (128, 64)


def test_apng_blend() -> None:
    with Image.open("Tests/images/apng/blend_op_source_solid.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    with Image.open("Tests/images/apng/blend_op_source_transparent.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 0, 0, 0)
        assert im.getpixel((64, 32)) == (0, 0, 0, 0)

    with Image.open("Tests/images/apng/blend_op_source_near_transparent.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 2)
        assert im.getpixel((64, 32)) == (0, 255, 0, 2)

    with Image.open("Tests/images/apng/blend_op_over.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    with Image.open("Tests/images/apng/blend_op_over_near_transparent.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)


def test_apng_blend_transparency() -> None:
    with Image.open("Tests/images/blend_transparency.png") as im:
        im.seek(1)
        assert im.getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("with_default_image", (False, True))
@pytest.mark.parametrize(
    "first_blend",
    (PngImagePlugin.Blend.OP_SOURCE, PngImagePlugin.Blend.OP_OVER),
)
def test_apng_rgba_over_alpha_composite(
    with_default_image: bool, first_blend: PngImagePlugin.Blend
) -> None:
    semi_transparent_red = Image.new("RGBA", (1, 1), (255, 0, 0, 128))
    default_image = Image.new("RGBA", (1, 1), (0, 255, 0, 255))
    frames = [
        {
            "image": semi_transparent_red.copy(),
            "disposal": PngImagePlugin.Disposal.OP_NONE,
            "blend": first_blend,
        },
        {
            "image": semi_transparent_red.copy(),
            "disposal": PngImagePlugin.Disposal.OP_NONE,
            "blend": PngImagePlugin.Blend.OP_OVER,
        },
        {
            "image": semi_transparent_red.copy(),
            "disposal": PngImagePlugin.Disposal.OP_NONE,
            "blend": PngImagePlugin.Blend.OP_OVER,
        },
    ]
    data = _make_apng(
        frames,
        (1, 1),
        default_image if with_default_image else None,
    )

    with Image.open(BytesIO(data)) as im:
        assert im.n_frames == len(frames) + int(with_default_image)
        offset = int(with_default_image)
        if with_default_image:
            assert im.getpixel((0, 0)) == (0, 255, 0, 255)

        for index, alpha in enumerate((128, 192, 224)):
            im.seek(offset + index)
            assert im.getpixel((0, 0)) == (255, 0, 0, alpha)

        # Seeking backwards and forwards must reproduce the same frames.
        im.seek(offset)
        assert im.getpixel((0, 0)) == (255, 0, 0, 128)
        im.seek(offset + 2)
        assert im.getpixel((0, 0)) == (255, 0, 0, 224)


def test_apng_rgba_disposal_updates_canvas_state() -> None:
    size = (4, 1)
    red = Image.new("RGBA", size, (255, 0, 0, 255))
    blue = Image.new("RGBA", (1, 1), (0, 0, 255, 255))
    green = Image.new("RGBA", (1, 1), (0, 255, 0, 128))
    frames = [
        {
            "image": red,
            "disposal": PngImagePlugin.Disposal.OP_NONE,
            "blend": PngImagePlugin.Blend.OP_SOURCE,
        },
        {
            "image": blue,
            "bbox": (1, 0, 2, 1),
            "disposal": PngImagePlugin.Disposal.OP_PREVIOUS,
            "blend": PngImagePlugin.Blend.OP_SOURCE,
        },
        {
            "image": green,
            "bbox": (1, 0, 2, 1),
            "disposal": PngImagePlugin.Disposal.OP_BACKGROUND,
            "blend": PngImagePlugin.Blend.OP_OVER,
        },
        {
            "image": blue,
            "bbox": (2, 0, 3, 1),
            "disposal": PngImagePlugin.Disposal.OP_NONE,
            "blend": PngImagePlugin.Blend.OP_SOURCE,
        },
    ]
    data = _make_apng(frames, size)

    with Image.open(BytesIO(data)) as im:
        im.seek(1)
        assert [im.getpixel((x, 0)) for x in range(4)] == [
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (255, 0, 0, 255),
            (255, 0, 0, 255),
        ]

        # OP_PREVIOUS restores the red pixel before OP_OVER is applied.
        im.seek(2)
        assert [im.getpixel((x, 0)) for x in range(4)] == [
            (255, 0, 0, 255),
            (127, 128, 0, 255),
            (255, 0, 0, 255),
            (255, 0, 0, 255),
        ]

        # OP_BACKGROUND clears only the previous frame's local region.
        im.seek(3)
        assert [im.getpixel((x, 0)) for x in range(4)] == [
            (255, 0, 0, 255),
            (0, 0, 0, 0),
            (0, 0, 255, 255),
            (255, 0, 0, 255),
        ]

        im.seek(2)
        assert im.getpixel((1, 0)) == (127, 128, 0, 255)
        im.seek(3)
        assert im.getpixel((1, 0)) == (0, 0, 0, 0)


def test_apng_chunk_order() -> None:
    with Image.open("Tests/images/apng/fctl_actl.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)


def test_apng_delay() -> None:
    with Image.open("Tests/images/apng/delay.png") as im:
        im.seek(1)
        assert im.info.get("duration") == 500.0
        im.seek(2)
        assert im.info.get("duration") == 1000.0
        im.seek(3)
        assert im.info.get("duration") == 500.0
        im.seek(4)
        assert im.info.get("duration") == 1000.0

    with Image.open("Tests/images/apng/delay_round.png") as im:
        im.seek(1)
        assert im.info.get("duration") == 500.0
        im.seek(2)
        assert im.info.get("duration") == 1000.0

    with Image.open("Tests/images/apng/delay_short_max.png") as im:
        im.seek(1)
        assert im.info.get("duration") == 500.0
        im.seek(2)
        assert im.info.get("duration") == 1000.0

    with Image.open("Tests/images/apng/delay_zero_denom.png") as im:
        im.seek(1)
        assert im.info.get("duration") == 500.0
        im.seek(2)
        assert im.info.get("duration") == 1000.0

    with Image.open("Tests/images/apng/delay_zero_numer.png") as im:
        im.seek(1)
        assert im.info.get("duration") == 0.0
        im.seek(2)
        assert im.info.get("duration") == 0.0
        im.seek(3)
        assert im.info.get("duration") == 500.0
        im.seek(4)
        assert im.info.get("duration") == 1000.0


def test_apng_num_plays() -> None:
    with Image.open("Tests/images/apng/num_plays.png") as im:
        assert im.info.get("loop") == 0

    with Image.open("Tests/images/apng/num_plays_1.png") as im:
        assert im.info.get("loop") == 1


def test_apng_mode() -> None:
    with Image.open("Tests/images/apng/mode_16bit.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.mode == "RGBA"
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (0, 0, 128, 255)
        assert im.getpixel((64, 32)) == (0, 0, 128, 255)

    with Image.open("Tests/images/apng/mode_grayscale.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.mode == "L"
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == 128
        assert im.getpixel((64, 32)) == 255

    with Image.open("Tests/images/apng/mode_grayscale_alpha.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.mode == "LA"
        im.seek(im.n_frames - 1)
        assert im.getpixel((0, 0)) == (128, 191)
        assert im.getpixel((64, 32)) == (128, 191)

    with Image.open("Tests/images/apng/mode_palette.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.mode == "P"
        im.seek(im.n_frames - 1)
        im_rgb = im.convert("RGB")
    assert im_rgb.getpixel((0, 0)) == (0, 255, 0)
    assert im_rgb.getpixel((64, 32)) == (0, 255, 0)

    with Image.open("Tests/images/apng/mode_palette_alpha.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.mode == "P"
        im.seek(im.n_frames - 1)
        im_rgba = im.convert("RGBA")
    assert im_rgba.getpixel((0, 0)) == (0, 255, 0, 255)
    assert im_rgba.getpixel((64, 32)) == (0, 255, 0, 255)

    with Image.open("Tests/images/apng/mode_palette_1bit_alpha.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.mode == "P"
        im.seek(im.n_frames - 1)
        im_rgba = im.convert("RGBA")
    assert im_rgba.getpixel((0, 0)) == (0, 0, 255, 128)
    assert im_rgba.getpixel((64, 32)) == (0, 0, 255, 128)


def test_apng_chunk_errors() -> None:
    with Image.open("Tests/images/apng/chunk_no_actl.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert not im.is_animated

    with pytest.warns(UserWarning, match="Invalid APNG"):
        im = Image.open("Tests/images/apng/chunk_multi_actl.png")
    assert isinstance(im, PngImagePlugin.PngImageFile)
    assert not im.is_animated
    im.close()

    with Image.open("Tests/images/apng/chunk_actl_after_idat.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert not im.is_animated

    with Image.open("Tests/images/apng/chunk_no_fctl.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        with pytest.raises(SyntaxError):
            im.seek(im.n_frames - 1)

    with Image.open("Tests/images/apng/chunk_repeat_fctl.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        with pytest.raises(SyntaxError):
            im.seek(im.n_frames - 1)

    with Image.open("Tests/images/apng/chunk_no_fdat.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        with pytest.raises(SyntaxError):
            im.seek(im.n_frames - 1)


def test_apng_syntax_errors() -> None:
    with pytest.warns(UserWarning, match="Invalid APNG"):
        im = Image.open("Tests/images/apng/syntax_num_frames_zero.png")
    assert isinstance(im, PngImagePlugin.PngImageFile)
    assert not im.is_animated
    with pytest.raises(OSError):
        im.load()
    im.close()

    with pytest.warns(UserWarning, match="Invalid APNG"):
        im = Image.open("Tests/images/apng/syntax_num_frames_zero_default.png")
    assert isinstance(im, PngImagePlugin.PngImageFile)
    assert not im.is_animated
    im.load()
    im.close()

    # we can handle this case gracefully
    with Image.open("Tests/images/apng/syntax_num_frames_low.png") as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)

    with pytest.raises(OSError):
        with Image.open("Tests/images/apng/syntax_num_frames_high.png") as im:
            assert isinstance(im, PngImagePlugin.PngImageFile)
            im.seek(im.n_frames - 1)
            im.load()

    with pytest.warns(UserWarning, match="Invalid APNG"):
        im = Image.open("Tests/images/apng/syntax_num_frames_invalid.png")
    assert isinstance(im, PngImagePlugin.PngImageFile)
    assert not im.is_animated
    im.load()
    im.close()


@pytest.mark.parametrize(
    "test_file",
    (
        "sequence_start.png",
        "sequence_gap.png",
        "sequence_repeat.png",
        "sequence_repeat_chunk.png",
        "sequence_reorder.png",
        "sequence_reorder_chunk.png",
        "sequence_fdat_fctl.png",
    ),
)
def test_apng_sequence_errors(test_file: str) -> None:
    with pytest.raises(SyntaxError):
        with Image.open(f"Tests/images/apng/{test_file}") as im:
            assert isinstance(im, PngImagePlugin.PngImageFile)
            im.seek(im.n_frames - 1)
            im.load()


def test_apng_save(tmp_path: Path) -> None:
    with Image.open("Tests/images/apng/single_frame.png") as im:
        test_file = tmp_path / "temp.png"
        im.save(test_file, save_all=True)

    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.load()
        assert not im.is_animated
        assert im.n_frames == 1
        assert im.get_format_mimetype() == "image/png"
        assert im.info.get("default_image") is None
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    with Image.open("Tests/images/apng/single_frame_default.png") as im:
        frames = [frame_im.copy() for frame_im in ImageSequence.Iterator(im)]
        frames[0].save(
            test_file, save_all=True, default_image=True, append_images=frames[1:]
        )

    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.load()
        assert im.is_animated
        assert im.n_frames == 2
        assert im.get_format_mimetype() == "image/apng"
        assert im.info.get("default_image")
        im.seek(1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)


def test_apng_save_alpha(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"

    im = Image.new("RGBA", (1, 1), (255, 0, 0, 255))
    im2 = Image.new("RGBA", (1, 1), (255, 0, 0, 127))
    im.save(test_file, save_all=True, append_images=[im2])

    with Image.open(test_file) as reloaded:
        assert reloaded.getpixel((0, 0)) == (255, 0, 0, 255)

        reloaded.seek(1)
        assert reloaded.getpixel((0, 0)) == (255, 0, 0, 127)


def test_apng_save_split_fdat(tmp_path: Path) -> None:
    # test to make sure we do not generate sequence errors when writing
    # frames with image data spanning multiple fdAT chunks (in this case
    # both the default image and first animation frame will span multiple
    # data chunks)
    test_file = tmp_path / "temp.png"
    with Image.open("Tests/images/old-style-jpeg-compression.png") as im:
        frames = [im.copy(), Image.new("RGBA", im.size, (255, 0, 0, 255))]
        im.save(
            test_file,
            save_all=True,
            default_image=True,
            append_images=frames,
        )
    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        im.seek(im.n_frames - 1)
        im.load()


def test_apng_save_duration_loop(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    with Image.open("Tests/images/apng/delay.png") as im:
        frames = []
        durations = []
        loop = im.info.get("loop")
        default_image = im.info.get("default_image")
        for i, frame_im in enumerate(ImageSequence.Iterator(im)):
            frames.append(frame_im.copy())
            if i != 0 or not default_image:
                durations.append(frame_im.info.get("duration", 0))
        frames[0].save(
            test_file,
            save_all=True,
            default_image=default_image,
            append_images=frames[1:],
            duration=durations,
            loop=loop,
        )

    with Image.open(test_file) as im:
        im.load()
        assert im.info.get("loop") == loop
        im.seek(1)
        assert im.info.get("duration") == 500.0
        im.seek(2)
        assert im.info.get("duration") == 1000.0
        im.seek(3)
        assert im.info.get("duration") == 500.0
        im.seek(4)
        assert im.info.get("duration") == 1000.0

    # test removal of duplicated frames
    frame = Image.new("RGBA", (128, 64), (255, 0, 0, 255))
    frame.save(
        test_file, save_all=True, append_images=[frame, frame], duration=[500, 100, 150]
    )
    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.n_frames == 1
        assert "duration" not in im.info

    different_frame = Image.new("RGBA", (128, 64))
    frame.save(
        test_file,
        save_all=True,
        append_images=[frame, different_frame],
        duration=[500, 100, 150],
    )
    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.n_frames == 2
        assert im.info["duration"] == 600

        im.seek(1)
        assert im.info["duration"] == 150

    # test info duration
    frame.info["duration"] = 300
    frame.save(test_file, save_all=True, append_images=[frame, different_frame])
    with Image.open(test_file) as im:
        assert isinstance(im, PngImagePlugin.PngImageFile)
        assert im.n_frames == 2
        assert im.info["duration"] == 600


def test_apng_save_duration_float(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    im = Image.new("1", (1, 1))
    im2 = Image.new("1", (1, 1), 1)
    im.save(test_file, save_all=True, append_images=[im2], duration=0.5)

    with Image.open(test_file) as reloaded:
        assert reloaded.info["duration"] == 0.5


def test_apng_save_large_duration(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    im = Image.new("1", (1, 1))
    im2 = Image.new("1", (1, 1), 1)
    with pytest.raises(ValueError, match="cannot write duration"):
        im.save(test_file, save_all=True, append_images=[im2], duration=65536000)


def test_apng_save_disposal(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    size = (128, 64)
    red = Image.new("RGBA", size, (255, 0, 0, 255))
    green = Image.new("RGBA", size, (0, 255, 0, 255))
    transparent = Image.new("RGBA", size, (0, 0, 0, 0))

    # test OP_NONE
    red.save(
        test_file,
        save_all=True,
        append_images=[green, transparent],
        disposal=PngImagePlugin.Disposal.OP_NONE,
        blend=PngImagePlugin.Blend.OP_OVER,
    )
    with Image.open(test_file) as im:
        im.seek(2)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    # test OP_BACKGROUND
    disposal = [
        PngImagePlugin.Disposal.OP_NONE,
        PngImagePlugin.Disposal.OP_BACKGROUND,
        PngImagePlugin.Disposal.OP_NONE,
    ]
    red.save(
        test_file,
        save_all=True,
        append_images=[red, transparent],
        disposal=disposal,
        blend=PngImagePlugin.Blend.OP_OVER,
    )
    with Image.open(test_file) as im:
        im.seek(2)
        assert im.getpixel((0, 0)) == (0, 0, 0, 0)
        assert im.getpixel((64, 32)) == (0, 0, 0, 0)

    disposal = [
        PngImagePlugin.Disposal.OP_NONE,
        PngImagePlugin.Disposal.OP_BACKGROUND,
    ]
    red.save(
        test_file,
        save_all=True,
        append_images=[green],
        disposal=disposal,
        blend=PngImagePlugin.Blend.OP_OVER,
    )
    with Image.open(test_file) as im:
        im.seek(1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    # test OP_PREVIOUS
    disposal = [
        PngImagePlugin.Disposal.OP_NONE,
        PngImagePlugin.Disposal.OP_PREVIOUS,
        PngImagePlugin.Disposal.OP_NONE,
    ]
    red.save(
        test_file,
        save_all=True,
        append_images=[green, red, transparent],
        default_image=True,
        disposal=disposal,
        blend=PngImagePlugin.Blend.OP_OVER,
    )
    with Image.open(test_file) as im:
        im.seek(3)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    disposal = [
        PngImagePlugin.Disposal.OP_NONE,
        PngImagePlugin.Disposal.OP_PREVIOUS,
    ]
    red.save(
        test_file,
        save_all=True,
        append_images=[green],
        disposal=disposal,
        blend=PngImagePlugin.Blend.OP_OVER,
    )
    with Image.open(test_file) as im:
        im.seek(1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    # test info disposal
    red.info["disposal"] = PngImagePlugin.Disposal.OP_BACKGROUND
    red.save(
        test_file,
        save_all=True,
        append_images=[Image.new("RGBA", (10, 10), (0, 255, 0, 255))],
    )
    with Image.open(test_file) as im:
        im.seek(1)
        assert im.getpixel((64, 32)) == (0, 0, 0, 0)


def test_apng_save_disposal_previous(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    size = (128, 64)
    blue = Image.new("RGBA", size, (0, 0, 255, 255))
    red = Image.new("RGBA", size, (255, 0, 0, 255))
    green = Image.new("RGBA", size, (0, 255, 0, 255))

    # test OP_NONE
    blue.save(
        test_file,
        save_all=True,
        append_images=[red, green],
        disposal=PngImagePlugin.Disposal.OP_PREVIOUS,
    )
    with Image.open(test_file) as im:
        assert im.getpixel((0, 0)) == (0, 0, 255, 255)

        im.seek(2)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)


def test_apng_save_blend(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"
    size = (128, 64)
    red = Image.new("RGBA", size, (255, 0, 0, 255))
    green = Image.new("RGBA", size, (0, 255, 0, 255))
    transparent = Image.new("RGBA", size, (0, 0, 0, 0))

    # test OP_SOURCE on solid color
    blend = [
        PngImagePlugin.Blend.OP_OVER,
        PngImagePlugin.Blend.OP_SOURCE,
    ]
    red.save(
        test_file,
        save_all=True,
        append_images=[red, green],
        default_image=True,
        disposal=PngImagePlugin.Disposal.OP_NONE,
        blend=blend,
    )
    with Image.open(test_file) as im:
        im.seek(2)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    # test OP_SOURCE on transparent color
    blend = [
        PngImagePlugin.Blend.OP_OVER,
        PngImagePlugin.Blend.OP_SOURCE,
    ]
    red.save(
        test_file,
        save_all=True,
        append_images=[red, transparent],
        default_image=True,
        disposal=PngImagePlugin.Disposal.OP_NONE,
        blend=blend,
    )
    with Image.open(test_file) as im:
        im.seek(2)
        assert im.getpixel((0, 0)) == (0, 0, 0, 0)
        assert im.getpixel((64, 32)) == (0, 0, 0, 0)

    # test OP_OVER
    red.save(
        test_file,
        save_all=True,
        append_images=[green, transparent],
        default_image=True,
        disposal=PngImagePlugin.Disposal.OP_NONE,
        blend=PngImagePlugin.Blend.OP_OVER,
    )
    with Image.open(test_file) as im:
        im.seek(1)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)
        im.seek(2)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        assert im.getpixel((64, 32)) == (0, 255, 0, 255)

    # test info blend
    red.info["blend"] = PngImagePlugin.Blend.OP_OVER
    red.save(test_file, save_all=True, append_images=[green, transparent])
    with Image.open(test_file) as im:
        im.seek(2)
        assert im.getpixel((0, 0)) == (0, 255, 0, 255)


@pytest.mark.parametrize("with_default_image", (False, True))
def test_apng_save_rgba_over_preserves_frames_and_durations(
    with_default_image: bool,
) -> None:
    red = Image.new("RGBA", (2, 1), (255, 0, 0, 128))
    cover = Image.new("RGBA", (2, 1), (0, 255, 0, 255))
    if with_default_image:
        base = cover
        append_images = [red.copy(), red.copy(), red.copy()]
    else:
        base = red
        append_images = [red.copy(), red.copy()]
    original_frames = [
        (image.tobytes(), image.info.copy()) for image in [base, *append_images]
    ]
    out = BytesIO()

    base.save(
        out,
        "PNG",
        save_all=True,
        default_image=with_default_image,
        append_images=append_images,
        disposal=PngImagePlugin.Disposal.OP_NONE,
        blend=[
            PngImagePlugin.Blend.OP_SOURCE,
            PngImagePlugin.Blend.OP_OVER,
            PngImagePlugin.Blend.OP_OVER,
        ],
        duration=[100, 200, 300],
    )
    data = out.getvalue()

    assert [
        (image.tobytes(), image.info) for image in [base, *append_images]
    ] == original_frames

    fctl = _get_fctl(data)
    assert len(fctl) == 3
    assert [frame[7] for frame in fctl] == [0, 0, 0]
    assert [frame[8] for frame in fctl] == [0, 1, 1]
    assert [frame[5] / frame[6] * 1000 for frame in fctl] == [100, 200, 300]

    with Image.open(BytesIO(data)) as im:
        assert im.n_frames == 3 + int(with_default_image)
        offset = int(with_default_image)
        if with_default_image:
            assert im.getpixel((0, 0)) == (0, 255, 0, 255)
        total_duration = 0
        for index, alpha in enumerate((128, 192, 224)):
            im.seek(offset + index)
            assert im.getpixel((0, 0)) == (255, 0, 0, alpha)
            total_duration += im.info["duration"]
        assert total_duration == 600

        im.seek(offset)
        assert im.getpixel((0, 0)) == (255, 0, 0, 128)
        im.seek(offset + 2)
        assert im.getpixel((0, 0)) == (255, 0, 0, 224)


def test_apng_save_optimizes_disposal_regions() -> None:
    size = (4, 1)

    def patched(color: tuple[int, int, int, int]) -> Image.Image:
        image = red.copy()
        image.paste(Image.new("RGBA", (1, 1), color), (1, 0))
        return image

    red = Image.new("RGBA", size, (255, 0, 0, 255))
    blue = patched((0, 0, 255, 255))
    green = patched((0, 255, 0, 128))
    cleared = patched((0, 0, 0, 0))
    out = BytesIO()

    red.save(
        out,
        "PNG",
        save_all=True,
        append_images=[blue, green, cleared],
        disposal=[
            PngImagePlugin.Disposal.OP_NONE,
            PngImagePlugin.Disposal.OP_PREVIOUS,
            PngImagePlugin.Disposal.OP_BACKGROUND,
            PngImagePlugin.Disposal.OP_NONE,
        ],
        blend=[
            PngImagePlugin.Blend.OP_SOURCE,
            PngImagePlugin.Blend.OP_SOURCE,
            PngImagePlugin.Blend.OP_OVER,
            PngImagePlugin.Blend.OP_SOURCE,
        ],
        duration=[100, 100, 100, 100],
    )
    data = out.getvalue()

    fctl = _get_fctl(data)
    assert len(fctl) == 4
    assert fctl[1][1:5] == (1, 1, 1, 0)
    assert fctl[1][7:] == (
        PngImagePlugin.Disposal.OP_PREVIOUS,
        PngImagePlugin.Blend.OP_SOURCE,
    )
    assert fctl[2][1:5] == (1, 1, 1, 0)
    assert fctl[2][7:] == (
        PngImagePlugin.Disposal.OP_BACKGROUND,
        PngImagePlugin.Blend.OP_OVER,
    )

    with Image.open(BytesIO(data)) as im:
        assert im.n_frames == 4
        assert sum(
            frame.info["duration"] for frame in ImageSequence.Iterator(im)
        ) == 400

        im.seek(1)
        assert [im.getpixel((x, 0)) for x in range(4)] == [
            (255, 0, 0, 255),
            (0, 0, 255, 255),
            (255, 0, 0, 255),
            (255, 0, 0, 255),
        ]
        im.seek(2)
        assert im.getpixel((1, 0)) == (127, 128, 0, 255)
        im.seek(3)
        assert [im.getpixel((x, 0)) for x in range(4)] == [
            (255, 0, 0, 255),
            (0, 0, 0, 0),
            (255, 0, 0, 255),
            (255, 0, 0, 255),
        ]


def test_apng_save_size(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"

    im = Image.new("L", (100, 100))
    im.save(test_file, save_all=True, append_images=[Image.new("L", (200, 200))])

    with Image.open(test_file) as reloaded:
        assert reloaded.size == (200, 200)


def test_compress_level() -> None:
    compress_level_sizes = {}
    for compress_level in (0, 9):
        out = BytesIO()

        im = Image.new("L", (100, 100))
        im.save(
            out,
            "PNG",
            save_all=True,
            append_images=[Image.new("L", (200, 200))],
            compress_level=compress_level,
        )

        compress_level_sizes[compress_level] = len(out.getvalue())

    assert compress_level_sizes[0] > compress_level_sizes[9]


def test_seek_after_close() -> None:
    im = Image.open("Tests/images/apng/delay.png")
    im.seek(1)
    im.close()

    with pytest.raises(ValueError):
        im.seek(0)


@pytest.mark.parametrize("mode", ("RGBA", "RGB", "P"))
@pytest.mark.parametrize("default_image", (True, False))
@pytest.mark.parametrize("duplicate", (True, False))
def test_different_modes_in_later_frames(
    mode: str, default_image: bool, duplicate: bool, tmp_path: Path
) -> None:
    test_file = tmp_path / "temp.png"

    im = Image.new("L", (1, 1))
    im.save(
        test_file,
        save_all=True,
        default_image=default_image,
        append_images=[im.convert(mode) if duplicate else Image.new(mode, (1, 1), 1)],
    )
    with Image.open(test_file) as reloaded:
        assert reloaded.mode == mode


def test_different_durations(tmp_path: Path) -> None:
    test_file = tmp_path / "temp.png"

    with Image.open("Tests/images/apng/different_durations.png") as im:
        for _ in range(3):
            im.seek(0)
            assert im.info["duration"] == 4000

            im.seek(1)
            assert im.info["duration"] == 1000

        im.save(test_file, save_all=True)

    with Image.open(test_file) as reloaded:
        assert reloaded.info["duration"] == 4000

        reloaded.seek(1)
        assert reloaded.info["duration"] == 1000
