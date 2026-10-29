import io
import struct
import zlib

import pytest
from conftest import image_bytes
from PIL import Image, ImageFile

from deepvision.frontend.uploads import UploadError, validate_upload

LIMITS = {"max_bytes": 1024 * 1024, "max_pixels": 10_000}


def test_valid_jpeg_and_png():
    jpeg = validate_upload("Photo.JPG", image_bytes("JPEG", size=(40, 30)), **LIMITS)
    png = validate_upload("scene.png", image_bytes("PNG", size=(20, 10)), **LIMITS)
    assert (jpeg.format, jpeg.width, jpeg.height, jpeg.content_type) == (
        "JPEG",
        40,
        30,
        "image/jpeg",
    )
    assert (png.format, png.width, png.height, png.content_type) == ("PNG", 20, 10, "image/png")
    assert validate_upload("x.jpeg", image_bytes("JPEG"), **LIMITS).format == "JPEG"


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("image.gif", image_bytes("GIF"), "Unsupported file type"),
        ("image.webp", image_bytes("WEBP"), "Unsupported file type"),
        ("noextension", image_bytes("PNG"), "Unsupported file type"),
        ("empty.png", b"", "empty"),
        ("renamed.jpg", image_bytes("PNG"), "not a JPEG image"),
        ("renamed.png", image_bytes("JPEG"), "not a PNG image"),
        ("notes.png", b"hello, this is text", "not a PNG image"),
        ("script.jpg", b"<script>alert(1)</script>", "not a JPEG image"),
        ("truncated.png", image_bytes("PNG")[:40], "not a valid image or is corrupted"),
        ("garbage.jpg", b"\xff\xd8\xff" + b"\x00" * 50, "not a valid image or is corrupted"),
    ],
)
def test_rejected_uploads(name, data, message):
    with pytest.raises(UploadError, match=message):
        validate_upload(name, data, **LIMITS)


def test_size_limit():
    data = image_bytes("PNG", size=(40, 30))
    with pytest.raises(UploadError, match="the limit is"):
        validate_upload("big.png", data, max_bytes=len(data) - 1, max_pixels=10_000)
    assert validate_upload("ok.png", data, max_bytes=len(data), max_pixels=10_000)


def huge_png_header(width: int, height: int) -> bytes:
    """A tiny PNG that declares huge dimensions: only the header is valid."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload)
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IEND", b"")


@pytest.mark.parametrize("size", [(200, 200), (30_000, 30_000), (100_000, 100_000)])
def test_pixel_limit_is_checked_from_the_header(size):
    # 100,000 x 100,000 would need ~30 GB to decode; the check must not decode anything.
    with pytest.raises(UploadError, match="pixels|too large"):
        validate_upload("bomb.png", huge_png_header(*size), **LIMITS)


def test_validation_does_not_decode_pixels(monkeypatch):
    png, jpeg = image_bytes("PNG"), image_bytes("JPEG")  # encoded before patching

    def fail(*args, **kwargs):
        raise AssertionError("pixel data must not be decoded during validation")

    monkeypatch.setattr(Image.Image, "load", fail)
    monkeypatch.setattr(ImageFile.ImageFile, "load", fail)
    assert validate_upload("ok.png", png, **LIMITS).width == 40
    assert validate_upload("ok.jpg", jpeg, **LIMITS).width == 40


def test_exif_or_cmyk_jpegs_are_accepted():
    buffer = io.BytesIO()
    Image.new("CMYK", (16, 16)).save(buffer, format="JPEG")
    assert validate_upload("cmyk.jpg", buffer.getvalue(), **LIMITS).format == "JPEG"
