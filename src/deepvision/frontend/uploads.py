"""Validation of uploaded files before they are previewed or sent to the API.

Checks, in order: extension (JPG, JPEG, PNG), non-empty, size limit, file signature
(magic bytes must match the extension, so a renamed file is rejected), then the image
header: Pillow reads only the header to get the format and dimensions, and the pixel count
is checked before anything is decoded, which rules out decompression bombs in the preview.
The API repeats its own checks; this layer keeps bad files out of the UI and the network.
"""

import io
import warnings
from dataclasses import dataclass
from pathlib import PurePath

from PIL import Image, UnidentifiedImageError

FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
SIGNATURES = {"JPEG": b"\xff\xd8\xff", "PNG": b"\x89PNG\r\n\x1a\n"}
CONTENT_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png"}
ACCEPTED_TYPES = ("jpg", "jpeg", "png")  # for st.file_uploader


class UploadError(ValueError):
    """A rejected upload; the message is shown to the user."""


@dataclass(frozen=True)
class ValidatedImage:
    name: str
    data: bytes
    format: str
    width: int
    height: int

    @property
    def content_type(self) -> str:
        return CONTENT_TYPES[self.format]


def _megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):.1f} MB"


def validate_upload(name: str, data: bytes, *, max_bytes: int, max_pixels: int) -> ValidatedImage:
    extension = PurePath(name).suffix.lower()
    expected = FORMATS.get(extension)
    if expected is None:
        raise UploadError("Unsupported file type. Upload a JPG, JPEG, or PNG image.")
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > max_bytes:
        raise UploadError(
            f"The file is {_megabytes(len(data))}; the limit is {_megabytes(max_bytes)}."
        )
    if not data.startswith(SIGNATURES[expected]):
        raise UploadError(
            f"The file content is not a {expected} image (it does not match {extension})."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:  # reads the header only
                width, height, actual = image.width, image.height, image.format
                if actual != expected:
                    raise UploadError(f"The file content is not a {expected} image.")
                if width * height > max_pixels:
                    raise UploadError(
                        f"The image has {width * height:,} pixels; the limit is {max_pixels:,}."
                    )
                image.verify()  # structural check without decoding the pixel data
    except UploadError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise UploadError("The image dimensions are too large.") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise UploadError("The file is not a valid image or is corrupted.") from None
    return ValidatedImage(name, data, expected, width, height)
