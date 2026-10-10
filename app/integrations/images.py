"""
Sanitizes images submitted through Citizen Corner before they leave this
server.

The browser-reported content type and filename are not trustworthy, and phone
photos routinely carry GPS coordinates and device details in their EXIF
metadata. Every upload is therefore opened and decoded as a real image,
re-encoded without any metadata, and given a generated filename.
"""

import io
import uuid

from PIL import Image, ImageOps

# Refuse absurdly large images (decompression bombs) before decoding them.
MAX_PIXELS = 40_000_000

_FORMATS = {
    "JPEG": ("image/jpeg", "jpg"),
    "PNG": ("image/png", "png"),
    "WEBP": ("image/webp", "webp"),
    # Re-saved as PNG (first frame only): GIF animation isn't worth the risk.
    "GIF": ("image/png", "png"),
}


class ImageRejected(ValueError):
    """The upload isn't a usable image."""


def sanitize_image(data: bytes) -> tuple[bytes, str, str]:
    """Returns (clean_bytes, content_type, safe_filename) or raises ImageRejected."""
    try:
        image = Image.open(io.BytesIO(data))
        fmt = image.format
        if fmt not in _FORMATS:
            raise ImageRejected(f"unsupported image format: {fmt!r}")
        width, height = image.size
        if width * height > MAX_PIXELS:
            raise ImageRejected("image dimensions are too large")
        image.load()  # fully decode now so a corrupt file fails here, not later
    except ImageRejected:
        raise
    except Exception as exc:  # noqa: BLE001 -- anything Pillow can't decode is "not an image"
        raise ImageRejected("file could not be read as an image") from exc

    # Apply the camera's rotation flag, then drop all metadata by re-encoding
    # from raw pixels into a fresh image.
    image = ImageOps.exif_transpose(image)
    content_type, ext = _FORMATS[fmt]
    out = io.BytesIO()
    if content_type == "image/jpeg":
        clean = Image.new("RGB", image.size)
        clean.paste(image.convert("RGB"))
        clean.save(out, format="JPEG", quality=90, optimize=True)
    elif content_type == "image/webp":
        clean = Image.new("RGBA", image.size)
        clean.paste(image.convert("RGBA"))
        clean.save(out, format="WEBP", quality=90)
    else:
        clean = Image.new("RGBA", image.size)
        clean.paste(image.convert("RGBA"))
        clean.save(out, format="PNG", optimize=True)
    return out.getvalue(), content_type, f"photo-{uuid.uuid4().hex[:12]}.{ext}"
