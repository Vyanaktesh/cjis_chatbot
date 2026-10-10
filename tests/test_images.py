import io
import re

import pytest
from PIL import Image

from app.integrations.images import MAX_PIXELS, ImageRejected, sanitize_image


def encode(img: Image.Image, fmt: str, **kwargs) -> bytes:
    out = io.BytesIO()
    img.save(out, format=fmt, **kwargs)
    return out.getvalue()


def jpeg_with_metadata(orientation: int | None = None) -> bytes:
    exif = Image.Exif()
    exif[0x010F] = "SecretPhoneMaker"  # camera make
    exif[0x0131] = "PrivateEditingApp"  # software
    if orientation:
        exif[0x0112] = orientation
    return encode(Image.new("RGB", (40, 20), "red"), "JPEG", exif=exif)


def test_camera_metadata_is_removed_from_jpegs():
    original = jpeg_with_metadata()
    assert b"SecretPhoneMaker" in original  # the test photo really carries metadata

    clean, content_type, filename = sanitize_image(original)

    assert content_type == "image/jpeg"
    assert b"SecretPhoneMaker" not in clean and b"PrivateEditingApp" not in clean
    assert len(Image.open(io.BytesIO(clean)).getexif()) == 0
    assert re.fullmatch(r"photo-[0-9a-f]{12}\.jpg", filename)


def test_a_rotated_phone_photo_is_turned_upright_before_the_flag_is_dropped():
    clean, _, _ = sanitize_image(jpeg_with_metadata(orientation=6))  # 6 = rotate 90 degrees
    assert Image.open(io.BytesIO(clean)).size == (20, 40)


def test_png_text_chunks_are_removed():
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("Author", "Someone Private")
    original = encode(Image.new("RGB", (10, 10), "blue"), "PNG", pnginfo=info)
    assert b"Someone Private" in original

    clean, content_type, filename = sanitize_image(original)
    assert content_type == "image/png" and filename.endswith(".png")
    assert b"Someone Private" not in clean


def test_gifs_are_converted_to_png():
    clean, content_type, filename = sanitize_image(encode(Image.new("P", (8, 8)), "GIF"))
    assert content_type == "image/png" and filename.endswith(".png")
    assert Image.open(io.BytesIO(clean)).format == "PNG"


def test_webp_is_kept_as_webp():
    clean, content_type, filename = sanitize_image(encode(Image.new("RGB", (8, 8), "green"), "WEBP"))
    assert content_type == "image/webp" and filename.endswith(".webp")


@pytest.mark.parametrize(
    "data",
    [
        b"<script>alert(1)</script>",  # not an image at all, whatever the browser claimed
        b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>",
        b"%PDF-1.7 fake pdf",
        b"",
    ],
)
def test_files_that_are_not_images_are_rejected(data):
    with pytest.raises(ImageRejected):
        sanitize_image(data)


def test_a_truncated_image_is_rejected():
    with pytest.raises(ImageRejected):
        sanitize_image(jpeg_with_metadata()[:60])


def test_an_enormous_image_is_rejected_without_decoding_it():
    side = int(MAX_PIXELS**0.5) + 100
    huge = encode(Image.new("1", (side, side)), "PNG")  # compresses to a tiny file
    assert len(huge) < 1_000_000
    with pytest.raises(ImageRejected):
        sanitize_image(huge)
