#!/usr/bin/env python3
"""
None of the 47 real sources are scanned/image-only PDFs, so the OCR
fallback path in app/extraction/pdf_extractor.py has no real-world example
to prove it against. This script builds a synthetic image-only PDF (text
rendered onto a blank image via PIL, saved as a PDF with NO real text
layer — a faithful stand-in for a scanned document) and runs it through
the real extractor, to prove the OCR path actually triggers and actually
recovers the text, rather than leaving it as untested code.

Usage:
    python scripts/test_ocr_fallback.py
"""

import io
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.extraction.pdf_extractor import extract_pdf  # noqa: E402

SAMPLE_TEXT = [
    "Document Checklist",
    "",
    "1. Copy of current valid passport",
    "2. Two passport-size photographs",
    "3. Proof of address (utility bill or lease)",
]


def build_synthetic_scanned_pdf() -> bytes:
    img = Image.new("RGB", (1200, 800), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()

    y = 60
    for line in SAMPLE_TEXT:
        draw.text((60, y), line, fill="black", font=font)
        y += 50

    buf = io.BytesIO()
    img.save(buf, "PDF")  # PIL's PDF writer embeds the image with NO text layer
    return buf.getvalue()


def main() -> int:
    pdf_bytes = build_synthetic_scanned_pdf()
    print(f"Synthetic scanned PDF: {len(pdf_bytes)} bytes, 1 page, no text layer (image only)\n")

    blocks = extract_pdf(pdf_bytes)

    print(f"Extracted {len(blocks)} block(s):\n")
    for b in blocks:
        print(f"  [{b.type.value:<10} ocr={b.source == 'ocr'}] {b.text!r}")

    ocr_blocks = [b for b in blocks if b.source == "ocr"]
    recovered_text = " ".join(b.text for b in blocks).lower()

    checks = {
        "OCR path was used (source='ocr' on at least one block)": len(ocr_blocks) > 0,
        "'Document Checklist' text recovered": "document checklist" in recovered_text,
        "'passport' text recovered": "passport" in recovered_text,
        "at least one list item recovered": any(b.type.value == "list_item" for b in blocks),
    }

    print("\n=== Checks ===")
    all_pass = True
    for label, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        all_pass = all_pass and passed

    print("\nALL OK" if all_pass else "\nOCR FALLBACK NOT WORKING AS EXPECTED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
