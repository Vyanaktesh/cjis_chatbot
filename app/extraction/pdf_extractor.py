"""
PDF -> structured Block list.

PDFs have no real semantic markup, so structure is inferred:
  - Words are grouped into visual lines (by y-position), each line's
    average font size is compared against the document's most common
    ("body") font size to classify it as a heading (larger), a list item
    (leading bullet/number pattern), or plain paragraph text. Consecutive
    paragraph lines are merged until a heading/list item/blank-line gap
    appears, approximating real paragraph breaks.
  - A page with no (or almost no) extractable text is treated as a scanned
    image: it's rasterized (via pdf2image/poppler) and run through
    pytesseract OCR instead. OCR output can't be reliably classified into
    headings (no font metadata survives), so OCR lines all come back as
    plain paragraphs, tagged `source="ocr"` so this is visible downstream.
"""

import io
import re
from collections import Counter
from statistics import median

import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes

from app.core.logging_config import get_logger
from app.extraction.blocks import Block, BlockType

logger = get_logger(__name__)

MIN_WORDS_FOR_NATIVE_TEXT = 5  # fewer real words than this => treat page as scanned/image-only

LIST_MARKER_RE = re.compile(
    r"^(?P<marker>[•●■\-\*]|\d+[\.\)]|\([a-zA-Z0-9]+\)|[a-zA-Z][\.\)])\s+(?P<rest>.+)$"
)


def extract_pdf(pdf_bytes: bytes) -> list[Block]:
    blocks: list[Block] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        body_size = _estimate_body_font_size(pdf)
        ocr_page_numbers = []

        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1, y_tolerance=3, extra_attrs=["size"])
            if len(words) < MIN_WORDS_FOR_NATIVE_TEXT:
                ocr_page_numbers.append(page.page_number)
                continue
            lines = _group_words_into_lines(words)
            blocks.extend(_classify_lines(lines, body_size, page.page_number))

        for page_number in ocr_page_numbers:
            logger.info(
                "page has no extractable text, falling back to OCR",
                extra={"fields": {"page_number": page_number}},
            )
            blocks.extend(_ocr_page(pdf_bytes, page_number))

    return _merge_paragraph_runs(blocks)


def _estimate_body_font_size(pdf) -> float:
    sizes = []
    for page in pdf.pages:
        for word in page.extract_words(extra_attrs=["size"]):
            sizes.append(round(word["size"]))
    if not sizes:
        return 10.0
    return Counter(sizes).most_common(1)[0][0]


def _group_words_into_lines(words: list[dict], y_tolerance: float = 3.0) -> list[list[dict]]:
    lines: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        placed = False
        for line in lines:
            if abs(line[0]["top"] - word["top"]) <= y_tolerance:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])
    lines.sort(key=lambda line: line[0]["top"])
    for line in lines:
        line.sort(key=lambda w: w["x0"])
    return lines


def _classify_lines(lines: list[list[dict]], body_size: float, page_number: int) -> list[Block]:
    blocks = []
    for line in lines:
        text = " ".join(w["text"] for w in line).strip()
        if not text:
            continue
        avg_size = median(w["size"] for w in line)

        if avg_size >= body_size * 1.4:
            blocks.append(Block(type=BlockType.HEADING, text=text, level=1, page_number=page_number))
            continue
        if avg_size >= body_size * 1.2:
            blocks.append(Block(type=BlockType.HEADING, text=text, level=2, page_number=page_number))
            continue
        if avg_size >= body_size * 1.08:
            blocks.append(Block(type=BlockType.HEADING, text=text, level=3, page_number=page_number))
            continue

        match = LIST_MARKER_RE.match(text)
        if match:
            blocks.append(
                Block(
                    type=BlockType.LIST_ITEM,
                    text=match.group("rest"),
                    list_ordinal=match.group("marker"),
                    page_number=page_number,
                )
            )
            continue

        blocks.append(Block(type=BlockType.PARAGRAPH, text=text, page_number=page_number))
    return blocks


def _merge_paragraph_runs(blocks: list[Block]) -> list[Block]:
    """Consecutive PARAGRAPH blocks (same page, native text) get merged into
    one, since each was really just one wrapped visual line of the same
    paragraph, not a new paragraph."""
    merged: list[Block] = []
    for block in blocks:
        prev = merged[-1] if merged else None
        same_page_and_source = (
            prev is not None
            and prev.page_number == block.page_number
            and prev.source == block.source
        )

        if same_page_and_source and prev.type == BlockType.PARAGRAPH and block.type == BlockType.PARAGRAPH:
            # two visually-wrapped lines of the same paragraph
            prev.text = f"{prev.text} {block.text}"
        elif same_page_and_source and prev.type == BlockType.LIST_ITEM and block.type == BlockType.PARAGRAPH:
            # an unmarked continuation line immediately after a list item —
            # without this, the wrapped tail of a requirement becomes a
            # floating paragraph block that could land in a DIFFERENT chunk
            # than the requirement it belongs to, which is exactly the
            # "split mid-requirement" failure this chunker is meant to avoid.
            prev.text = f"{prev.text} {block.text}"
        else:
            merged.append(block)
    return merged


def _ocr_page(pdf_bytes: bytes, page_number: int) -> list[Block]:
    images = convert_from_bytes(pdf_bytes, first_page=page_number, last_page=page_number, dpi=300)
    if not images:
        return []
    text = pytesseract.image_to_string(images[0])
    blocks = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = LIST_MARKER_RE.match(line)
        if match:
            blocks.append(
                Block(
                    type=BlockType.LIST_ITEM,
                    text=match.group("rest"),
                    list_ordinal=match.group("marker"),
                    page_number=page_number,
                    source="ocr",
                )
            )
        else:
            blocks.append(
                Block(type=BlockType.PARAGRAPH, text=line, page_number=page_number, source="ocr")
            )
    return blocks
