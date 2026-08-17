"""
Shared structured-text model produced by both extractors (HTML and PDF).
The chunker (app/chunking/chunker.py) operates on this, not on raw
markup/PDF bytes, so it doesn't need to know anything about where a block
came from.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BlockType(str, Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_ROW = "table_row"


@dataclass
class Block:
    type: BlockType
    text: str
    level: int = 0  # heading level 1-6; unused for other block types
    list_ordinal: Optional[str] = None  # e.g. "1.", "•", "a)" — preserved marker
    page_number: Optional[int] = None  # PDFs only; None for HTML
    source: str = "native"  # "native" (real text layer) or "ocr" (scanned fallback)
