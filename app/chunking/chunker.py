"""
Section-aware chunker: splits on headings/logical blocks first, and only
falls back to a token cap when a section runs long — never splitting the
text of a single block (a paragraph, a list item, a table row) across two
chunks. That's what guarantees a checklist requirement can't be cut in
half: a LIST_ITEM block's text is always kept whole; if a whole section's
list is too long to fit in one chunk, the split happens BETWEEN list
items, never inside one.

Token counts are a rough character-based estimate (~4 chars/token for
English) — good enough to size chunks sensibly now. Phase 4 will load the
real BGE-M3 tokenizer for embedding; if that ever disagrees meaningfully
with this estimate, re-tune TOKEN_CHAR_RATIO then, not before.
"""

from dataclasses import dataclass, field

from app.extraction.blocks import Block, BlockType

TOKEN_CHAR_RATIO = 4
DEFAULT_MAX_TOKENS = 450


@dataclass
class Chunk:
    text: str
    chunk_index: int = 0
    heading_trail: list[str] = field(default_factory=list)
    block_types: list[str] = field(default_factory=list)
    page_numbers: list[int] = field(default_factory=list)
    used_ocr: bool = False


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // TOKEN_CHAR_RATIO)


def chunk_blocks(blocks: list[Block], max_tokens: int = DEFAULT_MAX_TOKENS) -> list[Chunk]:
    sections = _split_into_sections(blocks)
    chunks: list[Chunk] = []
    for heading_trail, content_blocks in sections:
        chunks.extend(_chunk_section(content_blocks, heading_trail, max_tokens))
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
    return chunks


def _split_into_sections(blocks: list[Block]) -> list[tuple[list[str], list[Block]]]:
    """Groups content blocks under the heading trail active at that point.
    heading_trail is e.g. ["Document Checklist", "Required Documents"] —
    outer heading first, current heading last."""
    sections: list[tuple[list[str], list[Block]]] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    current_content: list[Block] = []

    def flush():
        trail = [text for _, text in heading_stack]
        if current_content:
            sections.append((trail, list(current_content)))
        current_content.clear()

    for block in blocks:
        if block.type == BlockType.HEADING:
            flush()
            while heading_stack and heading_stack[-1][0] >= block.level:
                heading_stack.pop()
            heading_stack.append((block.level, block.text))
        else:
            current_content.append(block)
    flush()
    return sections


def _render_block(block: Block) -> str:
    if block.type == BlockType.LIST_ITEM and block.list_ordinal:
        return f"{block.list_ordinal} {block.text}"
    return block.text


def _chunk_section(blocks: list[Block], heading_trail: list[str], max_tokens: int) -> list[Chunk]:
    if not blocks:
        return []

    prefix = " > ".join(heading_trail) + "\n\n" if heading_trail else ""
    prefix_tokens = estimate_tokens(prefix) if prefix else 0

    chunks: list[Chunk] = []
    current: list[Block] = []
    current_tokens = prefix_tokens

    def finalize(content_blocks: list[Block]) -> Chunk:
        body = "\n".join(_render_block(b) for b in content_blocks)
        return Chunk(
            text=(prefix + body) if prefix else body,
            heading_trail=list(heading_trail),
            block_types=[b.type.value for b in content_blocks],
            page_numbers=sorted({b.page_number for b in content_blocks if b.page_number is not None}),
            used_ocr=any(b.source == "ocr" for b in content_blocks),
        )

    for block in blocks:
        block_tokens = estimate_tokens(_render_block(block))
        if current and current_tokens + block_tokens > max_tokens:
            chunks.append(finalize(current))
            current = []
            current_tokens = prefix_tokens
        current.append(block)
        current_tokens += block_tokens

    if current:
        chunks.append(finalize(current))

    return chunks
