from app.chunking.chunker import (
    DEFAULT_MAX_TOKENS,
    HARD_MAX_TOKENS,
    TOKEN_CHAR_RATIO,
    chunk_blocks,
    estimate_tokens,
)
from app.extraction.blocks import Block, BlockType


def words(text: str) -> list[str]:
    return text.split()


def sentences(n: int) -> str:
    return " ".join(f"Applicants must submit document number {i} with the form." for i in range(n))


def test_a_normal_block_is_kept_whole():
    text = sentences(20)  # well under the hard limit
    chunks = chunk_blocks([Block(type=BlockType.PARAGRAPH, text=text)])
    assert [c.text for c in chunks] == [text]


def test_a_huge_block_is_split_into_bounded_chunks_without_losing_text():
    text = sentences(400)  # about 20,000 characters, like the real 22,585-character chunk
    assert estimate_tokens(text) > HARD_MAX_TOKENS

    chunks = chunk_blocks([Block(type=BlockType.PARAGRAPH, text=text)])

    assert len(chunks) > 1
    assert max(estimate_tokens(c.text) for c in chunks) <= DEFAULT_MAX_TOKENS
    assert words(" ".join(c.text for c in chunks)) == words(text)  # nothing dropped or duplicated
    assert all(c.text.rstrip().endswith(".") for c in chunks)  # cut at sentence ends


def test_one_giant_sentence_is_split_at_spaces_never_inside_a_word():
    text = " ".join(f"word{i}" for i in range(5000))  # no punctuation at all
    chunks = chunk_blocks([Block(type=BlockType.PARAGRAPH, text=text)])

    assert len(chunks) > 1
    assert max(len(c.text) for c in chunks) <= DEFAULT_MAX_TOKENS * TOKEN_CHAR_RATIO
    assert words(" ".join(c.text for c in chunks)) == words(text)


def test_the_list_marker_stays_on_the_first_piece_only():
    block = Block(type=BlockType.LIST_ITEM, text=sentences(400), list_ordinal="7.")
    chunks = chunk_blocks([block])
    assert len(chunks) > 1
    assert chunks[0].text.startswith("7. ")
    assert sum(c.text.count("7. ") for c in chunks) == 1


def test_headings_are_still_carried_into_each_piece():
    blocks = [
        Block(type=BlockType.HEADING, text="OCI fees", level=1),
        Block(type=BlockType.PARAGRAPH, text=sentences(400)),
    ]
    chunks = chunk_blocks(blocks)
    assert len(chunks) > 1
    assert all(c.heading_trail == ["OCI fees"] for c in chunks)
    assert all(c.text.startswith("OCI fees") for c in chunks)
