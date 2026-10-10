"""
HTML -> structured Block list.

Strips nav/footer/script/style/etc. (site chrome, not content), then walks
the remaining tree in document order, preserving heading hierarchy
(h1-h6), lists (nested lists handled by detaching a <li>'s nested
<ul>/<ol> before reading its own text, then recursing into it separately
so nested-list text is never duplicated into the parent item), basic
tables (one Block per row, cells joined with " | "), and definition lists
(one Block per term/definition pair).

Any other text -- bare text sitting directly inside a <div>/<section>/<td>
alongside inline tags like <span>/<a>/<br> -- is collected into paragraph
blocks too. Real consular pages frequently put their content there rather
than in <p> tags, and dropping it silently meant a source could "extract"
fine yet contribute nothing to the chatbot.
"""

from bs4 import BeautifulSoup, Comment, NavigableString

from app.extraction.blocks import Block, BlockType

# Site chrome / non-content tags, removed before extraction. `aside` is
# included because on government/vendor sites it's reliably used for
# "related links" / sidebar widgets rather than primary content. `form`
# itself is deliberately NOT stripped: ASP.NET-style sites wrap the whole
# page in one <form>, so removing it removed every word of content. Only the
# interactive controls inside forms are dropped.
STRIP_TAGS = [
    "script", "style", "noscript", "svg", "iframe", "nav", "footer", "aside",
    "button", "input", "select", "textarea", "option",
]

HEADING_TAGS = {f"h{i}": i for i in range(1, 7)}

# Tags whose text flows inline with the text around them (as opposed to
# starting a new block).
INLINE_TAGS = {
    "a", "abbr", "b", "br", "cite", "code", "em", "font", "i", "img", "label",
    "mark", "q", "small", "span", "strong", "sub", "sup", "time", "u", "wbr",
}


_SENTENCE_MARKS = set(".:;$")


def _looks_like_ui_label(text: str) -> bool:
    """Bare text that sits loose in a container is often just site furniture
    ("Menu", "Search", "Search this website"). Real content in that position
    is either a few words long or carries a digit or sentence punctuation
    ("Fee $100", "Apply online.") -- so very short fragments with neither are
    skipped. Text inside <p>/<li>/headings/tables is never filtered."""
    if len(text.split()) >= 4:
        return False
    return not any(c.isdigit() or c in _SENTENCE_MARKS for c in text)


def extract_html(html_bytes: bytes) -> list[Block]:
    soup = BeautifulSoup(html_bytes, "lxml")

    for tag_name in STRIP_TAGS:
        for el in soup.find_all(tag_name):
            el.decompose()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    root = soup.body or soup
    blocks: list[Block] = []
    _walk(root, blocks)
    return blocks


def _walk(node, blocks: list[Block], emit_inline: bool = True) -> None:
    """`emit_inline=False` is used inside a <p>, whose own text was already
    emitted as one block -- only nested lists/tables are picked up there, so
    the paragraph's text isn't emitted a second time."""
    pending: list[str] = []

    def flush() -> None:
        # Fragments keep their original spacing (like a browser renders them),
        # so "<b>$100</b>." doesn't become "$100 ."; whitespace is normalized
        # once here.
        text = " ".join("".join(pending).split())
        pending.clear()
        if text and emit_inline and not _looks_like_ui_label(text):
            blocks.append(Block(type=BlockType.PARAGRAPH, text=text))

    for child in getattr(node, "children", []):
        if isinstance(child, NavigableString):
            if type(child) is NavigableString:  # skip Doctype/CData/etc.
                pending.append(str(child))
            continue

        name = getattr(child, "name", None)
        if name is None:
            continue

        if name in INLINE_TAGS:
            pending.append(" " if name == "br" else child.get_text())
            continue

        flush()

        if name in HEADING_TAGS:
            text = child.get_text(" ", strip=True)
            if text:
                blocks.append(Block(type=BlockType.HEADING, text=text, level=HEADING_TAGS[name]))
            # headings aren't expected to contain nested block content; don't recurse

        elif name == "p":
            text = child.get_text(" ", strip=True)
            if text:
                blocks.append(Block(type=BlockType.PARAGRAPH, text=text))
            _walk(child, blocks, emit_inline=False)  # handles a (rare/invalid) nested list inside a <p>

        elif name in ("ul", "ol"):
            _walk_list(child, blocks, ordered=(name == "ol"))

        elif name == "table":
            _walk_table(child, blocks)

        elif name == "dl":
            _walk_definition_list(child, blocks)

        elif name in ("pre", "blockquote"):
            text = child.get_text(" ", strip=True)
            if text:
                blocks.append(Block(type=BlockType.PARAGRAPH, text=text))

        else:
            # div, section, span, article, header, body, form, etc. -- just a
            # container; recurse to find real content inside it.
            _walk(child, blocks, emit_inline)

    flush()


def _walk_list(list_tag, blocks: list[Block], ordered: bool) -> None:
    idx = 0
    for li in list_tag.find_all("li", recursive=False):
        idx += 1
        nested_lists = li.find_all(["ul", "ol"], recursive=False)
        for nested in nested_lists:
            nested.extract()  # detach so li.get_text() below doesn't include its text twice

        text = li.get_text(" ", strip=True)
        marker = f"{idx}." if ordered else "•"
        if text:
            blocks.append(Block(type=BlockType.LIST_ITEM, text=text, list_ordinal=marker))

        for nested in nested_lists:
            _walk_list(nested, blocks, ordered=(nested.name == "ol"))


def _walk_table(table_tag, blocks: list[Block]) -> None:
    for tr in table_tag.find_all("tr"):
        cells = tr.find_all(["td", "th"], recursive=False)
        texts = [c.get_text(" ", strip=True) for c in cells]
        row_text = " | ".join(t for t in texts if t)
        if row_text:
            blocks.append(Block(type=BlockType.TABLE_ROW, text=row_text))


def _walk_definition_list(dl_tag, blocks: list[Block]) -> None:
    """One paragraph per term/definition pair ("Fee: $100"); a term with
    several definitions yields one paragraph per definition."""
    term = ""
    for el in dl_tag.find_all(["dt", "dd"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name == "dt":
            term = text
        else:
            blocks.append(
                Block(type=BlockType.PARAGRAPH, text=f"{term}: {text}" if term else text)
            )
