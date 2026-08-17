"""
HTML -> structured Block list.

Strips nav/footer/script/style/etc. (site chrome, not content), then walks
the remaining tree in document order, preserving heading hierarchy
(h1-h6), lists (nested lists handled by detaching a <li>'s nested
<ul>/<ol> before reading its own text, then recursing into it separately
so nested-list text is never duplicated into the parent item), and basic
tables (one Block per row, cells joined with " | ").
"""

from bs4 import BeautifulSoup, Comment

from app.extraction.blocks import Block, BlockType

# Site chrome / non-content tags, removed before extraction. `aside` is
# included because on government/vendor sites it's reliably used for
# "related links" / sidebar widgets rather than primary content.
STRIP_TAGS = ["script", "style", "noscript", "svg", "iframe", "nav", "footer", "aside", "form", "button"]

HEADING_TAGS = {f"h{i}": i for i in range(1, 7)}


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


def _walk(node, blocks: list[Block]) -> None:
    for child in getattr(node, "children", []):
        name = getattr(child, "name", None)
        if name is None:
            continue  # NavigableString — text is picked up via the parent tag's get_text()

        if name in HEADING_TAGS:
            text = child.get_text(" ", strip=True)
            if text:
                blocks.append(Block(type=BlockType.HEADING, text=text, level=HEADING_TAGS[name]))
            # headings aren't expected to contain nested block content; don't recurse

        elif name == "p":
            text = child.get_text(" ", strip=True)
            if text:
                blocks.append(Block(type=BlockType.PARAGRAPH, text=text))
            _walk(child, blocks)  # handles a (rare/invalid) nested list inside a <p>

        elif name in ("ul", "ol"):
            _walk_list(child, blocks, ordered=(name == "ol"))

        elif name == "table":
            _walk_table(child, blocks)

        else:
            # div, section, span, article, header, body, etc. — just a
            # container; recurse to find real content inside it.
            _walk(child, blocks)


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
