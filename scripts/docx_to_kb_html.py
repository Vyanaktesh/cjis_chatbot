#!/usr/bin/env python3
"""
Turn a Word document supplied by the Consulate into a plain HTML page under
frontend/public/kb/, so it can be both

  * served by nginx at https://<site>/kb/... (the source chips in the chat
    link there, so a citizen can open what the answer was based on), and
  * ingested by scripts/ingest_kb_pages.py through the normal
    extract -> chunk -> embed pipeline, exactly like a fetched web page.

Word structure is mapped to HTML the HTML extractor already understands:
Heading 1/2/3 -> h1/h2/h3, numbered or bulleted paragraphs -> <ul><li>, and
FAQ-style "Q. ..." paragraphs -> <h2> so each answer is chunked under its
question. A question that appears twice in the document (copy-paste in the
source file) is kept only once. Everything else becomes <p>. No styling or
images are carried over -- only the text matters for the chatbot.

Usage:
    python scripts/docx_to_kb_html.py "OCI FAQs.docx" frontend/public/kb/oci/oci-faqs.html
"""

import argparse
import html
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
URL_RE = re.compile(r"https?://[^\s<>()]+[^\s<>().,;:]")

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin: 0; background: #f6f7f9; color: #1c2430; font: 16px/1.6 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width: 760px; margin: 0 auto; padding: 32px 16px 48px; }}
  h1 {{ font-size: 1.7rem; line-height: 1.25; margin: 0 0 1rem; }}
  h2 {{ font-size: 1.2rem; margin: 2rem 0 .5rem; }}
  h3 {{ font-size: 1.05rem; margin: 1.5rem 0 .4rem; }}
  a {{ color: #1f4f8f; overflow-wrap: anywhere; }}
  footer {{ max-width: 760px; margin: 0 auto; padding: 16px; border-top: 1px solid #d5dbe3; color: #5a6676; font-size: .875rem; }}
</style>
</head>
<body>
<main>
{body}
</main>
<footer>{footer}</footer>
</body>
</html>
"""

FOOTER = (
    "Information provided by the Consulate General of India, Atlanta, for the DOST assistant. "
    'Official OCI services: <a href="https://ociservices.gov.in/onlineOCI/">ociservices.gov.in</a>.'
)


def _inline(text: str) -> str:
    """Escape text and turn bare URLs into links."""
    out, last = [], 0
    for m in URL_RE.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        url = m.group(0)
        out.append(f'<a href="{html.escape(url)}">{html.escape(url)}</a>')
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


def read_paragraphs(docx_path: Path) -> list[tuple[str, str]]:
    """(kind, text) per non-empty paragraph; kind is h1/h2/h3/li/p."""
    root = ET.fromstring(zipfile.ZipFile(docx_path).read("word/document.xml"))
    items = []
    for p in root.iter(W + "p"):
        text = " ".join("".join(t.text or "" for t in p.iter(W + "t")).split())
        if not text:
            continue
        ppr = p.find(W + "pPr")
        style_el = ppr.find(W + "pStyle") if ppr is not None else None
        style = style_el.get(W + "val") if style_el is not None else ""
        if style in ("Heading1", "Title"):
            kind = "h1"
        elif style == "Heading2":
            kind = "h2"
        elif style in ("Heading3", "Heading4"):
            kind = "h3"
        elif ppr is not None and ppr.find(W + "numPr") is not None:
            kind = "li"
        elif re.match(r"^Q[.:]\s", text):
            kind, text = "h2", re.sub(r"^Q[.:]\s*", "", text)
        else:
            kind, text = "p", re.sub(r"^A[.:]\s*", "", text)
        items.append((kind, text))
    return items


def drop_repeated_questions(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Remove a heading section whose heading text already appeared earlier."""
    seen, out, skipping = set(), [], False
    for kind, text in items:
        if kind in ("h2", "h3"):
            key = text.casefold()
            skipping = key in seen
            seen.add(key)
        if not skipping:
            out.append((kind, text))
    return out


def render(items: list[tuple[str, str]]) -> tuple[str, str]:
    title = next((t for k, t in items if k == "h1"), "Consulate information")
    lines, in_list = [], False
    for kind, text in items:
        if kind == "li" and not in_list:
            lines.append("<ul>")
            in_list = True
        elif kind != "li" and in_list:
            lines.append("</ul>")
            in_list = False
        lines.append(f"<{kind}>{_inline(text)}</{kind}>" if kind != "li" else f"  <li>{_inline(text)}</li>")
    if in_list:
        lines.append("</ul>")
    return title, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("docx", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    items = drop_repeated_questions(read_paragraphs(args.docx))
    title, body = render(items)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(PAGE.format(title=html.escape(title), body=body, footer=FOOTER), encoding="utf-8")
    print(f"{args.out}: {len(items)} blocks, title {title!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
