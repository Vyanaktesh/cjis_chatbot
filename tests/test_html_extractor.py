from app.extraction.blocks import BlockType
from app.extraction.html_extractor import extract_html


def texts(html: str) -> list[str]:
    return [b.text for b in extract_html(html.encode())]


def test_bare_text_in_div_with_br_is_kept():
    out = texts("<html><body><div>Passport fee is $100.<br>Processing takes 10 days.</div></body></html>")
    assert out == ["Passport fee is $100. Processing takes 10 days."]


def test_page_wrapped_in_a_form_is_not_dropped():
    out = texts("<body><form id='aspnetForm'><p>Passport fee is $100.</p><ul><li>Photo</li></ul></form></body>")
    assert out == ["Passport fee is $100.", "Photo"]


def test_form_controls_are_still_removed():
    out = texts("<body><form><p>Search the site</p><input value='query'><select><option>Georgia</option></select><button>Go</button></form></body>")
    assert out == ["Search the site"]


def test_definition_list_pairs_term_with_definition():
    out = texts("<body><dl><dt>Fee</dt><dd>$100</dd><dt>Time</dt><dd>10 days</dd></dl></body>")
    assert out == ["Fee: $100", "Time: 10 days"]


def test_text_directly_in_section_and_span():
    out = texts("<body><section>Apply online at the portal.</section><span>Bring original documents.</span></body>")
    assert out == ["Apply online at the portal.", "Bring original documents."]


def test_inline_markup_stays_in_one_paragraph():
    out = texts("<body><div>Pay <strong>$100</strong> at the <a href='#'>VFS office</a>.</div></body>")
    assert out == ["Pay $100 at the VFS office."]


def test_paragraph_text_is_not_emitted_twice():
    out = texts("<body><div><p>Fee is $100.</p></div></body>")
    assert out == ["Fee is $100."]


def test_headings_lists_and_tables_still_work():
    blocks = extract_html(
        b"<body><h1>Fees</h1><p>Intro.</p><ol><li>One</li><li>Two<ul><li>Nested</li></ul></li></ol>"
        b"<table><tr><td>Adult</td><td>$100</td></tr></table></body>"
    )
    kinds = [(b.type, b.text) for b in blocks]
    assert kinds == [
        (BlockType.HEADING, "Fees"),
        (BlockType.PARAGRAPH, "Intro."),
        (BlockType.LIST_ITEM, "One"),
        (BlockType.LIST_ITEM, "Two"),
        (BlockType.LIST_ITEM, "Nested"),
        (BlockType.TABLE_ROW, "Adult | $100"),
    ]


def test_short_loose_ui_labels_are_skipped_but_short_facts_are_kept():
    out = texts("<body><div>Menu</div><div>Search this website</div><div>Fee $100</div><div>Apply online.</div></body>")
    assert out == ["Fee $100", "Apply online."]


def test_short_text_inside_real_blocks_is_never_filtered():
    out = texts("<body><p>Menu</p><ul><li>Photo</li></ul><table><tr><td>Adult</td></tr></table></body>")
    assert out == ["Menu", "Photo", "Adult"]


def test_site_chrome_and_non_content_is_removed():
    out = texts(
        "<!DOCTYPE html><html><head><title>t</title><style>p{}</style></head><body>"
        "<nav>Home | About</nav><script>var x=1;</script><!-- hidden --><p>Real content.</p>"
        "<aside>Related links</aside><footer>Copyright</footer></body></html>"
    )
    assert out == ["Real content."]
