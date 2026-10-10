from app.fetcher.orchestrate import same_extracted_text

def write_previous(tmp_path, html: str) -> str:
    relative = "data/raw/source/v1/page.html"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(html.encode())
    return relative


def test_a_page_whose_text_is_unchanged_is_not_a_new_version(tmp_path):
    # The raw bytes differ (a script/tracking token changed, comments, extra whitespace) but the
    # text a visitor would read is identical, so this must not count as a new version.
    old = "<html><body><h1>OCI fees</h1><p>The fee is $275.</p><script>var t=1</script><!-- a --></body></html>"
    new = "<html><body><h1>OCI fees</h1>\n<p>The fee is $275.</p><script>var t=2</script><!-- b --></body></html>"
    previous = write_previous(tmp_path, old)
    assert old.encode() != new.encode()
    assert same_extracted_text("html", previous, new.encode(), base_dir=tmp_path) is True


def test_a_real_content_change_is_still_a_new_version(tmp_path):
    previous = write_previous(tmp_path, "<html><body><p>The fee is $275.</p></body></html>")
    new = b"<html><body><p>The fee is $300.</p></body></html>"
    assert same_extracted_text("html", previous, new, base_dir=tmp_path) is False


def test_pdfs_are_still_compared_by_their_bytes(tmp_path):
    previous = write_previous(tmp_path, "%PDF-1.4 same text")
    assert same_extracted_text("pdf", previous, b"%PDF-1.4 same text", base_dir=tmp_path) is False


def test_a_missing_previous_file_counts_as_changed(tmp_path):
    assert same_extracted_text("html", "data/raw/gone/v1/page.html", b"<p>x</p>", base_dir=tmp_path) is False
