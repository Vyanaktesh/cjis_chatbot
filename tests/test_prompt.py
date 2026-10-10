import pytest

from app.generation.prompt import strip_thinking


@pytest.mark.parametrize(
    "raw, expected",
    [
        # ranges must keep their meaning (fees, timelines, office hours)
        ("Open Monday – Friday, 9 am – 5 pm [1].", "Open Monday to Friday, 9 am to 5 pm [1]."),
        ("Processing takes 10 – 15 working days [2].", "Processing takes 10 to 15 working days [2]."),
        ("The fee is $25 – $50 depending on type [1].", "The fee is $25 to $50 depending on type [1]."),
        ("Closed June – August.", "Closed June to August."),
        ("Validity is 5–10 years [1].", "Validity is 5-10 years [1]."),
        # clause breaks become commas
        ("Not valid for minors — an adult must apply [3].", "Not valid for minors, an adult must apply [3]."),
        ("It is free—no fee applies.", "It is free, no fee applies."),
        # a digit on only ONE side is a clause break, not a range
        ("Complete step 2 – submit the form.", "Complete step 2, submit the form."),
        # nothing to change
        ("Plain text with a hyphen-ated word.", "Plain text with a hyphen-ated word."),
    ],
)
def test_dashes_never_change_the_meaning_of_ranges(raw, expected):
    assert strip_thinking(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        # the exact shape that showed up in a real OCI answer
        (
            "**General Application Steps:**\n1.  **Online Application:** Complete the form [1].\n2.  **Physical Application:** Submit it [2].",
            "General Application Steps:\n1. Online Application: Complete the form [1].\n2. Physical Application: Submit it [2].",
        ),
        ("* Item one\n* Item two", "- Item one\n- Item two"),
        # indented sub-bullets under a numbered, bold step (shape of a real answer)
        (
            "3.  **Photograph and Signature:**\n    *   Upload a digital photo [1].\n    *   The photos must be identical [1].",
            "3. Photograph and Signature:\n    - Upload a digital photo [1].\n    - The photos must be identical [1].",
        ),
        ("* **Fee:** $100 [1]", "- Fee: $100 [1]"),
        ("## Steps\nApply online.", "Steps\nApply online."),
        ("Bring the *original* documents.", "Bring the original documents."),
        ("Apply at `ociservices.gov.in` today.", "Apply at ociservices.gov.in today."),
        ("See [the OCI portal](https://ociservices.gov.in) for details.", "See the OCI portal (https://ociservices.gov.in) for details."),
        # markdown cleanup and range handling work together
        ("**Hours:** Monday \u2013 Friday", "Hours: Monday to Friday"),
    ],
)
def test_markdown_symbols_are_removed_from_answers(raw, expected):
    assert strip_thinking(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "The fee is $100 [1][2].",
        "Compute 5 * 3 to get 15.",
        "Call 1-800-555-0100 * ext 5 for help.",
        "A hyphen-ated word and a plain list:\n- one\n- two",
        "1. First step\n2. Second step",
    ],
)
def test_ordinary_text_is_left_alone(text):
    assert strip_thinking(text) == text
