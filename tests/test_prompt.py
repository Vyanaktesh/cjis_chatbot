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
