"""Masking personal details before chat logs are stored (app/core/redact.py).
Examples are from real questions in the Oct 2026 test set, with made-up values."""
import pytest

from app.core.redact import find_personal_details, mask_personal_details


@pytest.mark.parametrize(
    "text, expected",
    [
        ('MY FILE NUMBER IS "CANT0D7B4N24" . IT HAS BEEN MORE THEN 12 WEEKS',
         'MY FILE NUMBER IS "[REFERENCE_NO]" . IT HAS BEEN MORE THEN 12 WEEKS'),
        ("My full name: Arivukkarasu Subramanian Could you please provide an update",
         "My full name: [NAME] Could you please provide an update"),
        ("my name is Priya Sharma and I need a PCC", "my name is [NAME] and I need a PCC"),
        ("I have Tourist / multiple Visa / VJ 4238612 / 09.09.2015 to 09.09.2025",
         "I have Tourist / multiple Visa / VJ [NUMBER] / 09.09.2015 to 09.09.2025"),
        ("Please call me at (404) 555-0182 or 404.555.0182", "Please call me at [PHONE] or [PHONE]"),
        ("my number is +91 98765 43210", "my number is [PHONE]"),
        ("write to ravi.k+oci@example.com", "write to [EMAIL]"),
        ("passport no Z1234567 expired", "passport no [PASSPORT_NO] expired"),
        ("my A-number is A123456789", "my A-number is [A_NUMBER]"),
        ("SSN 123-45-6789", "SSN [SSN]"),
        ("Fedex tracking ID is 7712 3456 7890", "Fedex tracking ID is [NUMBER]"),
    ],
)
def test_personal_details_are_masked(text, expected):
    assert mask_personal_details(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "MY i30 and i485 (adjustmentof status) both are in the process",
        "I am on H1B and my I-797A was approved in 2024",
        "So do I need to enclose Money order for $140 or $20?",
        "I am travelling to India from 4th december 2023 to 27th december 2023",
        "How long does a new OCI application take at the Atlanta consulate?",
        "Is the consulate open on Diwali?",
    ],
)
def test_ordinary_questions_are_left_alone(text):
    assert mask_personal_details(text) == text


def test_the_answer_loses_the_visitors_details_but_keeps_public_contacts():
    question = "My full name: Ravi Kumar, file CANT0D7B4N24. Call me on 404-555-0182."
    answer = (
        "- Ravi Kumar, I can't see the status of CANT0D7B4N24.\n"
        "- Call the VFS Global Atlanta helpline at +1 800 320 9693 or email oci.atlanta@mea.gov.in."
    )
    masked = mask_personal_details(answer, also=find_personal_details(question))
    assert "Ravi Kumar" not in masked and "CANT0D7B4N24" not in masked
    assert "+1 800 320 9693" in masked and "oci.atlanta@mea.gov.in" in masked
