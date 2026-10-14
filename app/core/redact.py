"""
Masks personal details in visitor text before it is stored (see
app/db/chat_log_repo.py). People type their own case into a public chat box:
emails, phone numbers, passport and application numbers, sometimes their name
("My full name: ..."). Those are replaced with labels such as [EMAIL] or
[PASSPORT_NO], the style already used in the Oct 2026 test set.

Best-effort by nature: it catches the formats visitors actually use, not every
possible personal detail (a name typed without "my name is" stays). That is
why the logs are also kept for a limited time and store no IP address.
"""

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# US / international phone numbers: +1 (404) 555-1234, 404-555-1234, 404.555.1234,
# 4045551234, +91 98765 43210.
_PHONE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)"
    r"|(?<![\w+])\+91[\s.-]?\d{5}[\s.-]?\d{5}(?!\w)"
)
_A_NUMBER = re.compile(r"\bA[-\s]?\d{8,9}\b")  # US alien registration number
_PASSPORT_NO = re.compile(r"\b[A-Za-z]\d{7}\b")  # Indian passport: one letter, seven digits
# OCI file / application references and similar codes: letters and digits
# mixed, e.g. CANT0D7B4N24.
_TOKEN = re.compile(r"\b[A-Za-z0-9]{8,20}\b")
# Any other run of six or more digits (tracking numbers, US passport numbers).
_LONG_NUMBER = re.compile(r"(?<![\w.])\d(?:[\s-]?\d){5,}(?![\w.])")
_NAME = re.compile(
    r"\b(?i:my\s+(?:full\s+)?name\s+is|full\s+name\s*(?:is)?\s*[:\-]?)\s*"
    r"([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,3})"
)
# Capitalised words that start the next sentence, not part of a name.
_NOT_NAME = {
    "could", "can", "please", "i", "my", "and", "the", "is", "what", "how", "when", "where", "why",
    "do", "does", "will", "would", "should", "thank", "thanks", "kindly", "help", "need", "want",
    "passport", "oci", "visa", "application", "file", "date",
}


def _name_part(captured: str) -> str:
    words = []
    for word in captured.split():
        if word.lower() in _NOT_NAME:
            break
        words.append(word)
    return " ".join(words)


def _is_reference(token: str) -> bool:
    digits = sum(c.isdigit() for c in token)
    letters = sum(c.isalpha() for c in token)
    return digits >= 3 and letters >= 2


def find_personal_details(text: str) -> list[tuple[str, str]]:
    """(matched text, label) for each personal detail found in `text`."""
    found: list[tuple[str, str]] = []
    for pattern, label in (
        (_EMAIL, "[EMAIL]"),
        (_SSN, "[SSN]"),
        (_PHONE, "[PHONE]"),
        (_A_NUMBER, "[A_NUMBER]"),
        (_PASSPORT_NO, "[PASSPORT_NO]"),
    ):
        found += [(m.group(0), label) for m in pattern.finditer(text)]
    found += [(m.group(0), "[REFERENCE_NO]") for m in _TOKEN.finditer(text) if _is_reference(m.group(0))]
    found += [(m.group(0), "[NUMBER]") for m in _LONG_NUMBER.finditer(text)]
    found += [(name, "[NAME]") for m in _NAME.finditer(text) if (name := _name_part(m.group(1)))]
    return found


def mask_personal_details(text: str, *, also: list[tuple[str, str]] | None = None) -> str:
    """`text` with personal details replaced by labels. `also`: details found
    elsewhere (the question) to remove from this text too, so an answer that
    repeats the visitor's number or name is masked while the Consulate's own
    public phone numbers and emails in it are kept."""
    if not text:
        return text
    details = find_personal_details(text) if also is None else list(also)
    # One label per value, the first pattern's (a phone number is also a long
    # number), then longest first so a value inside a longer one isn't split.
    labels: dict[str, str] = {}
    for value, label in details:
        labels.setdefault(value, label)
    for value in sorted(labels, key=len, reverse=True):
        text = text.replace(value, labels[value])
    return text
