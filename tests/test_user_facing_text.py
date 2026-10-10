"""Text users read must be clean: no em/en dashes and no markdown symbols.
Model answers are cleaned in app/generation/prompt.py (see test_prompt.py);
this guards the fixed messages the backend sends, so they can't regress."""
import re

import pytest

from app.generation import service

DASHES = ("—", "–")
MARKDOWN = re.compile(r"\*\*|__|^\s*[*+]\s|^\s*#{1,6}\s|`", re.MULTILINE)

FIXED_MESSAGES = {
    "GENERATION_UNAVAILABLE_ANSWER": service.GENERATION_UNAVAILABLE_ANSWER,
    "NO_CONTEXT_ANSWER": service.NO_CONTEXT_ANSWER,
    "OUT_OF_SCOPE_ANSWER": service.OUT_OF_SCOPE_ANSWER,
}


@pytest.mark.parametrize("name", FIXED_MESSAGES)
def test_fixed_backend_messages_are_clean(name):
    text = FIXED_MESSAGES[name]
    assert not any(d in text for d in DASHES), f"{name} contains an em/en dash"
    assert not MARKDOWN.search(text), f"{name} contains markdown symbols"
