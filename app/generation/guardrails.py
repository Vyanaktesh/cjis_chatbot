"""
Input-side guardrails, applied before retrieval or any model call.

Two deterministic screens that must not depend on the model doing the right
thing (they run first, every time):

1. Prompt-injection / jailbreak attempts -> a polite, on-scope redirect. The
   patterns are deliberately narrow so genuine consular questions are never
   caught. The system prompt (app/generation/prompt.py) is the second layer
   that handles anything these miss.
2. Self-harm / crisis messages -> a compassionate safety response with real
   resources, never routed through the RAG pipeline.

General rudeness/abuse is NOT pattern-matched here (too easy to false-positive
on words like "damn form"); the system prompt instructs the model to stay
professional and not engage with it. On-topic scoping is handled by the
relevance guardrail already in service.py.

screen_query returns (reply_text, kind) when a message should be answered
directly without retrieval/generation, or None to let it flow through normally.
"""

import re
from typing import Optional

# Clear attempts to override instructions, change persona, or exfiltrate the
# system prompt. Narrow by design -- see module docstring.
_INJECTION_PATTERNS = [
    r"ignore (?:all |any |the |your )*(?:previous|prior|above|earlier)[\w ]*?(?:instruction|rule|prompt)",
    r"disregard (?:all |any |the |your )*(?:previous |prior |above )*(?:instruction|rule|system prompt|guardrail)",
    r"forget (?:all |your |the )*(?:previous |above )*(?:instruction|rule)",
    r"(?:reveal|show|print|repeat|give me|tell me|what(?:'s| is)) (?:me )?(?:your|the) (?:system )?(?:prompt|instructions)",
    r"(?:your|the) system prompt",
    r"\bjailbreak\b",
    r"\bdeveloper mode\b",
    r"\bdan mode\b",
    r"\bdo anything now\b",
    r"pretend (?:you are|to be|that you)",
    r"act as (?:a |an )?(?:dan\b|unrestricted|jailbroken|different ai|evil)",
    r"bypass (?:your |the |all )*(?:rule|restriction|filter|guardrail|safety)",
    r"(?:without|no) (?:any )?(?:restriction|filter|guardrail)s",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

# High-precision self-harm / crisis indicators. Kept specific so ordinary
# phrases ("I'm dying to travel", "this form is killing me") don't trigger it.
_CRISIS_PATTERNS = [
    r"\bkill (?:myself|me)\b",
    r"\b(?:commit )?suicide\b",
    r"\b(?:want|going|plan) (?:to )?(?:die|end my life)\b",
    r"\bend(?:ing)? my life\b",
    r"\b(?:harm|hurt|cut) (?:myself|me)\b",
    r"\bno reason to live\b",
]
_CRISIS_RE = re.compile("|".join(_CRISIS_PATTERNS), re.IGNORECASE)

INJECTION_REPLY = (
    "I can only help with Indian consular services at the Consulate General of "
    "India, Atlanta, such as passport, OCI, and visa questions. How can I help "
    "you with those?"
)

CRISIS_REPLY = (
    "I'm really sorry you're going through this, and I want you to be safe. "
    "I'm only able to help with consular services, so I can't support you with "
    "this myself, but please reach out to someone who can right now. If you are "
    "in immediate danger, call 911. You can also call or text 988 (the Suicide "
    "and Crisis Lifeline in the US) any time to talk to someone. You are not "
    "alone, and help is available."
)


def screen_query(query: str) -> Optional[tuple[str, str]]:
    """Returns (reply, kind) if this message should be answered directly by a
    guardrail instead of the RAG pipeline, else None. `kind` is one of
    "crisis" or "injection" (used for logging and to tell the widget not to
    offer human escalation)."""
    text = query or ""
    # Crisis takes priority over everything else.
    if _CRISIS_RE.search(text):
        return CRISIS_REPLY, "crisis"
    if _INJECTION_RE.search(text):
        return INJECTION_REPLY, "injection"
    return None
