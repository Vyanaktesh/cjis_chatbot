"""Pydantic request bodies for the Phase 5 admin review API."""

from typing import Optional

from pydantic import BaseModel


class ReviewDecisionRequest(BaseModel):
    actor: str
    reason: Optional[str] = None
