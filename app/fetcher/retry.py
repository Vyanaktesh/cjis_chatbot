"""
Small, dependency-free retry-with-exponential-backoff helper.

Hand-rolled rather than pulling in a library: the whole point of this
module is to be auditable at a glance by whoever eventually operates this
on government infrastructure.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass
class RetryAttempt:
    attempt_number: int
    exception: Exception


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    on_attempt_failed: Callable[[RetryAttempt], None] | None = None,
) -> T:
    """
    Calls fn() up to max_attempts times. Sleeps base_delay_seconds *
    (backoff_factor ** attempt_index) between attempts. Re-raises the last
    exception if every attempt fails.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - intentionally broad, this is a generic retry wrapper
            last_exc = exc
            if on_attempt_failed:
                on_attempt_failed(RetryAttempt(attempt_number=attempt, exception=exc))
            if attempt < max_attempts:
                delay = base_delay_seconds * (backoff_factor ** (attempt - 1))
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc
