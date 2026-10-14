import pytest

from app.core.config import get_settings


@pytest.fixture(autouse=True)
def no_chat_logging(monkeypatch):
    """The tests run inside the backend container, next to a real database:
    without this, every test that calls /chat would write a row to chat_logs.
    tests/test_chat_logging.py switches it back on with a fake database."""
    monkeypatch.setattr(get_settings(), "chat_log_enabled", False)
