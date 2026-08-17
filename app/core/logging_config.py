"""
Structured (JSON-lines) logging setup shared across the project.

Every fetch attempt also gets a durable, queryable row in Postgres'
audit_log table (see app/db/audit_repo.py) — this stdout logger is for
real-time visibility while a script is running, not the system of record.
"""

import json
import logging
import sys
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # allow callers to pass structured fields via `extra={"fields": {...}}`
        if hasattr(record, "fields"):
            payload.update(record.fields)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    if root.handlers:
        # already configured (e.g. re-imported in the same process) — no-op
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
