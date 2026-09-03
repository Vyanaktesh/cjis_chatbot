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
        # allow callers to pass structured fields via `extra={"fields": {...}}`,
        # but never let them clobber the reserved keys above (which would
        # corrupt the schema for downstream log parsers)
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                if key not in payload:
                    payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


# Module-level flag so this is idempotent on OUR handler specifically. The
# old `if root.handlers: return` guard silently did nothing whenever another
# library (e.g. uvicorn) had already attached a root handler, so the
# JsonFormatter never applied and app logs came out unstructured in exactly
# the production setup that matters.
_configured = False


def _coerce_level(level: str) -> int:
    resolved = logging.getLevelName(str(level).upper())
    return resolved if isinstance(resolved, int) else logging.INFO


def configure_logging(level: str = "INFO") -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(_coerce_level(level))
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
