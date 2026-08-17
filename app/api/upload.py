"""
Phase 5: manual document upload, through the exact same
extract -> chunk -> embed -> index pipeline the bulk fetcher uses
(`app/ingestion/pipeline.embed_and_index_version`) — for content an admin
has on hand (a PDF checklist emailed by the office, a form with no stable
public URL, a correction to something the live fetch got wrong) rather
than something the automated fetcher can reach on its own.

Scoped to PDF only for this phase, matching the brief's "manual PDF
upload endpoint through same pipeline" — other formats can follow later
if actually needed.
"""

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.db.audit_repo import log_event
from app.db.connection import get_conn
from app.db.source_versions_repo import create_version, get_latest_version
from app.db.sources_repo import get_by_url, upsert_source
from app.embedding.bge_m3 import BgeM3Embedder
from app.ingestion.pipeline import embed_and_index_version
from app.vectorstore.qdrant_store import get_qdrant_client

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_STORAGE_DIR = REPO_ROOT / "data" / "raw"


def _safe_filename(name: str) -> str:
    name = Path(name).name  # strip any directory components
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "upload.pdf"


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    url: str = Form(
        ...,
        description="Canonical identifier for this content: a real URL if it has one, "
        "or a manual:// identifier if not. Re-uploading the same url with new content "
        "creates a new version and supersedes the old one, exactly like a re-fetch.",
    ),
    service_category: str = Form(...),
    canonical: bool = Form(False),
    title: Optional[str] = Form(None),
    source_group: Optional[str] = Form(None),
    jurisdiction: Optional[str] = Form(None, description="comma-separated, defaults to 'all'"),
    applicant_variant: Optional[str] = Form(None, description="comma-separated, optional"),
    notes: Optional[str] = Form(None),
    actor: str = Form("admin", description="who is performing this upload, for the audit log"),
):
    content = await file.read()
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="uploaded file does not look like a PDF (missing %PDF header)")

    jurisdiction_list = [j.strip() for j in jurisdiction.split(",")] if jurisdiction else ["all"]
    applicant_variant_list = [v.strip() for v in applicant_variant.split(",")] if applicant_variant else []

    content_hash = hashlib.sha256(content).hexdigest()

    with get_conn() as conn:
        try:
            source = get_by_url(conn, url)
            if source is None:
                source = upsert_source(
                    conn,
                    url=url,
                    source_type="pdf",
                    service_category=service_category,
                    canonical=canonical,
                    jurisdiction=jurisdiction_list,
                    applicant_variant=applicant_variant_list,
                    notes=notes,
                    title=title,
                    source_group=source_group or "Manual uploads",
                )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"could not create source: {exc}")

        latest = get_latest_version(conn, source.id)
        if latest is not None and latest.content_hash == content_hash:
            return {
                "source_id": str(source.id),
                "version": latest.version,
                "unchanged": True,
                "message": "content_hash matches the existing latest version — nothing new to embed",
            }

        next_version = (latest.version + 1) if latest else 1
        version_dir = RAW_STORAGE_DIR / str(source.id) / f"v{next_version}"
        version_dir.mkdir(parents=True, exist_ok=True)
        raw_path = version_dir / _safe_filename(file.filename or "upload.pdf")
        raw_path.write_bytes(content)

        version = create_version(
            conn,
            source_id=source.id,
            content_hash=content_hash,
            retrieval_date=datetime.now(timezone.utc),
            raw_content_path=str(raw_path.relative_to(REPO_ROOT)),
            fetch_status="success",
        )
        assert version.version == next_version, (
            f"version numbering mismatch for source {source.id}: computed {next_version}, "
            f"DB assigned {version.version}"
        )

        log_event(
            conn,
            entity_type="source_version",
            entity_id=version.id,
            action="manual_upload",
            actor=actor,
            details={
                "url": url,
                "content_hash": content_hash,
                "version": version.version,
                "filename": file.filename,
            },
        )

        try:
            embedder = BgeM3Embedder(batch_size=12)
            qdrant_client = get_qdrant_client()
            summary = embed_and_index_version(conn, qdrant_client, embedder, source, version)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"uploaded and versioned successfully, but extraction/embedding failed: {exc}",
            )

    return {
        "source_id": str(source.id),
        "version": version.version,
        "unchanged": False,
        **summary,
    }
