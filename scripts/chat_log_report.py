#!/usr/bin/env python3
"""
What visitors asked the chat widget (table chat_logs, see
app/core/chat_logging.py): a summary, the most-asked questions, and the
questions it could not answer, which are the knowledge-base gaps to raise with
the Consulate. Personal details are already masked in the stored text.

Usage (inside the backend container):
    python scripts/chat_log_report.py                 # last 7 days
    python scripts/chat_log_report.py --days 30
    python scripts/chat_log_report.py --days 60 --csv /tmp/chat_logs.csv
then, to open the CSV in Excel on your computer:
    docker cp rag_backend:/tmp/chat_logs.csv chat_logs.csv
"""

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.connection import get_conn, get_dict_cursor  # noqa: E402

COLUMNS = ["created_at", "session_id", "mode", "is_followup", "outcome", "question",
           "search_query", "answer", "source_urls", "retrieved_count", "latency_ms"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=7, help="how many days back (default 7)")
    parser.add_argument("--csv", help="also write every row in the period to this CSV file")
    parser.add_argument("--top", type=int, default=15, help="how many most-asked questions to list")
    args = parser.parse_args()

    with get_conn() as conn, get_dict_cursor(conn) as cur:
        cur.execute(
            f"SELECT {', '.join(COLUMNS)} FROM chat_logs "
            "WHERE created_at >= now() - make_interval(days => %s) ORDER BY created_at;",
            (args.days,),
        )
        rows = cur.fetchall()

    print(f"Chat questions in the last {args.days} day(s): {len(rows)}")
    if not rows:
        return 0
    sessions = len({r["session_id"] for r in rows if r["session_id"]})
    print(f"Conversations: {sessions}   voice: {sum(r['mode'] == 'voice' for r in rows)}   "
          f"follow-ups: {sum(r['is_followup'] for r in rows)}")
    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    if latencies:
        print(f"Response time: median {latencies[len(latencies) // 2] / 1000:.1f}s, "
              f"slowest {latencies[-1] / 1000:.1f}s")

    print("\nOutcome:")
    for outcome, n in Counter(r["outcome"] for r in rows).most_common():
        print(f"  {outcome:<13} {n:>5}  ({n / len(rows):.0%})")

    print("\nPer day:")
    for day, n in sorted(Counter(r["created_at"].date().isoformat() for r in rows).items()):
        print(f"  {day}  {n}")

    def norm(q: str) -> str:
        return re.sub(r"\s+", " ", q.strip().lower().rstrip("?.! "))

    new_questions = [r for r in rows if not r["is_followup"] and r["outcome"] != "smalltalk"]
    asked = Counter(norm(r["question"]) for r in new_questions)
    wording = {}
    for r in new_questions:
        wording.setdefault(norm(r["question"]), r["question"].strip())
    print(f"\nMost asked (same wording, top {args.top}):")
    for q, n in asked.most_common(args.top):
        print(f"  {n:>3} x  {wording[q][:110]}")

    gaps = [r for r in rows if r["outcome"] in ("declined", "out_of_scope")]
    print(f"\nNot answered ({len(gaps)}), newest first. Declined = on-topic but not in the knowledge base:")
    for r in reversed(gaps[-40:]):
        print(f"  {r['created_at']:%Y-%m-%d %H:%M}  {r['outcome']:<12}  {r['question'][:110]}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for r in rows:
                writer.writerow({**r, "source_urls": " ".join(r["source_urls"] or [])})
        print(f"\nWrote {len(rows)} row(s) to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
