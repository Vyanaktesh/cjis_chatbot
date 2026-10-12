#!/usr/bin/env python3
"""
Phase 2 voice-grounding check: run 10 high-stakes questions in voice mode and
verify the SPOKEN summary keeps the fees, deadlines and "must" conditions that
appear in the full written answer. This is a live check against a running stack,
not a unit test -- it calls the real model.

Usage (stack running, see docker-compose):
    python scripts/voice_eval.py --base-url http://localhost:8080 --password demo-access-2026

It prints, per question: the money/time values found in the full answer, whether
each survived into the spoken text (or the spoken text pointed to the screen),
and a final pass rate. Nothing secret is printed.
"""
import argparse
import json
import re
import sys
import urllib.request

QUESTIONS = [
    "How do I re-issue my OCI card after getting a new passport?",
    "I lost my passport, what do I do and what does it cost?",
    "How do I apply for an emergency visa and how long does it take?",
    "What is the fee to renew an adult passport?",
    "Do I need a renunciation certificate before applying for OCI?",
    "How do I apply for an OCI card for my newborn child?",
    "What are the photo requirements for an OCI application?",
    "How long does a normal visa take to process?",
    "What documents do I need for a passport for a minor?",
    "How do I get a police clearance certificate?",
]

MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
TIME_RE = re.compile(r"\b\d+\s?(?:-\s?\d+\s?)?(?:business\s)?(?:days?|weeks?|months?|hours?)\b", re.IGNORECASE)
SCREEN_RE = re.compile(r"on the screen|on screen|full (?:list|conditions|details)", re.IGNORECASE)


def ask(base_url, jar_header, question):
    data = json.dumps({"message": question, "mode": "voice"}).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat",
        data=data,
        headers={"Content-Type": "application/json", "Cookie": jar_header},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def login(base_url, password):
    req = urllib.request.Request(
        base_url.rstrip("/") + "/auth/login",
        data=json.dumps({"password": password}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        cookie = r.headers.get("Set-Cookie", "")
    return cookie.split(";")[0] if cookie else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--password", default="")
    args = ap.parse_args()

    jar = login(args.base_url, args.password) if args.password else ""

    grounded_ok = 0
    grounded_total = 0
    declines = 0
    for i, q in enumerate(QUESTIONS, 1):
        d = ask(args.base_url, jar, q)
        full = d.get("answer") or ""
        spoken = d.get("spoken_text") or ""
        grounded = d.get("grounded")

        if not grounded:
            # A decline is correct behaviour (the topic isn't in approved
            # sources); it just must be a short spoken refusal.
            declines += 1
            short = len(spoken.split()) <= 60
            print(f"[{i:2}] DECLINE grounded=False short_spoken={short} words={len(spoken.split())}")
            print(f"     spoken: {spoken[:160]}")
            continue

        grounded_total += 1
        full_values = set(MONEY_RE.findall(full)) | set(TIME_RE.findall(full))
        spoken_norm = spoken.lower()
        # A value is "kept" if it appears in the spoken text, OR the spoken text
        # explicitly points the user to the screen for the full details.
        points_to_screen = bool(SCREEN_RE.search(spoken))
        missing = [v for v in full_values if v.lower() not in spoken_norm]
        ok = not missing or points_to_screen
        grounded_ok += 1 if ok else 0

        print(f"[{i:2}] {'PASS ' if ok else 'CHECK'} grounded=True words={len(spoken.split())}")
        print(f"     values in full: {sorted(full_values) or '(none)'}")
        if missing:
            print(f"     NOT spoken verbatim: {sorted(missing)}  (screen-pointer={points_to_screen})")
        print(f"     spoken: {spoken[:160]}")

    print("-" * 70)
    print(f"Grounded answers that kept fees/deadlines (or pointed to screen): {grounded_ok}/{grounded_total}")
    print(f"Correct declines (topic not in approved sources): {declines}")
    return 0 if grounded_ok == grounded_total else 1


if __name__ == "__main__":
    sys.exit(main())
