#!/usr/bin/env python3
"""
Batch video generation for "Tales From My Mom's Basement" via the Runway API.

    pip install runwayml
    export RUNWAYML_API_SECRET="your_key_here"

    python generate.py --dry-run                 # cost estimate, no API calls
    python generate.py --shots shot_21           # single shot, moderation test
    python generate.py --shots intro_B1 --model gen4.5   # cheap draft
    python generate.py --shots shot_01-shot_23   # a range
    python generate.py                           # everything not already downloaded

Resumable: any shot with an existing .mp4 in OUTPUT_DIR is skipped. Delete the
file to force a re-roll. Every attempt is appended to manifest.csv.

======================== VERIFIED AGAINST SDK 5.13.0 =========================
Read out of the installed package, not guessed:

  endpoint   client.text_to_video.create(...)   <- NOT image_to_video, which
             requires prompt_image and has no text-only signature
  model      "seedance2"                        confirmed valid
  ratio      "1920:1080"                        confirmed in seedance2's list
  duration   plain int for seedance2 (other models restrict to 4/6/8)
  audio      bool - seedance2 makes native audio; set AUDIO=False to skip
  references [{"uri": ...}] - image refs for character consistency

  NO seed and NO negative_prompt on seedance2. Keep negatives in prompt text.
  Reruns are NOT reproducible - references are your only consistency lever.
==============================================================================
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ----------------------------- CONFIG ---------------------------------------

SHOTS_FILE = Path("shots.json")
OUTPUT_DIR = Path("./output")
MANIFEST = OUTPUT_DIR / "manifest.csv"

MODEL = "seedance2"        # unverified - see header
RATIO = "1920:1080"        # unverified - see header
DURATION = 5               # seconds
CONCURRENCY = 2            # raise once you know your tier's rate limit
MAX_RETRIES = 2            # transient errors only; moderation is not retried
AUDIO = True               # seedance2 generates native audio; False to skip it

# credits per second, from docs.dev.runwayml.com/guides/pricing
CREDITS_PER_SEC = {
    "seedance2": 40,        # 1080p (36 at 480p/720p)
    "seedance2_fast": 29,   # 480p/720p only
    "seedance2_mini": 16,   # 480p/720p only, 64 credit minimum
    "gen4.5": 12,
    "gen4_turbo": 5,
    "veo3.1": 20,
}
USD_PER_CREDIT = 0.01

# ----------------------------------------------------------------------------

try:
    from runwayml import RunwayML
except ImportError:
    sys.exit("runwayml SDK not installed.  pip install runwayml")

# The SDK exports this, but the path has moved between versions. Fall back to
# broad exception handling rather than crashing on an import.
try:
    from runwayml import TaskFailedError
except ImportError:
    class TaskFailedError(Exception):
        pass


def load_shots():
    if not SHOTS_FILE.exists():
        sys.exit(f"{SHOTS_FILE} not found. Run build_shots.py first.")
    return json.load(open(SHOTS_FILE))


def select(shots, spec):
    """--shots accepts: shot_21 | shot_21,shot_24 | shot_01-shot_23 | intro_B1"""
    if not spec:
        return shots
    ids = [s["id"] for s in shots]
    wanted = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part and part.count("-") == 1 and not part.startswith("-"):
            a, b = part.split("-")
            a, b = a.strip(), b.strip()
            if a in ids and b in ids:
                wanted += ids[ids.index(a):ids.index(b) + 1]
                continue
        if part in ids:
            wanted.append(part)
        else:
            print(f"  ! unknown shot id: {part}")
    return [s for s in shots if s["id"] in wanted]


def cost(model, duration):
    c = CREDITS_PER_SEC.get(model)
    if c is None:
        return None, None
    credits = c * duration
    return credits, credits * USD_PER_CREDIT


def log(row):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    new = not MANIFEST.exists()
    with open(MANIFEST, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "shot_id", "model", "ratio", "duration",
                        "status", "task_id", "credits", "usd", "file", "detail"])
        w.writerow(row)


def download(url, dest):
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    tmp.rename(dest)
    return dest.stat().st_size


def generate_one(client, shot, model, ratio, duration):
    sid = shot["id"]
    dest = OUTPUT_DIR / f"{sid}.mp4"
    credits, usd = cost(model, duration)
    stamp = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")

    for attempt in range(1, MAX_RETRIES + 2):
        task_id = ""
        try:
            kwargs = dict(
                model=model,
                prompt_text=shot["prompt"],
                ratio=ratio,
                duration=duration,
                audio=AUDIO,
            )
            if shot.get("references"):
                kwargs["references"] = [{"uri": u} for u in shot["references"]]

            task = client.text_to_video.create(**kwargs).wait_for_task_output()

            task_id = getattr(task, "id", "") or ""
            outputs = getattr(task, "output", None) or []
            if not outputs:
                raise RuntimeError("task succeeded but returned no output URL")

            # Output URLs are ephemeral. Download immediately.
            size = download(outputs[0], dest)
            log([stamp(), sid, model, ratio, duration, "ok", task_id,
                 credits, usd, dest.name, f"{size//1024}KB"])
            return sid, "ok", f"{size//1024}KB"

        except TaskFailedError as e:
            # Generation ran and was rejected - moderation, or a bad prompt.
            # Retrying costs money and will fail identically. Stop.
            detail = str(getattr(e, "task_details", e))[:300]
            log([stamp(), sid, model, ratio, duration, "failed", task_id,
                 credits, usd, "", detail])
            return sid, "failed", detail

        except Exception as e:
            detail = f"{type(e).__name__}: {e}"[:300]
            if attempt <= MAX_RETRIES:
                wait = 5 * attempt
                print(f"  {sid}: {detail} - retry {attempt}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
                continue
            log([stamp(), sid, model, ratio, duration, "error", task_id,
                 0, 0, "", detail])
            return sid, "error", detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", help="shot_21 | shot_01-shot_23 | a,b,c")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ratio", default=RATIO)
    ap.add_argument("--duration", type=int, default=DURATION)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="regenerate existing")
    ap.add_argument("--concurrency", type=int, default=CONCURRENCY)
    args = ap.parse_args()

    shots = select(load_shots(), args.shots)
    if not shots:
        sys.exit("No shots selected.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.force:
        pending = [s for s in shots if not (OUTPUT_DIR / f"{s['id']}.mp4").exists()]
        skipped = len(shots) - len(pending)
        if skipped:
            print(f"Skipping {skipped} already downloaded (use --force to redo)")
        shots = pending

    if not shots:
        sys.exit("Nothing to do - all selected shots already downloaded.")

    credits, usd = cost(args.model, args.duration)
    print(f"\nModel:    {args.model}")
    print(f"Ratio:    {args.ratio}   Duration: {args.duration}s")
    print(f"Shots:    {len(shots)}")
    if credits:
        print(f"Estimate: {credits * len(shots):,} credits  ~${usd * len(shots):,.2f}"
              f"   (${usd:.2f} each)")
    else:
        print(f"Estimate: unknown - '{args.model}' not in the price table")

    if args.dry_run:
        print("\n--dry-run, nothing sent:\n")
        for s in shots:
            print(f"  {s['id']:<12} {s['title'][:52]}")
            print(f"               {s['prompt'][:100]}...")
        return

    if not os.environ.get("RUNWAYML_API_SECRET"):
        sys.exit("RUNWAYML_API_SECRET is not set.")

    if len(shots) > 3:
        if input(f"\nGenerate {len(shots)} shots? [y/N] ").strip().lower() != "y":
            return

    client = RunwayML()
    results = {"ok": 0, "failed": 0, "error": 0}
    t0 = time.time()

    print()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(generate_one, client, s, args.model, args.ratio,
                        args.duration): s for s in shots
        }
        for i, fut in enumerate(as_completed(futures), 1):
            sid, status, detail = fut.result()
            results[status] = results.get(status, 0) + 1
            mark = {"ok": "OK  ", "failed": "FAIL", "error": "ERR "}[status]
            print(f"[{i}/{len(shots)}] {mark} {sid}  {detail[:70]}")

    mins = (time.time() - t0) / 60
    print(f"\nDone in {mins:.1f} min - "
          f"{results['ok']} ok, {results['failed']} rejected, {results['error']} errored")
    spent = results["ok"] * (usd or 0)
    print(f"Downloaded to {OUTPUT_DIR.resolve()}   ~${spent:,.2f} on successes")
    print(f"Log: {MANIFEST}")
    if results["failed"]:
        print("\nRejected shots are usually content moderation. Check the "
              "'detail' column in manifest.csv, rewrite the prompt, re-run "
              "with --shots <id>.")


if __name__ == "__main__":
    main()
