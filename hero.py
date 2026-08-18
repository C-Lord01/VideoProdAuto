#!/usr/bin/env python3
"""
Character reference workflow - generate a hero still per character, then
wire each one into the shots that character appears in.

    python hero.py --list                     # free: roster + shot coverage
    python hero.py --generate                 # all characters, 4 variants each
    python hero.py --generate goon_adult      # just one character
    python hero.py --generate --count 2 --seed 100
    python hero.py --pick goon_adult hero/goon_adult_3.png   # choose a variant
    python hero.py --upload                   # upload all picks, patch shots.json
    python hero.py --clear                    # strip all references

Typical run:
    python hero.py --generate                 # look at ./hero/
    python hero.py --pick goon_adult hero/goon_adult_2.png
    python hero.py --pick mom hero/mom_1.png      ... etc
    python hero.py --upload
    python generate.py --shots shot_21

--------------------------------------------------------------------------
EPHEMERAL UPLOADS EXPIRE. Runway deletes them after a period it does not
specify. Run --upload immediately before the batch, not the night before.
If a long batch starts failing with reference errors, re-run --upload and
resume - generate.py skips whatever it already downloaded.
--------------------------------------------------------------------------
"""

import argparse, json, sys, urllib.request
from pathlib import Path
from runwayml import RunwayML

SHOTS_FILE = Path("shots.json")
CHARS_FILE = Path("characters.json")
PICKS_FILE = Path("hero/picks.json")
HERO_DIR   = Path("./hero")
IMAGE_MODEL = "gen4_image"      # supports seed; reproducible


def chars():
    return json.load(open(CHARS_FILE))


def picks():
    return json.load(open(PICKS_FILE)) if PICKS_FILE.exists() else {}


def save_picks(p):
    HERO_DIR.mkdir(exist_ok=True)
    json.dump(p, open(PICKS_FILE, "w"), indent=2)


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    return dest


def cmd_list():
    C, shots = chars(), json.load(open(SHOTS_FILE))
    ids = {s["id"] for s in shots}
    covered = set()
    print(f"{'CHARACTER':<22} {'SHOTS':>5}  PICK")
    print("-" * 62)
    P = picks()
    for k, c in C.items():
        covered |= set(c["shots"])
        bad = [s for s in c["shots"] if s not in ids]
        mark = P.get(k, "-")
        print(f"{k:<22} {len(c['shots']):>5}  {Path(mark).name if mark != '-' else '-'}")
        if bad:
            print(f"{'':22} !! unknown shot ids: {bad}")
    clean = sorted(ids - covered)
    print("-" * 62)
    print(f"{len(covered)} shots get a reference, {len(clean)} stay clean:")
    print("  " + ", ".join(clean))


def cmd_generate(only, count, seed):
    c_all = chars()
    todo = {only: c_all[only]} if only else c_all
    if only and only not in c_all:
        sys.exit(f"Unknown character '{only}'. Options: {', '.join(c_all)}")
    client = RunwayML()
    HERO_DIR.mkdir(exist_ok=True)
    for key, c in todo.items():
        print(f"\n{c['name']}")
        for i in range(count):
            s = (seed + i) if seed is not None else None
            kw = dict(model=IMAGE_MODEL, prompt_text=c["prompt"], ratio=c["ratio"])
            if s is not None:
                kw["seed"] = s
            print(f"  {i+1}/{count} ({'seed='+str(s) if s is not None else 'random'}) ...", flush=True)
            try:
                t = client.text_to_image.create(**kw).wait_for_task_output()
                urls = getattr(t, "output", None) or []
                if not urls:
                    print("     no output"); continue
                d = download(urls[0], HERO_DIR / f"{key}_{i+1}.png")
                print(f"     -> {d}")
            except Exception as e:
                print(f"     failed: {type(e).__name__}: {e}")
    print(f"\nCandidates in {HERO_DIR.resolve()}")
    print("Pick each:  python hero.py --pick <character> hero/<file>.png")


VARIANT_POSES = {
    "3q":      "standing in a relaxed three-quarter view turned slightly to the left, full figure",
    "profile": "standing in full side profile facing left, full figure",
    "seated":  "sitting slumped in a plain office chair, full figure",
    "closeup": "head and shoulders portrait facing camera, upper body only",
    "raised":  "standing with both arms raised straight overhead, full figure",
    "back":    "standing with back to camera, full figure",
}


def cmd_variants(key, poses):
    """Generate the SAME character in new poses, using the picked still as a
    reference image. This is what you want instead of a seed - a seed repeats
    one image, a reference carries the character into new ones."""
    C, P = chars(), picks()
    if key not in C:
        sys.exit(f"Unknown character '{key}'. Options: {', '.join(C)}")
    if key not in P:
        sys.exit(f"No pick for '{key}' yet. Run --pick first.")

    src = Path(P[key])
    client = RunwayML()
    print(f"Uploading reference: {src}")
    with open(src, "rb") as f:
        up = client.uploads.create_ephemeral(file=f)
    uri = getattr(up, "url", None) or getattr(up, "uri", None) or getattr(up, "id", None)
    if not uri:
        print("upload response:", up)
        sys.exit("No URI on upload response.")

    want = poses or list(VARIANT_POSES)
    for name in want:
        if name not in VARIANT_POSES:
            print(f"  skip unknown pose '{name}' (have: {', '.join(VARIANT_POSES)})")
            continue
        prompt = (f"@char the same puppet character, {VARIANT_POSES[name]}, against a "
                  f"plain neutral grey studio backdrop, even soft studio lighting, sharp "
                  f"focus, no props, no background scenery.")
        print(f"  {name} ...", flush=True)
        try:
            t = client.text_to_image.create(
                model=IMAGE_MODEL,
                prompt_text=prompt,
                ratio=C[key]["ratio"],
                reference_images=[{"uri": uri, "tag": "char"}],
            ).wait_for_task_output()
            urls = getattr(t, "output", None) or []
            if not urls:
                print("     no output"); continue
            d = download(urls[0], HERO_DIR / f"{key}_pose_{name}.png")
            print(f"     -> {d}")
        except Exception as e:
            print(f"     failed: {type(e).__name__}: {e}")

    print(f"\nVariants in {HERO_DIR.resolve()}")
    print("These are alternate views of the SAME puppet. The --pick file stays")
    print("your canonical reference for the video batch.")


def cmd_pick(key, path):
    if key not in chars():
        sys.exit(f"Unknown character '{key}'. Options: {', '.join(chars())}")
    p = Path(path)
    if not p.exists():
        sys.exit(f"{p} not found")
    P = picks(); P[key] = str(p); save_picks(P)
    print(f"{key} -> {p}")
    missing = [k for k in chars() if k not in P]
    print(f"Still unpicked: {', '.join(missing) if missing else 'none - run --upload'}")


def cmd_upload():
    C, P = chars(), picks()
    if not P:
        sys.exit("No picks yet. Run --generate then --pick.")
    missing = [k for k in C if k not in P]
    if missing:
        print(f"WARNING: no pick for {', '.join(missing)} - those shots stay clean.\n")

    client = RunwayML()
    uris = {}
    for key, path in P.items():
        print(f"Uploading {key}: {path} ...")
        with open(path, "rb") as f:
            up = client.uploads.create_ephemeral(file=f)
        uri = getattr(up, "url", None) or getattr(up, "uri", None) or getattr(up, "id", None)
        if not uri:
            print("  upload response:", up)
            sys.exit("No URI on the upload response - inspect above.")
        uris[key] = uri
        print(f"  {uri}")

    shots = json.load(open(SHOTS_FILE))
    by_id = {s["id"]: s for s in shots}
    for s in shots:
        s.pop("references", None)
    tagged = 0
    for key, uri in uris.items():
        for sid in C[key]["shots"]:
            if sid in by_id:
                by_id[sid].setdefault("references", []).append(uri)
                tagged += 1
    json.dump(shots, open(SHOTS_FILE, "w"), indent=2)

    multi = [s["id"] for s in shots if len(s.get("references", [])) > 1]
    print(f"\n{tagged} references applied across "
          f"{sum(1 for s in shots if s.get('references'))} shots.")
    if multi:
        print(f"Multi-character shots: {', '.join(multi)}")
    print("\nRun the batch now - uploads expire.")


def cmd_clear():
    shots = json.load(open(SHOTS_FILE))
    n = sum(1 for s in shots if s.pop("references", None))
    json.dump(shots, open(SHOTS_FILE, "w"), indent=2)
    print(f"Cleared references from {n} shots.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--generate", nargs="?", const="", metavar="CHARACTER")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pick", nargs=2, metavar=("CHARACTER", "PATH"))
    ap.add_argument("--variants", metavar="CHARACTER",
                    help="same character, new poses, via reference image")
    ap.add_argument("--poses", nargs="*", metavar="POSE",
                    help=f"subset of: {', '.join(VARIANT_POSES)}")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--clear", action="store_true")
    a = ap.parse_args()

    if a.list:            cmd_list()
    elif a.generate is not None: cmd_generate(a.generate or None, a.count, a.seed)
    elif a.pick:          cmd_pick(*a.pick)
    elif a.variants:      cmd_variants(a.variants, a.poses)
    elif a.upload:        cmd_upload()
    elif a.clear:         cmd_clear()
    else:                 ap.print_help()
