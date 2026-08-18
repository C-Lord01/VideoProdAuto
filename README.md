# VideoProdAuto

Batch AI video generation pipeline for music video production, built against the
Runway ML API. Originally built for **"Tales From My Mom's Basement"** (C-LORD) —
a 47-shot satirical music video using consistent Muppet-style puppet characters
across every shot.

## What this does

Takes a structured shot list (`shots.json`) and a character roster
(`characters.json`), generates reference stills for each character, wires those
references into every shot that character appears in, then batch-generates video
for the full shot list — resumable, cost-tracked, and logged.

## Pipeline

hero.py --generate → generate character reference stills (gen4_image)
hero.py --pick → select the canonical still per character
hero.py --upload → upload picks, auto-tag references into shots.json
generate.py → batch text-to-video generation (seedance2)


## Key files

| File | Purpose |
|---|---|
| `generate.py` | Batch video generation. Resumable — skips shots already downloaded. Logs every attempt (cost, status, output) to `output/manifest.csv`. |
| `hero.py` | Character reference workflow: generate candidate stills, pick one per character, upload, and auto-wire references into the matching shots. |
| `shots.json` | Structured shot list — prompt, duration, and attached reference image(s) per shot. |
| `characters.json` | Character roster — reference prompt and full shot coverage list per character. |

## Why references, not seeds

`seedance2` (the model used here) doesn't support `seed` or `negative_prompt` —
re-rolls aren't reproducible on this model. The actual consistency mechanism is
image references (`references: [{"uri": ...}]`), generated once per character via
`gen4_image` (which does support seeds) and then reused across every shot that
character appears in. This is what keeps a puppet's fur color, eye size, and
costume identical across 47 independent generations.

## Notes

- Built and verified against `runwayml` SDK v5.13.0 — endpoint and parameter
  names in this repo are confirmed against the installed package, not assumed
  from docs.
- Reference image URIs in `shots.json` are ephemeral Runway upload tokens and
  expire quickly — they're left in as an example of the wiring, not usable
  credentials.
- Cost model: `seedance2` at 1080p runs ~40 credits/sec ($0.01/credit). A 5s
  shot costs ~$2. The 47-shot batch for this project ran ~$94.

## Status

This repo reflects a single production pass (batch generation complete,
re-roll/assembly ongoing). It's shared as an example of the pipeline
architecture, not a finished, general-purpose tool.