# Segmentation regression harness

Scores a **produced** segment JSON against a hand-labeled **ground-truth** segment
JSON so pipeline changes can be measured instead of eyeballed one clip at a time.

## Layout

```
test/
  video/                     # source .mp4 fixtures (large; not committed)
  segment/<stem>.json        # hand-verified GROUND TRUTH, one per video
  regression_harness.py      # this scorer (pure stdlib — no cv2/boto3/ffmpeg/Bedrock)
```

Ground truth and produced files share the same shape:

```json
{
  "video": "<stem>.mp4",
  "segments": [
    {"segment_start": 0, "segment_end": 20.7, "segment_type": "D", "title": "...", "caption": ""},
    {"segment_start": 20.7, "segment_end": 97.3, "segment_type": "introduction", "title": "...", "caption": "COAST TO COAST"}
  ],
  "programs": [ {"program_index": 0, "program_start": 20.7, "program_end": 1659.8} ]   // optional
}
```

## Why it's decoupled from the pipeline

Running the real pipeline needs ffmpeg, OpenCV, AWS Transcribe, and Bedrock (rate-
limited, minutes per video, non-deterministic). So the harness does **not** run the
pipeline. Instead:

1. **Produce** segment JSONs by running the pipeline (in the container / on ECS).
   Each run writes `segment/<stem>.json` to S3; download those to a local folder.
2. **Score** them against the ground truth with this harness (pure JSON→JSON, runs
   anywhere, incl. CI).

## Workflow: how to produce the JSONs to score

There are two loops, chosen by whether your change touches the LLM stages.

### Fast local loop (deterministic changes — no deploy, no AWS)

For changes to **dead-space detection** (`analyze_deadspace.py`, `detect_bars.py`,
`detect_static.py`), boundary **cues** (lexical/pause), the speaker **merge/split/
coalesce** logic, the **carve**, or **program grouping** — none of which need Bedrock —
use `run_local.py`. It runs ffmpeg + OpenCV + PySceneDetect (no AWS) and writes a
`segment/<stem>.json`.

```bash
pip install opencv-python-headless numpy scenedetect   # ffmpeg must also be on PATH
python run_local.py video/spc-mss1159-s02-i0005.mp4 -o out/
python run_local.py --video-dir video -o out/          # whole fixture set
python regression_harness.py --produced-dir out --truth-dir segment
```

**Two local modes**, chosen automatically per video:

- **fusion (no-Bedrock)** — used when a cached transcript `transcript/<stem>.json`
  exists (diarized turns `[{start,end,speaker,text}]`). Runs the transcript-driven
  stages: shot detection, lexical + pause + dead-span boundaries (NOT the Bedrock
  `analyze_program` boundaries), speaker merge/split, coalesce, and program grouping.
  The **speech-veto is on**, so dead-space precision is realistic. Segment **types are
  placeholders** (labeling needs Bedrock), so the output is marked
  `"labels_attempted": false`.
- **dead-only** — used when no transcript exists (or `--dead-only`). Emits only D
  spans; the speech-veto is off, so it *over*-detects dead space in talky regions.

Which metrics are meaningful locally:

| metric | fusion (no-Bedrock) | dead-only |
|---|---|---|
| dead-space  | ✅ realistic (speech-veto on) | ⚠️ over-detects (veto off) — use as lower bound |
| boundaries  | ✅ real (minus LLM boundaries — a lower bound vs. cloud) | n/a |
| type-accuracy | n/a (`labels_attempted: false`) | n/a |
| programs    | ✅ when grouping produces >1 program | n/a |

The harness is **mode-aware**: metrics the producer didn't attempt are reported
**`n/a`** and do NOT gate pass/fail (a `type-accuracy` of `n/a` on a no-Bedrock run,
not a misleading `0.00 FAIL`). Applicability rules:
- `dead-space` — needs D spans on either side.
- `boundaries` — needs content segments in the produced file.
- `type-accuracy` — needs content segments AND `labels_attempted != false`.

`type-accuracy` (and the Bedrock `analyze_program` boundaries) still require the full
cloud loop below.

### Full cloud loop (LLM-affecting changes)

For changes to **boundaries** (`analyze_program`), **labels/descriptions**
(`label_segments`), or **program labeling** — which need Bedrock:

1. Deploy the backend to dev (`cdk deploy`) so the ECS container has your change.
2. Process the test videos (upload to the dev bucket, or drop segment-request markers).
3. Download the resulting `segment/<stem>.json` from S3 to a local folder.
4. `python regression_harness.py --produced-dir <folder> --truth-dir segment`.

This loop exercises the full pipeline (all metrics meaningful) but is slow: container
rebuild, ECS cold start, Transcribe, and the ~10 RPM Bedrock cap across the fixtures.
LLM stages are also non-deterministic, so expect small run-to-run score wobble on
`boundaries` / `type-accuracy` even with no code change.

## Usage

```bash
# One produced file (auto-finds ground truth by stem in test/segment/)
python regression_harness.py /path/to/produced/spc-mss1159-s02-i0100.json

# Explicit ground truth
python regression_harness.py produced.json --truth test/segment/spc-mss1159-s02-i0100.json

# Batch: score every produced JSON in a directory
python regression_harness.py --produced-dir /tmp/pipeline_out --truth-dir test/segment

# Sanity check the harness itself (each ground truth scored against itself → all perfect)
python regression_harness.py --self-test

# Machine-readable output for CI
python regression_harness.py --produced-dir /tmp/out --json
```

Exit code is **nonzero if any fixture falls below the pass thresholds**, so it can
gate CI or a pre-deploy check.

## Metrics

| Metric | Meaning |
|---|---|
| **dead-space** | Precision/Recall/F1/IoU over `segment_type == "D"` time (spans merged first, so overlapping D markers don't inflate totals). Measures edit-out accuracy. |
| **boundaries** | Fraction of ground-truth content-segment boundary times a produced boundary lands within `--bound-tol` seconds of (recall), and vice versa (precision), plus F1. |
| **type-accuracy** | On content segments matched by dominant time overlap, fraction with the same `segment_type`. |
| **seg-count** | Produced vs. truth content-segment counts + delta (informational). |
| **programs** | Program-boundary P/R/F1 within tolerance when EITHER side has a `programs` array; `n/a` otherwise (single-program fixtures aren't penalized). |

## Pass thresholds

Defined in `THRESHOLDS` in `regression_harness.py` (starting points — tighten as the
pipeline improves):

- `dead_f1 >= 0.80`
- `boundary_f1 >= 0.70`
- `type_accuracy >= 0.70`

`--bound-tol` (default 2.0s) sets how close a produced boundary must be to count as a
match; loosen for coarse fixtures, tighten to demand precise cuts.

## Adding a fixture

1. Drop the source video in `test/video/<stem>.mp4`.
2. Create/verify `test/segment/<stem>.json` (hand-label, or start from a pipeline run
   and correct it in the Editor, then export).
3. It's picked up automatically by `--self-test` and batch mode (paired by stem).

## Notes / limitations

- The harness scores **structure and types**, not label *wording* (titles/descriptions
  are free text; comparing them needs a separate, fuzzier check).
- Program metrics compare program **boundaries**; program title/description quality is
  not scored here (that's a labeling concern for a later pass).
- `--self-test` scoring a file against itself must be perfect (1.00). If it isn't, the
  ground-truth file likely has overlapping/again D spans worth cleaning up.
