#!/usr/bin/env python3
"""Regression harness for the SCUA segmentation pipeline.

Scores a PRODUCED segment JSON against a hand-labeled GROUND-TRUTH segment JSON.
It is deliberately decoupled from HOW the produced JSON was made: you run the
pipeline (locally in the container, or on ECS) to write segment/<stem>.json, then
point this harness at that file. That keeps the harness pure-stdlib (no cv2, boto3,
ffmpeg, or Bedrock) so it runs anywhere and in CI, and lets every pipeline change be
measured against the fixtures in test/segment/ instead of eyeballed one clip at a time.

Segment JSON shape (both produced and ground truth):
    {
      "video": "<name>.mp4",
      "segments": [
        {"segment_start": float, "segment_end": float,
         "segment_type": "D" | "<content category>", "title", "caption", ...},
        ...
      ],
      "programs": [ {"program_index", "program_start", "program_end", ...}, ... ]  # optional
    }

Metrics (all in [0,1] unless noted):
  dead_space:   how well produced "D" spans match ground-truth "D" spans
                (precision, recall, F1, IoU over the union of D time).
  boundaries:   fraction of ground-truth CONTENT segment boundaries that a produced
                boundary lands within `--bound-tol` seconds of (recall), and vice
                versa (precision). Boundaries are segment start/end times.
  seg_count:    produced vs. truth content-segment counts (informational + delta).
  type_accuracy: on content segments matched by time overlap, fraction with the
                same segment_type.
  programs:     when EITHER side has a "programs" array — program-boundary recall/
                precision within `--bound-tol` (else reported as n/a).

Usage:
    # one file (auto-finds truth by stem in the truth dir)
    python regression_harness.py path/to/produced/spc-...-i0100.json
    # explicit truth
    python regression_harness.py produced.json --truth test/segment/spc-...-i0100.json
    # batch: score every produced JSON in a dir against test/segment/
    python regression_harness.py --produced-dir /tmp/out --truth-dir test/segment
    # self-test: score the ground truth against itself (must be perfect)
    python regression_harness.py --self-test

Exit code is nonzero if any fixture falls below the pass thresholds (see THRESHOLDS).
"""

import argparse
import glob
import json
import os
import sys

DEAD_TYPE = "D"

# Pass thresholds (tunable via CLI). A fixture PASSES when every gated metric meets
# its floor. These are starting points — tighten as the pipeline improves.
THRESHOLDS = {
    "dead_f1": 0.80,        # D-span F1
    "boundary_f1": 0.70,    # content-boundary F1 within tolerance
    "type_accuracy": 0.70,  # type agreement on matched content segments
}


# ── helpers ──────────────────────────────────────────────────────────────────
def _load(path):
    with open(path) as f:
        return json.load(f)


def _segs(doc):
    return doc.get("segments", []) or []


def _merge(spans):
    """Merge overlapping/adjacent (start,end) spans so totals/overlaps don't double
    count when the input has overlapping or touching spans."""
    spans = sorted(spans)
    out = []
    for a, b in spans:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _spans(segs, dead=False):
    """(start,end) spans for dead (type==D) or content (type!=D) segments.
    NOT merged — callers that need de-duplicated time totals merge explicitly."""
    out = []
    for s in segs:
        is_dead = s.get("segment_type") == DEAD_TYPE
        if is_dead == dead:
            a, b = float(s.get("segment_start", 0)), float(s.get("segment_end", 0))
            if b > a:
                out.append((a, b))
    return sorted(out)


def _total(spans):
    return sum(b - a for a, b in spans)


def _overlap_total(spans_a, spans_b):
    """Total overlapping seconds between two span lists (assumes each is sorted)."""
    total = 0.0
    j = 0
    for a0, a1 in spans_a:
        for b0, b1 in spans_b:
            if b1 <= a0:
                continue
            if b0 >= a1:
                break
            total += min(a1, b1) - max(a0, b0)
    return total


def dead_metrics(prod, truth):
    """Precision/recall/F1/IoU over dead-space (D) time. Spans are merged first so
    overlapping/adjacent D markers don't inflate totals past the real time covered."""
    p = _merge(_spans(_segs(prod), dead=True))
    t = _merge(_spans(_segs(truth), dead=True))
    pt, tt = _total(p), _total(t)
    ov = _overlap_total(p, t)
    precision = ov / pt if pt else (1.0 if tt == 0 else 0.0)
    recall = ov / tt if tt else (1.0 if pt == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else (1.0 if tt == 0 and pt == 0 else 0.0)
    union = pt + tt - ov
    iou = ov / union if union else 1.0
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou,
            "produced_dead_s": round(pt, 1), "truth_dead_s": round(tt, 1)}


def _boundaries(segs, dead=False):
    """Distinct segment start/end times for content (or dead) segments."""
    bs = set()
    for a, b in _spans(segs, dead=dead):
        bs.add(round(a, 2))
        bs.add(round(b, 2))
    return sorted(bs)


def _match_within(a_points, b_points, tol):
    """How many points in a_points have a b_points point within tol."""
    matched = 0
    for a in a_points:
        if any(abs(a - b) <= tol for b in b_points):
            matched += 1
    return matched


def boundary_metrics(prod, truth, tol):
    """Content-boundary precision/recall/F1 within `tol` seconds."""
    pb = _boundaries(_segs(prod))
    tb = _boundaries(_segs(truth))
    recall = _match_within(tb, pb, tol) / len(tb) if tb else (1.0 if not pb else 0.0)
    precision = _match_within(pb, tb, tol) / len(pb) if pb else (1.0 if not tb else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else (1.0 if not tb and not pb else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "produced_boundaries": len(pb), "truth_boundaries": len(tb)}


def _mid(a, b):
    return (a + b) / 2.0


def type_accuracy(prod, truth):
    """Type agreement on content segments matched by dominant time overlap.

    Each ground-truth content segment is matched to the produced content segment it
    overlaps most; agreement = same segment_type. Returns accuracy + matched count."""
    p_content = [s for s in _segs(prod) if s.get("segment_type") != DEAD_TYPE]
    t_content = [s for s in _segs(truth) if s.get("segment_type") != DEAD_TYPE]
    if not t_content:
        return {"accuracy": 1.0 if not p_content else 0.0, "matched": 0, "total": 0}
    agree = 0
    matched = 0
    for t in t_content:
        ta, tb = float(t["segment_start"]), float(t["segment_end"])
        best, best_ov = None, 0.0
        for p in p_content:
            pa, pb = float(p["segment_start"]), float(p["segment_end"])
            ov = max(0.0, min(tb, pb) - max(ta, pa))
            if ov > best_ov:
                best_ov, best = ov, p
        if best is not None:
            matched += 1
            if str(best.get("segment_type", "")).lower() == str(t.get("segment_type", "")).lower():
                agree += 1
    return {"accuracy": agree / matched if matched else 0.0,
            "matched": matched, "total": len(t_content)}


def program_metrics(prod, truth, tol):
    """Program-boundary precision/recall/F1 within tol, or None if neither side
    has a programs array (so single-program fixtures aren't penalized)."""
    pp = prod.get("programs") or []
    tp = truth.get("programs") or []
    if not pp and not tp:
        return None
    def bounds(progs):
        bs = set()
        for pr in progs:
            bs.add(round(float(pr.get("program_start", 0)), 2))
            bs.add(round(float(pr.get("program_end", 0)), 2))
        return sorted(bs)
    pb, tb = bounds(pp), bounds(tp)
    recall = _match_within(tb, pb, tol) / len(tb) if tb else (1.0 if not pb else 0.0)
    precision = _match_within(pb, tb, tol) / len(pb) if pb else (1.0 if not tb else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "produced_programs": len(pp), "truth_programs": len(tp)}


# ── scoring one fixture ────────────────────────────────────────────────────────
# Mode-aware: a metric is only scored/gated when the PRODUCED file actually attempted
# it. The deterministic local runner emits dead-space only (no content segments), so
# `boundaries`/`type_accuracy` are reported n/a and don't gate — instead of a
# misleading 0.00 FAIL. A full cloud run produces content, so those become live.
def score(prod, truth, bound_tol):
    p_content = len([s for s in _segs(prod) if s.get("segment_type") != DEAD_TYPE])
    t_content = len([s for s in _segs(truth) if s.get("segment_type") != DEAD_TYPE])
    p_dead = len(_spans(_segs(prod), dead=True))
    t_dead = len(_spans(_segs(truth), dead=True))

    dead = dead_metrics(prod, truth)
    bounds = boundary_metrics(prod, truth, bound_tol)
    types = type_accuracy(prod, truth)
    progs = program_metrics(prod, truth, bound_tol)

    # Applicability: a metric applies only if the produced file could have produced it.
    # - dead-space: applies if either side has any D spans (else nothing to compare).
    # - boundaries: applies only if the produced file has CONTENT segments.
    # - type_accuracy: applies only if the producer ATTEMPTED labels. The local no-
    #   Bedrock runner sets "labels_attempted": false (its segment types are placeholders),
    #   so type-accuracy is reported n/a there instead of failing on unlabeled content.
    labels_attempted = prod.get("labels_attempted", True)
    applies = {
        "dead": (p_dead > 0 or t_dead > 0),
        "boundaries": p_content > 0,
        "type_accuracy": p_content > 0 and labels_attempted,
    }

    checks = []
    if applies["dead"]:
        checks.append(dead["f1"] >= THRESHOLDS["dead_f1"])
    if applies["boundaries"]:
        checks.append(bounds["f1"] >= THRESHOLDS["boundary_f1"])
    if applies["type_accuracy"]:
        checks.append(types["accuracy"] >= THRESHOLDS["type_accuracy"])
    # PASS only if every APPLICABLE gated metric passes. If nothing applies (empty
    # produced file), that's not a pass — flag it so an empty producer isn't a silent PASS.
    passed = bool(checks) and all(checks)

    return {
        "dead": dead, "boundaries": bounds, "types": types, "programs": progs,
        "applies": applies,
        "labels_attempted": labels_attempted,
        "seg_count": {"produced": p_content, "truth": t_content, "delta": p_content - t_content},
        "passed": passed,
        "nothing_applicable": not checks,
    }


def print_report(name, r):
    d, b, t = r["dead"], r["boundaries"], r["types"]
    ap = r["applies"]
    status = "PASS" if r["passed"] else ("N/A " if r["nothing_applicable"] else "FAIL")
    print(f"\n[{status}] {name}")
    if ap["dead"]:
        print(f"  dead-space   F1={d['f1']:.2f}  P={d['precision']:.2f} R={d['recall']:.2f} "
              f"IoU={d['iou']:.2f}  (prod {d['produced_dead_s']}s / truth {d['truth_dead_s']}s)")
    else:
        print(f"  dead-space   n/a (no dead-space on either side)")
    if ap["boundaries"]:
        print(f"  boundaries   F1={b['f1']:.2f}  P={b['precision']:.2f} R={b['recall']:.2f} "
              f"(prod {b['produced_boundaries']} / truth {b['truth_boundaries']})")
    else:
        print(f"  boundaries   n/a (produced file has no content segments)")
    if ap["type_accuracy"]:
        print(f"  type-acc     {t['accuracy']:.2f}  ({t['matched']}/{t['total']} matched)")
    else:
        reason = ("labels not attempted (no-Bedrock run)"
                  if r.get("labels_attempted") is False else "produced file has no content segments")
        print(f"  type-acc     n/a ({reason})")
    sc = r["seg_count"]
    print(f"  seg-count    produced={sc['produced']} truth={sc['truth']} (delta {sc['delta']:+d})")
    if r["programs"] is not None:
        p = r["programs"]
        print(f"  programs     F1={p['f1']:.2f}  P={p['precision']:.2f} R={p['recall']:.2f} "
              f"(prod {p['produced_programs']} / truth {p['truth_programs']})")
    else:
        print(f"  programs     n/a (no program tier on either side)")


# ── CLI ────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TRUTH_DIR = os.path.join(HERE, "segment")


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def _find_truth(produced_path, truth_dir):
    cand = os.path.join(truth_dir, _stem(produced_path) + ".json")
    return cand if os.path.exists(cand) else None


def main():
    ap = argparse.ArgumentParser(description="Score produced segment JSON vs. ground truth.")
    ap.add_argument("produced", nargs="?", help="Produced segment JSON to score.")
    ap.add_argument("--truth", help="Ground-truth JSON (default: match by stem in --truth-dir).")
    ap.add_argument("--produced-dir", help="Batch: score every *.json in this dir.")
    ap.add_argument("--truth-dir", default=DEFAULT_TRUTH_DIR,
                    help="Ground-truth dir (default: test/segment).")
    ap.add_argument("--bound-tol", type=float, default=2.0,
                    help="Boundary match tolerance in seconds (default 2.0).")
    ap.add_argument("--self-test", action="store_true",
                    help="Score each ground-truth file against ITSELF (must be perfect).")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON results.")
    args = ap.parse_args()

    pairs = []  # (name, produced_doc, truth_doc)

    if args.self_test:
        for gt in sorted(glob.glob(os.path.join(args.truth_dir, "*.json"))):
            d = _load(gt)
            pairs.append((_stem(gt) + " (self)", d, d))
    elif args.produced_dir:
        for prod in sorted(glob.glob(os.path.join(args.produced_dir, "*.json"))):
            tp = _find_truth(prod, args.truth_dir)
            if not tp:
                print(f"WARN: no ground truth for {os.path.basename(prod)} in {args.truth_dir}",
                      file=sys.stderr)
                continue
            pairs.append((_stem(prod), _load(prod), _load(tp)))
    elif args.produced:
        tp = args.truth or _find_truth(args.produced, args.truth_dir)
        if not tp:
            print(f"ERROR: no ground truth found for {args.produced} (use --truth).", file=sys.stderr)
            return 2
        pairs.append((_stem(args.produced), _load(args.produced), _load(tp)))
    else:
        ap.print_help()
        return 2

    if not pairs:
        print("No fixtures to score.", file=sys.stderr)
        return 2

    results = {}
    any_fail = False
    for name, prod, truth in pairs:
        r = score(prod, truth, args.bound_tol)
        results[name] = r
        # A genuine FAIL is an applicable metric below threshold. An all-n/a fixture
        # is surfaced but does NOT fail the run (e.g. a dead-only local producer scored
        # against a fixture with no dead space).
        if not r["passed"] and not r["nothing_applicable"]:
            any_fail = True
        if not args.json:
            print_report(name, r)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        n_pass = sum(1 for r in results.values() if r["passed"])
        n_na = sum(1 for r in results.values() if r["nothing_applicable"])
        n_fail = sum(1 for r in results.values()
                     if not r["passed"] and not r["nothing_applicable"])
        summary = f"{n_pass} passed, {n_fail} failed"
        if n_na:
            summary += f", {n_na} n/a (no applicable metrics)"
        print(f"\n{'=' * 50}\nOVERALL: {summary} of {len(results)} fixtures "
              f"(thresholds: dead_F1>={THRESHOLDS['dead_f1']}, "
              f"boundary_F1>={THRESHOLDS['boundary_f1']}, "
              f"type_acc>={THRESHOLDS['type_accuracy']})")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
