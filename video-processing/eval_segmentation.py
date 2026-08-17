"""Evaluate segmentation quality against a human-annotated ground truth.

Gives two families of metrics, because neither alone tells the whole story:

  Boundary detection (the intuitive one) -- precision / recall / F1 at a time
    TOLERANCE: a predicted story boundary counts as a hit if it falls within `tol`
    seconds of a ground-truth boundary, matched ONE-TO-ONE (no double counting). This
    is the number to put on a slide: "story boundaries detected at 87% F1 (+/-5s)".

  Pk and WindowDiff (the standard text/temporal-segmentation metrics) -- window-based
    and LOWER is better. They discretize the timeline and slide a window, so a boundary
    that is a little early/late is penalized gently instead of as a full miss+false-
    alarm. WindowDiff is the more robust of the two (Pevzner & Hearst 2002).

Ground truth is just the SAME segment JSON the pipeline emits, with the boundaries a
human corrected. Any of these shapes loads as a boundary list:
  - pipeline output : {"segments": [{"segment_start": .., "segment_end": ..}, ...]}
  - labeled output  : {"segments": [{"start": .., "end": ..}, ...]}
  - bare boundaries : {"boundaries": [12.3, 45.6], "duration": 1600}   or   [12.3, 45.6]

Make a gold file to correct:
    python eval_segmentation.py --make-template segment/ID.json -o gold/ID.json
    # then hand-fix the segment_start times in gold/ID.json to the true boundaries

Evaluate:
    python eval_segmentation.py --pred segment/ID.json --gold gold/ID.json
    python eval_segmentation.py --pred p.json --gold g.json --tol 2 5 10 --step 1
    python eval_segmentation.py --manifest pairs.json     # micro/macro over many videos

`pairs.json` for --manifest = [{"name": "RCC_183", "pred": "...", "gold": "..."}, ...].
"""

import argparse
import bisect
import json
import math
import sys


# ------------------------------- load boundaries ------------------------------
def _seg_bounds(segs):
    """(segment starts, segment ends) from either {segment_start/segment_end} or
    {start/end} rows -- both shapes the pipeline emits."""
    starts, ends = [], []
    for s in segs:
        st = s.get("segment_start", s.get("start"))
        en = s.get("segment_end", s.get("end"))
        if st is None:
            continue
        starts.append(float(st))
        if en is not None:
            ends.append(float(en))
    return starts, ends


def load_boundaries(path, duration=None):
    """Return (sorted boundary times, duration) from a segment JSON, a {boundaries,
    duration} dict, or a bare list of times. Boundaries are every segment START; the
    trivial ones at ~0 and ~duration are dropped later, in evaluate()."""
    data = json.load(open(path))
    if isinstance(data, list):
        b = sorted(float(x) for x in data)
        return b, (duration if duration is not None else (max(b) if b else 0.0))
    if data.get("segments"):
        starts, ends = _seg_bounds(data["segments"])
        dur = duration or (max(ends) if ends else (max(starts) if starts else 0.0))
        return sorted(starts), float(dur)
    b = sorted(float(x) for x in data.get("boundaries", []))
    dur = duration or data.get("duration") or (max(b) if b else 0.0)
    return b, float(dur)


def _internal(bounds, duration, step):
    """Real internal boundaries only: drop the trivial start/end (within one step of 0
    or duration) and collapse duplicates that round to the same step."""
    return sorted({b for b in bounds if step < b < duration - step})


# ---------------------------- boundary P / R / F1 -----------------------------
def match_boundaries(pred, gold, tol):
    """One-to-one match within `tol` seconds (greedy by increasing distance, which is
    order-independent and optimal for 1-1 on the line). Returns
    (tp, fp, fn, precision, recall, f1)."""
    pairs = sorted((abs(p - g), i, j)
                   for i, p in enumerate(pred) for j, g in enumerate(gold)
                   if abs(p - g) <= tol)
    used_p, used_g, tp = set(), set(), 0
    for _, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    fp, fn = len(pred) - tp, len(gold) - tp
    precision = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    recall = tp / len(gold) if gold else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


# ------------------------------ Pk / WindowDiff -------------------------------
def _boundary_units(bounds, step, n):
    """Boundary times -> sorted unique unit indices strictly inside (0, n)."""
    return sorted({u for u in (int(round(b / step)) for b in bounds) if 0 < u < n})


def _labels(bunits, n):
    """Per-unit segment id: two units share a segment iff no boundary lies between
    them. `bunits` is the sorted boundary-unit list."""
    labels, seg, j = [0] * n, 0, 0
    for u in range(n):
        while j < len(bunits) and bunits[j] <= u:
            seg += 1
            j += 1
        labels[u] = seg
    return labels


def pk(ref_lab, hyp_lab, k):
    """Beeferman Pk: P(the two ends of a width-k probe are mislabeled same/different).
    Lower is better; 0 is perfect."""
    n, err, cnt = len(ref_lab), 0, 0
    for i in range(n - k):
        err += (ref_lab[i] == ref_lab[i + k]) != (hyp_lab[i] == hyp_lab[i + k])
        cnt += 1
    return err / cnt if cnt else 0.0


def windowdiff(ref_units, hyp_units, n, k):
    """Pevzner & Hearst WindowDiff: fraction of width-k windows where ref and hyp
    disagree on the NUMBER of boundaries inside. Lower is better; 0 is perfect."""
    def cnt_in(arr, a, b):                      # boundaries in (a, b]
        return bisect.bisect_right(arr, b) - bisect.bisect_right(arr, a)
    err, cnt = 0, 0
    for i in range(n - k):
        err += cnt_in(ref_units, i, i + k) != cnt_in(hyp_units, i, i + k)
        cnt += 1
    return err / cnt if cnt else 0.0


# --------------------------------- evaluate -----------------------------------
def evaluate(pred_b, gold_b, duration, tols=(2.0, 5.0), step=1.0):
    """Full report dict for one (pred, gold) pair."""
    pred = _internal(pred_b, duration, step)
    gold = _internal(gold_b, duration, step)
    n = max(1, int(math.ceil(duration / step)))
    ru, hu = _boundary_units(gold, step, n), _boundary_units(pred, step, n)
    # Window size k = half the average reference segment length (the standard choice).
    k = max(1, round(n / (len(ru) + 1) / 2))
    res = {
        "n_pred": len(pred), "n_gold": len(gold), "duration": round(duration, 1),
        "k_units": k, "step": step,
        "pk": pk(_labels(ru, n), _labels(hu, n), k),
        "windowdiff": windowdiff(ru, hu, n, k),
        "tol": {},
    }
    for tol in tols:
        tp, fp, fn, p, r, f1 = match_boundaries(pred, gold, tol)
        res["tol"][tol] = {"tp": tp, "fp": fp, "fn": fn,
                           "precision": p, "recall": r, "f1": f1}
    return res


def format_report(res, name=""):
    head = f"=== {name} ===" if name else "==="
    lines = [head,
             f"  duration {res['duration']}s | pred boundaries {res['n_pred']} | "
             f"gold boundaries {res['n_gold']}",
             f"  Pk {res['pk']:.3f} | WindowDiff {res['windowdiff']:.3f} "
             f"(k={res['k_units']} units of {res['step']}s; lower is better)"]
    for tol, m in res["tol"].items():
        lines.append(f"  +/-{tol:g}s: P {m['precision']:.3f}  R {m['recall']:.3f}  "
                     f"F1 {m['f1']:.3f}   (tp {m['tp']}, fp {m['fp']}, fn {m['fn']})")
    return "\n".join(lines)


# ------------------------------ manifest (batch) ------------------------------
def evaluate_manifest(items, tols, step):
    """Per-file reports + a corpus aggregate: MICRO P/R/F1 (pool tp/fp/fn across all
    videos) and MACRO-mean Pk/WindowDiff."""
    reports, agg = [], {t: {"tp": 0, "fp": 0, "fn": 0} for t in tols}
    pks, wds = [], []
    for it in items:
        pb, dur = load_boundaries(it["pred"], it.get("duration"))
        gb, gdur = load_boundaries(it["gold"], it.get("duration"))
        res = evaluate(pb, gb, it.get("duration") or gdur or dur, tols, step)
        reports.append((it.get("name", it["pred"]), res))
        pks.append(res["pk"])
        wds.append(res["windowdiff"])
        for t in tols:
            for f in ("tp", "fp", "fn"):
                agg[t][f] += res["tol"][t][f]
    micro = {}
    for t in tols:
        tp, fp, fn = agg[t]["tp"], agg[t]["fp"], agg[t]["fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        micro[t] = {"precision": p, "recall": r,
                    "f1": 2 * p * r / (p + r) if p + r else 0.0,
                    "tp": tp, "fp": fp, "fn": fn}
    macro = {"pk": sum(pks) / len(pks) if pks else 0.0,
             "windowdiff": sum(wds) / len(wds) if wds else 0.0, "n": len(items)}
    return reports, micro, macro


def make_template(pred_path, out_path):
    """Copy a prediction into a gold template to hand-correct (keeps segment_start/end
    + title so you just fix the boundary times)."""
    data = json.load(open(pred_path))
    segs = data.get("segments", [])
    out = {"video": data.get("video", ""), "_note": "GROUND TRUTH -- fix the boundaries",
           "segments": [{"segment_start": s.get("segment_start", s.get("start")),
                         "segment_end": s.get("segment_end", s.get("end")),
                         "title": s.get("title", s.get("label", ""))} for s in segs]}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {len(segs)}-segment gold template to {out_path} -- correct the "
          f"segment_start times, then run --pred/--gold.", file=sys.stderr)


# ------------------------------------ main ------------------------------------
def main():
    p = argparse.ArgumentParser(description="Score segmentation vs a ground truth "
                                            "(boundary P/R/F1 + Pk/WindowDiff).")
    p.add_argument("--pred", help="Predicted segment JSON (or boundary list).")
    p.add_argument("--gold", help="Ground-truth segment JSON (or boundary list).")
    p.add_argument("--manifest", help="JSON list of {name, pred, gold[, duration]} to "
                                      "aggregate over a whole test set.")
    p.add_argument("--make-template", help="Write a gold template from this prediction.")
    p.add_argument("-o", "--output", help="Output path for --make-template.")
    p.add_argument("--tol", type=float, nargs="+", default=[2.0, 5.0],
                   help="Boundary match tolerance(s) in seconds (default: 2 5).")
    p.add_argument("--duration", type=float, default=None,
                   help="Override program duration (else inferred from segment ends).")
    p.add_argument("--step", type=float, default=1.0,
                   help="Discretization step for Pk/WindowDiff, seconds (default: 1).")
    p.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    args = p.parse_args()
    tols = tuple(args.tol)

    if args.make_template:
        if not args.output:
            p.error("--make-template needs -o/--output")
        make_template(args.make_template, args.output)
        return

    if args.manifest:
        items = json.load(open(args.manifest))
        reports, micro, macro = evaluate_manifest(items, tols, args.step)
        if args.json:
            print(json.dumps({"per_file": [{"name": n, **r} for n, r in reports],
                              "micro": micro, "macro": macro}, indent=2))
            return
        for name, res in reports:
            print(format_report(res, name))
        print(f"\n=== AGGREGATE over {macro['n']} videos ===")
        print(f"  MACRO Pk {macro['pk']:.3f} | MACRO WindowDiff {macro['windowdiff']:.3f}")
        for t in tols:
            m = micro[t]
            print(f"  MICRO +/-{t:g}s: P {m['precision']:.3f}  R {m['recall']:.3f}  "
                  f"F1 {m['f1']:.3f}   (tp {m['tp']}, fp {m['fp']}, fn {m['fn']})")
        return

    if not (args.pred and args.gold):
        p.error("provide --pred and --gold (or --manifest, or --make-template)")

    pb, pdur = load_boundaries(args.pred, args.duration)
    gb, gdur = load_boundaries(args.gold, args.duration)
    duration = args.duration or gdur or pdur
    res = evaluate(pb, gb, duration, tols, args.step)
    print(json.dumps(res, indent=2) if args.json else format_report(res))


if __name__ == "__main__":
    main()
