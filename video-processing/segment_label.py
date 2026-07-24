"""Stage C of the audio-fusion pipeline: label each segment's content type.

Two model passes, both via Amazon Bedrock (project AWS creds, no ANTHROPIC_API_KEY):

  Pass 1 -- analyze_program(): ONE cheap text call over the whole diarized
    transcript that returns, for THIS program:
      (a) taxonomy  -- 4-8 GENERAL, reusable content types (open vocabulary), so
          the pipeline is not hard-wired to one show. News gets anchor/report/
          weather/sports; a talk show gets monologue/interview/performance; etc.
      (b) boundaries -- timestamps where the STORY/TOPIC changes, INCLUDING shifts
          that happen while one person keeps talking (a host ending an intro and
          starting the first story). Speaker-continuity merging alone misses these.

  Pass 2 -- label_segments(): ONE MULTIMODAL call. For each segment the model
    sees BOTH its words AND a representative frame, and labels it against the
    discovered taxonomy. Words carry the talky segments (interview vs PSA vs
    monologue look identical on screen); the frame carries the little/no-speech
    ones (music/dance, animation, a title card), where audio is blind.

Boundaries from Pass 1 (a) split long single-speaker segments and (b) stop Stage-D
from coalescing two adjacent same-type segments across a real topic change.

This account currently only has Haiku 4.5 enabled in Bedrock; request a stronger
model and bump BEDROCK_MODEL (bedrock.py) if labels/taxonomy are weak.

Usage (same AWS env as transcribe.py):
    python segment_label.py shots_fused.json transcript.json --video video.mp4 -o segments_labeled.json
    ... --no-discover               # use the built-in fallback taxonomy
    ... --categories "a,b,c"        # force a fixed taxonomy
    ... --no-topics                 # skip topic-shift splitting (speaker-merge only)
"""

import argparse
import json
import sys

import cv2

from segment_fuse import merge_by_speaker
from segment_content import _read_at, _encode_frame   # reuse the frame samplers
from bedrock import make_client, BEDROCK_MODEL, BEDROCK_REGION

# Fallback taxonomy when discovery is skipped/fails. Discovery replaces this
# per-video; it is intentionally generic, not tied to any one show.
DEFAULT_TAXONOMY = [
    {"type": "host segment", "definition": "host/anchor alone to camera or narrating over footage (intro, outro, links)"},
    {"type": "interview", "definition": "host in question-and-answer with a guest"},
    {"type": "field report", "definition": "a reported/produced story package on location"},
    {"type": "announcement", "definition": "a scripted public-service, advertising, or advocacy message"},
    {"type": "performance", "definition": "music, dance, or a stage performance with little speech"},
]

# Small generic synonym net so a spelled-out label still maps to a short type.
SYNONYMS = {"public service announcement": "announcement", "psa": "announcement",
            "commercial": "announcement", "advertisement": "announcement",
            "q&a": "interview", "music": "performance", "dance": "performance"}

# {host} = auto-detected host speaker. Deliberately show-agnostic.
PROMPT_HEADER = (
    "You are segmenting one video program into content-type segments. Below are "
    "consecutive segments in time order. For EACH you get the speakers present "
    "(diarization labels; the host/narrator is usually speaker {host}), the words "
    "spoken over it (from a transcript; \"(no speech)\" means music/no talking), and "
    "ONE representative video frame. Use BOTH signals: let the WORDS decide segments "
    "where people are talking (types that look identical on screen -- an interview, a "
    "monologue, a scripted announcement -- are told apart by what is said), and let "
    "the FRAME decide segments with little or no speech (music/performance, an "
    "animation, a title/name card, filler or leader footage), where audio is blind."
)

ANALYZE_PROMPT = (
    "Here is the full diarized transcript of ONE video program -- consecutive turns "
    "as [start_seconds] speaker: words.\n\n"
    "Do TWO things:\n"
    "1) taxonomy: propose 4-8 GENERAL, reusable CONTENT TYPES that would segment a "
    "program like this. Examples -- news: anchor segment, field report, weather, "
    "sports, interview, commercial; talk show: host monologue, interview, musical "
    "performance; documentary: narration, interview, archival footage. Use generic "
    "type names that also apply to OTHER episodes, NOT this episode's specific topics. "
    "Give each a one-line definition.\n"
    "2) boundaries: list approximate timestamps (seconds) where the STORY or TOPIC "
    "changes -- INCLUDING changes that happen while the same person keeps talking. "
    "Use the turn start times.\n\n"
    "Reply with ONLY this JSON:\n"
    '{"taxonomy": [{"type": "...", "definition": "..."}], "boundaries": [12.3, 45.6]}\n\n'
    "Transcript:\n{transcript}"
)


def detect_host(turns) -> str:
    """Host/narrator = most-talkative speaker (most total seconds). Show-agnostic."""
    dur = {}
    for t in turns:
        dur[t["speaker"]] = dur.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    return max(dur, key=dur.get) if dur else "spk_0"


def analyze_program(turns, client, model=BEDROCK_MODEL, max_types=8):
    """Pass 1: one text call -> (taxonomy, topic-boundary times) for this program."""
    lines = [f"[{t['start']:.0f}] {t['speaker']}: {t['text']}" for t in turns]
    prompt = ANALYZE_PROMPT.replace("{transcript}", "\n".join(lines))
    resp = client.messages.create(model=model, max_tokens=1200,
                                  messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in resp.content if b.type == "text")
    a, b = text.find("{"), text.rfind("}")
    try:
        obj = json.loads(text[a:b + 1])
    except Exception:
        obj = {}
    tax = [{"type": str(t.get("type", "")).strip(),
            "definition": str(t.get("definition", "")).strip()}
           for t in (obj.get("taxonomy") or []) if t.get("type")][:max_types]
    bounds = sorted(float(x) for x in (obj.get("boundaries") or [])
                    if isinstance(x, (int, float)))
    return (tax or DEFAULT_TAXONOMY), bounds


def segment_text(turns, start, end) -> tuple:
    """Words spoken during [start,end], plus the speakers, from transcript turns."""
    parts, speakers = [], []
    for t in turns:
        if min(t["end"], end) - max(t["start"], start) > 0:
            parts.append(t["text"])
            if t["speaker"] not in speakers:
                speakers.append(t["speaker"])
    return " ".join(parts).strip(), speakers


def split_on_topics(segments, boundaries, shots, min_piece=8.0, snap=3.0) -> list:
    """Split each speaker-continuity segment at topic boundaries that fall inside
    it. A boundary is snapped to a nearby shot cut when one is within `snap`
    seconds (keeps visual alignment), else used as-is (topic shifts in a monologue
    have no visual cut). Sub-pieces after the first are marked topic_start so
    Stage-D won't merge across them even if they share a label."""
    shot_starts = sorted({s["start"] for s in shots})
    out = []
    for seg in segments:
        cuts = []
        for b in boundaries:
            if seg["start"] + min_piece <= b <= seg["end"] - min_piece:
                near = [ss for ss in shot_starts
                        if seg["start"] < ss < seg["end"] and abs(ss - b) <= snap]
                cuts.append(round(min(near, key=lambda x: abs(x - b)) if near else b, 2))
        pts = [seg["start"]] + sorted(set(cuts)) + [seg["end"]]
        for k in range(len(pts) - 1):
            a, c = pts[k], pts[k + 1]
            if c - a < min_piece and out:          # too short -> extend previous
                out[-1]["end"] = c
                continue
            out.append({**seg, "start": a, "end": c, "topic_start": k > 0})
    return out


def canonical(label, types) -> str:
    """Map the model's free-text label onto the discovered type set, else 'other'."""
    r = (label or "").strip().lower()
    for c in types:
        if c.lower() in r or r in c.lower():
            return c
    for word, c in SYNONYMS.items():
        if word in r and c in types:
            return c
    return "other"


def _frame(cap, start, end, max_width) -> str:
    """A representative JPEG for a segment: the frame at its midpoint ('' on fail)."""
    if cap is None:
        return ""
    bgr = _read_at(cap, start + (end - start) / 2.0)
    return _encode_frame(bgr, max_width) if bgr is not None else ""


def build_task(taxonomy) -> str:
    """The Pass-2 instruction, built from the discovered taxonomy."""
    lines = "\n".join(f"- {t['type']}: {t['definition']}" for t in taxonomy)
    return ("\n\nClassify EACH segment into exactly one content type:\n" + lines +
            "\n- other: none of the above\n"
            "Then give a specific 3-6 word name.\n"
            "Reply with ONLY a JSON array, one object per segment IN ORDER:\n"
            '[{"i": 0, "label": "<one of the types>", "name": "<short name>"}, ...]')


def label_segments(segments, client, taxonomy, video_path=None, model=BEDROCK_MODEL,
                   max_width=512, host="spk_0") -> list:
    """Pass 2: one MULTIMODAL call -- each segment's words + a frame -> label + name,
    against the discovered taxonomy. Falls back to text-only if the video can't open."""
    types = [t["type"] for t in taxonomy]
    cap = cv2.VideoCapture(video_path) if video_path else None
    content = [{"type": "text", "text": PROMPT_HEADER.format(host=host)}]
    for i, s in enumerate(segments):
        spk = ",".join(s["speakers"]) or "-"
        txt = s["text"][:600] or "(no speech)"
        content.append({"type": "text",
                        "text": f"\nSegment {i}  [{s['start']:.0f}-{s['end']:.0f}s]  "
                                f"speakers={spk}\nWords: {txt}\nFrame:"})
        jpeg = _frame(cap, s["start"], s["end"], max_width)
        if jpeg:
            content.append({"type": "image", "source": {"type": "base64",
                            "media_type": "image/jpeg", "data": jpeg}})
    if cap is not None:
        cap.release()
    content.append({"type": "text", "text": build_task(taxonomy)})

    resp = client.messages.create(model=model, max_tokens=1600,
                                  messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    a, b = text.find("["), text.rfind("]")
    parsed = json.loads(text[a:b + 1]) if a >= 0 else []
    by_i = {int(o.get("i", k)): o for k, o in enumerate(parsed)}
    out = []
    for i, s in enumerate(segments):
        o = by_i.get(i, {})
        out.append({**s, "label": canonical(o.get("label"), types), "name": o.get("name", "").strip()})
    return out


def coalesce_labeled(labeled) -> list:
    """Stage D: merge ALL adjacent same-label segments into one content-type span,
    keeping the longest child's name. This is a CONTENT-TYPE view -- a run of
    consecutive "Interview Segment" shots becomes one interview, not one per
    sub-topic. (topic_start from split_on_topics only survives when the split
    produced a DIFFERENT label, e.g. a PSA embedded in a host segment; same-label
    topic splits re-merge here.)"""
    out = []
    for s in labeled:
        if out and out[-1]["label"] == s["label"]:
            if (s["end"] - s["start"]) > (out[-1]["end"] - out[-1]["start"]):
                out[-1]["name"] = s["name"]
            out[-1]["end"] = s["end"]
            for spk in s["speakers"]:
                if spk not in out[-1]["speakers"]:
                    out[-1]["speakers"].append(spk)
        else:
            out.append({**s, "speakers": list(s["speakers"])})
    return out


def main():
    p = argparse.ArgumentParser(description="Label each fused segment's content type (Stage C).")
    p.add_argument("shots_fused", help="shots_fused.json from segment_fuse.py")
    p.add_argument("transcript", help="transcript.json from transcribe.py")
    p.add_argument("-o", "--output", default="segments_labeled.json")
    p.add_argument("--host", default=None,
                   help="Host/narrator speaker label (default: auto = most-talkative speaker).")
    p.add_argument("--video", default=None,
                   help="Video for the per-segment frame (default: 'video' field in the fused JSON).")
    p.add_argument("--no-frames", action="store_true", help="Text-only labels (no frames).")
    p.add_argument("--no-discover", action="store_true",
                   help="Use the built-in fallback taxonomy instead of discovering one.")
    p.add_argument("--categories", default=None,
                   help="Force a fixed comma-separated taxonomy (overrides discovery).")
    p.add_argument("--no-topics", action="store_true",
                   help="Skip topic-shift splitting (speaker-continuity merge only).")
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--model", default=BEDROCK_MODEL)
    p.add_argument("--region", default=BEDROCK_REGION)
    args = p.parse_args()

    fused = json.load(open(args.shots_fused))
    shots = fused["shots"] if isinstance(fused, dict) else fused
    turns = json.load(open(args.transcript))
    video = None if args.no_frames else (args.video or (fused.get("video") if isinstance(fused, dict) else None))
    host = args.host or detect_host(turns)
    client = make_client(args.region)

    # Pass 1: discover taxonomy + topic boundaries (one call, unless both are off).
    taxonomy, boundaries = DEFAULT_TAXONOMY, []
    if not (args.no_discover and args.no_topics) and not args.categories:
        taxonomy, boundaries = analyze_program(turns, client, args.model)
    if args.categories:
        taxonomy = [{"type": c.strip(), "definition": ""} for c in args.categories.split(",")]
        if not args.no_topics:
            _, boundaries = analyze_program(turns, client, args.model)
    if args.no_discover and not args.categories:
        taxonomy = DEFAULT_TAXONOMY
    if args.no_topics:
        boundaries = []

    print(f"host/narrator = {host}", file=sys.stderr)
    print(f"discovered taxonomy ({len(taxonomy)}): "
          + ", ".join(t["type"] for t in taxonomy), file=sys.stderr)
    print(f"topic boundaries: {len(boundaries)}", file=sys.stderr)

    segments = merge_by_speaker(shots, host=host)
    segments = split_on_topics(segments, boundaries, shots)
    for s in segments:                       # attach clean per-segment transcript text
        s["text"], s["speakers"] = segment_text(turns, s["start"], s["end"])

    labeled = label_segments(segments, client, taxonomy, video, args.model, args.max_width, host)
    labeled = coalesce_labeled(labeled)

    print(f"\n{len(shots)} shots -> {len(labeled)} labeled segments:\n", file=sys.stderr)
    for s in labeled:
        name = f"  ({s['name']})" if s["name"] else ""
        print(f"  [{s['start']:7.1f}->{s['end']:7.1f}] ({s['end']-s['start']:6.1f}s) "
              f"{s['label']:20s}{name}")

    out = {"video": fused.get("video") if isinstance(fused, dict) else None,
           "host": host,
           "taxonomy": taxonomy,
           "segments": [{"start": round(s["start"], 2), "end": round(s["end"], 2),
                         "label": s["label"], "name": s["name"],
                         "speakers": s["speakers"]} for s in labeled]}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(labeled)} segments to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
