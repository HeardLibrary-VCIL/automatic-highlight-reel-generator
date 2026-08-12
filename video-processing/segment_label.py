"""Stage C/D of the audio-fusion pipeline: split the program into stories and label
each one's content type.

Boundaries come from three cheap cues, unioned in main.py:
  - analyze_program()  -- ONE Bedrock text call over the diarized transcript that
    returns (a) a 4-8 type taxonomy discovered for THIS program (open vocabulary,
    so nothing is hard-wired to one show) and (b) the story start times, including
    shifts that happen while one person keeps talking.
  - lexical_boundaries() -- TextTiling valleys in the transcript, to catch a story
    boundary that falls mid-shot (a host wrap-up rolling into a PSA, no camera cut).
  - pause_boundaries()   -- long silences between turns (a broadcast segment break).

Segments are split on those boundaries, each piece's host/guest role is recomputed
from who actually speaks (dominant_role), then:
  - label_segments()  -- ONE MULTIMODAL Bedrock call. Per segment the model sees the
    words AND a frame sampled near the start, and returns label + name + any on-screen
    caption it can read (a name lower-third / title card the transcript can't give).
    Words decide the talky types (interview vs PSA look identical on screen); the
    frame decides the little/no-speech ones (music, a title card).
  - coalesce_conversation() -- Stage D: merge by WHO is talking, so one interview
    stays one segment across camera cuts instead of one piece per backdrop.
  - boundary_confidence()   -- tag each final boundary by how many cues agree, for
    reviewer triage.

This account currently only has Haiku 4.5 enabled in Bedrock; request a stronger
model and bump BEDROCK_MODEL (bedrock.py) if labels/taxonomy are weak.

Usage (same AWS env as transcribe.py):
    python segment_label.py shots_fused.json transcript.json --video video.mp4 -o segments_labeled.json
    ... --no-discover               # use the built-in fallback taxonomy
    ... --categories "a,b,c"        # force a fixed taxonomy
    ... --no-topics                 # skip boundary splitting (speaker-merge only)
"""

import argparse
import json
import math
import re
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

# --- Scaling knobs: keep long programs inside each request's budget ---
# Pass 2 labels in batches of LABEL_BATCH segments/images per call, so a long program
# pages through several bounded calls instead of one oversized request whose reply
# would truncate away the later segments.
LABEL_BATCH = 8
# Pass 1 (analyze_program) splits a long transcript into OVERLAPPING windows, discovers
# taxonomy + boundaries per window, then stitches the results. A program that fits one
# window's budget is the plain single call (the ~24k-char test tapes stay single-window,
# so their behaviour is unchanged); only longer programs page through several windows.
ANALYZE_WINDOW_CHARS = 32000   # ~8k tokens of transcript text per window
ANALYZE_OVERLAP_FRAC = 0.2     # 20% overlap between adjacent windows
ANALYZE_EDGE_TURNS = 4         # drop a window's boundaries within this many turns of its
                               # own edge -- context-starved artifacts (incl. the "first
                               # segment start" every window emits); the overlap means a
                               # neighbouring window covers that region in ITS interior
ANALYZE_MERGE_TOL = 6.0        # s; cluster near-duplicate boundaries from the overlaps

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
    "2) boundaries: the start_seconds where each STORY/SEGMENT a human editor would "
    "mark begins. Start a NEW segment when the CONTENT PURPOSE changes, e.g.:\n"
    "   - a host/anchor finishes wrapping up one story (thanks a guest, signs off) and "
    "a DIFFERENT story or a scripted PSA/ad/dramatization begins -- EVEN IF the same "
    "host keeps talking and there is no camera cut;\n"
    "   - a scripted PSA / public-service dramatization starts or ends;\n"
    "   - the program moves to a new location, guest, or event.\n"
    "   Do NOT start a new segment for mere sub-topics inside ONE continuous interview "
    "or tour with the SAME guest (the same person showing different rooms is one "
    "segment). Use the turn start times; include the first content segment's start.\n\n"
    "Reply with ONLY this JSON:\n"
    '{"taxonomy": [{"type": "...", "definition": "..."}], "boundaries": [12.3, 45.6]}\n\n'
    "Transcript:\n{transcript}"
)


def detect_host(turns, bins=10) -> str:
    """Host/anchor = the speaker who INTERVENES MOST OFTEN and RECURS across the whole
    program -- not merely the one who talks the longest.

    Most total speaking time is a bad proxy: a guest in a long interview, or a scripted
    PSA monologue, can dominate the seconds while taking very few turns confined to one
    stretch. The broadcast-news role cue is turn COUNT (the anchor has the most
    interventions) weighted by temporal spread (the anchor recurs across the timeline; a
    guest is bunched in one block). Score each speaker by turns x fraction-of-program it
    appears in; ties fall back to raw turn count, then to a stable spk_0."""
    if not turns:
        return "spk_0"
    t0 = min(t["start"] for t in turns)
    span = max(max(t["end"] for t in turns) - t0, 1e-9)
    n_turns, seen = {}, {}
    for t in turns:
        spk = t["speaker"]
        n_turns[spk] = n_turns.get(spk, 0) + 1
        b = min(int(bins * (t["start"] - t0) / span), bins - 1)   # which time-bin this turn falls in
        seen.setdefault(spk, set()).add(b)
    return max(n_turns, key=lambda s: (n_turns[s] * len(seen[s]) / bins, n_turns[s]))


def _turn_windows(turns, window_chars=ANALYZE_WINDOW_CHARS,
                  overlap_frac=ANALYZE_OVERLAP_FRAC) -> list:
    """Slice `turns` into OVERLAPPING (i0, i1) index ranges, each ~window_chars of
    transcript text with ~overlap_frac overlap. A transcript that fits one window
    returns a single [(0, n)] range -- the plain single-call path.

    The overlap is what makes stitching sound: a boundary that lands near one window's
    edge (dropped there for want of context) sits in the NEXT window's trusted interior,
    so it is still recovered once, with full context on both sides."""
    n = len(turns)
    sizes = [len(t.get("text", "")) + 20 for t in turns]   # +~20 for the "[s] spk: " overhead
    if n <= 1 or sum(sizes) <= window_chars:
        return [(0, n)]
    overlap_chars = window_chars * overlap_frac
    windows, i = [], 0
    while i < n:
        j, acc = i, 0
        while j < n and acc < window_chars:
            acc += sizes[j]
            j += 1
        windows.append((i, j))
        if j >= n:
            break
        back, k = 0, j
        while k > i + 1 and back < overlap_chars:   # step start back by ~overlap_chars
            k -= 1
            back += sizes[k]
        i = k                                        # k > i guaranteed, so we always advance
    return windows


def _window_interior(turns, i0, i1, edge, is_first, is_last) -> tuple:
    """Time range of a window's TRUSTED interior. Boundaries outside it are context-
    starved edge artifacts (the first window keeps the real program start; the last
    keeps the true end); the overlap means a neighbour owns those edge regions."""
    lo = turns[i0]["start"] if is_first else turns[min(i0 + edge, i1 - 1)]["start"]
    hi = turns[i1 - 1]["end"] if is_last else turns[max(i1 - 1 - edge, i0)]["start"]
    return lo, hi


def _stitch_boundaries(cands, turns, tol=ANALYZE_MERGE_TOL) -> list:
    """Merge boundary candidates gathered from all windows: cluster values within `tol`
    seconds (the same true boundary reported by two overlapping windows), snap each
    cluster to the nearest turn start, and return sorted uniques."""
    if not cands:
        return []
    starts = sorted({t["start"] for t in turns})
    snap = (lambda b: min(starts, key=lambda s: abs(s - b))) if starts else (lambda b: b)
    clusters = []
    for b in sorted(cands):
        if clusters and b - clusters[-1][-1] <= tol:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    out = []
    for cl in clusters:
        rep = snap(cl[len(cl) // 2])                 # snap the cluster median to a turn start
        if not out or rep - out[-1] > tol:
            out.append(rep)
    return [round(x, 2) for x in out]


def _merge_taxonomies(tax_lists, max_types=8) -> list:
    """Fold the per-window taxonomies into one 4-8 type set: near-duplicate type names
    (substring either way, e.g. 'interview' vs 'guest interview') collapse together and
    the most frequently proposed types win, so the whole program shares one vocabulary."""
    seen, order = {}, []
    for t in tax_lists:
        name = str(t.get("type", "")).strip()
        if not name:
            continue
        key = name.lower()
        match = next((k for k in seen if k in key or key in k), None)
        if match:
            seen[match]["count"] += 1
        else:
            seen[key] = {"type": name, "definition": str(t.get("definition", "")).strip(),
                         "count": 1}
            order.append(key)
    ranked = sorted(order, key=lambda k: -seen[k]["count"])   # stable: ties keep first-seen order
    return [{"type": seen[k]["type"], "definition": seen[k]["definition"]}
            for k in ranked[:max_types]]


def _analyze_window(turn_slice, client, model, max_types) -> tuple:
    """One text call over a slice of turns -> (taxonomy, boundary times). The turn start
    times are kept verbatim, so every boundary is an absolute program timestamp."""
    lines = [f"[{t['start']:.0f}] {t['speaker']}: {t['text']}" for t in turn_slice]
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
    return tax, bounds


def analyze_program(turns, client, model=BEDROCK_MODEL, max_types=8):
    """Pass 1: discover a per-program taxonomy + story-boundary times.

    A long transcript is split into OVERLAPPING windows -- one call can't reliably
    reason over an hour of turns, and truncating each turn (the old stopgap) loses the
    content the boundaries are read from. Each window returns its own taxonomy +
    boundaries; the boundaries are then STITCHED (context-starved window-edge candidates
    dropped -- including the "first segment start" every window emits -- and the
    near-duplicates from the overlaps merged and snapped to turn starts), and the
    per-window taxonomies folded into one 4-8 type set. A program under one window's
    budget takes exactly the old single call, so short programs are unchanged."""
    windows = _turn_windows(turns)
    if len(windows) <= 1:
        tax, bounds = _analyze_window(turns, client, model, max_types)
        return (tax or DEFAULT_TAXONOMY), bounds

    cands, tax_all, last = [], [], len(windows) - 1
    for w, (i0, i1) in enumerate(windows):
        tax_w, bounds_w = _analyze_window(turns[i0:i1], client, model, max_types)
        tax_all.extend(tax_w)
        lo, hi = _window_interior(turns, i0, i1, ANALYZE_EDGE_TURNS,
                                  is_first=(w == 0), is_last=(w == last))
        cands.extend(b for b in bounds_w if lo <= b <= hi)
    bounds = _stitch_boundaries(cands, turns)
    return (_merge_taxonomies(tax_all, max_types) or DEFAULT_TAXONOMY), bounds


def segment_text(turns, start, end) -> tuple:
    """Words spoken during [start,end], plus the speakers, from transcript turns."""
    parts, speakers = [], []
    for t in turns:
        if min(t["end"], end) - max(t["start"], start) > 0:
            parts.append(t["text"])
            if t["speaker"] not in speakers:
                speakers.append(t["speaker"])
    return " ".join(parts).strip(), speakers


# Function words dropped before measuring lexical cohesion (content words carry topic).
_STOP = frozenset((
    "the a an and or but of to in on at for with as is are was were be been being it its "
    "this that these those i you he she they we me my your his her their our us him them "
    "there here what which who how when where why not no yes do does did have has had will "
    "would can could should may might must so if then out up down into about uh um okay ok "
    "well like know get got go going come one two").split())


def lexical_boundaries(turns, W=20, K=6, c=0.3):
    """TextTiling-style boundary candidates from vocabulary shift in the transcript.

    News story boundaries frequently fall in the MIDDLE of a shot (a host wraps up one
    story and starts the next without a camera cut), so shot cuts alone miss them. This
    is the classic text signal: split the diarized words into fixed pseudo-sentences,
    score each gap by the cosine similarity of the content-word histograms of the K
    blocks on either side, and return the timestamps of the deep similarity valleys
    (depth above mean + c*std). These are only CANDIDATES -- lexical cohesion over-
    segments spoken transcripts, so the speaker-continuity coalesce is what removes the
    noise; here we just make sure real mid-shot boundaries are on the table.
    """
    toks = []                                       # (timestamp, content word)
    for t in turns:
        ws = [w for w in re.findall(r"[a-z']+", t["text"].lower())
              if w not in _STOP and len(w) > 2]
        dur = max(t["end"] - t["start"], 0.01)
        for i, w in enumerate(ws):
            toks.append((t["start"] + dur * i / max(len(ws), 1), w))
    ps = [toks[i:i + W] for i in range(0, len(toks), W)]     # pseudo-sentences
    words = [[w for _, w in blk] for blk in ps]
    if len(ps) < 2 * K:
        return []

    def sim(i0, i1, j0, j1):
        a, b = {}, {}
        for k in range(i0, i1):
            for w in words[k]:
                a[w] = a.get(w, 0) + 1
        for k in range(j0, j1):
            for w in words[k]:
                b[w] = b.get(w, 0) + 1
        common = set(a) & set(b)
        da = math.sqrt(sum(v * v for v in a.values()))
        db = math.sqrt(sum(v * v for v in b.values()))
        return sum(a[w] * b[w] for w in common) / (da * db) if da and db else 0.0

    gaps = [(ps[g][0][0], sim(max(0, g - K), g, g, min(len(ps), g + K)))
            for g in range(1, len(ps))]
    sims = [s for _, s in gaps]
    depths = []
    for i, (ts, s) in enumerate(gaps):
        lp = s; j = i
        while j > 0 and sims[j - 1] >= sims[j]:
            j -= 1; lp = max(lp, sims[j])
        rp = s; j = i
        while j < len(sims) - 1 and sims[j + 1] >= sims[j]:
            j += 1; rp = max(rp, sims[j])
        depths.append((ts, (lp - s) + (rp - s)))
    dv = [d for _, d in depths]
    mean = sum(dv) / len(dv)
    std = (sum((x - mean) ** 2 for x in dv) / len(dv)) ** 0.5
    return [round(ts, 2) for i, (ts, d) in enumerate(depths)
            if d > mean + c * std and d == max(x[1] for x in depths[max(0, i - 1):i + 2])]


def dominant_role(turns, start, end, host, min_guest=6.0):
    """Role of [start,end] by who actually speaks the most: 'host' unless a non-host
    holds a substantial, comparable share of the speech. Recomputed AFTER boundary
    splitting, so a host wrap-up sliced off the front of a guest segment is re-tagged
    'host' and is not merged back into that guest's story by coalesce_conversation."""
    secs = {}
    for t in turns:
        ov = min(t["end"], end) - max(t["start"], start)
        if ov > 0:
            secs[t["speaker"]] = secs.get(t["speaker"], 0.0) + ov
    host_s = secs.get(host, 0.0)
    guest_s = max([v for k, v in secs.items() if k != host], default=0.0)
    return "guest" if guest_s >= max(min_guest, 0.4 * host_s) else "host"


def segment_guests(turns, start, end, host, min_secs=1.5):
    """Non-host speakers with at least `min_secs` of speech in [start,end]. The floor
    drops a fraction-of-a-second sliver of the NEXT speaker that a segment straddling a
    boundary picks up -- otherwise that sliver makes the segment share a guest with the
    following conversation and coalesce_conversation chains two distinct stories."""
    secs = {}
    for t in turns:
        ov = min(t["end"], end) - max(t["start"], start)
        if ov > 0 and t["speaker"] != host:
            secs[t["speaker"]] = secs.get(t["speaker"], 0.0) + ov
    return [s for s, v in secs.items() if v >= min_secs]


def pause_boundaries(turns, min_gap=1.5):
    """Boundary candidates at long SILENCES between speech turns. Broadcast segments
    are usually separated by a beat of silence (a pause or a stinger), so a gap of
    >= min_gap seconds is a cheap acoustic cue -- data we already have from the ASR
    timestamps, an unused signal. Returns the gap midpoints."""
    ts = sorted(turns, key=lambda t: t["start"])
    return [round((ts[i]["end"] + ts[i + 1]["start"]) / 2.0, 2)
            for i in range(len(ts) - 1)
            if ts[i + 1]["start"] - ts[i]["end"] >= min_gap]


def _near(t, points, tol):
    return any(abs(t - p) <= tol for p in points)


def boundary_confidence(segments, turns, shot_starts, model_bounds, lex_bounds,
                        host, tol=2.0, pause_min=0.8):
    """Annotate each segment's START with how many INDEPENDENT cues support it, so a
    reviewer can trust high-agreement cuts and scrutinize the shaky ones. Cues (all
    from signals we already computed): a shot cut, a speaker change, a silence pause,
    a lexical-cohesion valley, and the LLM boundary. Sets seg['cues'] (list) and
    seg['confidence'] ('high' >=3, 'medium' 2, 'low' <=1); the very first segment has
    no boundary before it and is left as 'high' by convention."""
    ts = sorted(turns, key=lambda t: t["start"])

    def speaker_change(t):
        before = [x for x in ts if x["end"] <= t + tol]
        after = [x for x in ts if x["start"] >= t - tol]
        return bool(before and after and before[-1]["speaker"] != after[0]["speaker"])

    def pause_at(t):
        for i in range(len(ts) - 1):
            gap = ts[i + 1]["start"] - ts[i]["end"]
            if gap >= pause_min and ts[i]["end"] - tol <= t <= ts[i + 1]["start"] + tol:
                return True
        return False

    for k, seg in enumerate(segments):
        if k == 0:
            seg["cues"], seg["confidence"] = [], "high"
            continue
        t = seg["start"]
        cues = []
        if _near(t, shot_starts, tol):
            cues.append("shot")
        if speaker_change(t):
            cues.append("speaker")
        if pause_at(t):
            cues.append("pause")
        if _near(t, lex_bounds, tol):
            cues.append("lexical")
        if _near(t, model_bounds, tol):
            cues.append("llm")
        seg["cues"] = cues
        seg["confidence"] = "high" if len(cues) >= 3 else "medium" if len(cues) == 2 else "low"
    return segments


def split_on_topics(segments, boundaries, shots, min_piece=8.0, snap=3.0) -> list:
    """Split each speaker-continuity segment at boundaries that fall inside it. A
    boundary is snapped to a nearby shot cut when one is within `snap` seconds (keeps
    the edit on a visual cut), else used as-is -- a topic shift in a monologue has no
    cut, which is exactly the mid-shot boundary lexical/pause candidates recover."""
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
            out.append({**seg, "start": a, "end": c})
    return out


def canonical(label, types) -> str:
    """Map the model's free-text label onto the discovered type set, else 'other'."""
    r = (label or "").strip().lower()
    if not r:
        # An empty label must NOT match: `"" in c` is always true and would silently
        # pin a blank/omitted segment to the first taxonomy type. Treat it as 'other'.
        return "other"
    for c in types:
        if c.lower() in r or r in c.lower():
            return c
    for word, c in SYNONYMS.items():
        if word in r and c in types:
            return c
    return "other"


def _frame(cap, start, end, max_width) -> str:
    """A JPEG sampled a few seconds INTO the segment ('' on fail). Name lower-thirds/
    chyrons are shown briefly (~3-7s) when a person is first introduced and then
    removed, so a midpoint frame misses them; a frame ~2-4s in (past the opening
    cut, while the caption is still up) catches the name/title AND still shows the
    on-screen subject for content classification."""
    if cap is None:
        return ""
    bgr = _read_at(cap, start + min(4.0, (end - start) / 3.0))
    return _encode_frame(bgr, max_width) if bgr is not None else ""


def build_task(taxonomy) -> str:
    """The Pass-2 instruction, built from the discovered taxonomy."""
    lines = "\n".join(f"- {t['type']}: {t['definition']}" for t in taxonomy)
    return ("\n\nClassify EACH segment into exactly one content type:\n" + lines +
            "\n- other: none of the above\n"
            "Then give a specific 3-6 word name.\n"
            "Also transcribe any ON-SCREEN TEXT visible in the frame -- a name/title "
            "caption (lower-third/chyron), a title card, or credits -- EXACTLY as shown; "
            'use "" if the frame has no readable overlaid text. This caption often gives '
            "the real speaker name or segment title, so prefer it when naming.\n"
            "Reply with ONLY a JSON array, one object per segment IN ORDER:\n"
            '[{"i": 0, "label": "<one of the types>", "name": "<short name>", '
            '"on_screen_text": "<caption or empty>"}, ...]')


def _parse_label_array(text):
    """Salvage the label objects from the model's reply. Tries a clean parse first,
    then falls back to scanning for complete top-level {...} objects so a reply that
    got truncated mid-array (max_tokens) still yields every FINISHED segment instead
    of losing all of them."""
    a, b = text.find("["), text.rfind("]")
    if a >= 0 and b > a:
        try:
            return json.loads(text[a:b + 1])
        except Exception:
            pass
    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objs.append(json.loads(text[start:i + 1]))
                except Exception:
                    pass
                start = None
    return objs


def _label_batch(batch, client, taxonomy, cap, model, max_width, host) -> list:
    """One MULTIMODAL Bedrock call over a slice of segments -> label dicts aligned to
    `batch` order (each segment's words + a frame -> label + name + on-screen text).
    Segments the reply omits (e.g. a truncated array) default to 'other'."""
    types = [t["type"] for t in taxonomy]
    content = [{"type": "text", "text": PROMPT_HEADER.format(host=host)}]
    for i, s in enumerate(batch):
        spk = ",".join(s["speakers"]) or "-"
        txt = s["text"][:600] or "(no speech)"
        content.append({"type": "text",
                        "text": f"\nSegment {i}  [{s['start']:.0f}-{s['end']:.0f}s]  "
                                f"speakers={spk}\nWords: {txt}\nFrame:"})
        jpeg = _frame(cap, s["start"], s["end"], max_width)
        if jpeg:
            content.append({"type": "image", "source": {"type": "base64",
                            "media_type": "image/jpeg", "data": jpeg}})
    content.append({"type": "text", "text": build_task(taxonomy)})

    resp = client.messages.create(model=model, max_tokens=4096,
                                  messages=[{"role": "user", "content": content}])
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    by_i = {int(o.get("i", k)): o for k, o in enumerate(_parse_label_array(text))}
    out = []
    for i, s in enumerate(batch):
        o = by_i.get(i, {})
        out.append({**s, "label": canonical(o.get("label"), types),
                    "name": o.get("name", "").strip(),
                    "on_screen_text": str(o.get("on_screen_text", "")).strip()})
    return out


def label_segments(segments, client, taxonomy, video_path=None, model=BEDROCK_MODEL,
                   max_width=512, host="spk_0", batch_size=LABEL_BATCH) -> list:
    """Pass 2: MULTIMODAL labeling of each segment (words + a frame) against the
    discovered taxonomy. Batched into `batch_size`-segment calls so a long program
    doesn't overflow one request's image/token budget or truncate the reply into
    losing its later segments; falls back to text-only when the video can't open. A
    batch that errors is retried once, then left with default labels so one bad call
    can't sink the rest of the timeline."""
    types = [t["type"] for t in taxonomy]
    cap = cv2.VideoCapture(video_path) if video_path else None
    out = []
    try:
        for start in range(0, len(segments), batch_size):
            batch = segments[start:start + batch_size]
            try:
                out.extend(_label_batch(batch, client, taxonomy, cap, model, max_width, host))
            except Exception:
                try:
                    out.extend(_label_batch(batch, client, taxonomy, cap, model, max_width, host))
                except Exception:
                    out.extend({**s, "label": canonical(None, types), "name": "",
                                "on_screen_text": ""} for s in batch)
    finally:
        if cap is not None:
            cap.release()
    return out


def coalesce_conversation(labeled, host="spk_0", segue_max=20.0) -> list:
    """Stage D (continuity view): merge segments into stories by WHO is talking,
    not by the per-shot visual label.

    A continuous interview/tour with one guest is ONE story even when the backdrop
    -- and therefore the frame-driven label -- keeps changing. (Without this a
    single interview gets chopped up every time the camera cuts to another room or
    a piece of B-roll.) The rule keys on the diarized speakers, which are stable
    across those cuts:

      * A run of role=guest segments that SHARE a non-host speaker merges into one
        story about that guest.
      * Host narration bracketed by the SAME guest on both sides is an internal
        bridge -> absorbed regardless of length (the host linking within one story).
      * Host narration between DIFFERENT guests is ambiguous: if it is SHORT
        (total <= segue_max s) it is that next guest's lead-in ("...I met with Bob
        Hon") and is attached to their story; if it is LONG it is a real narration
        and stands on its own. Host-only stretches with no guest story after them
        also stand alone.

    A role=host segment that merely has a guest speaker leak into it (diarization
    noise) is treated as host-only. Each merged story is labeled/named from the
    child covering the most seconds, so a short segue never renames the interview.
    The final same-label merge is applied ONLY to host-only stories (e.g. a run of
    studio event coverage); two different guests are never merged on label alone.
    """
    def guests(seg):
        if seg.get("role") != "guest":
            return set()
        # Prefer the min-duration guest set (set upstream from the turns) so a boundary
        # sliver of another speaker doesn't chain two conversations; fall back to the
        # raw speaker list when it isn't available (e.g. a hand-built segment).
        if seg.get("guests") is not None:
            return set(seg["guests"])
        return {s for s in seg.get("speakers", []) if s != host}

    def new_group(children):
        spk, gset = [], set()
        for c in children:
            gset |= guests(c)
            for s in c.get("speakers", []):
                if s not in spk:
                    spk.append(s)
        return {"start": children[0]["start"], "end": children[-1]["end"],
                "speakers": spk, "guests": gset, "_children": [dict(c) for c in children]}

    def absorb(g, seg):
        g["end"] = seg["end"]
        g["guests"] |= guests(seg)
        for s in seg.get("speakers", []):
            if s not in g["speakers"]:
                g["speakers"].append(s)
        g["_children"].append(dict(seg))

    groups, pending = [], []      # pending = buffered host-only (bridge or lead-in)
    for s in labeled:
        gs = guests(s)
        if not gs:                                               # host-only: buffer it
            pending.append(s)
            continue
        cur = groups[-1] if groups else None
        if cur and cur["guests"] and (gs & cur["guests"]):       # same guest resumes:
            for hb in pending:                                   # pending is a bridge
                absorb(cur, hb)                                  # (absorb, any length)
            pending = []
            absorb(cur, s)
        else:                                                    # new/different guest
            # Only the trailing short run of host narration is this guest's lead-in
            # ("...and I met with Bob Hon"); any earlier, longer narration before it
            # stands on its own rather than being dragged into the interview.
            lead, acc = [], 0.0
            while pending and acc + (pending[-1]["end"] - pending[-1]["start"]) <= segue_max:
                acc += pending[-1]["end"] - pending[-1]["start"]
                lead.insert(0, pending.pop())
            for hb in pending:                                   # the long remainder
                groups.append(new_group([hb]))
            pending = []
            groups.append(new_group(lead + [s]))
    for hb in pending:                                           # trailing, no guest after
        groups.append(new_group([hb]))

    # Label/name each group from the child covering the most seconds; keep the first
    # non-empty on-screen caption (a name/title card seen anywhere in the story).
    for g in groups:
        secs = {}
        for c in g["_children"]:
            lbl = c.get("label", "other")
            secs[lbl] = secs.get(lbl, 0.0) + (c["end"] - c["start"])
        g["label"] = max(secs, key=secs.get)
        g["name"] = max((c for c in g["_children"] if c.get("label", "other") == g["label"]),
                        key=lambda c: c["end"] - c["start"]).get("name", "")
        g["on_screen_text"] = next((c.get("on_screen_text", "") for c in g["_children"]
                                    if c.get("on_screen_text", "").strip()), "")

    # Merge only adjacent HOST-ONLY stories that share a label; guest stories stay
    # distinct even if two consecutive interviews happen to get the same label.
    out = []
    for g in groups:
        if (out and out[-1]["label"] == g["label"]
                and not out[-1]["guests"] and not g["guests"]):
            out[-1]["end"] = g["end"]
            out[-1]["on_screen_text"] = out[-1]["on_screen_text"] or g["on_screen_text"]
            for s in g["speakers"]:
                if s not in out[-1]["speakers"]:
                    out[-1]["speakers"].append(s)
        else:
            out.append(g)
    return [{"start": g["start"], "end": g["end"], "label": g["label"],
             "name": g["name"], "speakers": g["speakers"],
             "on_screen_text": g["on_screen_text"]} for g in out]


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
    else:
        boundaries = sorted(set(boundaries) | set(lexical_boundaries(turns)))

    print(f"host/narrator = {host}", file=sys.stderr)
    print(f"discovered taxonomy ({len(taxonomy)}): "
          + ", ".join(t["type"] for t in taxonomy), file=sys.stderr)
    print(f"boundaries (LLM + lexical): {len(boundaries)}", file=sys.stderr)

    segments = merge_by_speaker(shots, host=host)
    segments = split_on_topics(segments, boundaries, shots)
    for s in segments:                       # attach clean per-segment transcript text
        s["text"], s["speakers"] = segment_text(turns, s["start"], s["end"])
        s["role"] = dominant_role(turns, s["start"], s["end"], host)
        s["guests"] = segment_guests(turns, s["start"], s["end"], host)

    labeled = label_segments(segments, client, taxonomy, video, args.model, args.max_width, host)
    labeled = coalesce_conversation(labeled, host=host)

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
