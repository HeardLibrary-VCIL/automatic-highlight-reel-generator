"""Name the segments audio_segment.py found, and write the Editor's segment JSON.

The LLM's job here is the part signal processing can't do: deciding WHAT each
song is. Claude on Bedrock does not take audio, so it gets text evidence only:

  * what was SAID around each song (an MC's "this next one is ...", a radio
    host's back-announce, an oral-history narrator), from Amazon Transcribe,
  * the transcribed lyrics of the song itself (noisy: ASR of singing),
  * the original filename, which for archival items often carries the
    collection, performer, date or tape side.

The reply is validated against a pydantic schema (the same idea as Pydantic AI:
structured output enforced by a model, with one corrective retry on a bad
reply). pydantic is already installed as a dependency of the anthropic SDK.

Never blocks output: on any failure, segments keep plain titles ("Song 3").
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "us.anthropic.claude-sonnet-4-6")
LABEL_BATCH = int(os.environ.get("AUDIO_LABEL_BATCH", "40"))   # content segments per Bedrock call
SPEECH_CHARS = 1200       # transcript budget per speech segment in the prompt
LYRIC_CHARS = 400         # transcript budget per song (lyrics) in the prompt

CONTENT_KINDS = ("song", "speech", "applause")
KIND_TITLES = {"speech": "Speech", "applause": "Applause"}


# ───────────────────────────── output schema ─────────────────────────────────
class SegmentLabel(BaseModel):
    index: int
    title: str = Field(min_length=1, max_length=140)
    description: str = Field(default="", max_length=600)
    confidence: Literal["high", "medium", "low"] = "low"
    evidence: str = Field(default="", max_length=300)


class BatchLabels(BaseModel):
    segments: List[SegmentLabel]


class RecordingSummary(BaseModel):
    program_title: str = Field(default="", max_length=160)
    summary: str = Field(default="", max_length=1200)


SYSTEM = """You help archivists at Vanderbilt's Special Collections and Music Library catalogue
digitized audio recordings (reel-to-reel, cassette, LP and radio transfers, concerts, oral histories).
Software has already split the recording into segments and classified each one as song, speech or
applause. You name and briefly describe each segment so an archivist can review it.

Rules:
- Use only the evidence given: the filename, what is said in speech segments, and transcribed lyrics.
- Song titles: give a title only when an announcement names it, or the lyrics clearly quote a
  well-known song (for example a traditional spiritual or hymn whose refrain is recognizable). Put the
  supporting words in `evidence`. Otherwise title it descriptively, for example
  "Song 4 (untitled, choral)", and set confidence "low". Never invent titles or performers.
- Lyrics are automatic speech recognition of singing and may be garbled; do not over-read them.
- Speech titles say what the speech is: "MC introduces next selection", "Interview: early career".
- Keep titles under 80 characters and descriptions to one or two sentences.
- Reply with JSON only, no prose and no code fences."""


# ──────────────────────────────── helpers ────────────────────────────────────
def segment_text(turns, start, end):
    parts, speakers = [], []
    for t in turns or []:
        if min(float(t["end"]), end) - max(float(t["start"]), start) > 0:
            parts.append(t.get("text", ""))
            sp = t.get("speaker")
            if sp and sp not in speakers:
                speakers.append(sp)
    return " ".join(p for p in parts if p).strip(), speakers


def _clip(text, n):
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _fmt(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _parse_json(text):
    """The JSON object in a model reply (tolerates stray prose/code fences)."""
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        raise ValueError("no JSON object in reply")
    return json.loads(m.group(0))


def _create(client, **kwargs):
    """messages.create, with video-processing/bedrock.py's throttling backoff when
    it is importable (it is in the container); plain call otherwise (local runs)."""
    try:
        from bedrock import create_with_retry
    except ImportError:
        return client.messages.create(**kwargs)
    return create_with_retry(client, **kwargs)


def _ask(client, model, schema, user, max_tokens=4000):
    """One Bedrock call whose reply must validate against `schema`; on a bad
    reply, retry once with the validation error so the model can fix it."""
    messages = [{"role": "user", "content": user}]
    for attempt in range(2):
        resp = _create(client, model=model, max_tokens=max_tokens,
                       system=SYSTEM, messages=messages)
        text = "".join(getattr(b, "text", "") for b in resp.content)
        try:
            return schema.model_validate(_parse_json(text))
        except (ValueError, ValidationError) as e:
            if attempt:
                raise
            log.warning("Label reply failed validation (%s); retrying once", str(e)[:200])
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content":
                          f"That reply was not valid: {str(e)[:800]}\nReturn only the corrected JSON."}]


# ────────────────────────────── labeling ─────────────────────────────────────
def describe_segments(regions, turns):
    """Prompt-ready view of every region (dead space included, for context)."""
    rows, song_no = [], 0
    for i, r in enumerate(regions):
        text, speakers = segment_text(turns, r.start, r.end)
        row = {"index": i, "start": _fmt(r.start), "end": _fmt(r.end),
               "seconds": round(r.end - r.start), "kind": r.kind}
        if r.kind == "song":
            song_no += 1
            row["song_number"] = song_no
            if text:
                row["lyrics_asr"] = _clip(text, LYRIC_CHARS)
        elif r.kind == "speech":
            row["transcript"] = _clip(text, SPEECH_CHARS)
            if speakers:
                row["speakers"] = speakers
        rows.append(row)
    return rows


def label_audio(regions, turns, client, original_filename, duration, model=CLAUDE_MODEL):
    """Returns {"segments": {index: SegmentLabel}, "program_title", "summary"}."""
    rows = describe_segments(regions, turns)
    content_idx = [r["index"] for r in rows if r["kind"] in CONTENT_KINDS]
    labels = {}
    t0 = time.time()
    for b in range(0, len(content_idx), LABEL_BATCH):
        want = content_idx[b: b + LABEL_BATCH]
        lo, hi = max(0, want[0] - 2), min(len(rows), want[-1] + 3)   # neighbours as context
        user = (
            f"Recording: {original_filename} (duration {_fmt(duration)})\n"
            f"Segments {lo}-{hi - 1} of {len(rows)}:\n"
            + json.dumps(rows[lo:hi], ensure_ascii=False, indent=1)
            + f"\n\nLabel exactly these segment indexes: {want}.\n"
            'Return {"segments": [{"index": int, "title": str, "description": str, '
            '"confidence": "high"|"medium"|"low", "evidence": str}]}'
        )
        out = _ask(client, model, BatchLabels, user)
        for s in out.segments:
            if s.index in want:
                labels[s.index] = s
    log.info("[audio-label] %d/%d content segments labeled in %.1fs",
             len(labels), len(content_idx), time.time() - t0)

    summary = RecordingSummary()
    try:
        outline = [{"index": r["index"], "kind": r["kind"], "start": r["start"],
                    "title": labels[r["index"]].title if r["index"] in labels else r["kind"]}
                   for r in rows if r["kind"] in CONTENT_KINDS]
        speech = " ".join(r.get("transcript", "") for r in rows if r["kind"] == "speech")
        user = (
            f"Recording: {original_filename} (duration {_fmt(duration)})\n"
            f"Segment outline: {json.dumps(outline, ensure_ascii=False)}\n"
            f"Spoken content (excerpt): {_clip(speech, 3000)}\n\n"
            "Give the recording a catalogue title and a 2-4 sentence summary of what it contains "
            "(type of event or programme, performers or speakers if stated, number of songs). "
            'Return {"program_title": str, "summary": str}'
        )
        summary = _ask(client, model, RecordingSummary, user, max_tokens=800)
    except Exception as e:
        log.warning("[audio-label] summary skipped: %s", e)
    return {"segments": labels, "program_title": summary.program_title, "summary": summary.summary}


# ──────────────────────────── segment JSON ───────────────────────────────────
def build_audio_segment_json(s3_key, regions, turns, labels, duration):
    """Same shape the Editor already reads for video (segment/{stem}.json)."""
    from audio_segment import DEAD_KINDS, DEAD_TITLES

    stem = Path(s3_key).name.rsplit(".", 1)[0]
    seg_labels = (labels or {}).get("segments", {})
    segments, song_no, dead_total = [], 0, 0.0
    for i, r in enumerate(regions):
        if r.kind in DEAD_KINDS:
            dead_total += r.end - r.start
            segments.append({
                "segment_start": r.start, "segment_end": r.end, "segment_type": "D",
                "title": DEAD_TITLES.get(r.kind, "Dead space (auto-detected)"),
                "content_label": r.kind,
            })
            continue
        if r.kind == "song":
            song_no += 1
        lab: Optional[SegmentLabel] = seg_labels.get(i)
        text, speakers = segment_text(turns, r.start, r.end)
        default_title = f"Song {song_no}" if r.kind == "song" else KIND_TITLES.get(r.kind, "Content")
        segments.append({
            "segment_start": r.start,
            "segment_end": r.end,
            "segment_type": r.kind,
            "title": lab.title if lab else default_title,
            "description": lab.description if lab else "",
            "content_label": r.kind,
            "speakers": speakers,
            # For songs this is ASR of the lyrics; for speech, what was said.
            "transcript": text,
            "caption": "",
            # Boundary trust (how the cut BEFORE this segment was found) and,
            # separately, how sure the model is about the title.
            "confidence": r.confidence,
            "boundary_cues": r.cues,
            "title_confidence": lab.confidence if lab else "",
            "title_evidence": lab.evidence if lab else "",
            **({"song_number": song_no} if r.kind == "song" else {}),
        })
    out = {
        "video": stem,                      # key name kept for Editor compatibility
        "media_type": "audio",
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auto_detected": True,
        "content_segmented": True,
        "trim_status": "OK",
        "duration": round(duration, 2),
        "kept_pct": round(100.0 * (1 - dead_total / duration), 1) if duration else 0.0,
        "song_count": song_no,
        "segments": segments,
    }
    if labels and labels.get("summary"):
        out["summary"] = labels["summary"]
    if labels and labels.get("program_title"):
        out["program_title"] = labels["program_title"]
    return out
