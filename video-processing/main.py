"""ECS entry point: download video from S3, detect dead space, segment content, write results.

Integrated with SCUA-Video-Editing Amplify frontend:
  - Input  : S3_BUCKET / S3_KEY pointing at uploaded video under `video/` prefix
  - Output : segment JSON  → `segment/{basename}.json`  (for the Editor timeline)
             review marker → `review/{basename}.txt`    (if NEEDS_REVIEW)

Pipeline stages:
  1. Dead-space detection (analyze_deadspace): find head/tail dead regions
  2. Audio-visual fusion segmentation:
       A. transcribe   -- Amazon Transcribe reads the video straight from S3
                          (already uploaded by the frontend) -> diarized turns
       B. fuse         -- PySceneDetect shot cuts + join words/speakers onto each
                          shot, then merge shots by speaker continuity
       C. segment/label-- boundaries from an LLM pass + lexical (TextTiling) + pause
                          cues; split, recompute host/guest role, then ONE multimodal
                          Bedrock call labels each segment from its words AND a frame
                          (also reading any on-screen caption)
       D. coalesce     -- merge by speaker continuity so one interview stays one
                          segment across camera cuts; score each boundary's confidence
  3. Write combined segment JSON to S3 for the Editor UI

All model calls go through Amazon Bedrock using the ECS task role, so no
ANTHROPIC_API_KEY is needed anywhere in the deployed stack.

The actual video trimming is triggered later by the user from the Editor UI,
after they review and adjust the auto-detected segments.
"""
import json
import os
import re
import subprocess
import sys
import logging
import tempfile
import traceback
import time
import uuid
from dataclasses import replace
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from analyze_deadspace import analyze, analyze_full_video, apply_trim

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Output prefixes — match SCUA Amplify storage paths
SEGMENT_PREFIX = os.environ.get("SEGMENT_PREFIX", "segment").strip("/")
TRIM_MODE = os.environ.get("TRIM_MODE", "black")
# Content-type segmentation. Credentials come from the ECS task role (Bedrock +
# Transcribe), so "auto" simply means on; set CONTENT_SEGMENT=off to get
# dead-space-only segments.
CONTENT_SEGMENT = os.environ.get("CONTENT_SEGMENT", "auto")  # "auto", "on", "off"
SEGMENT_DETECTOR = os.environ.get("SEGMENT_DETECTOR", "adaptive")  # "adaptive" or "content"
SEGMENT_MIN_SHOT = float(os.environ.get("SEGMENT_MIN_SHOT", "1.5"))
TRANSCRIBE_LANGUAGE = os.environ.get("TRANSCRIBE_LANGUAGE", "en-US")
MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", "10"))
# Region for boto3 service clients. The container sets AWS_REGION, but botocore
# resolves from AWS_DEFAULT_REGION -- and Transcribe (unlike S3) has no global
# fallback, so we pass region_name explicitly rather than rely on either var.
REGION = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
          or "us-east-1")
# Full-video dead-space detection tuning
MERGE_GAP = float(os.environ.get("MERGE_GAP", "2.0"))
MIN_DEAD_DUR = float(os.environ.get("MIN_DEAD_DUR", "3.0"))
MIN_CONTENT_DUR = float(os.environ.get("MIN_CONTENT_DUR", "5.0"))
BLACK_PIC_TH = float(os.environ.get("BLACK_PIC_TH", "0.995"))
MAX_FRAME_WIDTH = int(os.environ.get("MAX_FRAME_WIDTH", "768"))
# Set FULL_VIDEO_SCAN=on to detect dead space throughout (not just head/tail)
FULL_VIDEO_SCAN = os.environ.get("FULL_VIDEO_SCAN", "on").lower() in ("on", "true", "1")


# Normalized content categories used as `segment_type` in the output. Claude's
# per-program taxonomy is open-vocabulary, so its free-text labels are folded into
# this stable, finite set; the frontend color-codes segments by these values. Dead
# space keeps its own "D" code (handled separately, not classified by Claude).
CONTENT_CATEGORIES = (
    "interview", "host", "report", "announcement", "performance",
    "sports", "introduction", "closing", "credits", "other",
)

# Substring keywords → normalized category. Checked in order; first hit wins.
_CATEGORY_KEYWORDS = (
    ("interview", "interview"), ("q&a", "interview"), ("q and a", "interview"),
    ("sport", "sports"), ("game", "sports"), ("match", "sports"),
    ("credit", "credits"),
    ("intro", "introduction"), ("teaser", "introduction"), ("open", "introduction"),
    ("closing", "closing"), ("outro", "closing"), ("sign-off", "closing"),
    ("sign off", "closing"), ("wrap-up", "closing"), ("wrap up", "closing"),
    ("announce", "announcement"), ("psa", "announcement"),
    ("public service", "announcement"), ("commercial", "announcement"),
    ("advert", "announcement"), ("advocacy", "announcement"), ("ad ", "announcement"),
    ("perform", "performance"), ("music", "performance"), ("dance", "performance"),('performer', 'performance'),
    ("song", "performance"),
    ("report", "report"), ("field", "report"), ("package", "report"), ("story", "report"),
    ("host", "host"), ("anchor", "host"), ("monologue", "host"),
    ("narrat", "host"), ("desk", "host"), ("link", "host"),
)


def _match_category_keywords(text: str) -> str:
    """First keyword hit in `text` → its category, else 'other'. Shared by the
    label classifier and the title/description fallback."""
    r = (text or "").strip().lower()
    if not r:
        return "other"
    if r in CONTENT_CATEGORIES:
        return r
    for kw, cat in _CATEGORY_KEYWORDS:
        if kw in r:
            return cat
    return "other"


def normalize_category(label: str, *fallback_texts: str) -> str:
    """Fold a free-text content label (from Claude's per-program taxonomy) into one
    of CONTENT_CATEGORIES. Unmatched or empty labels become 'other'.

    When the LABEL itself maps to 'other' (uninformative, e.g. Claude wrote "concert"
    or a generic phrase with no category keyword), fall back to keyword-matching the
    provided `fallback_texts` (the segment's title + description) so a segment plainly
    described as a performance/interview/etc. still gets that type. The label always
    wins when it is specific — the fallback only fires on an 'other' label."""
    cat = _match_category_keywords(label)
    if cat != "other":
        return cat
    for text in fallback_texts:
        cat = _match_category_keywords(text)
        if cat != "other":
            return cat
    return "other"


def segment_key(s3_key: str) -> str:
    """Map an input key to its segment JSON key under segment/."""
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    return f"{SEGMENT_PREFIX}/{stem}.json"


# ── Program grouping (tape -> programs -> segments) ──────────────────────────
# A tape may hold several recordings ("programs"). Content segments are grouped
# into programs; a program often keeps the SAME speakers throughout even across
# intervening clips, so a tape splice (snow) does NOT by itself start a new
# program. A new program begins only where a snow splice COINCIDES with a change
# of on-screen speakers OR a new title-card caption. Dead space and title-card-
# only markers are NOT assigned to any program. Videos that are a single (or
# nearly single) content segment get no program tier at all.
PROGRAM_MIN_SEGMENTS = int(os.environ.get("PROGRAM_MIN_SEGMENTS", "3"))  # below this: no program tier

# A caption that looks like a program/title-card slate (short, mostly caps/among
# the few standalone words): a strong "new program starts here" signal.
def _looks_like_title_card(caption: str) -> bool:
    c = (caption or "").strip()
    if not c:
        return False
    # A slate is short (a title, not a lower-third sentence) and largely uppercase.
    letters = [ch for ch in c if ch.isalpha()]
    if not letters:
        return False
    upper_frac = sum(ch.isupper() for ch in letters) / len(letters)
    return len(c) <= 60 and upper_frac >= 0.7


def group_into_programs(segments, snow_spans, tol=2.0):
    """Assign a program_index to each CONTENT segment and return the programs list.

    `segments` is the final flat list (content + 'D' markers). `snow_spans` is a
    list of (start, end) for detected snow (the only splice kind that can delimit
    programs). Returns (programs, changed) where `programs` is
    [{program_index, program_start, program_end}] (labels filled in Phase 2) and
    `changed` is False when no program tier applies (single/near-single segment):
    in that case NO program_index is assigned.

    Boundary rule: walking content segments in time order, a NEW program starts at
    segment i (i>0) only if a snow splice lies between segment i-1 and i AND either
    the speaker roster changed or segment i carries a title-card caption. Snow with
    unchanged speakers and no title card stays the SAME program (an internal splice).
    """
    content = [s for s in segments if s.get("segment_type") != "D"]
    # No program tier for a single / near-single content segment.
    if len(content) < PROGRAM_MIN_SEGMENTS:
        return [], False

    def snow_between(a_end, b_start):
        lo, hi = min(a_end, b_start) - tol, max(a_end, b_start) + tol
        return any(ss <= hi and se >= lo for ss, se in (snow_spans or []))

    programs = []
    idx = -1
    prev = None
    for s in content:
        start_new = prev is None
        if prev is not None:
            roster_prev = set(prev.get("speakers", []) or [])
            roster_cur = set(s.get("speakers", []) or [])
            roster_changed = bool(roster_prev) and bool(roster_cur) and roster_prev.isdisjoint(roster_cur)
            has_title_card = _looks_like_title_card(s.get("caption", ""))
            if snow_between(prev["segment_end"], s["segment_start"]) and (roster_changed or has_title_card):
                start_new = True
        if start_new:
            idx += 1
            programs.append({"program_index": idx,
                             "program_start": s["segment_start"],
                             "program_end": s["segment_end"]})
        else:
            programs[-1]["program_end"] = s["segment_end"]
        s["program_index"] = idx
        prev = s

    # If everything landed in one program, there's no meaningful tier -> drop it.
    if len(programs) <= 1:
        for s in content:
            s.pop("program_index", None)
        return [], False
    return programs, True


_ORIG_NAME_CACHE: dict = {}


def resolve_original_filename(s3_client, bucket: str, key: str) -> str:
    """Best-effort ORIGINAL (user-facing) filename for this video, used both for
    logging AND as `original-filename` object metadata stamped on every derivative,
    so each S3 object (despite its UUID key) traces back to the archival item.

    Objects are stored under a UUID key (video/<uuid>.mp4), so the key filename is
    not the name the user uploaded. We recover the real name from S3 object
    metadata when present -- a `Content-Disposition: ...filename="..."` or a user
    metadata field ('original-filename' / 'originalfilename' / 'filename') -- and
    fall back to the key's own filename otherwise. Cached per (bucket,key) so the
    head_object runs once per run. Never raises; logging must not break processing."""
    cache_key = (bucket, key)
    if cache_key in _ORIG_NAME_CACHE:
        return _ORIG_NAME_CACHE[cache_key]
    resolved = _resolve_original_filename_uncached(s3_client, bucket, key)
    _ORIG_NAME_CACHE[cache_key] = resolved
    return resolved


def _resolve_original_filename_uncached(s3_client, bucket: str, key: str) -> str:
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        cd = head.get("ContentDisposition") or ""
        m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd)
        if m:
            return Path(m.group(1)).name
        meta = {k.lower(): v for k, v in (head.get("Metadata") or {}).items()}
        for mk in ("original-filename", "originalfilename", "filename", "display-name"):
            if meta.get(mk):
                return Path(meta[mk]).name
    except Exception as e:
        log.warning("Could not read S3 metadata for original filename (%s: %s); "
                    "using key filename", type(e).__name__, e)
    return Path(key).name


def safe_output_name(name: str, fallback_stem: str) -> str:
    """An S3-safe basename (with a single .mp4) for the trimmed clip. Honors the
    reviewer's chosen name from the trim request but strips path parts and unsafe
    chars; falls back to '{fallback_stem}_trimmed.mp4' when no usable name is given."""
    base = Path((name or "").strip()).name
    base = re.sub(r"\.mp4$", "", base, flags=re.IGNORECASE).strip()
    base = re.sub(r"[^A-Za-z0-9._ -]", "", base).strip()
    return f"{base}.mp4" if base else f"{fallback_stem}_trimmed.mp4"


def _should_run_content_segmentation() -> bool:
    """Whether to run the fusion segmentation. Bedrock/Transcribe auth comes from
    the ECS task role, so there is no key to check -- only the explicit off switch."""
    return CONTENT_SEGMENT != "off"


def turns_to_vtt(turns: list) -> str:
    """Convert speaker turns to WebVTT format (for Aviary/captioning)."""
    lines = ["WEBVTT", ""]
    for i, t in enumerate(turns, 1):
        sh, sm, ss = int(t["start"] // 3600), int((t["start"] % 3600) // 60), t["start"] % 60
        eh, em, es = int(t["end"] // 3600), int((t["end"] % 3600) // 60), t["end"] % 60
        lines.append(str(i))
        lines.append(f"{sh:02d}:{sm:02d}:{ss:06.3f} --> {eh:02d}:{em:02d}:{es:06.3f}")
        speaker = t.get("speaker", "")
        text = t.get("text", "")
        lines.append(f"<v {speaker}>{text}" if speaker and speaker != "spk_?" else text)
        lines.append("")
    return "\n".join(lines)


def extract_audio(input_video: str, output_audio: str) -> str:
    """Extract the audio track to an mp4/aac clip for Amazon Transcribe, used when the
    uploaded container isn't one Transcribe reads directly (avi/mkv/wmv/ts/...). Raises
    if the source has no audio stream, which the caller treats as "no usable audio" and
    drops to visual segmentation."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-i", input_video, "-vn", "-c:a", "aac",
         "-b:a", "128k", output_audio],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, check=True,
    )
    return output_audio


def has_audio_stream(input_video: str) -> bool:
    """True if the video has at least one audio stream. Silent videos (e.g. a
    game clip with no commentary) have none — we skip transcription for those
    and segment visually, avoiding a wasted/failing Transcribe cycle."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", input_video],
            stderr=subprocess.DEVNULL).decode().strip()
        return bool(out)
    except Exception:
        return False


def get_transcript_turns(local_video_path, s3_bucket, s3_key):
    """Transcribe the video (or load cached transcript). Returns diarized turns:
    a list of {"start","end","speaker","text"}. Cached to transcript/{stem}.json so
    it only runs Transcribe once even though both the dead-space pre-check and the
    fusion segmentation need it. Returns [] if no usable audio/speech."""
    import boto3 as _boto3
    from transcribe import start_job, wait, fetch_result, to_speaker_turns, media_format

    stem = Path(s3_key).name.rsplit(".", 1)[0]
    s3_client = _boto3.client("s3", region_name=REGION)
    transcript_cache_key = f"transcript/{stem}.json"

    # Cache check
    try:
        resp = s3_client.get_object(Bucket=s3_bucket, Key=transcript_cache_key)
        turns = json.loads(resp["Body"].read().decode("utf-8"))
        if turns:
            log.info(f"[A] Loaded cached transcript ({len(turns)} turns)")
            return turns
    except Exception:
        pass

    transcribe_client = _boto3.client("transcribe", region_name=REGION)
    job_name = f"scua-{stem}-{int(time.time())}-{uuid.uuid4().hex[:8]}"[:200]
    media_uri = f"s3://{s3_bucket}/{s3_key}"
    if media_format(s3_key) is None:
        audio_key = f"edit/{stem}.transcribe.mp4"
        audio_path = Path(local_video_path).with_name(f"{stem}.transcribe.mp4")
        extract_audio(str(local_video_path), str(audio_path))
        s3_client.upload_file(str(audio_path), s3_bucket, audio_key)
        media_uri = f"s3://{s3_bucket}/{audio_key}"
        log.info(f"[A] extracted audio -> s3://{s3_bucket}/{audio_key}")
    log.info(f"[A] Transcribe job {job_name} on {media_uri}")
    start_job(transcribe_client, media_uri, job_name=job_name,
              language=TRANSCRIBE_LANGUAGE, max_speakers=MAX_SPEAKERS)
    job = wait(transcribe_client, job_name)
    turns = to_speaker_turns(fetch_result(job, s3_client))
    if not turns:
        return []
    # Cache transcript + VTT (stamped with the source original filename for traceability)
    try:
        orig_meta = {"original-filename": resolve_original_filename(s3_client, s3_bucket, s3_key)}
        s3_client.put_object(Bucket=s3_bucket, Key=transcript_cache_key,
                             Body=json.dumps(turns, indent=2).encode("utf-8"),
                             ContentType="application/json", Metadata=orig_meta)
        s3_client.put_object(Bucket=s3_bucket,
                             Key=transcript_cache_key.replace(".json", ".vtt"),
                             Body=turns_to_vtt(turns).encode("utf-8"),
                             ContentType="text/vtt", Metadata=orig_meta)
        log.info(f"[A] Cached transcript + VTT")
    except Exception as cache_err:
        log.warning(f"[A] Could not cache transcript: {cache_err}")
    return turns


def speech_windows_from_turns(turns, merge_gap=5.0):
    """Collapse diarized turns into merged speech time windows [(start,end)].
    Adjacent turns within merge_gap seconds are joined so brief pauses between
    sentences don't fragment the windows.

    merge_gap is deliberately small (5s): a longer gap between turns means a real
    silent stretch (e.g. a black-screen filler or bars leader between segments),
    which must NOT be absorbed into a speech window — otherwise the dead-space
    veto would hide those mid-video dead spans. Only bridge true sentence pauses."""
    if not turns:
        return []
    spans = sorted((float(t["start"]), float(t["end"])) for t in turns)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


# The visual detectors (bars/snow/black/white) are audio-INDEPENDENT, so they mark
# a leader dead even when it plays LOUD noise/tone. Color-bars/countdown/tape leaders
# are exactly that: a vivid but non-program picture over a loud 1kHz tone or hiss.
# These constants gate the head-leader guard below.
HEAD_LEADER_MAX = float(os.environ.get("HEAD_LEADER_MAX", "180.0"))  # only reclassify a leader within the first N s
HEAD_LEADER_MIN = float(os.environ.get("HEAD_LEADER_MIN", "2.0"))    # ignore trivially short head regions


def head_leader_end(regions, *, max_head=HEAD_LEADER_MAX, gap=3.0):
    """End time of a leading NON-PROGRAM leader (color bars / snow / black / white)
    that begins at (or very near) t=0, or 0.0 if there is none.

    Uses ONLY the visual detector regions, so a leader with LOUD audio (a tone or
    noise over color bars) is still recognized as dead -- audio level and any
    speech Transcribe may have hallucinated over the noise are irrelevant here.
    Contiguous leader regions separated by <= `gap` are chained so a bars->black
    ->snow leader counts as one block."""
    lead = [(r.start, r.end) for r in regions
            if r.kind in ("bars", "snow", "black", "white")]
    lead.sort()
    end = 0.0
    for s, e in lead:
        if s <= max(HEAD_LEADER_MIN, end) + gap:   # starts at 0 or chains onto the run
            end = max(end, e)
        elif s > end:
            break                                   # a real content gap -> leader is over
        if end >= max_head:
            break
    return end if end >= HEAD_LEADER_MIN else 0.0


def run_fusion_segmentation(local_video_path, s3_bucket, s3_key, prop, turns=None,
                            dead_spans=None, client=None):
    """Stages A-D: transcribe -> shots+fuse -> discover taxonomy/topics -> label.

    Transcribe reads the ORIGINAL S3 object (the frontend already uploaded it), so
    nothing is re-uploaded; the local copy is only used for shot detection and for
    sampling one frame per segment. Returns (labeled_segments, taxonomy, turns) --
    the diarized `turns` are handed back so the segment JSON can carry each
    segment's transcript for the reviewer.
    """
    import boto3 as _boto3
    from transcribe import start_job, wait, fetch_result, to_speaker_turns, media_format
    from segment_shots import detect_shots
    from segment_fuse import attach, merge_by_speaker
    from segment_label import (analyze_program, detect_host, segment_text,
                               split_on_topics, label_segments, coalesce_conversation,
                               lexical_boundaries, dominant_role, segment_guests,
                               pause_boundaries, boundary_confidence)
    from bedrock import make_client

    stem = Path(s3_key).name.rsplit(".", 1)[0]
    import boto3 as _boto3
    s3_client = _boto3.client("s3", region_name=REGION)

    # --- A. transcribe straight from S3 (diarized) — reuse pre-fetched turns ---
    if turns is None:
        turns = get_transcript_turns(local_video_path, s3_bucket, s3_key)
    if not turns:
        raise RuntimeError("Transcribe returned no speech turns")

    # --- B. shots + join words onto them ---
    t0 = time.time()
    window = (prop.content_start, prop.content_end)
    shots = detect_shots(str(local_video_path), window, SEGMENT_DETECTOR, SEGMENT_MIN_SHOT)
    fused = attach(shots, [(t["start"], t["end"], t["speaker"], t["text"]) for t in turns])
    log.info(f"[B] {len(fused)} shots in window {window[0]:.1f}-{window[1]:.1f}s "
             f"({time.time() - t0:.1f}s)")

    # --- C. discover taxonomy + multi-cue boundaries, split, then label multimodally ---
    t0 = time.time()
    # Reuse the caller's Bedrock client when provided (one per video across fusion +
    # Phase 2 labeling); otherwise make one so standalone calls still work.
    if client is None:
        client = make_client()
    host = detect_host(turns)
    taxonomy, llm_bounds = analyze_program(turns, client)
    # Two more cue sources, both free from data we already have: TextTiling lexical
    # valleys catch mid-shot boundaries (a host wrap-up rolling into a PSA with no
    # camera cut); long silences between turns mark broadcast segment breaks. The
    # speaker-continuity coalesce removes any that fall inside one conversation.
    lex_bounds = lexical_boundaries(turns)
    pause_bounds = pause_boundaries(turns)
    # A detected tape splice (black/bars/snow dead span) is a HARD boundary: content
    # on either side of it is a different item, so add each dead span's edges to the
    # boundary set. This forces split_on_topics to cut at splices the LLM is blind to
    # (it only sees transcript+frames), preventing a segment from spanning or extending
    # through a splice. Only mid-window edges matter (head/tail are handled elsewhere).
    dead_bounds = []
    for ds in (dead_spans or []):
        dead_bounds += [round(ds.start, 2), round(ds.end, 2)]
    boundaries = sorted(set(llm_bounds) | set(lex_bounds) | set(pause_bounds) | set(dead_bounds))
    log.info(f"[C] host={host} | taxonomy: " + ", ".join(t["type"] for t in taxonomy)
             + f" | {len(boundaries)} boundaries (LLM {len(llm_bounds)} + lexical "
             f"{len(lex_bounds)} + pause {len(pause_bounds)} + dead {len(dead_bounds)})")

    segments = merge_by_speaker(fused, host=host)
    segments = split_on_topics(segments, boundaries, fused)
    for s in segments:
        s["text"], s["speakers"] = segment_text(turns, s["start"], s["end"])
        # Re-tag role by who actually dominates this (possibly newly split) span, so a
        # host wrap-up sliced off a guest segment reads as host, not the guest's story.
        s["role"] = dominant_role(turns, s["start"], s["end"], host)
        s["guests"] = segment_guests(turns, s["start"], s["end"], host)

    labeled = label_segments(segments, client, taxonomy, str(local_video_path), host=host)
    # --- D. coalesce by conversation continuity (guest speaker), not visual label:
    # a continuous interview stays ONE segment even when the camera keeps cutting.
    # Dead spans are passed so the merge NEVER bridges across a tape splice, even for
    # the same speaker — a splice always separates stories.
    labeled = coalesce_conversation(labeled, host=host, dead_spans=dead_spans)
    # Score each final boundary by how many independent cues agree, for reviewer triage.
    shot_starts = sorted({s["start"] for s in fused})
    labeled = boundary_confidence(labeled, turns, shot_starts, llm_bounds, lex_bounds, host)
    log.info(f"[C/D] {len(segments)} -> {len(labeled)} labeled segments "
             f"({time.time() - t0:.1f}s)")
    return labeled, taxonomy, turns


def run_visual_segmentation(local_video_path, prop):
    """Segment a video that has NO usable audio by CONTENT TYPE — the standard
    two-level approach (shot-boundary detection -> group shots into scenes) rather
    than fixed-interval chopping:

        PySceneDetect shots -> classify each shot from a frame with Claude/Bedrock
        (dance performance / football game / interview / political ad / ... / other,
        plus a short free-text description) -> merge ADJACENT same-type shots into
        one scene -> name each scene by its content type.

    So segments read as "Interview", "Football game", "Commercial", etc. instead of
    generic "Scene N", and the boundaries fall where the CONTENT actually changes.
    No transcript needed, so this covers the silent / no-speech case.

    Honest limit: a video that is one uniform type throughout (e.g. a raw fixed-
    camera game) is correctly ONE scene — fine-grained sub-segmentation of a single
    type (e.g. per-play) needs a domain signal like scoreboard OCR, not implemented.
    Raises if nothing could be classified so the caller can drop to dead-space only.
    """
    from segment_shots import detect_shots, label_shots, _shots_to_segments
    from segment_content import (confirm_targets, smooth_labels, consolidate_segments,
                                  CATEGORIES, OTHER, other_note)

    window = (prop.content_start, prop.content_end)
    shots = detect_shots(str(local_video_path), window, SEGMENT_DETECTOR, SEGMENT_MIN_SHOT)

    # One Bedrock vision call per shot -> (shot_start, content-type, short description).
    labeled = label_shots(str(local_video_path), shots)
    kept = {round(ss, 2) for ss, _, _ in labeled}
    shots = [(ss, ee) for ss, ee in shots if round(ss, 2) in kept]
    if not shots:
        raise RuntimeError("visual classification produced no labelled shots")

    cats = [(ss, c) for ss, c, _ in labeled]
    cats = smooth_labels(confirm_targets(cats, CATEGORIES, min_windows=1))
    segments = _shots_to_segments(shots, cats)      # merge adjacent same-type shots
    segments = consolidate_segments(segments, 0, 0.0)
    for s in segments:
        if s.label == OTHER:
            s.note = other_note(s, labeled)          # a description to name the 'other' span

    log.info(f"Visual content-type segmentation: {len(labeled)} shots -> {len(segments)} scenes")
    for s in segments:
        note = getattr(s, "note", "")
        log.info(f"  [{s.start:.1f}s -> {s.end:.1f}s] {s.label}"
                 + (f" ({note})" if s.label == OTHER and note else ""))
    return [{
        "start": round(s.start, 2),
        "end": round(s.end, 2),
        # Known content type -> 'C' segment titled by the type; 'other' -> 'I'
        # segment titled by the model's short description (build_segment_json).
        "label": s.label,
        "name": (getattr(s, "note", "") or "") if s.label == OTHER else "",
        "speakers": [],
        "on_screen_text": "",
        "confidence": "",
        "cues": ["shot", "vlm"],
    } for s in segments]


def build_segment_json(s3_key: str, prop, content_segments=None, taxonomy=None,
                       turns=None, mid_dead_spans=None, client=None) -> dict:
    """Build a segment JSON matching SCUA Editor format.

    `mid_dead_spans` are DeadSpan objects that fall BETWEEN content (e.g. a color
    bars leader or black gap mid-program). They're inserted as "D" segments so the
    Editor shows them; without this they'd be swallowed by the content window.

    `content_segments` are the fusion-pipeline dicts
    {start, end, label, name, speakers}; the Editor gets the specific `name` as
    the title (e.g. "Triangle Inn resort tour") and the content type as
    `content_label`. Without them we fall back to simple D/I dead-space segments.

    `turns` are the diarized transcript rows; when present, each content segment
    carries a `transcript` of the words spoken over it so a reviewer can verify
    the label and boundaries against what is actually said.
    """
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    segments = []

    # Recompute transcript from `turns` against each FINAL segment span rather than
    # trusting the seg["text"] carried on the dict: that text predates coalescing
    # and would be truncated to the first merged child.
    if turns:
        from segment_label import segment_text

        def _transcript(start, end):
            return segment_text(turns, start, end)[0]
    else:
        def _transcript(start, end):
            return ""

    if content_segments:
        # Use the fusion segmentation results, keeping head/tail dead-space markers
        if prop.content_start > 1.0:
            segments.append({
                "segment_start": 0,
                "segment_end": round(prop.content_start, 2),
                "segment_type": "D",
                "title": "Head dead space (auto-detected)",
            })

        for seg in content_segments:
            label = seg.get("label", "other")
            name = (seg.get("name") or "").strip()
            description = (seg.get("description") or "").strip()
            # segment_type is the NORMALIZED content category (see CONTENT_CATEGORIES),
            # folded from Claude's per-program label. The frontend color-codes by it.
            # When the label is uninformative ('other'), fall back to the title +
            # description so a segment plainly described as a performance/interview/etc.
            # still gets that type. (Dead-space segments keep the separate "D" code.)
            category = normalize_category(label, name, description)
            segments.append({
                "segment_start": round(seg["start"], 2),
                "segment_end": round(seg["end"], 2),
                "segment_type": category,
                "title": name or label.title(),
                "description": (seg.get("description") or "").strip(),
                "content_label": label,
                "speakers": seg.get("speakers", []),
                "transcript": _transcript(seg["start"], seg["end"]),
                # On-screen caption (name/title read off the frame) + how much the
                # boundary before this segment is trusted, for reviewer triage.
                "caption": seg.get("on_screen_text", ""),
                "confidence": seg.get("confidence", ""),
                "boundary_cues": seg.get("cues", []),
            })

        if prop.duration - prop.content_end > 1.0:
            segments.append({
                "segment_start": round(prop.content_end, 2),
                "segment_end": round(prop.duration, 2),
                "segment_type": "D",
                "title": "Tail dead space (auto-detected)",
            })
    else:
        # Fallback: simple dead-space-only segments
        if prop.content_start > 1.0:
            segments.append({
                "segment_start": 0,
                "segment_end": round(prop.content_start, 2),
                "segment_type": "D",
                "title": "Head dead space (auto-detected)",
            })

        segments.append({
            "segment_start": round(prop.content_start, 2),
            "segment_end": round(prop.content_end, 2),
            "segment_type": "I",
            "title": "Content",
        })

        if prop.duration - prop.content_end > 1.0:
            segments.append({
                "segment_start": round(prop.content_end, 2),
                "segment_end": round(prop.duration, 2),
                "segment_type": "D",
                "title": "Tail dead space (auto-detected)",
            })

    # Inject mid-video dead spans (color bars leaders, black gaps between segments).
    # These fall inside the content window, so they'd otherwise be hidden. For each
    # mid dead span, carve it out of any overlapping content segment and add a "D"
    # marker, then re-sort. Head/tail spans (at the very edges) are already handled
    # above, so skip anything touching 0 or duration.
    _DEAD_LABELS = {"bars": "Color bars (auto-detected)",
                    "snow": "Video static/snow (auto-detected)",
                    "black": "Black gap (auto-detected)",
                    "white": "White gap (auto-detected)",
                    "mixed": "Dead space (auto-detected)"}
    if mid_dead_spans:
        # Carve a content segment around a dead span WITHOUT cloning its transcript
        # onto both halves. Each carved piece gets its transcript RE-SLICED to its own
        # (narrower) span — otherwise a segment spanning a black gap ends up duplicated
        # on both sides (same transcript/title), and an over-long content tail extending
        # into a dead stretch becomes a phantom copy on the dead side. A carved piece
        # that comes out blank (no words, no caption) is itself dead space (it was only
        # a clone of a neighbour's metadata over a silent stretch), so it is relabeled
        # as a "D" marker instead of a bogus content segment — keeping the timeline
        # gapless while not manufacturing content out of dead space.
        def _carve_piece(seg, new_start, new_end):
            piece = dict(seg)
            piece["segment_start"], piece["segment_end"] = new_start, new_end
            piece["transcript"] = _transcript(new_start, new_end)
            return piece

        def _is_blank_phantom(piece):
            # Blank = no transcribed words and no on-screen caption. A real silent
            # performance carries a caption/label/frame, so this only catches pieces
            # that are genuinely empty content (dead-region residue from the carve).
            return not (piece.get("transcript") or "").strip() and not (piece.get("caption") or "").strip()

        def _dead_piece(new_start, new_end):
            return {"segment_start": new_start, "segment_end": new_end,
                    "segment_type": "D", "title": "Dead space (auto-detected)"}

        for ds in mid_dead_spans:
            ds_start, ds_end = round(ds.start, 2), round(ds.end, 2)
            if ds_end - ds_start < 0.5:
                continue
            carved = []
            for seg in segments:
                # Dead markers already placed pass through untouched.
                if seg.get("segment_type") == "D":
                    carved.append(seg)
                    continue
                s0, s1 = seg["segment_start"], seg["segment_end"]
                # No overlap → keep as-is
                if ds_end <= s0 or ds_start >= s1:
                    carved.append(seg)
                    continue
                # Overlap: keep the portion(s) of this segment outside the dead span,
                # each with its transcript re-sliced to the kept span. A carved piece
                # that comes out blank is relabeled as dead space (keeps timeline gapless).
                if s0 < ds_start:
                    left = _carve_piece(seg, s0, ds_start)
                    carved.append(_dead_piece(s0, ds_start) if _is_blank_phantom(left) else left)
                if s1 > ds_end:
                    right = _carve_piece(seg, ds_end, s1)
                    carved.append(_dead_piece(ds_end, s1) if _is_blank_phantom(right) else right)
                # (the middle, [ds_start,ds_end], is replaced by the dead marker below)
            carved.append({
                "segment_start": ds_start,
                "segment_end": ds_end,
                "segment_type": "D",
                "title": _DEAD_LABELS.get(ds.kind, "Dead space (auto-detected)"),
            })
            segments = carved
        # Re-sort and drop any zero/negative-length slivers from carving
        segments = [s for s in segments if s["segment_end"] - s["segment_start"] > 0.05]
        segments.sort(key=lambda s: s["segment_start"])

    # Guarantee the timeline is a gapless partition of [0, duration]: snap the first
    # segment to 0 and the last to the full duration, so a sub-second head/tail sliver
    # below the dead-space-marker threshold isn't left as an uncovered gap that the
    # Editor silently drops when trimming (only listed segments are kept).
    if segments:
        segments[0]["segment_start"] = 0
        segments[-1]["segment_end"] = round(prop.duration, 2)

    # Group content segments into programs (tape -> programs -> segments). Only when
    # there's real content segmentation and more than one/near-one content segment;
    # snow splices delimit programs ONLY when they coincide with a speaker change or a
    # new title card (a program keeps its speakers across intervening clips/splices).
    programs = []
    if content_segments:
        snow_spans = [(round(ds.start, 2), round(ds.end, 2))
                      for ds in (mid_dead_spans or []) if getattr(ds, "kind", "") == "snow"]
        programs, _ = group_into_programs(segments, snow_spans)

    # Phase 2 (best-effort Bedrock): title each program (title cards + intro/closing),
    # and generate a brief high-level summary for EVERY video. Empty/skip on any failure
    # or when no client is available — never blocks segment JSON output.
    video_summary = ""
    single_program = {}
    if content_segments:
        try:
            from segment_label import (label_programs, summarize_video,
                                        title_single_program)
            if programs:
                label_programs(programs, segments, client)
            else:
                # Single-program tape: still give it ONE program title (from the same
                # title-card / intro / closing signals) so the video has a program name.
                single_program = title_single_program(segments, client)
            video_summary = summarize_video(programs, segments, client,
                                            program_title=single_program.get("title", ""))
        except Exception as e:
            log.warning(f"Program/summary labeling skipped: {e}")

    out = {
        "video": stem,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "auto_detected": True,
        "content_segmented": bool(content_segments),
        "trim_status": prop.status,
        "duration": round(prop.duration, 2),
        "kept_pct": round(prop.kept_pct, 1),
        "segments": segments,
    }
    if video_summary:
        # High-level catalogue summary of the whole video (generated for every video).
        out["summary"] = video_summary
    if programs:
        # Program tier with LLM-guessed title/description (Phase 2), human-editable.
        # Absent for single-program / single-segment videos.
        out["programs"] = programs
    elif single_program.get("title"):
        # Single-program tape: one program name for the whole video (human-editable).
        out["program_title"] = single_program["title"]
        if single_program.get("description"):
            out["program_description"] = single_program["description"]
        if single_program.get("confidence"):
            out["program_confidence"] = single_program["confidence"]
    if taxonomy:
        # The per-video content types the model discovered, so the Editor can show
        # (and let a human correct) the vocabulary the labels came from.
        out["taxonomy"] = taxonomy
    return out


def write_review_marker(s3_client, bucket: str, s3_key: str, prop) -> None:
    """Drop a review marker so a human can find flagged files."""
    stem = Path(s3_key).name.rsplit(".", 1)[0]
    original_filename = resolve_original_filename(s3_client, bucket, s3_key)
    body = (
        f"original_filename: {original_filename}\n"
        f"source: s3://{bucket}/{s3_key}\n"
        f"status: {prop.status}\n"
        f"proposed content: {prop.content_start:.2f}s -> {prop.content_end:.2f}s\n"
        f"kept: {prop.kept_pct:.0f}%\n"
        + "".join(f"note: {n}\n" for n in prop.notes)
    )
    try:
        s3_client.put_object(Bucket=bucket, Key=f"review/{stem}.txt",
                             Body=body.encode("utf-8"),
                             Metadata={"original-filename": original_filename})
    except Exception as e:
        log.warning(f"Could not write review marker: {e}")


# --- Main Orchestrator ---
def main():
    pipeline_start_time = time.time()
    mode = os.environ.get("MODE", "detect")

    s3_bucket = os.environ.get("S3_BUCKET")
    s3_key = os.environ.get("S3_KEY")
    if not s3_bucket or not s3_key:
        log.error("S3_BUCKET and S3_KEY environment variables are required.")
        sys.exit(1)

    if mode == "trim":
        run_trim(s3_bucket, s3_key, pipeline_start_time)
    else:
        run_detect(s3_bucket, s3_key, pipeline_start_time)


def run_trim(s3_bucket, s3_key, pipeline_start_time):
    """Trim video based on user-approved segments from a trim request."""
    log.info("=== SCUA Video Trimmer Starting ===")
    trim_request_key = os.environ.get("TRIM_REQUEST_KEY", "")
    s3_client = boto3.client("s3")
    original_filename = resolve_original_filename(s3_client, s3_bucket, s3_key)
    log.info(f"Trimming original video '{original_filename}' (s3://{s3_bucket}/{s3_key})")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        try:
            # Load trim request
            log.info(f"Loading trim request: s3://{s3_bucket}/{trim_request_key}")
            resp = s3_client.get_object(Bucket=s3_bucket, Key=trim_request_key)
            trim_data = json.loads(resp["Body"].read().decode("utf-8"))
            keep_segments = trim_data.get("keep_segments", [])
            video_name = trim_data.get("video", "unknown")

            if not keep_segments:
                log.error("No keep_segments in trim request")
                sys.exit(1)

            # Download video
            filename = Path(s3_key).name
            local_video_path = temp_dir / filename
            log.info(f"Downloading video: s3://{s3_bucket}/{s3_key}")
            s3_client.download_file(s3_bucket, s3_key, str(local_video_path))

            # Trim: concatenate all keep segments
            trimmed_path = temp_dir / f"{video_name}_trimmed.mp4"

            if len(keep_segments) == 1:
                # Single segment — simple cut
                seg = keep_segments[0]
                log.info(f"Trimming single segment: {seg['start']:.2f}s -> {seg['end']:.2f}s")
                apply_trim(str(local_video_path), str(trimmed_path), seg["start"], seg["end"])
            else:
                # Multiple segments — cut each then concatenate
                clip_paths = []
                for i, seg in enumerate(keep_segments):
                    clip_path = temp_dir / f"clip_{i:03d}.mp4"
                    log.info(f"  Cutting clip {i}: {seg['start']:.2f}s -> {seg['end']:.2f}s")
                    apply_trim(str(local_video_path), str(clip_path), seg["start"], seg["end"])
                    clip_paths.append(clip_path)

                # Write concat list
                concat_file = temp_dir / "concat.txt"
                with open(concat_file, "w") as f:
                    for cp in clip_paths:
                        f.write(f"file '{cp}'\n")

                # Concatenate with a RE-ENCODE (not -c copy). The clips are already
                # re-encoded with identical settings, but concatenating with -c copy can
                # still glitch at boundaries (timestamp/GOP discontinuities); re-encoding
                # the joined stream guarantees clean cuts. -c:a aac is a no-op when the
                # source has no audio (silent clips), so this also handles no-audio video.
                log.info(f"Concatenating {len(clip_paths)} clips (re-encode)...")
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-y", "-f", "concat", "-safe", "0",
                     "-i", str(concat_file),
                     "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                     "-c:a", "aac", "-b:a", "128k",
                     str(trimmed_path)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL, check=True,
                )

            # Prefer the explicit output_key from the frontend: the trimmed object
            # is named by the trimmed Video row's permanent id (edit/<id>.mp4), so
            # it's decoupled from any display name and stable across renames. Fall
            # back to the legacy sanitized-name path for older requests.
            output_key = trim_data.get("output_key") or \
                f"edit/{safe_output_name(trim_data.get('output_name'), video_name)}"
            log.info(f"Uploading trimmed video to s3://{s3_bucket}/{output_key} "
                     f"(original-filename='{original_filename}')")
            # Derivatives carry the SOURCE original's filename so every object traces
            # back to the archival item, despite the UUID key.
            s3_client.upload_file(str(trimmed_path), s3_bucket, output_key,
                                  ExtraArgs={"Metadata": {"original-filename": original_filename}})

            log.info(f"=== SCUA Video Trimmer Finished Successfully in "
                     f"{time.time() - pipeline_start_time:.2f}s ===")

        except ClientError as e:
            log.error(f"S3 error: {e.response['Error']['Message']}")
            log.error(traceback.format_exc())
            sys.exit(1)
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            log.error(traceback.format_exc())
            sys.exit(1)


def run_detect(s3_bucket, s3_key, pipeline_start_time):
    """Detect dead space and write segment JSON."""
    log.info("=== SCUA Segment Detector Starting ===")
    s3_client = boto3.client("s3")
    original_filename = resolve_original_filename(s3_client, s3_bucket, s3_key)
    log.info(f"Processing original video '{original_filename}' "
             f"(s3://{s3_bucket}/{s3_key}, mode={TRIM_MODE})")

    with tempfile.TemporaryDirectory() as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        try:
            filename = Path(s3_key).name
            stem = filename.rsplit(".", 1)[0]
            local_video_path = temp_dir / filename
            log.info(f"Downloading video to {local_video_path}...")
            s3_client.download_file(s3_bucket, s3_key, str(local_video_path))

            # --- STAGE 0: TRANSCRIBE FIRST (so speech can veto false dead space) ---
            # VHS audio is low-level and often reads as "silent" to ffmpeg's
            # silencedetect, so we can't rely on audio energy to tell dark program
            # from dead space. The transcript is authoritative: if Transcribe found
            # speech in a range, that range is content regardless of black frames.
            prefetched_turns = None
            speech_windows = []
            video_has_audio = True
            if _should_run_content_segmentation():
                video_has_audio = has_audio_stream(str(local_video_path))
                if not video_has_audio:
                    # Silent video (e.g. a game clip with no commentary): skip
                    # transcription entirely and let Stage 2 segment visually.
                    log.info("--- Stage 0 (Transcribe): no audio stream; "
                             "skipping transcription, will segment visually ---")
                else:
                    try:
                        prefetched_turns = get_transcript_turns(
                            local_video_path, s3_bucket, s3_key)
                        speech_windows = speech_windows_from_turns(prefetched_turns)
                        log.info(f"--- Stage 0 (Transcribe): {len(prefetched_turns or [])} turns, "
                                 f"{len(speech_windows)} speech windows ---")
                    except Exception as e:
                        log.warning(f"Stage 0 transcription failed ({e}); "
                                    f"dead-space runs without speech veto")

            # --- STAGE 1: PROBE ---
            stage1_start = time.time()
            if FULL_VIDEO_SCAN:
                # Full-video dead-space detection: finds ALL dead spans throughout
                full_analysis = analyze_full_video(
                    str(local_video_path), mode=TRIM_MODE,
                    merge_gap=MERGE_GAP, min_dead_dur=MIN_DEAD_DUR,
                    min_content_dur=MIN_CONTENT_DUR, black_pic_th=BLACK_PIC_TH,
                    speech_windows=speech_windows)
                # Build a Proposal-compatible object for the rest of the pipeline
                # using the first content span start and last content span end
                if full_analysis.content_spans:
                    cs_start = full_analysis.content_spans[0].start
                    cs_end = full_analysis.content_spans[-1].end
                else:
                    cs_start, cs_end = 0.0, full_analysis.duration
                from analyze_deadspace import Proposal
                prop = Proposal(full_analysis.duration, cs_start, cs_end,
                                full_analysis.status, full_analysis.notes,
                                full_analysis.regions)
                # Head-leader guard: a color-bars/snow/black leader that plays LOUD
                # noise or tone can slip through (loud audio defeats silence, and
                # Transcribe may hallucinate "speech" over the noise, vetoing the dead
                # span) -- leaving content_start pinned at the leader. Reclassify the
                # leading visual-dead run as dead space directly from the audio-blind
                # visual detectors, and drop any (spurious) speech windows sitting
                # inside it so they can't re-absorb it downstream.
                lead_end = head_leader_end(full_analysis.regions)
                if lead_end > cs_start + 0.5:
                    log.info(f"  HEAD-LEADER guard: reclassifying 0->{lead_end:.1f}s "
                             f"as dead (visual leader; audio ignored)")
                    prop = replace(prop, content_start=lead_end)
                    cs_start = lead_end
                    speech_windows = [(s, e) for (s, e) in speech_windows if e > lead_end + 0.5]
                # Mid-video dead spans (bars/black/snow between content) — these fall
                # inside [cs_start, cs_end] so build_segment_json must inject them as
                # "D" markers, else they're hidden inside the content window.
                mid_dead_spans = [
                    ds for ds in full_analysis.dead_spans
                    if ds.start > cs_start + 0.5 and ds.end < cs_end - 0.5
                ]
                for ds in mid_dead_spans:
                    log.info(f"  MID-DEAD [{ds.start:.1f}s -> {ds.end:.1f}s] {ds.kind}")
                log.info(
                    f"--- Stage 1 (Full-video probe) completed in "
                    f"{time.time() - stage1_start:.2f}s ---\n"
                    f"  Duration: {full_analysis.duration:.1f}s | "
                    f"Dead: {len(full_analysis.dead_spans)} spans | "
                    f"Content: {len(full_analysis.content_spans)} spans | "
                    f"Status: {full_analysis.status}")
                for ds in full_analysis.dead_spans:
                    log.info(f"  DEAD [{ds.start:.1f}s -> {ds.end:.1f}s] {ds.kind}")
            else:
                prop = analyze(str(local_video_path), mode=TRIM_MODE)
                full_analysis = None
                mid_dead_spans = []  # head/tail-only mode has no mid-video dead spans
                log.info(
                    f"--- Stage 1 (Probe) completed in {time.time() - stage1_start:.2f}s --- "
                    f"content {prop.content_start:.2f}s -> {prop.content_end:.2f}s | "
                    f"head {prop.head_trim:.2f}s, tail {prop.tail_trim:.2f}s | "
                    f"kept {prop.kept_pct:.0f}% | status={prop.status}")
            for n in prop.notes:
                log.info(f"  note: {n}")

            # --- STAGE 2: AUDIO-VISUAL FUSION SEGMENTATION ---
            stage2_start = time.time()
            content_segments, taxonomy, transcript_turns = None, None, None

            # ONE Bedrock client per video, shared across fusion labeling (Stage 2) and
            # Phase 2 program-title/summary labeling (build_segment_json). Best-effort:
            # if it can't be created, segmentation still runs (labels/summary stay empty).
            label_client = None
            if _should_run_content_segmentation():
                try:
                    from bedrock import make_client
                    label_client = make_client()
                except Exception as e:
                    log.warning(f"Bedrock client unavailable (labels/summary will be empty): {e}")

            # Segmentation is decoupled from the dead-space VERDICT: a NEEDS_REVIEW trim
            # (odd leader, short clip, over-eager detectors) should still yield a
            # segmented timeline, not one big block. When the trim is OK we segment the
            # trusted content window; otherwise we segment the WHOLE file and emit no
            # head/tail dead-space markers, since we didn't trust the trim. `seg_prop`
            # drives BOTH the segmentation window and the head/tail markers so they can
            # never disagree; for the OK case it is `prop` itself (no behaviour change).
            seg_prop = (prop if prop.status == "OK" and prop.kept > 0
                        else replace(prop, content_start=0.0, content_end=prop.duration))

            if _should_run_content_segmentation() and seg_prop.kept > 0 and not video_has_audio:
                # Silent video: skip fusion (needs a transcript) and segment by
                # content type visually — labels scenes "Football game", etc.
                log.info("Running visual content-type segmentation (no audio)...")
                try:
                    content_segments = run_visual_segmentation(local_video_path, seg_prop)
                    taxonomy, transcript_turns = None, None
                    log.info(f"Visual segmentation: {len(content_segments)} scenes "
                             f"in {time.time() - stage2_start:.2f}s")
                except Exception as e2:
                    log.warning(f"Visual segmentation failed (falling back to dead-space only): {e2}")
                    log.warning(traceback.format_exc())
                    content_segments, taxonomy, transcript_turns = None, None, None
            elif _should_run_content_segmentation() and seg_prop.kept > 0:
                if prop.status != "OK":
                    log.info(f"Trim status={prop.status}; segmenting whole file "
                             f"0->{prop.duration:.1f}s (no auto dead-space markers)")
                log.info("Running audio-visual fusion segmentation...")
                try:
                    content_segments, taxonomy, transcript_turns = run_fusion_segmentation(
                        local_video_path, s3_bucket, s3_key, seg_prop,
                        turns=prefetched_turns,
                        dead_spans=(full_analysis.dead_spans if full_analysis else None),
                        client=label_client)
                    log.info(f"Fusion segmentation complete: {len(content_segments)} segments "
                             f"in {time.time() - stage2_start:.2f}s")
                    for s in content_segments:
                        log.info(f"  [{s['start']:.2f}s -> {s['end']:.2f}s] "
                                 f"{s['label']} ({s.get('name', '')})")
                    # Cache transcript + VTT to S3 for reuse on re-segmentation
                    if transcript_turns:
                        transcript_s3_key = f"transcript/{stem}.json"
                        try:
                            _orig_meta = {"original-filename": original_filename}
                            s3_client.put_object(
                                Bucket=s3_bucket, Key=transcript_s3_key,
                                Body=json.dumps(transcript_turns, indent=2).encode("utf-8"),
                                ContentType="application/json", Metadata=_orig_meta)
                            s3_client.put_object(
                                Bucket=s3_bucket, Key=transcript_s3_key.replace(".json", ".vtt"),
                                Body=turns_to_vtt(transcript_turns).encode("utf-8"),
                                ContentType="text/vtt", Metadata=_orig_meta)
                            log.info(f"  Cached transcript + VTT to s3://{s3_bucket}/{transcript_s3_key}")
                        except Exception as cache_err:
                            log.warning(f"  Could not cache transcript: {cache_err}")
                except Exception as e:
                    # No usable audio (silent clip / no audio stream / Transcribe
                    # failed) or a broken fusion stage: fall back to VISUAL shot
                    # segmentation so the video is still segmented instead of one
                    # big block. Only if THAT fails too do we drop to dead-space only.
                    log.warning(f"Fusion segmentation unavailable ({e}); trying visual shot segmentation")
                    try:
                        content_segments = run_visual_segmentation(local_video_path, seg_prop)
                        taxonomy, transcript_turns = None, None
                        log.info(f"Visual shot segmentation: {len(content_segments)} shots "
                                 f"in {time.time() - stage2_start:.2f}s")
                    except Exception as e2:
                        log.warning(f"Visual segmentation also failed (falling back to dead-space only): {e2}")
                        log.warning(traceback.format_exc())
                        content_segments, taxonomy, transcript_turns = None, None, None
            elif not _should_run_content_segmentation():
                log.info("Content segmentation disabled (CONTENT_SEGMENT=off)")
            else:
                log.info("Skipping content segmentation (empty content span)")

            # --- STAGE 3: WRITE SEGMENTS ---
            stage3_start = time.time()
            seg_s3_key = segment_key(s3_key)

            if prop.status == "NEEDS_REVIEW":
                write_review_marker(s3_client, s3_bucket, s3_key, prop)

            # Always write segment JSON so the Editor can display it. Use `seg_prop`
            # (not `prop`) so the head/tail dead-space markers match the window the
            # segments were actually computed over.
            # Phase 2 program-title + video-summary labeling reuses the SAME Bedrock
            # client created above (one per video); best-effort, so a None client just
            # leaves labels/summary empty.
            seg_json = build_segment_json(s3_key, seg_prop, content_segments, taxonomy,
                                          transcript_turns, mid_dead_spans=mid_dead_spans,
                                          client=label_client)
            log.info(f"Uploading segment JSON to s3://{s3_bucket}/{seg_s3_key}")
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=seg_s3_key,
                Body=json.dumps(seg_json, indent=2).encode("utf-8"),
                ContentType="application/json",
                Metadata={"original-filename": original_filename},
            )

            log.info(f"--- Stage 3 (Write Segments) completed in "
                     f"{time.time() - stage3_start:.2f}s ---")

        except ClientError as e:
            log.error(f"S3 error: {e.response['Error']['Message']}")
            log.error(traceback.format_exc())
            sys.exit(1)
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            log.error(traceback.format_exc())
            sys.exit(1)

    log.info(f"=== SCUA Segment Detector Finished Successfully in "
             f"{time.time() - pipeline_start_time:.2f}s ===")


if __name__ == "__main__":
    main()
