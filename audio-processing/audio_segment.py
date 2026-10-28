"""Audio-only segmentation for SCUA / Music Library recordings.

Finds where each song starts and ends in an audio file, and marks the non-program
stretches (silence, tape hiss/static, test tones) as dead space, so the Editor
can show an audio file on the same timeline it uses for video.

Why classical DSP (numpy + ffmpeg) instead of a neural model
------------------------------------------------------------
The container already ships ffmpeg and numpy, so this adds NO new dependency and
NO model weights: the image stays the same size and the code runs on the same
CPU-only Fargate/EC2 task as the video pipeline. Each detector is a few lines of
arithmetic on a spectrogram, so a reviewer can see exactly why a boundary was
placed and tune it with an env var. Speech detection leans on Amazon Transcribe
(already run for every file), so the acoustic classifier only has to answer the
easier question "is this music or talk?". If real SCUA tapes show it confusing
classes, a pretrained classifier (YAMNet via onnxruntime) can replace
`classify_rows` without touching anything else.

Pipeline (all streaming; memory stays flat for multi-hour tapes)
---------------------------------------------------------------
1. ffmpeg decodes to 16 kHz mono float32 and pipes it here in blocks.
2. Every HOP (0.5 s) we compute one feature row: loudness, a 20 ms loudness
   envelope, spectral flatness (noise vs tone), spectral flux, a 12-bin chroma
   (pitch-class profile), 24 log-spaced band energies, a pure-tone ratio, and
   spectral-peak persistence (do partials stay in the same FFT bin 32 ms later?).
3. Each row is classified as silence / hiss / tone / static / applause /
   speech / music from those features over a few seconds of context, then
   smoothed (sustained tones, static and silence are dead space).
4. Rows become regions; short blips are absorbed into their neighbours.
5. Music regions are split into songs at (a) quiet gaps >= SONG_GAP_MIN, and
   (b) novelty peaks: points where the average timbre/harmony of the preceding
   NOVELTY_WIN seconds differs sharply from the following NOVELTY_WIN seconds
   (Foote's checkerboard novelty, computed in O(n) as the distance between the
   left- and right-window feature means). Novelty cuts are marked low confidence.

Run locally (no AWS needed):
    python audio_segment.py some_tape.wav            # prints the segment table
    python audio_segment.py some_tape.wav --json     # segment list as JSON
"""
from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass, field, asdict

import numpy as np

# ── Tunables (env-overridable so the stack can adjust without a code change) ──
SR = 16000
HOP_S = 0.5                                     # one feature row per HOP_S seconds
HOP = int(SR * HOP_S)
ENV_SUB = 320                                   # 20 ms loudness-envelope sub-frames
N_SHORT, H_SHORT = 1024, 512                    # 64 ms frames for flatness/flux/bands
N_LONG = 4096                                   # 256 ms frames for chroma + tone
LOOKAHEAD = N_LONG                              # extra samples a row may read past its hop

def _env_f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)

MIN_DEAD_DUR = _env_f("AUDIO_MIN_DEAD_DUR", 3.0)      # silence/hiss/tone/static shorter than this is not dead space
SONG_GAP_MIN = _env_f("AUDIO_SONG_GAP_MIN", 1.0)      # a quiet gap this long inside music separates two songs
MIN_SONG_DUR = _env_f("AUDIO_MIN_SONG_DUR", 30.0)     # novelty never cuts a song shorter than this
NOVELTY_WIN = _env_f("AUDIO_NOVELTY_WIN", 15.0)       # seconds compared on each side of a candidate cut
NOVELTY_K = _env_f("AUDIO_NOVELTY_K", 5.0)            # peak must exceed median + K * robust std
NOVELTY_MIN = _env_f("AUDIO_NOVELTY_MIN", 0.30)       # ...and this absolute distance (0..2 scale)
NOVELTY_ON = os.environ.get("AUDIO_NOVELTY", "on").lower() in ("on", "true", "1")
NOISE_FLAT = _env_f("AUDIO_NOISE_FLAT", 0.40)        # spectral flatness above this = broadband noise (static/applause)
MIN_REGION = _env_f("AUDIO_MIN_REGION", 4.0)          # speech/music blips shorter than this are absorbed
MIN_APPLAUSE = _env_f("AUDIO_MIN_APPLAUSE", 2.0)      # applause shorter than this is absorbed
QUIET_MARGIN_DB = _env_f("AUDIO_QUIET_MARGIN_DB", 6.0)   # quiet = within this many dB of the noise floor...
QUIET_BELOW_PROG_DB = _env_f("AUDIO_QUIET_BELOW_PROG_DB", 18.0)  # ...and at least this far below typical program level

DEAD_KINDS = ("silence", "hiss", "tone", "static")
DEAD_TITLES = {
    "silence": "Silence (auto-detected)",
    "hiss": "Tape hiss / room noise (auto-detected)",
    "tone": "Test tone (auto-detected)",
    "static": "Static / noise (auto-detected)",
}


# ─────────────────────────────── decoding ────────────────────────────────────
def get_duration(path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path], stderr=subprocess.DEVNULL).decode().strip()
    return float(out)


def is_audio_only(path: str) -> bool:
    """True when the file has an audio stream and no REAL video stream.

    Cover art embedded in mp3/m4a/flac shows up to ffprobe as a one-frame video
    stream flagged `attached_pic`; that is still an audio file."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type:stream_disposition=attached_pic", "-of", "json", path],
            stderr=subprocess.DEVNULL).decode()
        streams = json.loads(out).get("streams", [])
    except Exception:
        return False
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    has_video = any(s.get("codec_type") == "video"
                    and not (s.get("disposition") or {}).get("attached_pic")
                    for s in streams)
    return has_audio and not has_video


def _pcm_blocks(path: str, block_s: float = 60.0):
    """Yield float32 mono 16 kHz blocks of the file's audio via an ffmpeg pipe."""
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-vn", "-ac", "1",
         "-ar", str(SR), "-f", "f32le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    nbytes = int(block_s * SR) * 4
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if not buf:
                break
            yield np.frombuffer(buf[: len(buf) - len(buf) % 4], dtype=np.float32)
    finally:
        proc.stdout.close()
        proc.wait()


# ──────────────────────────── feature extraction ─────────────────────────────
_WIN_S = np.hanning(N_SHORT).astype(np.float32)
_WIN_L = np.hanning(N_LONG).astype(np.float32)
_FREQ_S = np.fft.rfftfreq(N_SHORT, 1 / SR)
_FREQ_L = np.fft.rfftfreq(N_LONG, 1 / SR)
_FLAT_BAND = (_FREQ_S >= 150) & (_FREQ_S <= 6000)
_BAND_EDGES = np.geomspace(60, 7600, 25)
_BAND_IDX = [np.where((_FREQ_S >= lo) & (_FREQ_S < hi))[0] for lo, hi in zip(_BAND_EDGES[:-1], _BAND_EDGES[1:])]
_CHROMA_MASK = (_FREQ_L >= 65) & (_FREQ_L <= 2100)
_CHROMA_BIN = (np.round(12 * np.log2(np.maximum(_FREQ_L, 1) / 440.0)) % 12).astype(int)
_TONE_MASK = _FREQ_L >= 100
_PEAK_IDX = np.where((_FREQ_S >= 150) & (_FREQ_S <= 2000))[0]


def _peak_persistence(P: np.ndarray, top: int = 8, floor_nat: float = 7.0) -> float:
    """Share of a frame's strongest spectral peaks (150-2000 Hz) that sit in the
    SAME FFT bin 32 ms later. Held musical notes keep their partials in place;
    speech pitch glides continuously, so its harmonics move between bins.
    This is the single most useful music/speech cue in our test material."""
    B = np.log(P[:, _PEAK_IDX])
    inner = B[:, 1:-1]
    is_pk = (inner > B[:, :-2]) & (inner >= B[:, 2:]) & (inner > B.max(axis=1, keepdims=True) - floor_nat)
    score = np.where(is_pk, inner, -np.inf)
    k = min(top, score.shape[1])
    keep = np.zeros_like(is_pk)
    idx = np.argpartition(-score, k - 1, axis=1)[:, :k]
    np.put_along_axis(keep, idx, True, axis=1)
    keep &= is_pk
    a, b = keep[:-1], keep[1:]
    n_a = a.sum(axis=1)
    ok = n_a > 0
    if not ok.any():
        return 0.0
    return float(np.mean((a & b).sum(axis=1)[ok] / n_a[ok]))


@dataclass
class Rows:
    """Per-HOP feature matrix for a whole file (one row per HOP_S seconds)."""
    t: np.ndarray            # row start times (s)
    rms_db: np.ndarray       # loudness, dBFS
    env: np.ndarray          # [n, HOP/ENV_SUB] 20 ms RMS envelope (linear)
    flat: np.ndarray         # spectral flatness 0..1 (1 = white noise)
    flux: np.ndarray         # mean positive spectral change between 64 ms frames
    bands: np.ndarray        # [n, 24] log band energies
    chroma: np.ndarray       # [n, 12] L2-normalised pitch-class profile
    tone: np.ndarray         # share of energy in the strongest +-3 bins (1 = pure tone)
    tone_hz: np.ndarray      # frequency of that strongest bin
    persist: np.ndarray      # spectral-peak persistence 0..1 (held notes -> high)
    duration: float = 0.0


def _row_features(x: np.ndarray):
    """Features for one row: x holds HOP samples plus LOOKAHEAD context."""
    core = x[:HOP]
    rms = float(np.sqrt(np.mean(core.astype(np.float64) ** 2)) + 1e-10)
    env = np.sqrt(np.mean(core.reshape(-1, ENV_SUB).astype(np.float64) ** 2, axis=1))

    starts = range(0, HOP, H_SHORT)
    frames = np.stack([x[s:s + N_SHORT] for s in starts]) * _WIN_S
    P = np.abs(np.fft.rfft(frames, axis=1)) ** 2 + 1e-12
    Pb = P[:, _FLAT_BAND]
    flat = float(np.mean(np.exp(np.mean(np.log(Pb), axis=1)) / np.mean(Pb, axis=1)))
    L = np.log(P)
    flux = float(np.mean(np.maximum(np.diff(L, axis=0), 0).mean(axis=1))) if len(L) > 1 else 0.0
    Pm = P.mean(axis=0)
    bands = np.log(np.array([Pm[i].sum() for i in _BAND_IDX]) + 1e-10)

    lf = np.stack([x[0:N_LONG], x[HOP // 2:HOP // 2 + N_LONG]]) * _WIN_L
    PL = (np.abs(np.fft.rfft(lf, axis=1)) ** 2).mean(axis=0) + 1e-12
    chroma = np.bincount(_CHROMA_BIN[_CHROMA_MASK], weights=PL[_CHROMA_MASK], minlength=12)
    chroma = chroma / (np.linalg.norm(chroma) + 1e-12)
    PT = PL * _TONE_MASK
    k = int(np.argmax(PT))
    tone = float(PT[max(0, k - 3):k + 4].sum() / PT.sum())
    return (20 * math.log10(rms), env, flat, flux, bands, chroma, tone, float(_FREQ_L[k]),
            _peak_persistence(P))


def extract_rows(path: str) -> Rows:
    """Stream the file through ffmpeg and compute one feature row per HOP."""
    cols = {k: [] for k in ("rms_db", "env", "flat", "flux", "bands", "chroma", "tone", "tone_hz", "persist")}
    buf = np.zeros(0, dtype=np.float32)
    n_samples = 0

    def consume(b, final=False):
        while len(b) >= HOP + LOOKAHEAD or (final and len(b) > 0):
            x = b[:HOP + LOOKAHEAD]
            if len(x) < HOP + LOOKAHEAD:
                x = np.pad(x, (0, HOP + LOOKAHEAD - len(x)))
            for key, v in zip(cols, _row_features(x)):
                cols[key].append(v)
            b = b[HOP:]
            if final and len(b) <= 0:
                break
        return b

    for blk in _pcm_blocks(path):
        n_samples += len(blk)
        buf = consume(np.concatenate([buf, blk]))
    # Flush the tail. Only rows that START inside the real audio are kept.
    n_rows_expected = int(math.ceil(n_samples / HOP))
    while len(cols["rms_db"]) < n_rows_expected and len(buf) > 0:
        buf = consume(buf, final=True)

    n = len(cols["rms_db"])
    if n == 0:
        raise RuntimeError("no audio decoded")
    return Rows(
        t=np.arange(n) * HOP_S,
        rms_db=np.array(cols["rms_db"]),
        env=np.stack(cols["env"]),
        flat=np.array(cols["flat"]),
        flux=np.array(cols["flux"]),
        bands=np.stack(cols["bands"]),
        chroma=np.stack(cols["chroma"]),
        tone=np.array(cols["tone"]),
        tone_hz=np.array(cols["tone_hz"]),
        persist=np.array(cols["persist"]),
        duration=n_samples / SR,
    )


# ───────────────────────────── classification ────────────────────────────────
def _windowed(a: np.ndarray, half: int, fn):
    """fn over a centred window of +-half rows (clipped at the edges)."""
    return np.array([fn(a[max(0, i - half): i + half + 1]) for i in range(len(a))])


def quiet_threshold(rms_db: np.ndarray) -> float:
    """Level below which a row counts as 'quiet' (gap between items).

    Archival tape never reaches digital silence between songs; it sits on a hiss
    floor. So the threshold is relative: a few dB above the file's own noise
    floor, but always well below its typical program level."""
    live = rms_db[rms_db > -85]
    if len(live) == 0:
        return -60.0
    floor = float(np.percentile(live, 3))
    prog = float(np.percentile(live, 60))
    return min(floor + QUIET_MARGIN_DB, prog - QUIET_BELOW_PROG_DB)


def classify_rows(R: Rows, speech_mask: np.ndarray | None = None):
    """Label every row. Returns (labels, scores) where labels are strings.

    speech_mask (optional) is True where Amazon Transcribe heard words. Words are
    evidence for speech, but sung lyrics are transcribed too, so they only tip a
    row that the acoustics leave ambiguous; clearly musical rows stay music."""
    n = len(R.t)
    qth = quiet_threshold(R.rms_db)
    quiet = R.rms_db < qth
    live = R.rms_db[R.rms_db > qth]
    prog = float(np.percentile(live, 60)) if len(live) else -30.0

    # Low-energy ratio: share of 20 ms sub-frames well below the local mean.
    # Speech pauses between syllables/words -> high; sustained music -> low.
    env = R.env
    half = 4                                            # +-2 s context
    ler = np.empty(n)
    env_cv = np.empty(n)
    for i in range(n):
        w = env[max(0, i - half): i + half + 1].ravel()
        m = w.mean() + 1e-12
        ler[i] = np.mean(w < 0.5 * m)
        env_cv[i] = w.std() / m
    # Harmonic stability: how alike consecutive chroma profiles are. Held notes
    # and chords repeat; speech and noise do not.
    cs = np.r_[1.0, np.sum(R.chroma[1:] * R.chroma[:-1], axis=1)]
    chroma_stab = _windowed(cs, half, np.mean)
    # Rhythmic periodicity: peak autocorrelation of the (log) loudness envelope
    # at beat-like lags (0.3-2 s) over ~8 s. Drums and steady pulses score high;
    # speech has no regular pulse. Without this, percussive music (lots of
    # short gaps between hits) looks like speech to the low-energy ratio.
    periodicity = _envelope_periodicity(env)
    flat_w = _windowed(R.flat, 2, np.mean)
    # Burstiness WITHIN each 0.5 s row (not across the context window, which
    # would see a level step at every boundary): steady static ~0.05, applause
    # (overlapping claps) ~0.8.
    row_cv = env.std(axis=1) / (env.mean(axis=1) + 1e-12)
    burst = _windowed(row_cv, 1, np.median)
    tone_w = _windowed(R.tone, 1, np.min)
    # A sustained pure tone keeps the same frequency row to row.
    hz = R.tone_hz
    hz_stable = np.r_[False, np.abs(np.diff(hz)) <= np.maximum(4.0, 0.02 * hz[1:])]
    hz_stable = hz_stable | np.r_[hz_stable[1:], False]

    # Speech-vs-music score: positive -> speech. Each term is centred on the
    # value that separated the classes in our labelled test clips (README,
    # "Audio Recordings"; re-check against real SCUA tapes and adjust).
    persist = _windowed(R.persist, half, np.mean)
    speech_score = (2.0 * (ler - 0.28)
                    + 2.0 * (0.66 - chroma_stab)
                    + 6.0 * (0.28 - persist)
                    - 3.0 * np.maximum(0.0, periodicity - 0.40))
    if speech_mask is not None:
        speech_score = speech_score + 0.5 * speech_mask.astype(float)

    labels = np.empty(n, dtype=object)
    for i in range(n):
        if quiet[i]:
            labels[i] = "silence" if R.rms_db[i] < -60 or flat_w[i] < 0.25 else "hiss"
        elif tone_w[i] > 0.6 and hz_stable[i] and (R.tone_hz[i] >= 250 or R.rms_db[i] > prog - 6):
            # Line-up/test tones are recorded at reference level. A quiet low
            # tone (50/60/120 Hz) is mains hum under the program, not a tone.
            labels[i] = "tone"
        elif flat_w[i] > NOISE_FLAT:
            # Broadband noise: stationary -> tape static; bursty -> applause.
            labels[i] = "applause" if burst[i] > 0.35 else "static"
        else:
            labels[i] = "speech" if speech_score[i] > 0 else "music"

    # Majority-smooth speech/music over ~3 s so single rows don't flicker.
    sm = labels.copy()
    for i in range(n):
        if labels[i] in ("speech", "music"):
            w = labels[max(0, i - 3): i + 4]
            sp = np.sum(w == "speech")
            mu = np.sum(w == "music")
            if sp != mu:
                sm[i] = "speech" if sp > mu else "music"
    scores = {"ler": ler, "chroma_stab": chroma_stab, "env_cv": env_cv,
              "periodicity": periodicity, "persist": persist, "flat": flat_w,
              "speech_score": speech_score, "quiet_th": qth}
    return sm, scores


def _envelope_periodicity(env: np.ndarray, half_rows: int = 8,
                          min_lag_s: float = 0.3, max_lag_s: float = 2.0) -> np.ndarray:
    sub_s = ENV_SUB / SR
    lo, hi = int(min_lag_s / sub_s), int(max_lag_s / sub_s)
    lenv = np.log(env + 1e-6)
    out = np.zeros(len(env))
    for i in range(len(env)):
        w = lenv[max(0, i - half_rows): i + half_rows + 1].ravel()
        if len(w) <= hi + 1:
            continue
        w = w - w.mean()
        spec = np.fft.rfft(w, 2 * len(w))
        ac = np.fft.irfft(np.abs(spec) ** 2)[:len(w)]
        if ac[0] <= 0:
            continue
        out[i] = float(np.max(ac[lo:hi] / ac[0]))
    return out


# ─────────────────────────────── regions ─────────────────────────────────────
@dataclass
class Region:
    start: float
    end: float
    kind: str                       # song | speech | applause | silence | hiss | tone | static
    cues: list = field(default_factory=list)   # what put the boundary at `start`
    confidence: str = ""

    @property
    def dur(self):
        return self.end - self.start


def _runs(labels, t, duration):
    out = []
    i = 0
    n = len(labels)
    while i < n:
        j = i
        while j + 1 < n and labels[j + 1] == labels[i]:
            j += 1
        end = t[j + 1] if j + 1 < n else duration
        out.append(Region(float(t[i]), float(min(end, duration)), str(labels[i])))
        i = j + 1
    return out


def _merge_same(regs):
    out = []
    for r in regs:
        if out and out[-1].kind == r.kind:
            out[-1].end = r.end
        else:
            out.append(r)
    return out


def _min_dur(r: Region) -> float:
    if r.kind in DEAD_KINDS:
        # A short quiet gap inside music is a song separator, not dead space;
        # keep it as long as it can split songs (handled in build_regions).
        return SONG_GAP_MIN if r.kind in ("silence", "hiss") else MIN_DEAD_DUR
    return MIN_APPLAUSE if r.kind == "applause" else MIN_REGION


def _absorb_short(regs):
    """Fold regions shorter than their kind's minimum into a neighbour,
    shortest first, until none remain (keeps the timeline gapless)."""
    regs = _merge_same(regs)
    while True:
        cand = [(r.dur, i) for i, r in enumerate(regs) if r.dur < _min_dur(r)]
        if not cand or len(regs) == 1:
            return regs
        _, i = min(cand)
        r = regs[i]
        left = regs[i - 1] if i > 0 else None
        right = regs[i + 1] if i + 1 < len(regs) else None
        # Prefer a neighbour whose kind matches the other side (A x A -> A),
        # else the longer neighbour.
        if left and right and left.kind == right.kind:
            tgt = left
        elif left and right:
            tgt = left if left.dur >= right.dur else right
        else:
            tgt = left or right
        if tgt is left:
            left.end = r.end
        else:
            right.start = r.start
        del regs[i]
        regs = _merge_same(regs)


def novelty_curve(R: Rows, i0: int, i1: int):
    """Foote novelty for rows [i0, i1): distance between the mean feature vector
    of the NOVELTY_WIN seconds before each row and the NOVELTY_WIN seconds after.

    With L2-normalised rows and a box checkerboard kernel, the classic
    self-similarity-matrix novelty reduces exactly to |mu_left - mu_right|^2,
    so this is O(n) and never builds the n x n matrix."""
    X = np.hstack([R.chroma[i0:i1],
                   _znorm(R.bands[i0:i1]) * 0.35,
                   _znorm(R.flat[i0:i1, None]) * 0.35])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    k = max(2, int(round(NOVELTY_WIN / HOP_S)))
    c = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
    n = len(X)
    nov = np.zeros(n)
    for i in range(k, n - k):
        mu_l = (c[i] - c[i - k]) / k
        mu_r = (c[i + k] - c[i]) / k
        nov[i] = float(np.sum((mu_l - mu_r) ** 2))
    return nov


def _znorm(a):
    return (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-9)


def novelty_cuts(R: Rows, reg: Region):
    """Song boundaries inside one continuous music region (no gap between songs)."""
    i0, i1 = int(reg.start / HOP_S), int(math.ceil(reg.end / HOP_S))
    if (reg.end - reg.start) < 2 * MIN_SONG_DUR:
        return []
    nov = novelty_curve(R, i0, i1)
    k = max(2, int(round(NOVELTY_WIN / HOP_S)))
    core = nov[k:len(nov) - k]
    if len(core) < 3:
        return []
    med = float(np.median(core))
    mad = float(np.median(np.abs(core - med))) * 1.4826 + 1e-9
    th = max(med + NOVELTY_K * mad, NOVELTY_MIN)
    cuts = []
    for i in np.argsort(-nov):
        if nov[i] < th:
            break
        t = R.t[i0 + i]
        if t - reg.start < MIN_SONG_DUR or reg.end - t < MIN_SONG_DUR:
            continue
        if any(abs(t - c) < MIN_SONG_DUR for c in cuts):
            continue
        cuts.append(float(t))
    return sorted(cuts)


def build_regions(R: Rows, labels) -> list:
    """Turn row labels into the final region list (songs split, blips absorbed)."""
    regs = _absorb_short(_runs(list(labels), R.t, R.duration))

    # Quiet/noise runs shorter than MIN_DEAD_DUR are not dead space. When music
    # sits on both sides they mark a song change: split the gap between the two
    # songs and record the cue. Otherwise fold them into a neighbour (a pause in
    # speech, a breath before applause).
    out = []
    i = 0
    while i < len(regs):
        r = regs[i]
        if r.kind in DEAD_KINDS and r.dur < MIN_DEAD_DUR:
            prev = out[-1] if out else None
            nxt = regs[i + 1] if i + 1 < len(regs) else None
            mid = (r.start + r.end) / 2
            if prev and nxt and prev.kind == "music" and nxt.kind == "music":
                prev.end = mid
                nxt.start = mid
                nxt.cues = ["gap"]
                nxt.confidence = "high" if r.dur >= 2.0 else "medium"
            elif prev:
                prev.end = r.end
            elif nxt:
                nxt.start = r.start
            else:
                out.append(r)
            i += 1
            continue
        out.append(r)
        i += 1
    out = _merge_same_keep_cues(out)

    # Split long music runs at novelty peaks (song -> song with no gap).
    final = []
    for r in out:
        if r.kind == "music" and NOVELTY_ON:
            cuts = novelty_cuts(R, r)
            edges = [r.start] + cuts + [r.end]
            for j, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
                piece = Region(a, b, "music", list(r.cues) if j == 0 else ["novelty"],
                               r.confidence if j == 0 else "low")
                final.append(piece)
        else:
            final.append(r)

    # Cues for boundaries not set above: what kind of thing ended before this one.
    for prev, cur in zip(final[:-1], final[1:]):
        if not cur.cues:
            cur.cues = [prev.kind if prev.kind not in DEAD_KINDS else "dead"]
            cur.confidence = cur.confidence or "high"
    for r in final:
        r.kind = "song" if r.kind == "music" else r.kind
        r.start, r.end = round(r.start, 2), round(r.end, 2)
    if final:
        final[0].start = 0.0
        final[-1].end = round(R.duration, 2)
    return final


def _merge_same_keep_cues(regs):
    """Merge adjacent same-kind regions EXCEPT music split by a gap (two songs)."""
    out = []
    for r in regs:
        if out and out[-1].kind == r.kind and not (r.kind == "music" and r.cues):
            out[-1].end = r.end
        else:
            out.append(r)
    return out


def speech_mask_from_turns(turns, t: np.ndarray) -> np.ndarray:
    """Rows that overlap a transcribed turn."""
    m = np.zeros(len(t), dtype=bool)
    for tr in turns or []:
        a, b = float(tr["start"]), float(tr["end"])
        m |= (t + HOP_S > a) & (t < b)
    return m


def segment_audio(path: str, turns=None):
    """Full DSP pass: returns (regions, rows, labels, scores)."""
    R = extract_rows(path)
    mask = speech_mask_from_turns(turns, R.t) if turns else None
    labels, scores = classify_rows(R, mask)
    return build_regions(R, labels), R, labels, scores


# ─────────────────────────────────── CLI ─────────────────────────────────────
def _fmt(t):
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Segment an audio file into songs / speech / dead space.")
    p.add_argument("audio")
    p.add_argument("--json", action="store_true", help="print regions as JSON")
    p.add_argument("--rows", action="store_true", help="also print per-row features (debug)")
    a = p.parse_args(argv)
    regs, R, labels, sc = segment_audio(a.audio)
    if a.rows:
        for i in range(len(R.t)):
            print(f"{R.t[i]:7.1f} {labels[i]:9s} db={R.rms_db[i]:6.1f} flat={sc['flat'][i]:.2f} "
                  f"ler={sc['ler'][i]:.2f} cstab={sc['chroma_stab'][i]:.2f} cv={sc['env_cv'][i]:.2f} "
                  f"per={sc['periodicity'][i]:.2f} pp={sc['persist'][i]:.2f} tone={R.tone[i]:.2f}@{R.tone_hz[i]:.0f} s={sc['speech_score'][i]:+.2f}")
    if a.json:
        print(json.dumps([asdict(r) for r in regs], indent=2))
    else:
        print(f"duration {_fmt(R.duration)}  quiet<{sc['quiet_th']:.1f} dBFS  {len(regs)} regions")
        for r in regs:
            print(f"  {_fmt(r.start)} -> {_fmt(r.end)}  {r.kind:9s} {','.join(r.cues):10s} {r.confidence}")


if __name__ == "__main__":
    main()
