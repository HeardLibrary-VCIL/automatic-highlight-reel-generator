# audio-processing

Segments **audio-only** recordings (`.wav .mp3 .flac .m4a .aac .ogg .oga .opus .aif .aiff .wma`,
or any file with no real video stream) into **songs, speech, applause** and **dead space**
(silence, tape hiss, test tones, static), names each song with Claude, and writes the same
`segment/{id}.json` the SCUA Editor already reads for video.

It runs inside the same ECS container as `video-processing/`. `video-processing/main.py`
downloads every upload, checks it with `is_audio_only()`, and hands audio files to
`audio_pipeline.run_audio_detect()`; video files never touch this folder.

| File | What it does |
|------|--------------|
| `audio_pipeline.py` | Entry point called by `main.py`: Transcribe → segment → label → write JSON |
| `audio_segment.py` | DSP segmenter (numpy + ffmpeg): one feature row per 0.5 s → labels → regions → song splits |
| `audio_label.py` | Claude on Bedrock names each segment from speech + lyrics; pydantic-validated, one retry; builds the segment JSON |
| `test_audio.py` | Unit tests on synthesized audio (no downloads, no AWS) |
| `eval/make_tape.py` | Builds a labelled test tape from public music/speech clips |
| `eval/eval_audio.py` | Scores the segmenter against ground truth (row accuracy, boundary precision/recall) |
| `requirements.txt` | numpy, pydantic (ffmpeg comes from the image) |

## How boundaries are found

- **Dead space**: loudness relative to the file's own noise floor (tape sits on hiss, never
  digital silence); a stable dominant spectral peak at reference level is a test tone (quiet
  low hum under music is not); high spectral flatness is noise: steady = static (dead),
  bursty = applause (kept).
- **Music vs speech**: low-energy-frame ratio, chroma stability, spectral-peak persistence
  (held notes keep their partials in the same FFT bin; speech pitch glides) and rhythmic
  periodicity. Transcribe's speech spans add a nudge toward speech; sung lyrics are
  transcribed too, so they can't override clear music.
- **Song → song**: a quiet gap ≥ 1 s inside music, applause or speech between songs, or a
  novelty peak (mean timbre/harmony of the 15 s before vs after a point). Novelty cuts are
  marked `confidence: "low"`, `boundary_cues: ["novelty"]`.

## Run it locally (no AWS)

```bash
cd audio-processing
pip3 install -r requirements.txt               # ffmpeg must be installed too
python3 -m unittest test_audio -v              # 12 tests on synthetic audio
python3 audio_segment.py some_tape.wav         # print the segment table
python3 audio_segment.py some_tape.wav --rows  # per-0.5 s features, for tuning
```

Score against ground truth (public clips, MIT/Apache licensed):

```bash
git clone --depth 1 https://github.com/tyiannak/pyAudioAnalysis  samples/pyAA
git clone --depth 1 https://github.com/ina-foss/inaSpeechSegmenter samples/isS
python3 eval/make_tape.py samples /tmp/tape.wav
python3 eval/eval_audio.py /tmp/tape.wav /tmp/tape.truth.json --oracle-speech
```

On that tape: 98.5% of time labelled correctly, 12/12 boundaries within ±2 s, no false
cuts (with Transcribe's speech spans; 95.1% from audio alone). A 70-minute file takes about
18 s and 60 MB of RAM on one CPU. Re-tune on real SCUA tapes before trusting the numbers.

## Tuning (container env vars)

| Variable | Default | Meaning |
|----------|---------|---------|
| `AUDIO_PIPELINE` | `on` | `off` sends audio files down the video path instead (read by `main.py`) |
| `AUDIO_SONG_GAP_MIN` | `1.0` | A quiet gap this long (s) inside music separates two songs |
| `AUDIO_MIN_DEAD_DUR` | `3.0` | Silence/hiss/tone/static shorter than this (s) is not dead space |
| `AUDIO_MIN_SONG_DUR` | `30` | Novelty never cuts a song shorter than this (s) |
| `AUDIO_NOVELTY` | `on` | Split gapless songs at novelty peaks |
| `AUDIO_NOVELTY_WIN` | `15` | Seconds compared on each side of a candidate cut |
| `AUDIO_NOVELTY_K` | `5` | Peak must exceed median + K × robust spread |
| `AUDIO_NOISE_FLAT` | `0.40` | Spectral flatness above this = broadband noise |
| `AUDIO_QUIET_MARGIN_DB` | `6` | Quiet = within this many dB of the noise floor… |
| `AUDIO_QUIET_BELOW_PROG_DB` | `18` | …and at least this far below typical program level |
| `AUDIO_LABEL_BATCH` | `40` | Content segments per Bedrock call |

## Upgrade path

All classification is in `audio_segment.classify_rows`. If real tapes confuse it, a
pretrained classifier (YAMNet via onnxruntime, ~15 MB) can replace that one function;
regions, labelling and the JSON stay unchanged.
