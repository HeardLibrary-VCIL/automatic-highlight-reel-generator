"""Tests for the audio pipeline (audio_segment.py + audio_label.py).

Run from audio-processing/:  python3 -m unittest test_audio -v

Self-contained: the test audio is synthesized here (tones, hiss, applause and
two different "songs" built from harmonic notes), so nothing is downloaded and
no AWS call is made — the Bedrock client is a fake. Accuracy on REAL recordings
(speech, real music) is measured separately with eval/eval_audio.py.
"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import audio_segment as A

try:                                    # pydantic ships with the anthropic SDK in the image
    import audio_label as L
except ImportError:                     # bare dev machine: skip only the labeling tests
    L = None

SR = 16000
HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")
rng = np.random.default_rng(3)


def _db(x, level):
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-12) * 10 ** (level / 20)).astype(np.float32)


def tone(sec, hz=1000, level=-14):
    t = np.arange(int(sec * SR)) / SR
    return _db(np.sin(2 * np.pi * hz * t), level)


def noise(sec, level):
    return _db(rng.standard_normal(int(sec * SR)), level)


def applause(sec, level=-20):
    n = int(sec * SR)
    y = np.zeros(n)
    for _ in range(int(sec * 40)):
        p = rng.integers(0, n - 400)
        m = rng.integers(80, 400)
        y[p:p + m] += rng.standard_normal(m) * np.exp(-np.arange(m) / (m / 4)) * rng.uniform(0.3, 1)
    return _db(y, level)


def song(sec, root_hz, harmonics, note_s, drums, level=-20):
    """A melody over a chord pad: held notes with stable partials, like music."""
    n = int(sec * SR)
    t = np.arange(n) / SR
    y = np.zeros(n)
    scale = [0, 2, 4, 7, 9, 12]
    note_len = int(note_s * SR)
    for k, start in enumerate(range(0, n, note_len)):
        f = root_hz * 2 ** (scale[rng.integers(len(scale))] / 12)
        seg = slice(start, min(n, start + note_len))
        tt = t[seg] - t[start]
        env = np.minimum(1, tt / 0.02) * np.exp(-tt / (note_s * 2))
        y[seg] += sum(w * np.sin(2 * np.pi * f * (h + 1) * tt) for h, w in enumerate(harmonics)) * env
    chord_len = int(2.0 * SR)
    for c, start in enumerate(range(0, n, chord_len)):
        seg = slice(start, min(n, start + chord_len))
        tt = t[seg]
        base = root_hz / 2 * 2 ** ([0, 5, 7, 3][c % 4] / 12)
        y[seg] += 0.3 * sum(np.sin(2 * np.pi * base * r * tt) for r in (1, 1.25, 1.5))
    if drums:
        beat = int(note_s * SR)
        for p in range(0, n - 800, beat):
            y[p:p + 800] += 0.4 * rng.standard_normal(800) * np.exp(-np.arange(800) / 150)
    return _db(y, level)


def write_wav(path, parts, floor_db=-55):
    y = np.concatenate(parts)
    y = y + noise(len(y) / SR, floor_db)[: len(y)]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                    "-i", "-", str(path)], input=np.clip(y, -1, 1).astype(np.float32).tobytes(),
                   check=True)


def kinds(regions):
    return [r.kind for r in regions]


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not installed")
class SegmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        # tone 0-6 | hiss 6-10 | song A 10-70 | gap 70-72 | song B 72-132 |
        # applause 132-137 | quiet 137-145 | static 145-151
        cls.tape = cls.tmp / "tape.wav"
        write_wav(cls.tape, [
            tone(6), noise(4, -52),
            song(60, 220, [1, .6, .4, .2], 0.5, drums=True),
            noise(2, -52),
            song(60, 330, [1, .2], 0.375, drums=False),
            applause(5), noise(8, -52), noise(6, -22),
        ])
        cls.regions, cls.rows, _, _ = A.segment_audio(str(cls.tape))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def near(self, t, target, tol=1.5):
        self.assertLessEqual(abs(t - target), tol, f"{t} not within {tol}s of {target}")

    def test_finds_every_item_in_order(self):
        self.assertEqual(kinds(self.regions),
                         ["tone", "hiss", "song", "song", "applause", "hiss", "static"])

    def test_boundaries_land_on_the_real_changes(self):
        starts = [r.start for r in self.regions]
        for got, want in zip(starts[1:], [6, 10, 71, 132, 137, 145]):
            self.near(got, want)

    def test_short_gap_splits_songs_without_becoming_dead_space(self):
        b = self.regions[3]
        self.assertEqual(b.kind, "song")
        self.assertEqual(b.cues, ["gap"])

    def test_timeline_is_a_gapless_partition(self):
        self.assertEqual(self.regions[0].start, 0.0)
        self.assertAlmostEqual(self.regions[-1].end, self.rows.duration, places=1)
        for a, b in zip(self.regions[:-1], self.regions[1:]):
            self.assertEqual(a.end, b.start)

    def test_novelty_splits_gapless_songs_but_not_one_song(self):
        path = self.tmp / "gapless.wav"
        write_wav(path, [song(70, 220, [1, .6, .4, .2], 0.5, drums=True),
                         song(70, 330, [1, .2], 0.375, drums=False)])
        regs, _, _, _ = A.segment_audio(str(path))
        songs = [r for r in regs if r.kind == "song"]
        self.assertEqual(len(songs), 2, regs)
        self.assertEqual(songs[1].cues, ["novelty"])
        self.assertEqual(songs[1].confidence, "low")
        self.near(songs[1].start, 70, tol=3)

        one = self.tmp / "one.wav"
        write_wav(one, [song(150, 262, [1, .5, .3], 0.5, drums=True)])
        regs, _, _, _ = A.segment_audio(str(one))
        self.assertEqual(kinds(regs), ["song"])

    def test_quiet_mains_hum_is_not_a_test_tone(self):
        path = self.tmp / "hum.wav"
        hum = tone(150, hz=120, level=-40)
        music = song(150, 220, [1, .6, .4, .2], 0.5, drums=False)
        write_wav(path, [music + hum])
        regs, _, _, _ = A.segment_audio(str(path))
        self.assertNotIn("tone", kinds(regs))


@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not installed")
class AudioOnlyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def ff(self, *args):
        subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)

    def test_detects_audio_only_files(self):
        wav, mp4, m4a = self.tmp / "a.wav", self.tmp / "v.mp4", self.tmp / "cover.m4a"
        self.ff("-f", "lavfi", "-i", "sine=d=2", str(wav))
        self.ff("-f", "lavfi", "-i", "testsrc=d=2:s=160x120", "-f", "lavfi", "-i", "sine=d=2",
                "-shortest", str(mp4))
        cover = self.tmp / "cover.png"
        self.ff("-f", "lavfi", "-i", "color=c=red:s=64x64", "-frames:v", "1", str(cover))
        self.ff("-i", str(wav), "-i", str(cover), "-map", "0", "-map", "1", "-c:a", "aac",
                "-c:v", "png", "-disposition:v", "attached_pic", str(m4a))
        self.assertTrue(A.is_audio_only(str(wav)))
        self.assertFalse(A.is_audio_only(str(mp4)))
        self.assertTrue(A.is_audio_only(str(m4a)), "cover art must not make it a video")


class FakeClient:
    """Stands in for AnthropicBedrock: replays canned replies, records prompts."""
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.prompts.append(kw["messages"])
        return SimpleNamespace(content=[SimpleNamespace(text=self.replies.pop(0))])


@unittest.skipIf(L is None, "pydantic not installed (pip install -r requirements.txt)")
class LabelTest(unittest.TestCase):
    def setUp(self):
        R = A.Region
        self.regions = [
            R(0.0, 8.0, "tone"),
            R(8.0, 20.0, "speech", ["dead"], "high"),
            R(20.0, 200.0, "song", ["speech"], "high"),
            R(200.0, 205.0, "applause", ["music"], "high"),
            R(205.0, 380.0, "song", ["gap"], "medium"),
        ]
        self.turns = [
            {"start": 9.0, "end": 19.0, "speaker": "spk_0",
             "text": "Good evening. We open with Steal Away."},
            {"start": 40.0, "end": 60.0, "speaker": "spk_1", "text": "steal away steal away to Jesus"},
        ]

    def test_bad_reply_is_retried_then_labels_apply(self):
        good = json.dumps({"segments": [
            {"index": 1, "title": "MC welcome", "description": "Introduces the concert.",
             "confidence": "high", "evidence": "Good evening"},
            {"index": 2, "title": "Steal Away", "description": "Choral spiritual.",
             "confidence": "high", "evidence": "We open with Steal Away"},
            {"index": 3, "title": "Applause", "confidence": "high"},
            {"index": 4, "title": "Song 2 (untitled, choral)", "confidence": "low"},
        ]})
        summary = json.dumps({"program_title": "Concert, 1971", "summary": "A choral concert."})
        client = FakeClient(['{"segments": [{"index": 1}]}', good, summary])   # first reply invalid
        labels = L.label_audio(self.regions, self.turns, client, "Fisk_concert_1971.wav", 380.0)

        self.assertEqual(len(client.prompts), 3)                 # bad + retry + summary
        self.assertIn("not valid", client.prompts[1][-1]["content"])
        self.assertEqual(labels["segments"][2].title, "Steal Away")
        self.assertEqual(labels["program_title"], "Concert, 1971")

        out = L.build_audio_segment_json("video/Fisk_concert_1971.wav", self.regions,
                                         self.turns, labels, 380.0)
        segs = out["segments"]
        self.assertEqual(out["media_type"], "audio")
        self.assertEqual(out["song_count"], 2)
        self.assertEqual([s["segment_type"] for s in segs], ["D", "speech", "song", "applause", "song"])
        self.assertEqual(segs[0]["title"], "Test tone (auto-detected)")
        self.assertEqual(segs[2]["title"], "Steal Away")
        self.assertIn("steal away", segs[2]["transcript"])
        self.assertEqual(segs[4]["song_number"], 2)
        self.assertAlmostEqual(out["kept_pct"], 97.9, places=1)

    def test_prompt_only_contains_lyrics_and_speech_not_dead_space_text(self):
        rows = L.describe_segments(self.regions, self.turns)
        self.assertNotIn("transcript", rows[0])
        self.assertEqual(rows[2]["song_number"], 1)
        self.assertIn("steal away", rows[2]["lyrics_asr"])

    def test_without_labels_titles_fall_back(self):
        out = L.build_audio_segment_json("video/x.mp3", self.regions, [], {}, 380.0)
        self.assertEqual([s["title"] for s in out["segments"][1:]],
                         ["Speech", "Song 1", "Applause", "Song 2"])
        self.assertNotIn("summary", out)



@unittest.skipUnless(HAVE_FFMPEG, "ffmpeg/ffprobe not installed")
@unittest.skipIf(L is None, "pydantic not installed (pip install -r requirements.txt)")
class PipelineTest(unittest.TestCase):
    """run_audio_detect end to end, with fake S3 / Transcribe / Bedrock."""

    def setUp(self):
        import audio_pipeline
        self.P = audio_pipeline
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.wav = self.tmp / "Fisk_sideA.wav"
        write_wav(self.wav, [tone(6), noise(4, -52),
                             song(60, 220, [1, .6, .4, .2], 0.5, drums=True), noise(2, -52),
                             song(60, 330, [1, .2], 0.375, drums=False), noise(6, -52)])
        self.put = {}
        self.s3 = SimpleNamespace(put_object=lambda **kw: self.put.update(kw))

    def test_writes_segment_json_with_labels(self):
        def create(**kw):
            msg = kw["messages"][-1]["content"]
            if "Label exactly these segment indexes" in msg:
                import re
                idx = json.loads(re.search(r"indexes: (\[[^\]]*\])", msg).group(1))
                txt = json.dumps({"segments": [{"index": i, "title": f"Item {i}"} for i in idx]})
            else:
                txt = json.dumps({"program_title": "Side A", "summary": "Two songs."})
            return SimpleNamespace(content=[SimpleNamespace(text=txt)])
        client = SimpleNamespace(messages=SimpleNamespace(create=create))

        out = self.P.run_audio_detect(self.wav, self.s3, "bkt", "video/Fisk_sideA.wav",
                                      "Fisk_sideA.wav", get_turns=lambda *a: [],
                                      make_client=lambda: client)
        self.assertEqual(self.put["Key"], "segment/Fisk_sideA.json")
        self.assertEqual(json.loads(self.put["Body"]), out)
        self.assertEqual(out["song_count"], 2)
        self.assertEqual(out["program_title"], "Side A")
        songs = [s for s in out["segments"] if s["segment_type"] == "song"]
        self.assertTrue(all(s["title"].startswith("Item ") for s in songs))

    def test_failures_still_write_a_plain_timeline(self):
        def boom(*a, **k):
            raise RuntimeError("service down")
        out = self.P.run_audio_detect(self.wav, self.s3, "bkt", "video/Fisk_sideA.wav",
                                      "Fisk_sideA.wav", get_turns=boom, make_client=boom,
                                      seg_key="segment/custom.json")
        self.assertEqual(self.put["Key"], "segment/custom.json")
        self.assertEqual([s["title"] for s in out["segments"] if s["segment_type"] == "song"],
                         ["Song 1", "Song 2"])

if __name__ == "__main__":
    unittest.main()
