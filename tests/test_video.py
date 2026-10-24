"""Camera / phone videos: clarion takes the first audio track (via ffmpeg), keeping it in sync."""
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from clarion import audio_io
from clarion.pipeline import Project
from clarion.settings import PRESETS, SR

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

CLICK_S = 3.0


def _video(tmp_path, name: str, acodec: list[str]) -> tuple:
    """10 s of quiet noise with a click at 3.000 s, muxed with 12 s of test video (longer than the audio)."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((10 * SR, 2)) * 0.01).astype(np.float32)
    x[int(CLICK_S * SR)] = 0.9
    wav = tmp_path / "track.wav"
    sf.write(wav, x, SR, subtype="FLOAT")
    out = tmp_path / name
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=25:duration=12",
                    "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "mpeg4", *acodec, str(out)], check=True)
    return out, x


@pytest.mark.parametrize("name, acodec, tol", [
    ("camera.MOV", ["-c:a", "pcm_s24le"], 0),        # uncompressed sound, as many cameras record
    ("phone.mp4", ["-c:a", "aac", "-b:a", "192k"], 2),  # AAC: ffmpeg removes the encoder delay
])
def test_video_sound_loads_in_sync(tmp_path, name, acodec, tol):
    path, x = _video(tmp_path, name, acodec)
    mm = audio_io.load_to_memmap(path, tmp_path / "o.npy")
    assert abs(len(mm) - len(x)) <= 1024                  # the audio's length, not the 12 s of video
    click = int(np.argmax(np.abs(mm[:, 0])))
    assert abs(click - CLICK_S * SR) <= tol               # same moment as in the camera file


def test_video_restores_to_wav_of_track_length(tmp_path, no_dfn):
    path, x = _video(tmp_path, "clip.mov", ["-c:a", "pcm_s16le"])
    proj = Project(path)
    proj.prepare()
    y, sr = sf.read(proj.export(PRESETS["Gentle"], tmp_path / "out.wav"))
    assert sr == SR and abs(len(y) - len(x)) <= 1024
