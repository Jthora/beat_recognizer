"""Overlays an audible click at each detected beat time onto the original
audio, so a human can *listen* to whether beat-dsp's output actually lines
up with the music -- the real spot-check mechanism Phase 3's go/no-go
depends on. Numbers and confidence scores alone don't tell you whether a
detected "beat" musically makes sense; hearing the clicks land on/off the
groove does.

No new heavyweight audio dependency: decodes via the `ffmpeg` binary
(already a hard requirement for this whole project, confirmed installed in
docs/ARCHITECTURE_AND_LICENSING_RESEARCH's Phase 2.5 work) piped to raw
PCM, and writes the result with Python's built-in `wave` module.
"""

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44_100
CLICK_FREQ_HZ = 1000.0
CLICK_DURATION_S = 0.04
MUSIC_ATTENUATION = 0.5  # so the click stands out clearly


def decode_to_mono_f32(audio_path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(audio_path),
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _click_waveform(sample_rate: int) -> np.ndarray:
    n = int(CLICK_DURATION_S * sample_rate)
    t = np.arange(n) / sample_rate
    envelope = np.exp(-t / (CLICK_DURATION_S / 5))  # fast decay
    return (np.sin(2 * np.pi * CLICK_FREQ_HZ * t) * envelope).astype(np.float32)


def render_click_track(
    audio_path: Path,
    beat_times_s: list[float],
    out_path: Path,
    sample_rate: int = SAMPLE_RATE,
) -> None:
    audio = decode_to_mono_f32(audio_path, sample_rate) * MUSIC_ATTENUATION
    click = _click_waveform(sample_rate)

    for t in beat_times_s:
        start = int(t * sample_rate)
        end = min(start + len(click), len(audio))
        if start >= len(audio):
            continue
        audio[start:end] += click[: end - start]

    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("out_path", type=Path)
    parser.add_argument("--beats-json", type=Path, help="JSON file containing a list of beat times in seconds")
    parser.add_argument("--beats", type=str, help="comma-separated beat times in seconds")
    args = parser.parse_args()

    if args.beats_json:
        beats = json.loads(args.beats_json.read_text())
    elif args.beats:
        beats = [float(x) for x in args.beats.split(",")]
    else:
        raise SystemExit("pass --beats-json or --beats")

    render_click_track(args.audio_path, beats, args.out_path)
    print(f"Wrote {args.out_path}")
