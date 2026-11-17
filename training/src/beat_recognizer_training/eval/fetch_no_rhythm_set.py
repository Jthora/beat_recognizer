"""Production-readiness item 1: every eval set in this plan (GiantSteps,
Ballroom, Hainsworth, GTZAN) is rhythmic by construction. `has_rhythm`'s
false-positive rate -- confidently claiming a beat where none exists -- has
never been measured against real arrhythmic material, even though spec's
honesty principle is built specifically around not doing that. Two halves,
combined (neither alone is enough -- synthetic-only misses real audio's
spectral character; curated-only can't guarantee zero accidental
periodicity the way silence/noise can):

- Synthetic: silence, white/pink noise, sparse non-periodic onsets.
  Airtight ground truth by construction, zero licensing question.
- Curated real audio: ESC-50 (Piczak, karolpiczak/ESC-50 on GitHub),
  2,000 5-second environmental recordings, CC BY-NC (confirmed directly
  from the repo) -- fine for eval-only use, same treatment this plan
  already gives GTZAN/Hainsworth/Ballroom's commercial-origin audio.
  Only genuinely non-periodic categories are pulled -- explicitly
  excluding anything with inherent oscillation (clock_tick, clock_alarm,
  engine/helicopter/chainsaw's blade-pass periodicity, chirping_birds/
  crickets/insects' quasi-periodic chirps, footsteps/clapping's natural
  rhythm) since including those would make a "false positive" ambiguous
  rather than a clean miss.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import requests

SAMPLE_RATE = 44_100

# Categories confirmed non-periodic by nature -- see this module's
# docstring for what was deliberately excluded and why.
ESC50_SAFE_CATEGORIES = [
    "rain", "sea_waves", "wind", "thunderstorm", "crackling_fire",
    "breathing", "coughing", "sneezing", "laughing", "crying_baby",
    "snoring", "toilet_flush", "pouring_water", "glass_breaking",
]
ESC50_CSV_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
ESC50_AUDIO_BASE = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/"


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())


def generate_synthetic_clips(dest_dir: Path, duration_s: float = 10.0, seed: int = 0) -> list[dict]:
    rng = np.random.RandomState(seed)
    n = int(duration_s * SAMPLE_RATE)
    entries = []

    silence = np.zeros(n, dtype=np.float32)
    _write_wav(dest_dir / "silence.wav", silence)
    entries.append({"audio_path": str(dest_dir / "silence.wav"), "category": "synthetic_silence"})

    white = (rng.randn(n) * 0.1).astype(np.float32)
    _write_wav(dest_dir / "white_noise.wav", white)
    entries.append({"audio_path": str(dest_dir / "white_noise.wav"), "category": "synthetic_white_noise"})

    # Pink-ish noise via a cheap 1/f approximation (cumulative sum of white
    # noise, high-pass filtered to avoid pure random-walk drift) -- good
    # enough for "does this trigger false rhythm," not a real DSP need.
    pink_raw = np.cumsum(rng.randn(n))
    pink = (pink_raw - np.convolve(pink_raw, np.ones(2000) / 2000, mode="same"))
    pink = (pink / (np.abs(pink).max() + 1e-9) * 0.2).astype(np.float32)
    _write_wav(dest_dir / "pink_noise.wav", pink)
    entries.append({"audio_path": str(dest_dir / "pink_noise.wav"), "category": "synthetic_pink_noise"})

    # Sparse, deliberately non-periodic onsets (random inter-onset
    # intervals, no regular spacing) -- tests that isolated transients
    # alone don't get mistaken for a tempo.
    sparse = np.zeros(n, dtype=np.float32)
    t = 0.0
    click = np.exp(-np.arange(int(0.02 * SAMPLE_RATE)) / (SAMPLE_RATE * 0.005))
    while t < duration_s - 0.1:
        t += rng.uniform(0.3, 2.5)  # random, not evenly spaced
        start = int(t * SAMPLE_RATE)
        end = min(start + len(click), n)
        if start < n:
            sparse[start:end] += click[: end - start]
    _write_wav(dest_dir / "sparse_random_onsets.wav", sparse)
    entries.append({"audio_path": str(dest_dir / "sparse_random_onsets.wav"), "category": "synthetic_sparse_onsets"})

    return entries


def fetch_esc50_safe_subset(dest_dir: Path, per_category: int = 8) -> list[dict]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(ESC50_CSV_URL, timeout=30)
    resp.raise_for_status()
    rows = [line.split(",") for line in resp.text.splitlines()[1:] if line.strip()]
    # columns: filename, fold, target, category, esc10, src_file, take

    by_category: dict[str, list[str]] = {}
    for row in rows:
        filename, category = row[0], row[3]
        if category in ESC50_SAFE_CATEGORIES:
            by_category.setdefault(category, []).append(filename)

    entries = []
    for category in ESC50_SAFE_CATEGORIES:
        filenames = by_category.get(category, [])[:per_category]
        for filename in filenames:
            dest_path = dest_dir / filename
            if not dest_path.exists():
                audio_resp = requests.get(ESC50_AUDIO_BASE + filename, timeout=30)
                audio_resp.raise_for_status()
                dest_path.write_bytes(audio_resp.content)
            entries.append({"audio_path": str(dest_path), "category": f"esc50:{category}"})

    print(f"Fetched {len(entries)} ESC-50 clips across {len(ESC50_SAFE_CATEGORIES)} safe categories")
    return entries


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("eval_data/no_rhythm"))
    parser.add_argument("--per-category", type=int, default=8)
    args = parser.parse_args()

    entries = generate_synthetic_clips(args.dest / "synthetic") + fetch_esc50_safe_subset(
        args.dest / "esc50", per_category=args.per_category
    )

    index_path = args.dest / "index.jsonl"
    with index_path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"Wrote {len(entries)} entries to {index_path}")
