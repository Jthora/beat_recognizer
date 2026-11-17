"""Production-readiness item 1/2, iteration 2 step 5: ESC-50 clips are 5
seconds -- barely enough for the tracker to build history (`beat-dsp`'s own
`MAX_HISTORY_SECONDS` is 8.0). The 36-37% false-positive rate measured
against them might not represent what real, longer-duration no-rhythm audio
(ambient background noise during actual gameplay isn't 5 seconds) would
show. This concatenates same-category ESC-50 clips into longer synthetic
tracks and re-runs the leading candidate rules to check whether the
false-positive rate actually drops with more available audio.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from beat_recognizer_training.eval.confidence_prototype import (
    REPO_ROOT,
    _rate,
    load_no_rhythm_and_ballroom,
    summarize_set,
)
from beat_recognizer_training.eval.fetch_no_rhythm_set import _write_wav

CONCAT_COUNT = 3  # 3x 5s clips ~= 15s, comfortably past MAX_HISTORY_SECONDS


def build_concatenated_set(no_rhythm_paths: list[Path], dest_dir: Path, n: int = CONCAT_COUNT) -> list[Path]:
    """Groups ESC-50 paths by category (parsed from the parent-relative
    naming already used when fetched) and concatenates every `n` clips
    within a category into one longer file -- real audio glued together,
    not synthesized, so it's still a genuine test of longer-duration
    no-rhythm content, not a different kind of signal."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    by_category: dict[str, list[Path]] = defaultdict(list)
    for p in no_rhythm_paths:
        if "esc50" not in str(p):
            continue  # synthetic clips (silence/noise) don't need this -- already as long as we want
        by_category[p.parent.name].append(p)

    out_paths = []
    for category, paths in by_category.items():
        paths = sorted(paths)
        for i in range(0, len(paths) - n + 1, n):
            group = paths[i : i + n]
            samples = np.concatenate([sf.read(str(p), dtype="float32")[0] for p in group])
            out_path = dest_dir / f"{category}_{i // n:02d}.wav"
            _write_wav(out_path, samples)
            out_paths.append(out_path)
    return out_paths


if __name__ == "__main__":
    no_rhythm_paths, ballroom_paths = load_no_rhythm_and_ballroom()
    dest_dir = REPO_ROOT / "training" / "eval_data" / "no_rhythm" / "esc50_concat"

    print(f"Building {CONCAT_COUNT}x-concatenated clips...")
    concat_paths = build_concatenated_set(no_rhythm_paths, dest_dir)
    print(f"{len(concat_paths)} concatenated clips built")

    print("Scoring original (5s) clips...")
    original_summaries = summarize_set([p for p in no_rhythm_paths if "esc50" in str(p)])
    print("Scoring concatenated (~15s) clips...")
    concat_summaries = summarize_set(concat_paths)

    rules = {
        "confidence>0.25 (current, shipped)": lambda s: s.final_confidence > 0.25,
        "stability_4s>0.6": lambda s: s.stability_4s > 0.6,
        "debounced_25": lambda s: s.debounced_25,
    }
    print(f"\n{'rule':<35} {'5s clips FP':<18} {'~15s clips FP':<18}")
    for name, rule in rules.items():
        orig_hits, orig_n = _rate(original_summaries, rule)
        concat_hits, concat_n = _rate(concat_summaries, rule)
        orig_pct = f"{orig_hits}/{orig_n} ({orig_hits / orig_n:.0%})" if orig_n else "n/a"
        concat_pct = f"{concat_hits}/{concat_n} ({concat_hits / concat_n:.0%})" if concat_n else "n/a"
        print(f"{name:<35} {orig_pct:<18} {concat_pct:<18}")

    out_path = REPO_ROOT / "training/eval/DURATION_SENSITIVITY.jsonl"
    with out_path.open("w") as f:
        for s in original_summaries + concat_summaries:
            f.write(json.dumps(s.__dict__) + "\n")
    print(f"\nWrote {out_path}")
