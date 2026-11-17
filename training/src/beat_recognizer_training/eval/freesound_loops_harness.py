"""BPM/octave accuracy improvement program, Stage 0b: a real octave-accuracy
baseline from Freesound Loops (`data/FREESOUND_LOOPS_LABELS.jsonl`, 7,189
tracks) -- confirmed ground-truth BPM, no DSP estimation step in the loop
(1,693 expert-verified, 5,496 from producer upload-time tags), unlike every
other source this project's accuracy numbers have come from so far.

Distinct from `harness.py`'s GiantSteps scoring in one deliberate way:
GiantSteps is *the* long-standing reference set (Phase 2.5c's own numbers
came from it), so this doesn't replace it -- it adds a second, independent,
much larger real-ground-truth set specifically so Stages 1-4 of the
accuracy program get validated against two datasets, not one, catching an
overfit-to-GiantSteps risk that's existed silently since Phase 2.5c.

Reuses `harness.py`'s `accuracy1`/`accuracy2`/`run_cli` directly rather than
reimplementing the same formulas a second time.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from beat_recognizer_training.eval.harness import CLI_PATH, accuracy1, accuracy2, run_cli

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class LoopResult:
    audio_path: str
    true_bpm: float
    bpm_source: str
    detected_bpm: float | None
    alternate_bpm: float | None
    accuracy1: bool
    accuracy2: bool
    accuracy2_with_alternate: bool


def load_loops(labels_path: Path) -> list[dict]:
    with labels_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def score_loop(entry: dict) -> LoopResult:
    audio_path = Path(entry["audio_path"])
    true_bpm = entry["bpm"]
    output = run_cli(audio_path, mode="offline")
    if output is None or output.get("bpm") is None:
        return LoopResult(str(audio_path), true_bpm, entry["bpm_source"], None, None, False, False, False)

    detected = output["bpm"]
    alt = output.get("alternate_bpm")
    a1 = accuracy1(true_bpm, detected)
    a2 = accuracy2(true_bpm, detected)
    a2_alt = a2 or (alt is not None and accuracy2(true_bpm, alt))
    return LoopResult(str(audio_path), true_bpm, entry["bpm_source"], detected, alt, a1, a2, a2_alt)


def run_sample(labels_path: Path, n: int, seed: int = 42) -> list[LoopResult]:
    """`run_cli` in harness.py expects an mp3 needing conversion; Freesound
    Loops are already WAV, so this calls it directly with a WAV path --
    `run_cli`'s own `mp3_to_wav` step is skipped by construction since these
    files don't go through `score_track`, which is GiantSteps-specific."""
    loops = load_loops(labels_path)
    rng = random.Random(seed)
    sample = rng.sample(loops, min(n, len(loops))) if n < len(loops) else loops

    results = []
    for i, entry in enumerate(sample):
        results.append(score_loop(entry))
        if (i + 1) % 100 == 0:
            print(f"  scored {i + 1}/{len(sample)}")
    return results


def summarize(results: list[LoopResult]) -> dict:
    n = len(results)
    scored = [r for r in results if r.detected_bpm is not None]
    return {
        "n": n,
        "scored": len(scored),
        "accuracy1": sum(r.accuracy1 for r in scored) / n if n else 0.0,
        "accuracy2": sum(r.accuracy2 for r in scored) / n if n else 0.0,
        "accuracy2_with_alternate": sum(r.accuracy2_with_alternate for r in scored) / n if n else 0.0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("../data/FREESOUND_LOOPS_LABELS.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/FREESOUND_LOOPS_RESULTS.jsonl"))
    parser.add_argument("-n", type=int, default=1000, help="sample size (fixed seed, reproducible across stages)")
    args = parser.parse_args()

    if not CLI_PATH.exists():
        raise SystemExit(f"beat-dsp-cli not built at {CLI_PATH} -- run `cargo build --release -p beat-dsp` first")

    results = run_sample(args.labels, args.n)
    s = summarize(results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in results:
            f.write(json.dumps(r.__dict__) + "\n")

    print(f"\nFreesound Loops baseline (n={s['n']}, {s['scored']} scored):")
    print(f"  Accuracy1: {s['accuracy1']:.1%}")
    print(f"  Accuracy2: {s['accuracy2']:.1%}")
    print(f"  Accuracy2+alt: {s['accuracy2_with_alternate']:.1%}")
    print(f"\nWrote {args.out}")
