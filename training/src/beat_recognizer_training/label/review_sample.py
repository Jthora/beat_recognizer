"""Phase 3's uncertainty-driven spot-check: instead of reviewing a flat
random sample, prioritize exactly the labels most likely to be wrong (low
confidence, or an octave-alternate present) — the build plan's explicit
departure from flat-random sampling. Renders a click-track WAV per selected
track (via label.click_track) into a review batch a human listens to, plus
a checklist to record verdicts in.

This prepares the batch; it doesn't (can't) perform the listening itself.

Superseded by pipeline.py's prepare_tier_review_batch() wherever an
ensemble cross-check exists -- that version drives off training_tier
(which folds in independent-tracker agreement, a stronger signal than
confidence alone) instead of just confidence/alternate. This module still
works standalone from labels alone, before any ensemble data exists.
"""

from __future__ import annotations

import csv
from pathlib import Path

from beat_recognizer_training.label.click_track import render_click_track
from beat_recognizer_training.label.dsp_teacher import LabelRecord, load_labels


def select_review_sample(labels_path: Path, n: int, seed: int = 0, n_best: int = 0) -> list[LabelRecord]:
    """`n` worst-confidence (most likely wrong) labels, plus optionally
    `n_best` of the highest-confidence labels for direct contrast -- the
    point isn't just "are the bad ones bad," it's "does confidence actually
    track quality at all," which needs both ends of the range to check."""
    import random

    records = [r for r in load_labels(labels_path) if r.has_rhythm and r.confidence is not None]
    flagged = [r for r in records if r.needs_review]
    unflagged = [r for r in records if not r.needs_review]

    # Worst-confidence-first within the flagged set -- these are the labels
    # most likely to actually be wrong, so review budget goes there first.
    flagged.sort(key=lambda r: r.confidence)

    sample = flagged[:n]
    if len(sample) < n:
        # Pad with a random sample of unflagged tracks, as a sanity check
        # that "confident" labels are also actually trustworthy, not just
        # the ambiguous ones.
        rng = random.Random(seed)
        sample += rng.sample(unflagged, min(n - len(sample), len(unflagged)))

    if n_best:
        by_confidence_desc = sorted(records, key=lambda r: -r.confidence)
        already = {r.audio_path for r in sample}
        best = [r for r in by_confidence_desc if r.audio_path not in already][:n_best]
        sample = sample + best

    return sample


def prepare_review_batch(labels_path: Path, out_dir: Path, n: int, seed: int = 0, n_best: int = 0) -> Path:
    sample = select_review_sample(labels_path, n, seed, n_best=n_best)
    out_dir.mkdir(parents=True, exist_ok=True)

    checklist_path = out_dir / "checklist.csv"
    with checklist_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "click_track_file",
                "group",
                "audio_path",
                "bpm",
                "confidence",
                "alternate_bpm",
                "verdict (fill in: ok/bad/unsure)",
            ]
        )
        for i, record in enumerate(sample):
            group = "low_confidence" if i < len(sample) - n_best else "high_confidence"
            click_filename = f"{i:03d}_{group}_{Path(record.audio_path).stem}.wav"
            render_click_track(Path(record.audio_path), record.beat_times_s, out_dir / click_filename)
            writer.writerow(
                [click_filename, group, record.audio_path, record.bpm, record.confidence, record.alternate_bpm, ""]
            )
            if (i + 1) % 10 == 0:
                print(f"  rendered {i + 1}/{len(sample)}")

    print(f"Prepared {len(sample)} review clips in {out_dir}, checklist at {checklist_path}")
    return checklist_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/LABELS.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/review_batch"))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--n-best", type=int, default=0, help="also include this many of the highest-confidence tracks, for contrast")
    args = parser.parse_args()

    prepare_review_batch(args.labels, args.out, args.n, n_best=args.n_best)
