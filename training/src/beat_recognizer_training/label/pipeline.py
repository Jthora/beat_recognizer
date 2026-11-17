"""Unifies dsp_teacher.py (bootstrap), ensemble.py (cross-check), and
review_sample.py (human review) into one repeatable entry point. These were
three scripts glued together by hand across this project's label-quality
work -- including rewriting the same checklist-enrichment logic inline
twice on the 500-track sample -- which is fine at sample scale and not
something to repeat at the full ~4,000-track corpus.

`training_tier` assigns each label a place in Phase 4's curriculum (see
the build plan's Phase 3/Phase 4 revision):
  strong        - both librosa methods agree with dsp AND each other
  trusted       - at least one librosa method agrees with dsp
  needs_review  - no independent corroboration; human review required
  reviewed_ok   - was needs_review, human confirmed ok by ear/eye
  reviewed_bad  - was needs_review, human confirmed wrong
  excluded      - no rhythm detected (has_rhythm=False)

Only strong/trusted/reviewed_ok are eligible for training.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from beat_recognizer_training.data.verify_corpus import load_active_corpus
from beat_recognizer_training.label.click_track import render_click_track
from beat_recognizer_training.label.dsp_teacher import LabelRecord, bootstrap_labels, load_labels
from beat_recognizer_training.label.ensemble import EnsembleRecord, cross_check, load_ensemble

REPO_ROOT = Path(__file__).resolve().parents[4]
VERDICT_COLUMN = "verdict (fill in: ok/bad/unsure)"
TRAINING_TIERS = frozenset({"strong", "trusted", "reviewed_ok"})


@dataclass
class TieredRecord:
    audio_path: str
    bpm: float | None
    confidence: float | None
    alternate_bpm: float | None
    beat_times_s: list[float]
    librosa_bt_bpm: float | None
    librosa_plp_bpm: float | None
    agree_count: int
    agree_total: int
    training_tier: str


def tier_labels(labels: list[LabelRecord], ensemble_by_path: dict[str, EnsembleRecord]) -> list[TieredRecord]:
    tiered = []
    for r in labels:
        if not r.has_rhythm:
            tiered.append(TieredRecord(r.audio_path, None, None, None, [], None, None, 0, 0, "excluded"))
            continue

        e = ensemble_by_path.get(r.audio_path)
        if e is None or e.agree_total == 0:
            tier = "trusted" if (r.confidence is not None and r.confidence >= 0.6) else "needs_review"
            bt_bpm, plp_bpm, agree_count, agree_total = None, None, 0, 0
        else:
            bt_bpm, plp_bpm = e.librosa_bt_bpm, e.librosa_plp_bpm
            agree_count, agree_total = e.agree_count, e.agree_total
            if agree_count == agree_total == 2:
                tier = "strong"
            elif agree_count >= 1:
                tier = "trusted"
            else:
                tier = "needs_review"

        tiered.append(
            TieredRecord(
                r.audio_path, r.bpm, r.confidence, r.alternate_bpm, r.beat_times_s,
                bt_bpm, plp_bpm, agree_count, agree_total, tier,
            )
        )
    return tiered


def merge_review_verdicts(tiered: list[TieredRecord], checklist_path: Path) -> list[TieredRecord]:
    """Upgrades/downgrades needs_review tracks based on a review_viewer
    checklist.csv's recorded verdicts. A no-op if the checklist doesn't
    exist yet or a track has no verdict recorded."""
    if not checklist_path.exists():
        return tiered

    verdicts: dict[str, str] = {}
    with checklist_path.open(newline="") as f:
        for row in csv.DictReader(f):
            v = row.get(VERDICT_COLUMN, "").strip()
            if v in ("ok", "bad"):
                verdicts[row["audio_path"]] = v

    updated = []
    for r in tiered:
        v = verdicts.get(r.audio_path)
        if v == "ok":
            r = replace(r, training_tier="reviewed_ok")
        elif v == "bad":
            r = replace(r, training_tier="reviewed_bad")
        updated.append(r)
    return updated


def save_tiered(tiered: list[TieredRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in tiered:
            f.write(json.dumps(asdict(r)) + "\n")


def load_tiered(path: Path) -> list[TieredRecord]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(TieredRecord(**json.loads(line)))
    return records


def select_training_set(tiered: list[TieredRecord]) -> list[TieredRecord]:
    """The real Phase 4 training-set query."""
    return [r for r in tiered if r.training_tier in TRAINING_TIERS]


def bootstrap_and_tier(
    corpus: list[Path],
    labels_path: Path,
    ensemble_path: Path,
    tiered_path: Path,
    checklist_path: Path,
) -> list[TieredRecord]:
    bootstrap_labels(corpus, labels_path)
    cross_check(labels_path, ensemble_path)

    labels = load_labels(labels_path)
    ensemble_by_path = {r.audio_path: r for r in load_ensemble(ensemble_path)}
    tiered = tier_labels(labels, ensemble_by_path)
    tiered = merge_review_verdicts(tiered, checklist_path)
    save_tiered(tiered, tiered_path)
    return tiered


def prepare_tier_review_batch(
    tiered: list[TieredRecord], out_dir: Path, n: int = 30, n_best: int = 10
) -> Path | None:
    """Same uncertainty-driven + contrast-sample design as the original
    review_sample.py, driven by training_tier instead of a standalone
    needs_review flag, and writing the ensemble columns directly instead
    of a post-hoc enrichment pass."""
    needs_review = sorted(
        (r for r in tiered if r.training_tier == "needs_review" and r.confidence is not None),
        key=lambda r: r.confidence,
    )
    sample = needs_review[:n]

    if n_best:
        resolved = [r for r in tiered if r.training_tier in ("strong", "trusted") and r.confidence is not None]
        resolved.sort(key=lambda r: -r.confidence)
        already = {r.audio_path for r in sample}
        sample = sample + [r for r in resolved if r.audio_path not in already][:n_best]

    if not sample:
        print("Nothing needs review -- skipping review batch.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    checklist_path = out_dir / "checklist.csv"
    if checklist_path.exists():
        # This function is meant to be re-run as tiers evolve (e.g. after
        # a bigger corpus pass changes what's still needs_review), but it
        # writes to the same path merge_review_verdicts() just read from --
        # overwriting in place would silently discard any verdict
        # merge_review_verdicts() doesn't recognize (e.g. "unsure", a
        # deliberate third state meaning "looked at it, not ready to
        # decide") the moment a fresh batch is generated. Caught live: a
        # prior 40-clip run's 8 "unsure" rows were lost this way before
        # this backup existed. Timestamped rename, not overwrite-in-place,
        # so nothing already-reviewed is ever silently discarded.
        import shutil
        from datetime import datetime, timezone

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = out_dir / f"checklist_{stamp}.csv.bak"
        shutil.move(str(checklist_path), str(backup_path))
        print(f"Existing checklist had unprocessed verdicts -- backed up to {backup_path}")
    with checklist_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "click_track_file", "group", "audio_path", "bpm", "confidence", "alternate_bpm",
                "librosa_bt_bpm", "librosa_plp_bpm", "agree_count", "ensemble_agrees", VERDICT_COLUMN,
            ]
        )
        for i, record in enumerate(sample):
            click_filename = f"{i:03d}_{record.training_tier}_{Path(record.audio_path).stem}.wav"
            render_click_track(Path(record.audio_path), record.beat_times_s, out_dir / click_filename)
            writer.writerow(
                [
                    click_filename, record.training_tier, record.audio_path, record.bpm, record.confidence,
                    record.alternate_bpm, record.librosa_bt_bpm, record.librosa_plp_bpm,
                    f"{record.agree_count}/{record.agree_total}" if record.agree_total else "",
                    record.agree_count >= 1 if record.agree_total else "",
                    "",
                ]
            )
            if (i + 1) % 10 == 0:
                print(f"  rendered {i + 1}/{len(sample)}")

    print(f"Prepared {len(sample)} review clips in {out_dir}, checklist at {checklist_path}")
    return checklist_path


def main(args) -> None:
    corpus = load_active_corpus(args.manifest, args.health)
    if args.limit:
        corpus = corpus[: args.limit]

    tiered = bootstrap_and_tier(corpus, args.labels, args.ensemble, args.tiered, args.checklist)

    counts = Counter(r.training_tier for r in tiered)
    print("\nTier counts:")
    for tier in ("strong", "trusted", "needs_review", "reviewed_ok", "reviewed_bad", "excluded"):
        print(f"  {tier}: {counts.get(tier, 0)}")

    trainable = select_training_set(tiered)
    print(f"\n{len(trainable)}/{len(tiered)} eligible for training ({len(trainable) / len(tiered):.1%})")

    if not args.skip_review_batch:
        prepare_tier_review_batch(tiered, args.review_dir, n=args.review_n, n_best=args.review_n_best)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/PROVENANCE.jsonl")
    parser.add_argument("--health", type=Path, default=REPO_ROOT / "data/CORPUS_HEALTH.jsonl")
    parser.add_argument("--labels", type=Path, default=REPO_ROOT / "data/LABELS.jsonl")
    parser.add_argument("--ensemble", type=Path, default=REPO_ROOT / "data/ENSEMBLE.jsonl")
    parser.add_argument("--tiered", type=Path, default=REPO_ROOT / "data/LABELS_TIERED.jsonl")
    parser.add_argument("--checklist", type=Path, default=REPO_ROOT / "data/review_batch/checklist.csv")
    parser.add_argument("--review-dir", type=Path, default=REPO_ROOT / "data/review_batch")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of tracks (omit for the full active corpus)")
    parser.add_argument("--review-n", type=int, default=30)
    parser.add_argument("--review-n-best", type=int, default=10)
    parser.add_argument("--skip-review-batch", action="store_true")
    main(parser.parse_args())
