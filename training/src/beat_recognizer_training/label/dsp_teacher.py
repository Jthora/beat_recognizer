"""Phase 3: bootstrap beat labels for the corpus using beat-dsp's
deterministic tracker — deliberately not an NC-licensed ML model, so
there's no licensing gray area to manage (see
docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md §4 and the build plan's Phase
3). Writes *every* scored track, not just confident ones — filtering for
actual training use is a separate, later step (`select_confident_labels`),
so this bootstrap output stays a complete, non-destructive record.

Phase 5 repeats this module's role using the Phase 4 trained model instead
(now fully in-house, so no licensing question either way).

Resumable like the Phase 2 fetch scripts: skips audio_paths already present
in the output file, since bootstrapping ~4,000 tracks is exactly the kind
of long-running job that has been interrupted mid-run before in this
project (see the Jamendo pull's history).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from beat_recognizer_training.data.verify_corpus import load_active_corpus
from beat_recognizer_training.eval.harness import mp3_to_wav, run_cli

# Below this, or whenever an octave-alternate is present, a label is
# flagged for human review rather than trusted outright -- see
# data/CORPUS_CHARACTERIZATION.md: on a 300-track sample, only 53% of this
# corpus clears 0.5 confidence and 73.3% carry an alternate, so expect most
# tracks to be flagged, not a small minority.
CONFIDENCE_REVIEW_THRESHOLD = 0.6


@dataclass
class LabelRecord:
    audio_path: str
    has_rhythm: bool
    bpm: float | None
    confidence: float | None
    alternate_bpm: float | None
    alternate_confidence: float | None
    beat_times_s: list[float]
    needs_review: bool


def bootstrap_label(audio_path: Path, tmp_dir: Path) -> LabelRecord:
    wav_path = tmp_dir / (audio_path.stem + ".wav")
    try:
        mp3_to_wav(audio_path, wav_path)
        output = run_cli(wav_path)
    finally:
        wav_path.unlink(missing_ok=True)

    if output is None or output.get("bpm") is None:
        return LabelRecord(str(audio_path), False, None, None, None, None, [], False)

    confidence = output["confidence"]
    alternate_bpm = output["alternate_bpm"]
    needs_review = confidence < CONFIDENCE_REVIEW_THRESHOLD or alternate_bpm is not None

    return LabelRecord(
        str(audio_path),
        True,
        output["bpm"],
        confidence,
        alternate_bpm,
        output["alternate_confidence"],
        output["beats_s"],
        needs_review,
    )


def already_labeled_paths(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    paths = set()
    with out_path.open() as f:
        for line in f:
            if line.strip():
                paths.add(json.loads(line)["audio_path"])
    return paths


def bootstrap_labels(corpus: list[Path], out_path: Path) -> None:
    done = already_labeled_paths(out_path)
    remaining = [p for p in corpus if str(p) not in done]
    print(f"{len(done)} already labeled, {len(remaining)} remaining")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with out_path.open("a") as f:
            for i, audio_path in enumerate(remaining):
                record = bootstrap_label(audio_path, tmp_dir)
                f.write(json.dumps(asdict(record)) + "\n")
                f.flush()
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(remaining)}")

    print(f"Wrote labels to {out_path}")


def load_labels(labels_path: Path) -> list[LabelRecord]:
    records = []
    with labels_path.open() as f:
        for line in f:
            if line.strip():
                records.append(LabelRecord(**json.loads(line)))
    return records


def select_confident_labels(labels_path: Path, min_confidence: float = 0.5) -> list[LabelRecord]:
    """The actual Phase 4 training-set query: has_rhythm, confidence above
    threshold, not flagged for review (or reviewed and accepted -- review
    outcomes aren't tracked yet, see review_sample.py's TODO)."""
    return [
        r
        for r in load_labels(labels_path)
        if r.has_rhythm and r.confidence is not None and r.confidence >= min_confidence
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/PROVENANCE.jsonl"))
    parser.add_argument("--health", type=Path, default=Path("data/CORPUS_HEALTH.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/LABELS.jsonl"))
    args = parser.parse_args()

    corpus = load_active_corpus(args.manifest, args.health)
    bootstrap_labels(corpus, args.out)
