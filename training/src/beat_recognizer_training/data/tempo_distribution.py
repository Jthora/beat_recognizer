"""Phase 2.5b: rough tempo-distribution estimate across the acquired
corpus, using beat-dsp-cli itself (now real-audio-validated per Phase 2.5a)
as a proxy -- no ground truth here, just checking whether the corpus is
obviously skewed (e.g. everything clustering at 120-130 BPM) before that
becomes an invisible bias baked into training data. Informal by design; the
real accuracy numbers live in Phase 2.5a's harness against GiantSteps.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

from beat_recognizer_training.data.verify_corpus import load_active_corpus
from beat_recognizer_training.eval.harness import mp3_to_wav, run_cli


def sample_tempo_distribution(
    manifest_path: Path, health_path: Path, sample_size: int, seed: int = 0
) -> list[dict]:
    corpus = load_active_corpus(manifest_path, health_path)
    rng = random.Random(seed)
    sample = rng.sample(corpus, min(sample_size, len(corpus)))
    print(f"Sampling {len(sample)} / {len(corpus)} active corpus files")

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, audio_path in enumerate(sample):
            wav_path = tmp_dir / (audio_path.stem + ".wav")
            try:
                mp3_to_wav(audio_path, wav_path)
                output = run_cli(wav_path)
            finally:
                wav_path.unlink(missing_ok=True)

            if output and output["bpm"] is not None:
                results.append(
                    {
                        "audio_path": str(audio_path),
                        "bpm": output["bpm"],
                        "confidence": output["confidence"],
                        "has_alternate": output["alternate_bpm"] is not None,
                    }
                )
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(sample)}")

    return results


def report(results: list[dict]) -> None:
    bpms = [r["bpm"] for r in results]
    confident = [r for r in results if r["confidence"] > 0.5]
    print(f"\n{len(results)} tracks scored, {len(confident)} with confidence > 0.5")

    buckets = {
        "<70": 0, "70-90": 0, "90-110": 0, "110-130": 0,
        "130-150": 0, "150-170": 0, "170-200": 0, ">200": 0,
    }
    for bpm in bpms:
        if bpm < 70: buckets["<70"] += 1
        elif bpm < 90: buckets["70-90"] += 1
        elif bpm < 110: buckets["90-110"] += 1
        elif bpm < 130: buckets["110-130"] += 1
        elif bpm < 150: buckets["130-150"] += 1
        elif bpm < 170: buckets["150-170"] += 1
        elif bpm < 200: buckets["170-200"] += 1
        else: buckets[">200"] += 1

    print("\nDetected-tempo distribution (informal, no ground truth):")
    for label, count in buckets.items():
        bar = "#" * int(count / max(1, len(bpms)) * 100)
        print(f"  {label:>10}: {count:>4}  {bar}")

    alt_rate = sum(r["has_alternate"] for r in results) / len(results) if results else 0
    print(f"\n{alt_rate:.1%} of sampled tracks had an octave-alternate hypothesis exposed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/PROVENANCE.jsonl"))
    parser.add_argument("--health", type=Path, default=Path("data/CORPUS_HEALTH.jsonl"))
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("data/TEMPO_DISTRIBUTION_SAMPLE.jsonl"))
    args = parser.parse_args()

    results = sample_tempo_distribution(args.manifest, args.health, args.sample_size)
    with args.out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    report(results)
