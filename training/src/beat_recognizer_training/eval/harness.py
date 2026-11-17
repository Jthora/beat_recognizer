"""Phase 2.5a: scores the actual `beat-dsp-cli` binary (not a Python
reimplementation) against real annotated audio. Currently wired to
GiantSteps Tempo (SMC is deferred -- see fetch_eval_sets.py's docstring for
why). Implements the standard MIREX Accuracy1/Accuracy2 tempo metrics
directly rather than mir_eval.tempo.detection, whose API is shaped for a
different ground-truth format (two weighted reference tempi) than what
GiantSteps provides (a single BPM value) -- these are simple, well-defined
formulas (McKinney et al. 2007), not a home-grown metric.

Accuracy1: correct if the estimate is within `tol` (default 4%) of the true
tempo. Accuracy2: also correct if it's within tolerance of a standard
octave/related error (2x, 0.5x, 3x, 1/3x the true tempo) -- the same
tempo-octave-ambiguity tolerance the field has used for two decades,
directly relevant given beat-dsp now exposes its own octave-alternate
hypothesis (see tempo.rs's `alternate` field): Accuracy2-with-alternate
below checks whether *either* of our two reported hypotheses is correct,
which is arguably the more honest metric for a system that surfaces
ambiguity instead of silently resolving it.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CLI_PATH = REPO_ROOT / "target" / "release" / "beat-dsp-cli"
OCTAVE_MULTIPLES = (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)


@dataclass
class TrackResult:
    audio_path: str
    true_bpm: float
    detected_bpm: float | None
    confidence: float | None
    alternate_bpm: float | None
    alternate_confidence: float | None
    accuracy1: bool
    accuracy2: bool
    accuracy2_with_alternate: bool


def accuracy1(true_bpm: float, est_bpm: float, tol: float = 0.04) -> bool:
    return abs(est_bpm - true_bpm) <= tol * true_bpm


def accuracy2(true_bpm: float, est_bpm: float, tol: float = 0.04) -> bool:
    return any(accuracy1(true_bpm * m, est_bpm, tol) for m in OCTAVE_MULTIPLES)


def mp3_to_wav(mp3_path: Path, wav_path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(mp3_path), "-ar", "44100", "-ac", "1", "-f", "wav", str(wav_path)],
        check=True,
    )


def run_cli(wav_path: Path, mode: str = "offline") -> dict | None:
    result = subprocess.run(
        [str(CLI_PATH), str(wav_path), "--mode", mode],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"  CLI failed on {wav_path}: {result.stderr.strip()}")
        return None
    return json.loads(result.stdout)


def score_track(audio_path: Path, true_bpm: float, tmp_dir: Path, mode: str = "offline") -> TrackResult:
    wav_path = tmp_dir / (audio_path.stem + ".wav")
    try:
        mp3_to_wav(audio_path, wav_path)
        output = run_cli(wav_path, mode=mode)
    finally:
        wav_path.unlink(missing_ok=True)

    if output is None or output["bpm"] is None:
        return TrackResult(str(audio_path), true_bpm, None, None, None, None, False, False, False)

    detected = output["bpm"]
    alt = output["alternate_bpm"]
    a1 = accuracy1(true_bpm, detected)
    a2 = accuracy2(true_bpm, detected)
    a2_alt = a2 or (alt is not None and accuracy2(true_bpm, alt))

    return TrackResult(
        str(audio_path),
        true_bpm,
        detected,
        output["confidence"],
        alt,
        output["alternate_confidence"],
        a1,
        a2,
        a2_alt,
    )


def run_harness(index_path: Path, results_path: Path) -> None:
    if not CLI_PATH.exists():
        raise SystemExit(f"beat-dsp-cli not built at {CLI_PATH} -- run `cargo build --release -p beat-dsp` first")

    results = run_mode(index_path, "offline")

    n = len(results)
    scored = [r for r in results if r.detected_bpm is not None]
    a1_rate = sum(r.accuracy1 for r in scored) / n if n else 0.0
    a2_rate = sum(r.accuracy2 for r in scored) / n if n else 0.0
    a2_alt_rate = sum(r.accuracy2_with_alternate for r in scored) / n if n else 0.0

    report_lines = [
        "# GiantSteps Tempo -- beat-dsp real-audio evaluation (Phase 2.5a)",
        "",
        f"{n} tracks, {len(scored)} scored successfully, {n - len(scored)} CLI failures.",
        "",
        f"- **Accuracy1** (within 4% of true tempo): {a1_rate:.1%}",
        f"- **Accuracy2** (within 4% of true tempo or a standard octave multiple): {a2_rate:.1%}",
        f"- **Accuracy2-with-alternate** (Accuracy2, but the `alternate_bpm` hypothesis "
        f"also counts as correct when present): {a2_alt_rate:.1%}",
        "",
        "This is evaluation-only: GiantSteps audio is never used for training, per "
        "docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md §4 and the build plan's licensing note.",
        "",
        "| audio | true_bpm | detected_bpm | confidence | alternate_bpm | acc1 | acc2 | acc2+alt |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        report_lines.append(
            f"| {Path(r.audio_path).name} | {r.true_bpm:.1f} | "
            f"{r.detected_bpm if r.detected_bpm is not None else 'FAIL'} | "
            f"{r.confidence if r.confidence is not None else '-'} | "
            f"{r.alternate_bpm if r.alternate_bpm is not None else '-'} | "
            f"{'✓' if r.accuracy1 else ' '} | {'✓' if r.accuracy2 else ' '} | "
            f"{'✓' if r.accuracy2_with_alternate else ' '} |"
        )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(report_lines) + "\n")
    print(f"\nAccuracy1: {a1_rate:.1%}  Accuracy2: {a2_rate:.1%}  Accuracy2+alt: {a2_alt_rate:.1%}")
    print(f"Wrote {results_path}")


def run_mode(index_path: Path, mode: str) -> list[TrackResult]:
    """Scores every track in a GiantSteps-shaped index (audio_path,
    true_bpm) under a single CLI mode -- shared by run_streaming_vs_offline
    below so offline and streaming go through the exact same code path,
    the only variable being --mode."""
    pairs = []
    with index_path.open() as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                pairs.append((Path(entry["audio_path"]), entry["true_bpm"]))

    results: list[TrackResult] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, (audio_path, true_bpm) in enumerate(pairs):
            results.append(score_track(audio_path, true_bpm, tmp_dir, mode=mode))
            if (i + 1) % 50 == 0:
                print(f"  [{mode}] scored {i + 1}/{len(pairs)}")
    return results


def run_streaming_vs_offline(index_path: Path, results_path: Path) -> None:
    """Production-readiness item 2: every accuracy number in this project
    so far came from --mode offline. The causal streaming tracker -- what
    actually runs during live playback -- has only ever been checked
    against synthetic click tracks. This runs both modes over the same
    index so the accuracy gap (whatever it is) becomes an explicit, tracked
    number instead of an unknown."""
    if not CLI_PATH.exists():
        raise SystemExit(f"beat-dsp-cli not built at {CLI_PATH} -- run `cargo build --release -p beat-dsp` first")

    offline_results = run_mode(index_path, "offline")
    streaming_results = run_mode(index_path, "streaming")

    def summarize(results: list[TrackResult]) -> dict:
        n = len(results)
        scored = [r for r in results if r.detected_bpm is not None]
        return {
            "n": n,
            "scored": len(scored),
            "accuracy1": sum(r.accuracy1 for r in scored) / n if n else 0.0,
            "accuracy2": sum(r.accuracy2 for r in scored) / n if n else 0.0,
            "accuracy2_with_alternate": sum(r.accuracy2_with_alternate for r in scored) / n if n else 0.0,
        }

    off = summarize(offline_results)
    stream = summarize(streaming_results)

    report_lines = [
        f"# {index_path.parent.name} -- offline vs. streaming (production readiness item 2)",
        "",
        f"{off['n']} tracks. Offline: {off['scored']} scored. Streaming: {stream['scored']} scored.",
        "",
        "| metric | offline | streaming | gap |",
        "|---|---|---|---|",
        f"| Accuracy1 | {off['accuracy1']:.1%} | {stream['accuracy1']:.1%} | {stream['accuracy1'] - off['accuracy1']:+.1%} |",
        f"| Accuracy2 | {off['accuracy2']:.1%} | {stream['accuracy2']:.1%} | {stream['accuracy2'] - off['accuracy2']:+.1%} |",
        f"| Accuracy2+alt | {off['accuracy2_with_alternate']:.1%} | {stream['accuracy2_with_alternate']:.1%} | "
        f"{stream['accuracy2_with_alternate'] - off['accuracy2_with_alternate']:+.1%} |",
        "",
        "Streaming numbers reflect the *final* estimate after processing the whole track in "
        "512-sample chunks through the causal tracker -- a lower bound on real-time behavior, "
        "since production also cares about how fast that estimate converges after track start, "
        "which this doesn't measure (see the build plan's production-readiness item 2 for the "
        "time-to-lock follow-up).",
    ]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n".join(report_lines) + "\n")
    print(f"\nOffline  Acc1={off['accuracy1']:.1%} Acc2={off['accuracy2']:.1%}")
    print(f"Streaming Acc1={stream['accuracy1']:.1%} Acc2={stream['accuracy2']:.1%}")
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("eval_data/giantsteps/index.jsonl"))
    parser.add_argument("--results", type=Path, default=Path("eval/RESULTS.md"))
    parser.add_argument("--streaming-comparison", action="store_true", help="also run production-readiness item 2")
    args = parser.parse_args()

    run_harness(args.index, args.results)
    if args.streaming_comparison:
        run_streaming_vs_offline(args.index, args.results.parent / "STREAMING_VS_OFFLINE.md")
