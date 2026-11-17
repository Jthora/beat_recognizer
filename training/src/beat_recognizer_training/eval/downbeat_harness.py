"""Phase 3.5's go/no-go: validates the new tatum/measure/downbeat capability
(`crates/beat-dsp/src/{tempo,phase,lib}.rs`) against Ballroom (698 tracks)
and Hainsworth (222 tracks), both of which already carry real, human-
annotated `downbeat_times_s` ground truth (confirmed present directly, not
assumed) alongside the `beat_times_s` this project's existing tempo-
accuracy harness already uses.

Metric: `mir_eval.beat.f_measure` (the standard MIREX 70ms-window beat
F-measure) applied to the downbeat-only subsequence on each side --
mir_eval has no dedicated downbeat module (checked directly), and treating
downbeat tracking as a sparser instance of the same beat-tracking
evaluation problem is standard published practice, not a project-specific
invention. Beat-level F-measure is also computed alongside, as a sanity
check that isolates whether any drop is downbeat-specific rather than a
general tracking regression.

Motivated by a real smoke-test finding (build plan, Phase 3.5): one
Ballroom track's tempo came back essentially exact but its downbeats were
spaced every 2 beats instead of the true 4 -- this harness exists to find
out whether that's common or an outlier before attempting any fix to
`tempo.rs`'s `find_metrical_level` ratio-preference order.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import mir_eval
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
CLI_PATH = REPO_ROOT / "target" / "release" / "beat-dsp-cli"


def _resolve(base: Path, raw_path: str) -> Path:
    """Same discipline as confidence_prototype.py's own `_resolve` --
    index files disagree on whether audio_path is relative to the repo
    root or to training/, so resolve explicitly against a known base
    instead of trusting the process's cwd."""
    p = Path(raw_path)
    return p if p.is_absolute() or p.exists() else base / p


@dataclass
class TrackResult:
    audio_path: str
    dataset: str
    n_ref_beats: int
    n_ref_downbeats: int
    beat_f_measure: float | None
    downbeat_f_measure: float | None


def run_cli_offline(wav_path: Path) -> dict | None:
    result = subprocess.run(
        [str(CLI_PATH), str(wav_path), "--mode", "offline"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        print(f"  CLI failed on {wav_path}: {result.stderr.strip()}")
        return None
    return json.loads(result.stdout)


def score_track(entry: dict, dataset: str) -> TrackResult:
    wav_path = _resolve(REPO_ROOT, entry["audio_path"])
    ref_beats = np.array(sorted(entry["beat_times_s"]))
    ref_downbeats = np.array(sorted(entry["downbeat_times_s"]))

    output = run_cli_offline(wav_path)
    if output is None or output.get("bpm") is None:
        return TrackResult(str(wav_path), dataset, len(ref_beats), len(ref_downbeats), None, None)

    est_beats = np.array(output.get("beats_s", []))
    est_downbeats = np.array(output.get("downbeats_s", []))

    def safe_f_measure(ref: np.ndarray, est: np.ndarray) -> float | None:
        if len(ref) < 2 or len(est) == 0:
            return None
        ref_trimmed = mir_eval.beat.trim_beats(ref)
        est_trimmed = mir_eval.beat.trim_beats(est)
        if len(ref_trimmed) < 2:
            return None
        return float(mir_eval.beat.f_measure(ref_trimmed, est_trimmed))

    return TrackResult(
        str(wav_path),
        dataset,
        len(ref_beats),
        len(ref_downbeats),
        safe_f_measure(ref_beats, est_beats),
        safe_f_measure(ref_downbeats, est_downbeats),
    )


def load_index(index_path: Path) -> list[dict]:
    with index_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def run_dataset(index_path: Path, dataset: str, limit: int | None = None) -> list[TrackResult]:
    entries = load_index(index_path)
    entries = [e for e in entries if e.get("downbeat_times_s")]
    if limit:
        entries = entries[:limit]
    print(f"{dataset}: {len(entries)} tracks with downbeat annotations")

    results = []
    for i, entry in enumerate(entries):
        results.append(score_track(entry, dataset))
        if (i + 1) % 50 == 0:
            print(f"  [{dataset}] scored {i + 1}/{len(entries)}")
    return results


def summarize(results: list[TrackResult]) -> dict:
    beat_scores = [r.beat_f_measure for r in results if r.beat_f_measure is not None]
    downbeat_scores = [r.downbeat_f_measure for r in results if r.downbeat_f_measure is not None]
    return {
        "n": len(results),
        "n_beat_scored": len(beat_scores),
        "n_downbeat_scored": len(downbeat_scores),
        "beat_f_measure": sum(beat_scores) / len(beat_scores) if beat_scores else 0.0,
        "downbeat_f_measure": sum(downbeat_scores) / len(downbeat_scores) if downbeat_scores else 0.0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ballroom-index", type=Path, default=Path("eval_data/ballroom/index.jsonl"))
    parser.add_argument("--hainsworth-index", type=Path, default=Path("eval_data/hainsworth/index.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/DOWNBEAT_RESULTS.md"))
    parser.add_argument("--limit", type=int, default=None, help="cap tracks per dataset, for a quick smoke run")
    args = parser.parse_args()

    if not CLI_PATH.exists():
        raise SystemExit(f"beat-dsp-cli not built at {CLI_PATH} -- run `cargo build --release -p beat-dsp` first")

    ballroom = run_dataset(args.ballroom_index, "ballroom", limit=args.limit)
    hainsworth = run_dataset(args.hainsworth_index, "hainsworth", limit=args.limit)
    combined = ballroom + hainsworth

    b = summarize(ballroom)
    h = summarize(hainsworth)
    c = summarize(combined)

    report_lines = [
        "# Phase 3.5 downbeat validation -- Ballroom + Hainsworth",
        "",
        "Beat F-measure is a sanity check (isolates whether any gap is "
        "downbeat-specific); downbeat F-measure is the real go/no-go question.",
        "",
        "| dataset | tracks | beat F-measure | downbeat F-measure |",
        "|---|---|---|---|",
        f"| Ballroom | {b['n']} ({b['n_downbeat_scored']} scored) | {b['beat_f_measure']:.1%} | {b['downbeat_f_measure']:.1%} |",
        f"| Hainsworth | {h['n']} ({h['n_downbeat_scored']} scored) | {h['beat_f_measure']:.1%} | {h['downbeat_f_measure']:.1%} |",
        f"| **Combined** | {c['n']} ({c['n_downbeat_scored']} scored) | {c['beat_f_measure']:.1%} | {c['downbeat_f_measure']:.1%} |",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report_lines) + "\n")

    out_jsonl = args.out.with_suffix(".jsonl")
    with out_jsonl.open("w") as f:
        for r in combined:
            f.write(json.dumps(r.__dict__) + "\n")

    print(f"\nBallroom:   beat={b['beat_f_measure']:.1%} downbeat={b['downbeat_f_measure']:.1%}")
    print(f"Hainsworth: beat={h['beat_f_measure']:.1%} downbeat={h['downbeat_f_measure']:.1%}")
    print(f"Combined:   beat={c['beat_f_measure']:.1%} downbeat={c['downbeat_f_measure']:.1%}")
    print(f"\nWrote {args.out} and {out_jsonl}")
