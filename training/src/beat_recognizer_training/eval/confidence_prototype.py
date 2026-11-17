"""Production-readiness item 2's confidence-signal exploration (see the
build plan's Phase 3.5b, including "Iteration 2"): prototypes candidate
replacements for raw instantaneous confidence in Python against data
already collected, *before* committing to any change in shipped Rust
tracker behavior. `MIN_RHYTHM_CONFIDENCE`'s 0.25 threshold and
`has_rhythm`'s current gating are untouched by this module; it only reads
`beat-dsp-cli --mode trace`'s diagnostic output.

Signals tested, all derivable from data `tempo.rs`/`phase.rs` already
compute and previously discarded (see `TempoEstimate.prominence` and
`PhaseEstimate.best_score`, added purely for this prototype):

- prominence: the winning candidate's score relative to the mean of every
  other candidate scored that frame (tempo.rs).
- whole-trace stability: mean resultant length of confidence-weighted
  *tempo* vectors across the whole trace -- `|mean_k(confidence_k *
  exp(i*angle_k))| / mean_k(confidence_k)`, the standard circular-
  statistics coherence measure, where `angle_k = 2*pi*frac(log2(bpm_k))`.

  Important correction made *during* the first prototype pass, not
  before: the first version of this used `beat_phase` (progress toward
  the next beat) as the angle. That's wrong -- `beat_phase` sweeps the
  full [0, 1) circle once per beat *by design*, even for a perfectly-
  locked tracker, so its resultant vector averages toward ~0 regardless
  of tracking quality (confirmed directly: a correctly-tracked real
  Ballroom clip scored stability=0.008, near the theoretical floor,
  purely from the sweep, not from any real instability). What's actually
  wanted is consistency of the *tempo estimate* over time, not the
  beat-phase counter. `log2(bpm)`'s fractional part is the fix: it also
  elegantly makes octave-equivalent tempi (60/120/240 BPM, etc.) land at
  the same angle, so it doesn't penalize legitimate octave-alternation
  the way a raw BPM comparison would.

  Iteration 2 finding: whole-trace stability performed *worse* than
  simple confidence overall (every FP reduction cost more TP than it was
  worth) -- diagnosed cause: the startup convergence transient (a real
  track's tempo estimate swinging 41->304 BPM in its first second) drags
  down the average even for tracks that lock on well afterward.

- windowed stability: the same mean-resultant-length computation, but
  restricted to the trailing N seconds of the trace instead of the whole
  thing -- tests whether excluding the startup transient fixes whole-
  trace stability's underperformance.
- debounced (hysteresis) has_rhythm: `has_rhythm` only flips false->true
  once confidence has cleared the threshold for K consecutive frames --
  the simplest untried alternative, standard DSP debouncing, no complex-
  vector math at all.
- entropy (post-Iteration-3 continued research): normalized Shannon entropy
  of the full candidate-lag score distribution (`tempo.rs`), a different
  reduction of the same `scores` data prominence already uses -- close in
  spirit to MIRToolbox's "pulse entropy" pulse-clarity feature (Lartillot,
  Eerola, Toiviainen & Fornari 2008). Not redundant with prominence by
  construction: two distributions can share a winner-vs-mean ratio while
  differing in how concentrated the rest of the mass is.

  Direction confirmed empirically, and it's the *opposite* of the naive
  "peakiness" intuition (see the full rationale in tempo.rs where it's
  computed): high entropy = a real periodic signal (score mass spread
  across harmonics/subharmonics of the true period); low entropy = no
  true periodicity (mass concentrated on a few incidental bumps). By far
  the strongest signal found across all iterations of this exploration --
  see the build plan's Phase 3.5b for the full comparison table.
"""

from __future__ import annotations

import cmath
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CLI_PATH = REPO_ROOT / "target" / "release" / "beat-dsp-cli"
TRACE_CACHE_DIR = REPO_ROOT / "training" / "eval" / "_trace_cache"


def run_trace(wav_path: Path) -> list[dict]:
    result = subprocess.run(
        [str(CLI_PATH), str(wav_path), "--mode", "trace"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return []
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def run_trace_cached(wav_path: Path, cache_dir: Path = TRACE_CACHE_DIR) -> list[dict]:
    """Iteration 2 re-analyzes the same tracks several different ways
    (windowed stability at several window sizes, debounce at several K
    values, failure-case inspection) -- caching the raw per-chunk trace
    once per track avoids re-invoking the CLI for each of those, which
    would otherwise mean re-running the full tracker over every track
    several times for no new information."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(wav_path).encode()).hexdigest()[:16]
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    frames = run_trace(wav_path)
    cache_path.write_text(json.dumps(frames))
    return frames


def angle_of(bpm: float) -> float:
    return (math.log2(bpm) % 1.0) * 2 * math.pi


def resultant_stability(frames: list[dict]) -> float:
    """Mean resultant length of confidence-weighted tempo vectors over
    whatever frames are passed in -- shared by whole-trace and windowed
    stability, which differ only in which frames they pass here."""
    scored = [f for f in frames if f["bpm"] > 0]
    if not scored:
        return 0.0
    vectors = [f["confidence"] * cmath.exp(1j * angle_of(f["bpm"])) for f in scored]
    mean_vector = sum(vectors) / len(vectors)
    mean_magnitude = sum(f["confidence"] for f in scored) / len(scored)
    return abs(mean_vector) / mean_magnitude if mean_magnitude > 1e-9 else 0.0


def windowed_stability(frames: list[dict], window_s: float) -> float:
    if not frames:
        return 0.0
    cutoff_ms = frames[-1]["t_ms"] - window_s * 1000
    window = [f for f in frames if f["t_ms"] >= cutoff_ms]
    return resultant_stability(window)


def debounced_has_rhythm(frames: list[dict], k: int, threshold: float = 0.25) -> bool:
    """True only if the last k frames all cleared `threshold` -- standard
    debouncing, no history beyond a short trailing run of booleans."""
    if len(frames) < k:
        return False
    tail = frames[-k:]
    return all(f["confidence"] > threshold for f in tail)


def asymmetric_hysteresis(frames: list[dict], enter: float, exit_: float, hangover: int = 0) -> bool:
    """Iteration 3 item 1: standard VAD hangover-scheme state machine, not
    the single-symmetric-threshold debounce above. `has_rhythm` requires
    confidence > `enter` (a stricter bar) to become true; once true, it
    only reverts to false after confidence stays <= `exit_` (a more
    lenient bar) for more than `hangover` consecutive frames -- the same
    asymmetric-threshold-plus-hangover shape VAD literature uses to avoid
    chopping brief low-confidence dips. Returns the final state after the
    whole trace, same convention as debounced_has_rhythm, for comparable
    FP/TP-rate reporting."""
    state = False
    below_exit_run = 0
    for f in frames:
        conf = f["confidence"]
        if not state:
            if conf > enter:
                state = True
                below_exit_run = 0
        elif conf <= exit_:
            below_exit_run += 1
            if below_exit_run > hangover:
                state = False
        else:
            below_exit_run = 0
    return state


@dataclass
class TraceSummary:
    audio_path: str
    final_confidence: float | None
    final_prominence: float | None
    final_entropy: float | None
    stability_whole: float | None
    stability_2s: float | None
    stability_4s: float | None
    stability_6s: float | None
    debounced_10: bool | None
    debounced_25: bool | None
    debounced_50: bool | None
    hyst_35_15_0: bool | None
    hyst_30_15_25: bool | None
    hyst_35_20_25: bool | None
    has_alternate: bool | None
    final_bpm: float | None
    n_frames: int
    duration_s: float


def summarize(wav_path: Path) -> TraceSummary:
    frames = run_trace_cached(wav_path)
    if not frames:
        return TraceSummary(*([str(wav_path)] + [None] * 15 + [0, 0.0]))

    last = frames[-1]
    return TraceSummary(
        audio_path=str(wav_path),
        final_confidence=last["confidence"],
        final_prominence=last["prominence"],
        final_entropy=last.get("entropy"),
        stability_whole=resultant_stability(frames),
        stability_2s=windowed_stability(frames, 2.0),
        stability_4s=windowed_stability(frames, 4.0),
        stability_6s=windowed_stability(frames, 6.0),
        debounced_10=debounced_has_rhythm(frames, 10),
        debounced_25=debounced_has_rhythm(frames, 25),
        debounced_50=debounced_has_rhythm(frames, 50),
        hyst_35_15_0=asymmetric_hysteresis(frames, 0.35, 0.15, 0),
        hyst_30_15_25=asymmetric_hysteresis(frames, 0.30, 0.15, 25),
        hyst_35_20_25=asymmetric_hysteresis(frames, 0.35, 0.20, 25),
        has_alternate=last["has_alternate"],
        final_bpm=last["bpm"],
        n_frames=len(frames),
        duration_s=last["t_ms"] / 1000,
    )


def summarize_set(wav_paths: list[Path]) -> list[TraceSummary]:
    if not CLI_PATH.exists():
        raise SystemExit(f"beat-dsp-cli not built at {CLI_PATH} -- run `cargo build --release -p beat-dsp` first")
    return [summarize(p) for p in wav_paths]


def _rate(summaries: list[TraceSummary], predicate) -> tuple[int, int]:
    scored = [s for s in summaries if s.final_confidence is not None]
    hits = sum(1 for s in scored if predicate(s))
    return hits, len(scored)


def report_rules(no_rhythm: list[TraceSummary], ballroom: list[TraceSummary]) -> None:
    rules = {
        "confidence>0.25 (current, shipped)": lambda s: s.final_confidence > 0.25,
        "stability_whole>0.6": lambda s: s.stability_whole > 0.6,
        "stability_2s>0.6": lambda s: s.stability_2s > 0.6,
        "stability_4s>0.6": lambda s: s.stability_4s > 0.6,
        "stability_6s>0.6": lambda s: s.stability_6s > 0.6,
        "confidence>0.25 AND stability_4s>0.5": lambda s: s.final_confidence > 0.25 and s.stability_4s > 0.5,
        "debounced_10 (10 consecutive frames)": lambda s: s.debounced_10,
        "debounced_25 (25 consecutive frames)": lambda s: s.debounced_25,
        "debounced_50 (50 consecutive frames)": lambda s: s.debounced_50,
        "hyst enter=0.35 exit=0.15 hangover=0": lambda s: s.hyst_35_15_0,
        "hyst enter=0.30 exit=0.15 hangover=25": lambda s: s.hyst_30_15_25,
        "hyst enter=0.35 exit=0.20 hangover=25": lambda s: s.hyst_35_20_25,
        "entropy>0.90": lambda s: s.final_entropy is not None and s.final_entropy > 0.90,
        "entropy>0.93": lambda s: s.final_entropy is not None and s.final_entropy > 0.93,
        "entropy>0.95": lambda s: s.final_entropy is not None and s.final_entropy > 0.95,
        "confidence>0.25 AND entropy>0.93": lambda s: s.final_confidence > 0.25 and s.final_entropy is not None and s.final_entropy > 0.93,
        "debounced_25 AND entropy>0.93": lambda s: s.debounced_25 and s.final_entropy is not None and s.final_entropy > 0.93,
    }
    print(f"\n{'rule':<42} {'no-rhythm FP':<16} {'ballroom TP':<16}")
    for name, rule in rules.items():
        fp_hits, fp_n = _rate(no_rhythm, rule)
        tp_hits, tp_n = _rate(ballroom, rule)
        fp_pct = f"{fp_hits}/{fp_n} ({fp_hits / fp_n:.0%})" if fp_n else "n/a"
        tp_pct = f"{tp_hits}/{tp_n} ({tp_hits / tp_n:.0%})" if tp_n else "n/a"
        print(f"{name:<42} {fp_pct:<16} {tp_pct:<16}")


def _resolve(base: Path, raw_path: str) -> Path:
    """Index files disagree on whether audio_path is relative to the repo
    root or to training/ (GiantSteps/Ballroom use the former, no_rhythm the
    latter) -- resolve explicitly against a known base instead of trusting
    the process's cwd, since that's what actually broke the first run of
    the streaming-vs-offline harness earlier this session."""
    p = Path(raw_path)
    return p if p.is_absolute() or p.exists() else base / p


def load_no_rhythm_and_ballroom() -> tuple[list[Path], list[Path]]:
    training_root = REPO_ROOT / "training"
    no_rhythm_index = training_root / "eval_data/no_rhythm/index.jsonl"
    ballroom_index = training_root / "eval_data/ballroom/index.jsonl"

    no_rhythm_paths = [
        _resolve(training_root, json.loads(l)["audio_path"]) for l in no_rhythm_index.open() if l.strip()
    ]
    ballroom_paths = [
        _resolve(REPO_ROOT, json.loads(l)["audio_path"]) for l in ballroom_index.open() if l.strip()
    ][:100]
    return no_rhythm_paths, ballroom_paths


if __name__ == "__main__":
    no_rhythm_paths, ballroom_paths = load_no_rhythm_and_ballroom()

    print("Scoring no-rhythm set...")
    no_rhythm_summaries = summarize_set(no_rhythm_paths)
    print("Scoring Ballroom sample...")
    ballroom_summaries = summarize_set(ballroom_paths)

    report_rules(no_rhythm_summaries, ballroom_summaries)

    out_path = REPO_ROOT / "training/eval/CONFIDENCE_PROTOTYPE.jsonl"
    with out_path.open("w") as f:
        for s in no_rhythm_summaries + ballroom_summaries:
            f.write(json.dumps(s.__dict__) + "\n")
    print(f"\nWrote {out_path}")
