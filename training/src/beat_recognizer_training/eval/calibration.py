"""Production-readiness item 3: SMC (the eval set built specifically to be
hard/ambiguous, which calibration checking needs) is dead with no mirror
(see fetch_eval_sets.py's docstring). Two substitutes, combined:

- Hainsworth's own "rubato (Rub)" subset (26 tracks among the 222 already
  downloaded, confirmed via data.txt's `tempocond` field -- not bundled in
  fetch_eval_sets.py's index.jsonl, re-derived here) -- expressive tempo
  variation is genuinely hard for beat trackers, and unlike a proxy this is
  an external, citable hard set with real ground truth.
- The ensemble pipeline's own `needs_review` tier (tracks where beat-dsp
  and librosa disagree) -- directly relevant to *this system's* actual
  failure modes, though without independent ground truth this half is a
  confidence-distribution sanity check, not a true accuracy calibration
  check. Kept separate in the report for that reason, not blended into one
  number that would overstate what it's actually measuring.

A real calibration check (low confidence correlating with actual error)
needs a trained model to be meaningful in the way Phase 6 originally
intended -- run now against the DSP tracker instead, which still answers a
real, useful question in the meantime: does the *existing* confidence
score already track difficulty, or does Phase 4 need to fix that from
scratch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import requests

from beat_recognizer_training.eval.confidence_prototype import summarize as trace_summarize
from beat_recognizer_training.eval.harness import accuracy1, accuracy2, run_cli
from beat_recognizer_training.label.dsp_teacher import load_labels
from beat_recognizer_training.label.ensemble import load_ensemble
from beat_recognizer_training.label.pipeline import tier_labels

HAINSWORTH_DATA_URL = "https://raw.githubusercontent.com/audiocontentanalysis/dataset-hainsworth-tempo/main/data.txt"


def find_rubato_tracks(hainsworth_audio_dir: Path) -> list[dict]:
    resp = requests.get(HAINSWORTH_DATA_URL, timeout=30)
    resp.raise_for_status()
    rows = [line for line in resp.text.splitlines() if line.strip() and not line.startswith("#")]

    existing = {p.name for p in hainsworth_audio_dir.glob("*.wav")}
    rubato = []
    for row in rows:
        fields = [f.strip() for f in row.split("<sep>")]
        wav_name, artist, title, tempocond = fields[0], fields[1], fields[2], fields[6]
        beats = [float(x) for x in fields[10].split(",")]
        if wav_name not in existing or "rubato" not in tempocond.lower():
            continue
        # True tempo derived from the annotated beat times themselves
        # (median inter-beat interval) rather than trusting data.txt's own
        # bpm1/bpm2 fields, which are themselves one of several possible
        # metrical-level interpretations -- same principle as this
        # project's own `alternate` octave-ambiguity handling.
        # beats here are still raw sample positions; caller resolves sr.
        rubato.append(
            {
                "wav_name": wav_name,
                "artist": artist,
                "title": title,
                "beat_samples": beats,
            }
        )
    return rubato


@dataclass
class CalibrationResult:
    label: str
    source: str  # "hainsworth_rubato" | "ensemble_needs_review"
    dsp_bpm: float | None
    dsp_confidence: float | None
    dsp_prominence: float | None
    dsp_entropy: float | None
    true_bpm: float | None  # None for the ensemble half -- no independent ground truth
    accuracy1: bool | None
    accuracy2: bool | None


def score_hainsworth_rubato(hainsworth_audio_dir: Path) -> list[CalibrationResult]:
    """Uses --mode offline (run_cli's default) deliberately, not trace --
    this is what makes it an apples-to-apples re-check of the *exact*
    algorithm that produced the original inverted-confidence finding
    (Ellis DP via offline.rs, not the streaming comb-filter path), now that
    OfflineResult.prominence surfaces the same value `tempo.rs` already
    computed. See this module's iteration-2 addition below for the
    streaming-side (stability/debounce) re-check, which is a genuinely
    different algorithm and is labeled as such."""
    import soundfile as sf

    rubato = find_rubato_tracks(hainsworth_audio_dir)
    results = []
    for track in rubato:
        wav_path = hainsworth_audio_dir / track["wav_name"]
        sr = sf.info(str(wav_path)).samplerate
        beat_times = sorted(s / sr for s in track["beat_samples"])
        intervals = sorted(b - a for a, b in zip(beat_times, beat_times[1:]))
        median_interval = intervals[len(intervals) // 2]
        true_bpm = 60.0 / median_interval

        output = run_cli(wav_path)
        if output is None or output.get("bpm") is None:
            results.append(
                CalibrationResult(
                    track["wav_name"], "hainsworth_rubato", None, None, None, None, true_bpm, False, False
                )
            )
            continue

        a1 = accuracy1(true_bpm, output["bpm"])
        a2 = accuracy2(true_bpm, output["bpm"])
        results.append(
            CalibrationResult(
                track["wav_name"],
                "hainsworth_rubato",
                output["bpm"],
                output["confidence"],
                output.get("prominence"),
                output.get("entropy"),
                true_bpm,
                a1,
                a2,
            )
        )
    return results


@dataclass
class StreamingCalibrationResult:
    """Separate from CalibrationResult on purpose -- this measures the
    causal/streaming comb-filter path (via --mode trace), a genuinely
    different algorithm than the offline Ellis DP path CalibrationResult
    measures. Conflating the two would misrepresent which algorithm each
    finding is actually about."""

    label: str
    dsp_confidence: float | None
    dsp_entropy: float | None
    stability_4s: float | None
    true_bpm: float | None
    accuracy2: bool | None


def score_hainsworth_rubato_streaming(hainsworth_audio_dir: Path) -> list[StreamingCalibrationResult]:
    import soundfile as sf

    rubato = find_rubato_tracks(hainsworth_audio_dir)
    results = []
    for track in rubato:
        wav_path = hainsworth_audio_dir / track["wav_name"]
        sr = sf.info(str(wav_path)).samplerate
        beat_times = sorted(s / sr for s in track["beat_samples"])
        intervals = sorted(b - a for a, b in zip(beat_times, beat_times[1:]))
        median_interval = intervals[len(intervals) // 2]
        true_bpm = 60.0 / median_interval

        summary = trace_summarize(wav_path)
        if summary.final_confidence is None:
            results.append(StreamingCalibrationResult(track["wav_name"], None, None, None, true_bpm, None))
            continue
        a2 = accuracy2(true_bpm, summary.final_bpm) if summary.final_bpm else False
        results.append(
            StreamingCalibrationResult(
                track["wav_name"], summary.final_confidence, summary.final_entropy, summary.stability_4s, true_bpm, a2
            )
        )
    return results


def score_ensemble_needs_review(labels_path: Path, ensemble_path: Path) -> list[CalibrationResult]:
    labels = load_labels(labels_path)
    ensemble_by_path = {r.audio_path: r for r in load_ensemble(ensemble_path)}
    tiered = tier_labels(labels, ensemble_by_path)
    return [
        CalibrationResult(
            label=Path(r.audio_path).name,
            source="ensemble_needs_review",
            dsp_bpm=r.bpm,
            dsp_confidence=r.confidence,
            dsp_prominence=None,
            dsp_entropy=None,
            true_bpm=None,
            accuracy1=None,
            accuracy2=None,
        )
        for r in tiered
        if r.training_tier == "needs_review"
    ]


def _mean_by_correctness(rows, value_attr: str, correct_attr: str = "accuracy2") -> tuple[float | None, float | None, int, int]:
    correct = [getattr(r, value_attr) for r in rows if getattr(r, correct_attr) and getattr(r, value_attr) is not None]
    wrong = [getattr(r, value_attr) for r in rows if not getattr(r, correct_attr) and getattr(r, value_attr) is not None]
    c_mean = sum(correct) / len(correct) if correct else None
    w_mean = sum(wrong) / len(wrong) if wrong else None
    return c_mean, w_mean, len(correct), len(wrong)


def report(results: list[CalibrationResult]) -> None:
    rubato = [r for r in results if r.source == "hainsworth_rubato" and r.dsp_bpm is not None]
    needs_review = [r for r in results if r.source == "ensemble_needs_review"]

    print(f"\n=== Hainsworth rubato, OFFLINE mode ({len(rubato)}/26 scored) — real ground truth ===")
    print("(exact original methodology -- Ellis DP via offline.rs, same algorithm as the first inverted-confidence finding)")
    a1_rate = sum(r.accuracy1 for r in rubato) / len(rubato) if rubato else 0.0
    a2_rate = sum(r.accuracy2 for r in rubato) / len(rubato) if rubato else 0.0
    print(f"Accuracy1: {a1_rate:.1%}  Accuracy2: {a2_rate:.1%}")

    c_mean, w_mean, c_n, w_n = _mean_by_correctness(rubato, "dsp_confidence")
    print(f"confidence: correct={c_mean and round(c_mean, 3)} (n={c_n})  wrong={w_mean and round(w_mean, 3)} (n={w_n})")
    p_mean_c, p_mean_w, pc_n, pw_n = _mean_by_correctness(rubato, "dsp_prominence")
    print(f"prominence: correct={p_mean_c and round(p_mean_c, 3)} (n={pc_n})  wrong={p_mean_w and round(p_mean_w, 3)} (n={pw_n})")
    e_mean_c, e_mean_w, ec_n, ew_n = _mean_by_correctness(rubato, "dsp_entropy")
    print(f"entropy:    correct={e_mean_c and round(e_mean_c, 3)} (n={ec_n})  wrong={e_mean_w and round(e_mean_w, 3)} (n={ew_n})")
    print(
        "Well-calibrated signal should show correct > wrong; a small or inverted gap "
        "means that signal doesn't track difficulty on genuinely hard material."
    )

    print(f"\n=== Ensemble needs_review tier ({len(needs_review)} tracks) — no independent ground truth ===")
    confs = [r.dsp_confidence for r in needs_review if r.dsp_confidence is not None]
    if confs:
        print(f"Mean confidence: {sum(confs) / len(confs):.3f}  (range {min(confs):.3f}-{max(confs):.3f})")
    print(
        "Sanity check only: these are tracks independent trackers already disagree on, "
        "so a low mean confidence here would be consistent (not proof of calibration), "
        "and a high mean would be a real warning sign worth investigating."
    )


def report_streaming(results: list[StreamingCalibrationResult]) -> None:
    scored = [r for r in results if r.dsp_confidence is not None]
    print(f"\n=== Hainsworth rubato, STREAMING mode ({len(scored)}/26 scored) — different algorithm, labeled as such ===")
    print("(causal comb-filter path via --mode trace, NOT the offline Ellis DP path above)")
    c_mean, w_mean, c_n, w_n = _mean_by_correctness(scored, "dsp_confidence")
    print(f"confidence:    correct={c_mean and round(c_mean, 3)} (n={c_n})  wrong={w_mean and round(w_mean, 3)} (n={w_n})")
    e_mean, ew_mean, e_n, ew_n = _mean_by_correctness(scored, "dsp_entropy")
    print(f"entropy:       correct={e_mean and round(e_mean, 3)} (n={e_n})  wrong={ew_mean and round(ew_mean, 3)} (n={ew_n})")
    sc_mean, sw_mean, sc_n, sw_n = _mean_by_correctness(scored, "stability_4s")
    print(f"stability_4s:  correct={sc_mean and round(sc_mean, 3)} (n={sc_n})  wrong={sw_mean and round(sw_mean, 3)} (n={sw_n})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hainsworth-audio", type=Path, default=Path("eval_data/hainsworth/audio"))
    parser.add_argument("--labels", type=Path, default=Path("../data/LABELS.jsonl"))
    parser.add_argument("--ensemble", type=Path, default=Path("../data/ENSEMBLE.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/CALIBRATION_RESULTS.jsonl"))
    args = parser.parse_args()

    results = score_hainsworth_rubato(args.hainsworth_audio) + score_ensemble_needs_review(args.labels, args.ensemble)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in results:
            f.write(json.dumps(r.__dict__) + "\n")
    report(results)

    streaming_results = score_hainsworth_rubato_streaming(args.hainsworth_audio)
    streaming_out = args.out.parent / "CALIBRATION_STREAMING_RESULTS.jsonl"
    with streaming_out.open("w") as f:
        for r in streaming_results:
            f.write(json.dumps(r.__dict__) + "\n")
    report_streaming(streaming_results)

    print(f"\nWrote {args.out} and {streaming_out}")
