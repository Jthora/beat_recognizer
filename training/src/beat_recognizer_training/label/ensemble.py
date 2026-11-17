"""Phase 3 augmentation: cross-checks beat-dsp's bootstrapped labels against
two independent algorithms already inside librosa -- beat_track (Ellis-style
dynamic programming) and plp (Predominant Local Pulse, a structurally
different Fourier-tempogram approach). Unlike every NC-licensed tool this
project otherwise treats as benchmark-only (see
docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md §4), librosa is ISC-licensed --
no NonCommercial or ShareAlike restriction -- so its output is safe to fold
directly into label-quality decisions, not just compare against.

madmom's classical comb-filter path and pyACA (both otherwise license-safe)
were also tried as further ensemble members and ruled out -- see
pyproject.toml's dependency comments for why (madmom: broken against
Python 3.12/NumPy 2.x; pyACA: a real bug plus a default parameter that's a
genuine performance/memory trap on real-length tracks).

Agreement between independently-engineered trackers is a stronger
correctness signal than any one tracker's own self-reported confidence:
this is meant to shrink dsp_teacher.py's needs_review set before spending
human review time on it, and to surface any "confident" label that doesn't
actually hold up independently -- something a self-reported confidence
score alone can't catch. beat_track and plp share one decode + onset-
envelope computation per track (~2.8s combined vs. ~10s+ run separately),
since this is meant to eventually run across the full ~4,000-track corpus.

Resumable like dsp_teacher.py's bootstrap, for the same reason.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import find_peaks

from beat_recognizer_training.eval.harness import accuracy2
from beat_recognizer_training.label.dsp_teacher import load_labels


@dataclass
class EnsembleRecord:
    audio_path: str
    dsp_bpm: float
    dsp_confidence: float
    dsp_needs_review: bool
    librosa_bt_bpm: float | None  # beat_track (dynamic programming)
    librosa_plp_bpm: float | None  # predominant local pulse
    agree_count: int  # how many of the 2 librosa estimates octave-tolerantly agree with dsp_bpm
    agree_total: int  # how many librosa estimates were actually computable
    agrees: bool  # agree_count >= 1 -- at least one independent method corroborates dsp


def run_librosa_estimates(audio_path: Path) -> tuple[float | None, float | None]:
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # librosa returns a numpy array in some versions, a scalar in others.
    bt_bpm = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    bt_bpm = bt_bpm if bt_bpm > 0 else None

    pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    peaks, _ = find_peaks(pulse, height=0.5)
    peak_times = librosa.frames_to_time(peaks, sr=sr)
    intervals = np.diff(peak_times)
    plp_bpm = float(60.0 / np.median(intervals)) if len(intervals) >= 2 else None

    return bt_bpm, plp_bpm


def already_checked_paths(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    paths = set()
    with out_path.open() as f:
        for line in f:
            if line.strip():
                paths.add(json.loads(line)["audio_path"])
    return paths


def cross_check(labels_path: Path, out_path: Path) -> None:
    records = [r for r in load_labels(labels_path) if r.has_rhythm and r.bpm is not None]
    done = already_checked_paths(out_path)
    remaining = [r for r in records if r.audio_path not in done]
    print(f"{len(done)} already cross-checked, {len(remaining)} remaining")

    def octave_agree(a: float, b: float) -> bool:
        return accuracy2(a, b) or accuracy2(b, a)

    with out_path.open("a") as f:
        for i, record in enumerate(remaining):
            try:
                bt_bpm, plp_bpm = run_librosa_estimates(Path(record.audio_path))
            except Exception as e:
                print(f"  librosa failed on {record.audio_path}: {e}")
                bt_bpm, plp_bpm = None, None

            estimates = [e for e in (bt_bpm, plp_bpm) if e is not None]
            agree_count = sum(octave_agree(record.bpm, e) for e in estimates)
            out = EnsembleRecord(
                record.audio_path,
                record.bpm,
                record.confidence,
                record.needs_review,
                bt_bpm,
                plp_bpm,
                agree_count,
                len(estimates),
                agree_count >= 1,
            )
            f.write(json.dumps(asdict(out)) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(remaining)}")

    print(f"Wrote cross-check results to {out_path}")


def load_ensemble(path: Path) -> list[EnsembleRecord]:
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(EnsembleRecord(**json.loads(line)))
    return records


def select_trusted_labels(labels_path: Path, ensemble_path: Path) -> list:
    """The real Phase 4 training-set query, superseding dsp_teacher.py's
    confidence-only select_confident_labels() once an ensemble cross-check
    exists for a label set. Motivated by the 500-track sample's actual
    numbers: DSP-confidence >= 0.6 alone let a 26% wrong-rate through
    (7/27), while independent-tracker agreement is a harder, cross-checked
    bar that's now been validated against that same sample -- so agreement
    is the gate, not a self-reported confidence score. Labels librosa
    couldn't score at all fall back to the confidence-only rule rather than
    being silently dropped.
    """
    labels_by_path = {r.audio_path: r for r in load_labels(labels_path)}
    ensemble_by_path = {r.audio_path: r for r in load_ensemble(ensemble_path)}

    trusted = []
    for path, record in labels_by_path.items():
        if not record.has_rhythm:
            continue
        cross_check = ensemble_by_path.get(path)
        if cross_check is None or cross_check.agree_total == 0:
            if record.confidence is not None and record.confidence >= 0.6:
                trusted.append(record)
        elif cross_check.agrees:
            trusted.append(record)
    return trusted


def report(path: Path) -> None:
    records = load_ensemble(path)
    n = len(records)
    scored = [r for r in records if r.agree_total > 0]
    agree_rate = sum(r.agrees for r in scored) / len(scored) if scored else 0.0
    both_agree = sum(r.agree_count == 2 for r in records if r.agree_total == 2)
    both_total = sum(r.agree_total == 2 for r in records)

    promotable = [r for r in records if r.dsp_needs_review and r.agrees]
    demotable = [r for r in records if not r.dsp_needs_review and r.agree_total > 0 and not r.agrees]

    print(f"\n{n} labels cross-checked, {len(scored)} had at least one librosa estimate succeed")
    print(f"Agreement rate (at least one of {{beat_track, plp}} matches dsp, octave-tolerant): {agree_rate:.1%}")
    if both_total:
        print(f"Both librosa methods agree with dsp AND each other: {both_agree}/{both_total}")
    print(
        f"needs_review -> agreement found (promotion candidates): "
        f"{len(promotable)} / {sum(r.dsp_needs_review for r in records)}"
    )
    print(
        f"confident -> no librosa agreement (demotion candidates): "
        f"{len(demotable)} / {sum(not r.dsp_needs_review for r in records)}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("data/LABELS.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/ENSEMBLE.jsonl"))
    args = parser.parse_args()

    cross_check(args.labels, args.out)
    report(args.out)
