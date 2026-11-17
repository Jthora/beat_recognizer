"""Further confidence-signal research (post-Iteration-3): every signal
tested so far (confidence, prominence, entropy, stability, debouncing) is
derived from a *single*, full-band onset envelope. Real literature on
multi-band onset detection ("multi-band detection is useful... low-
frequency kicks and high-frequency hats produce onsets a single-band
detector misses"; band-splitting is a standard technique for separating
genuine musical events from broadband noise/texture) suggests a genuinely
different kind of signal: do independent frequency bands *agree* on
periodicity, not just how strong periodicity looks in one band.

Hypothesis: real music has coordinated multi-band rhythm (kick locks to
hi-hat locks to harmonic rhythm, roughly the same underlying pulse across
bands). Broadband textures that fool single-band confidence (rain,
breathing, traffic) shouldn't show that coordination -- each band's
"loudest" apparent periodicity should be closer to independent noise.

This is a fast, from-scratch Python feasibility check using librosa
directly (not a faithful reimplementation of tempo.rs's tie-break logic --
that's only worth doing in Rust if this shows real separation). Bands are
drum-kit-inspired (kick/snare/hihat ranges), not arbitrary: low ~20-250Hz,
mid ~250-2000Hz, high ~2000-8000Hz. Per-band "confidence" is the band's own
top-lag normalized autocorrelation score; cross-band coherence reuses the
exact same resultant-length math already validated for stability_4s
(confidence_prototype.py), just aggregating across the 3 bands instead of
across time frames -- same technique, different population.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

SR = 22_050
N_FFT = 2048
HOP_LENGTH = 512
BPM_MIN, BPM_MAX = 40.0, 300.0

BANDS = {
    "low": (20.0, 250.0),
    "mid": (250.0, 2000.0),
    "high": (2000.0, 8000.0),
}


def band_onset_envelope(S: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (freqs >= lo) & (freqs < hi)
    band = S[mask]
    flux = np.diff(band, axis=1)
    flux = np.clip(flux, 0, None).sum(axis=0)
    std = flux.std()
    return flux / std if std > 1e-9 else flux


def top_lag_and_score(onset_env: np.ndarray, sr: float) -> tuple[float, float]:
    """Normalized autocorrelation peak within [BPM_MIN, BPM_MAX] -- a quick
    argmax, not tempo.rs's full harmonic-averaged/tie-broken estimate
    (deliberately: this is a feasibility check, see module docstring)."""
    hop_s = HOP_LENGTH / sr
    n = len(onset_env)
    centered = onset_env - onset_env.mean()
    ac0 = float(np.dot(centered, centered)) / n
    if ac0 <= 1e-9:
        return 0.0, 0.0

    min_lag = max(1, round((60.0 / BPM_MAX) / hop_s))
    max_lag = min(n - 1, round((60.0 / BPM_MIN) / hop_s))
    if max_lag <= min_lag:
        return 0.0, 0.0

    best_lag, best_score = min_lag, -1.0
    for lag in range(min_lag, max_lag + 1):
        m = n - lag
        r = float(np.dot(centered[:m], centered[lag : lag + m])) / m / ac0
        if r > best_score:
            best_lag, best_score = lag, r

    bpm = 60.0 / (best_lag * hop_s)
    return bpm, max(best_score, 0.0)


def angle_of(bpm: float) -> float:
    return (math.log2(bpm) % 1.0) * 2 * math.pi


@dataclass
class MultibandSummary:
    audio_path: str
    band_bpms: dict
    band_scores: dict
    cross_band_coherence: float
    mean_band_score: float


def summarize(wav_path: Path) -> MultibandSummary:
    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    band_bpms, band_scores = {}, {}
    for name, (lo, hi) in BANDS.items():
        env = band_onset_envelope(S, freqs, lo, hi)
        bpm, score = top_lag_and_score(env, sr)
        band_bpms[name] = bpm
        band_scores[name] = score

    vectors = [band_scores[b] * cmath.exp(1j * angle_of(band_bpms[b])) for b in BANDS if band_bpms[b] > 0]
    mean_score = sum(band_scores.values()) / len(band_scores)
    if vectors and mean_score > 1e-9:
        coherence = abs(sum(vectors) / len(vectors)) / mean_score
    else:
        coherence = 0.0

    return MultibandSummary(
        audio_path=str(wav_path),
        band_bpms=band_bpms,
        band_scores=band_scores,
        cross_band_coherence=coherence,
        mean_band_score=mean_score,
    )


def summarize_set(wav_paths: list[Path]) -> list[MultibandSummary]:
    out = []
    for p in wav_paths:
        try:
            out.append(summarize(p))
        except Exception as e:
            print(f"  failed on {p}: {e}")
    return out


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from beat_recognizer_training.eval.confidence_prototype import load_no_rhythm_and_ballroom

    no_rhythm_paths, ballroom_paths = load_no_rhythm_and_ballroom()
    print(f"{len(no_rhythm_paths)} no-rhythm, {len(ballroom_paths)} ballroom")

    print("Scoring no-rhythm set (multi-band)...")
    nr = summarize_set(no_rhythm_paths)
    print("Scoring ballroom sample (multi-band)...")
    br = summarize_set(ballroom_paths)

    def pctiles(xs, ps=(0, 10, 25, 50, 75, 90, 100)):
        xs = sorted(xs)
        n = len(xs)
        return {p: round(xs[min(n - 1, int(p / 100 * n))], 3) for p in ps}

    nr_coh = [s.cross_band_coherence for s in nr]
    br_coh = [s.cross_band_coherence for s in br]
    print(f"\nno-rhythm cross_band_coherence percentiles: {pctiles(nr_coh)}")
    print(f"ballroom cross_band_coherence percentiles:   {pctiles(br_coh)}")

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        nr_fp = sum(1 for c in nr_coh if c > thresh)
        br_tp = sum(1 for c in br_coh if c > thresh)
        print(
            f"coherence>{thresh}: no-rhythm FP {nr_fp}/{len(nr_coh)} ({nr_fp/len(nr_coh):.0%})  "
            f"ballroom TP {br_tp}/{len(br_coh)} ({br_tp/len(br_coh):.0%})"
        )

    out_path = Path(__file__).resolve().parents[3] / "eval" / "MULTIBAND_COHERENCE.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for s in nr + br:
            f.write(json.dumps(s.__dict__) + "\n")
    print(f"\nWrote {out_path}")
