//! Deterministic (non-ML) beat/tempo tracking: onset detection, autocorrelation/
//! comb-filter tempo estimation, and causal phase tracking. Everything here
//! is reimplemented from published DSP algorithms (spectral-flux onset
//! detection, comb-filter periodicity, Ellis' 2007 dynamic-programming beat
//! tracker), so it carries no training-data licensing exposure — see
//! docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md.

pub mod offline;
pub mod onset;
pub mod phase;
pub mod tempo;

use std::collections::VecDeque;

/// How much onset-envelope history the streaming tracker keeps. Needs to
/// comfortably cover at least one full period at the slowest supported
/// tempo (40 BPM = 1.5s/beat) with margin for the recency-weighted phase
/// tracker to have real history to weigh.
const MAX_HISTORY_SECONDS: f64 = 8.0;

/// Below this, `has_rhythm` reports `false` even though `tempo::estimate_tempo`
/// found *some* candidate periodicity — a comb-filter/autocorrelation peak
/// exists for almost any non-silent input (including plain noise), so
/// "an estimate exists" alone was never a real rhythm/no-rhythm gate despite
/// `has_rhythm`'s doc comment promising one. Calibrated against real data,
/// not guessed: a 116-clip no-rhythm set (silence, white/pink noise, sparse
/// random onsets, ESC-50 environmental recordings — rain, breathing,
/// snoring, etc.) had 100% false positives with no gate at all; 0.25 cuts
/// that to 36% while costing zero true positives against a 100-track real
/// Ballroom sample (whose confidence never dropped below 0.279). The
/// remaining overlap between the two populations' confidence distributions
/// means this is a real, honest, partial mitigation, not a full fix —
/// separating them completely likely needs a stability/consistency signal
/// over multiple frames, not just a single-frame confidence threshold.
const MIN_RHYTHM_CONFIDENCE: f32 = 0.25;

/// Mirrors the `BeatEstimate` shape from `docs/WASM_BEAT_MODULE_SPEC.md` §4,
/// extended per `docs/WASM_BEAT_MODULE_ENHANCEMENTS.md` §2 with an explicit
/// no-rhythm state and octave-alternate exposure — both additive, neither
/// changes the original fields' meaning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatEstimate {
    /// Whether any periodicity was detected at all. `false` means "no
    /// discernible rhythm" — silence, talk radio, pure noise, or simply not
    /// enough history yet — a state distinct from "detected, but low
    /// confidence," which `confidence` alone can't disambiguate (a caller
    /// can't tell 0.0 apart from "barely detected" otherwise). When
    /// `false`, `bpm`/`beat_phase`/`confidence`/`alternate_*` are
    /// meaningless and should be ignored, not treated as a low-quality
    /// reading.
    pub has_rhythm: bool,
    pub bpm: f32,
    pub beat_phase: f32,
    pub confidence: f32,
    /// Milliseconds since the tracker was constructed or last reset — not
    /// wall-clock time. Callers align this to their own audio clock using
    /// whatever offset they already track (e.g. `AudioContext.currentTime`
    /// at the moment `reset()` was called).
    pub next_beat_time_ms: f64,
    pub is_downbeat: Option<bool>,
    /// The other tempo-octave candidate (e.g. 76 vs 152 BPM), present only
    /// when genuinely ambiguous — see `tempo::TempoEstimate::alternate`.
    /// Flattened into two `Option<f32>`s rather than a nested struct so
    /// this maps cleanly onto wasm-bindgen getters later.
    pub alternate_bpm: Option<f32>,
    pub alternate_confidence: Option<f32>,
}

impl Default for BeatEstimate {
    /// `has_rhythm: false` signals "no reading yet" rather than a guess —
    /// honest uncertainty is the whole point (spec §6), including before
    /// any audio has arrived or while there isn't enough history to trust
    /// a reading.
    fn default() -> Self {
        Self {
            has_rhythm: false,
            bpm: 0.0,
            beat_phase: 0.0,
            confidence: 0.0,
            next_beat_time_ms: 0.0,
            is_downbeat: None,
            alternate_bpm: None,
            alternate_confidence: None,
        }
    }
}

/// Streaming deterministic tracker: onset detection -> comb-filter tempo
/// estimation -> causal phase tracking, recomputed each time enough new
/// audio has arrived to complete another onset-detector hop.
pub struct DeterministicTracker {
    onset: onset::OnsetDetector,
    envelope: VecDeque<f32>,
    max_history_frames: usize,
    sample_rate: u32,
    total_samples: u64,
    current: BeatEstimate,
    // Diagnostic-only, not part of the shipped BeatEstimate shape (see
    // production-readiness item 2's confidence-signal exploration in the
    // build plan) -- the raw per-call outputs `recompute_estimate` already
    // computes internally, retained here instead of discarded so
    // `beat-dsp-cli --mode trace` can inspect them without duplicating the
    // envelope-buffering logic this struct already owns.
    last_tempo: Option<tempo::TempoEstimate>,
    last_phase: Option<phase::PhaseEstimate>,
}

impl DeterministicTracker {
    pub fn new() -> Self {
        Self {
            onset: onset::OnsetDetector::new(),
            envelope: VecDeque::new(),
            max_history_frames: 0,
            sample_rate: 0,
            total_samples: 0,
            current: BeatEstimate::default(),
            last_tempo: None,
            last_phase: None,
        }
    }

    /// Diagnostic-only accessors backing `beat-dsp-cli --mode trace` -- see
    /// the `last_tempo`/`last_phase` fields' doc comment.
    pub fn last_tempo_estimate(&self) -> Option<tempo::TempoEstimate> {
        self.last_tempo
    }

    pub fn last_phase_estimate(&self) -> Option<phase::PhaseEstimate> {
        self.last_phase
    }

    pub fn reset(&mut self) {
        self.onset.reset();
        self.envelope.clear();
        self.total_samples = 0;
        self.current = BeatEstimate::default();
        self.last_tempo = None;
        self.last_phase = None;
    }

    pub fn process_chunk(&mut self, samples: &[f32], sample_rate: u32) {
        if sample_rate != self.sample_rate {
            self.sample_rate = sample_rate;
            let frames_per_second = sample_rate as f64 / onset::HOP_SIZE as f64;
            self.max_history_frames = (MAX_HISTORY_SECONDS * frames_per_second).ceil() as usize;
        }

        self.total_samples += samples.len() as u64;
        let onsets = self.onset.push_samples(samples);
        if onsets.is_empty() {
            return;
        }

        for v in onsets {
            self.envelope.push_back(v);
        }
        while self.envelope.len() > self.max_history_frames {
            self.envelope.pop_front();
        }

        self.recompute_estimate();
    }

    fn recompute_estimate(&mut self) {
        let hop_ms = onset::hop_duration_ms(self.sample_rate);
        let env = self.envelope.make_contiguous();

        let Some(t) =
            tempo::estimate_tempo(env, hop_ms, tempo::DEFAULT_BPM_MIN, tempo::DEFAULT_BPM_MAX)
        else {
            self.current = BeatEstimate::default();
            self.last_tempo = None;
            self.last_phase = None;
            return;
        };
        self.last_tempo = Some(t);

        let now_ms = self.total_samples as f64 / self.sample_rate as f64 * 1000.0;
        let period = t.period_frames.round().max(1.0) as usize;
        let alternate_bpm = t.alternate.map(|a| a.bpm);
        let alternate_confidence = t.alternate.map(|a| a.confidence);
        // Gate on the tempo estimator's own confidence, not phase-tracking
        // success/failure below — a real periodicity can exist even when
        // phase locking hasn't converged yet, and conversely a spurious
        // comb-filter peak (noise, silence-adjacent hiss) shouldn't count
        // as rhythm just because *some* candidate period came out of it.
        let has_rhythm = t.confidence > MIN_RHYTHM_CONFIDENCE;

        let phase_result = phase::estimate_phase(env, period, hop_ms);
        self.last_phase = phase_result;

        // Downbeat: Phase 3.5's DSP capability expansion, reusing
        // `estimate_phase` a second time at the measure scale instead of a
        // new phase-tracking mechanism -- see phase.rs's
        // `beat_accent_sequence` doc comment and the build plan's Phase
        // 3.5 for the full rationale. Only attempted when `measure` exists
        // (a real grouping was found) and represents a genuine multi-beat
        // group (>=2) -- a `measure` at 1x the tactus period can't happen
        // by construction (`find_metrical_level` excludes lag==chosen.0),
        // but the `.max(1)` below guards the division regardless.
        let is_downbeat = phase_result.and_then(|p| {
            t.measure.and_then(|m| {
                let beats_per_measure = (m.period_frames / t.period_frames).round().max(1.0) as usize;
                if beats_per_measure < 2 {
                    return None;
                }
                let accents = phase::beat_accent_sequence(env, period, p.beat_phase);
                let beat_duration_ms = period as f64 * hop_ms;
                phase::estimate_phase(&accents, beats_per_measure, beat_duration_ms)
                    .map(|measure_phase| measure_phase.beat_phase < 0.5 / beats_per_measure as f32)
            })
        });

        self.current = match phase_result {
            Some(p) => BeatEstimate {
                has_rhythm,
                bpm: t.bpm,
                beat_phase: p.beat_phase,
                confidence: t.confidence,
                next_beat_time_ms: now_ms + p.next_beat_offset_ms,
                is_downbeat,
                alternate_bpm,
                alternate_confidence,
            },
            None => BeatEstimate {
                has_rhythm,
                bpm: t.bpm,
                beat_phase: 0.0,
                confidence: 0.0,
                next_beat_time_ms: now_ms,
                is_downbeat: None,
                alternate_bpm,
                alternate_confidence,
            },
        };
    }

    pub fn current_estimate(&self) -> BeatEstimate {
        self.current
    }

    /// Diagnostic-only: the raw `OfflineResult` (carries `prominence`,
    /// unlike `BeatEstimate`) behind `analyze_offline` below, exposed
    /// separately so `beat-dsp-cli`'s offline mode can surface it without
    /// changing `analyze_offline`'s signature -- that method is the actual
    /// shipped WASM API (see `beat-wasm/src/lib.rs`), not something to
    /// touch for a diagnostic need.
    pub fn analyze_offline_raw(&self, buffer: &[f32], sample_rate: u32) -> Option<offline::OfflineResult> {
        let hop_ms = onset::hop_duration_ms(sample_rate);
        let mut detector = onset::OnsetDetector::new();
        let envelope = detector.push_samples(buffer);
        offline::analyze(&envelope, hop_ms)
    }

    /// Full-buffer analysis via the from-scratch Ellis (2007)
    /// dynamic-programming beat tracker — a separate, non-causal pass that
    /// generally out-performs the streaming path when the whole track is
    /// available up front (spec §3).
    pub fn analyze_offline(&self, buffer: &[f32], sample_rate: u32) -> Vec<BeatEstimate> {
        let hop_ms = onset::hop_duration_ms(sample_rate);
        let mut detector = onset::OnsetDetector::new();
        let envelope = detector.push_samples(buffer);

        let Some(result) = offline::analyze(&envelope, hop_ms) else {
            return Vec::new();
        };
        let alternate_bpm = result.alternate.map(|a| a.bpm);
        let alternate_confidence = result.alternate.map(|a| a.confidence);
        // Same gate as the streaming path (see MIN_RHYTHM_CONFIDENCE) for
        // consistency, though only the streaming path's threshold has been
        // calibrated against real no-rhythm audio so far -- the offline DP
        // backtrace always produces *some* beat sequence once it runs at
        // all, the same "an estimate exists isn't evidence of rhythm"
        // pattern the streaming fix addressed, but its confidence scale
        // hasn't been separately validated the same way.
        let has_rhythm = result.confidence > MIN_RHYTHM_CONFIDENCE;

        // Downbeat: same reuse chain as the streaming path
        // (`recompute_estimate`), simpler here since `beat_frames` already
        // gives every beat's exact position -- nothing to reconstruct from
        // a `beat_phase`. `estimate_phase` only reports the offset of the
        // *last* entry in the accent sequence, so every other beat's
        // is-it-a-downbeat status is derived from that one offset via
        // modular arithmetic (each earlier beat is exactly one position
        // further back in the beats-since-last-downbeat cycle).
        let downbeat_info: Option<(usize, usize)> = result.measure.and_then(|m| {
            let tactus_period_frames = (60_000.0 / (result.bpm as f64 * hop_ms)).round().max(1.0);
            let beats_per_measure = (m.period_frames as f64 / tactus_period_frames).round().max(1.0) as usize;
            if beats_per_measure < 2 || result.beat_frames.len() < beats_per_measure {
                return None;
            }
            let accents = phase::beat_accent_sequence_at_frames(&envelope, &result.beat_frames);
            let beat_duration_ms = tactus_period_frames * hop_ms;
            let phase_est = phase::estimate_phase(&accents, beats_per_measure, beat_duration_ms)?;
            let offset = (phase_est.beat_phase * beats_per_measure as f32).round() as usize % beats_per_measure;
            Some((offset, beats_per_measure))
        });
        let n_beats = result.beat_frames.len();

        result
            .beat_frames
            .into_iter()
            .enumerate()
            .map(|(i, frame_idx)| {
                let is_downbeat = downbeat_info
                    .map(|(offset, beats_per_measure)| (n_beats - 1 - i) % beats_per_measure == offset);
                BeatEstimate {
                    has_rhythm,
                    bpm: result.bpm,
                    beat_phase: 0.0,
                    confidence: result.confidence,
                    next_beat_time_ms: frame_idx as f64 * hop_ms,
                    is_downbeat,
                    alternate_bpm,
                    alternate_confidence,
                }
            })
            .collect()
    }
}

impl Default for DeterministicTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_click_track(bpm: f32, n_beats: usize, sample_rate: u32) -> Vec<f32> {
        let period_samples = (60.0 / bpm * sample_rate as f32) as usize;
        let mut samples = vec![0.0f32; period_samples * n_beats];
        // A decaying burst on the order of a real percussive transient
        // (~6ms) — long enough to naturally spread across the onset
        // detector's frame boundaries the way real audio does, unlike a
        // handful-of-samples delta, which is shorter than one hop and
        // pathologically exercises frame-quantization rounding.
        let burst_len = (sample_rate as f32 * 0.006) as usize;
        for beat in 0..n_beats {
            let start = beat * period_samples;
            for i in 0..burst_len.min(samples.len() - start) {
                samples[start + i] = 1.0 - (i as f32 / burst_len as f32);
            }
        }
        samples
    }

    #[test]
    fn fresh_tracker_reports_no_rhythm() {
        let tracker = DeterministicTracker::new();
        let est = tracker.current_estimate();
        assert!(
            !est.has_rhythm,
            "fresh tracker should report no rhythm detected yet"
        );
        assert_eq!(est.confidence, 0.0);
    }

    #[test]
    fn silence_reports_no_rhythm_not_low_confidence() {
        let mut tracker = DeterministicTracker::new();
        // Enough silent samples to complete several onset-detector hops.
        tracker.process_chunk(&[0.0; 44_100], 44_100);
        let est = tracker.current_estimate();
        assert!(
            !est.has_rhythm,
            "silence should report no rhythm, not a low-confidence guess"
        );
    }

    #[test]
    fn reset_returns_to_default() {
        let mut tracker = DeterministicTracker::new();
        tracker.process_chunk(&[0.0; 512], 44_100);
        tracker.reset();
        assert_eq!(tracker.current_estimate(), BeatEstimate::default());
    }

    #[test]
    fn streaming_converges_on_known_bpm_fed_in_small_chunks() {
        let sample_rate = 44_100;
        let target_bpm = 128.0;
        let track = synthetic_click_track(target_bpm, 24, sample_rate);

        let mut tracker = DeterministicTracker::new();
        for chunk in track.chunks(256) {
            tracker.process_chunk(chunk, sample_rate);
        }

        let est = tracker.current_estimate();
        assert!(est.has_rhythm, "expected rhythm to be detected");
        assert!(
            (est.bpm - target_bpm).abs() < 3.0,
            "expected ~{target_bpm} BPM, got {}",
            est.bpm
        );
        assert!(
            est.confidence > 0.5,
            "expected reasonable confidence, got {}",
            est.confidence
        );
    }

    #[test]
    fn offline_mode_recovers_known_bpm_and_beat_count() {
        let sample_rate = 44_100;
        let target_bpm = 100.0;
        let track = synthetic_click_track(target_bpm, 20, sample_rate);

        let tracker = DeterministicTracker::new();
        let estimates = tracker.analyze_offline(&track, sample_rate);

        assert!(!estimates.is_empty());
        assert!(
            estimates.iter().all(|e| e.has_rhythm),
            "every detected beat should report has_rhythm"
        );
        let avg_bpm: f32 = estimates.iter().map(|e| e.bpm).sum::<f32>() / estimates.len() as f32;
        assert!(
            (avg_bpm - target_bpm).abs() < 3.0,
            "expected ~{target_bpm} BPM, got {avg_bpm}"
        );
        // Roughly one estimate per beat (allow some slack for windowing).
        assert!(
            estimates.len() >= 15 && estimates.len() <= 24,
            "got {} estimates",
            estimates.len()
        );
    }

    /// Same construction as `synthetic_click_track`, but every `accent_every`th
    /// beat's burst is at full strength and the others are scaled down --
    /// a real 4/4-style downbeat accent pattern, not a uniform click train.
    fn accented_click_track(bpm: f32, n_beats: usize, sample_rate: u32, accent_every: usize, weak_scale: f32) -> Vec<f32> {
        let mut samples = synthetic_click_track(bpm, n_beats, sample_rate);
        let period_samples = (60.0 / bpm * sample_rate as f32) as usize;
        let burst_len = (sample_rate as f32 * 0.006) as usize;
        for beat in 0..n_beats {
            if beat % accent_every != 0 {
                let start = beat * period_samples;
                for i in 0..burst_len.min(samples.len() - start) {
                    samples[start + i] *= weak_scale;
                }
            }
        }
        samples
    }

    #[test]
    fn streaming_detects_downbeat_on_accented_track() {
        // End-to-end through the real streaming pipeline (chunk-by-chunk
        // through the actual onset detector, not synthetic envelope arrays
        // handed directly to tempo.rs/phase.rs) -- confirms Phase 3.5's
        // reuse chain (tempo.rs's `measure` -> phase.rs's
        // `beat_accent_sequence` -> `estimate_phase` again) survives
        // through the onset detector's own smoothing/windowing, not just
        // in isolated unit tests against idealized envelopes.
        //
        // Checks *every* processed chunk rather than only the final state:
        // a first attempt asserted on the state right after the stream
        // ended, which failed not because of a real bug but because
        // `synthetic_click_track`'s truncating `as usize` cast on
        // `period_samples` means beat positions drift slightly relative to
        // the onset detector's hop grid over many beats, so the stream
        // doesn't necessarily end exactly *at* a beat (confirmed live:
        // `beat_phase` was 0.988 at the end, i.e. almost a full period
        // *past* the last detected beat, not near a real one at all).
        // Sampling every chunk and only checking the instants where
        // `beat_phase` is genuinely near zero avoids depending on exactly
        // where the stream happens to stop.
        //
        // accent_every=2 (not 4): a second attempt used 4, which hit the
        // *exact* same fixture-construction pitfall already found and
        // fixed for `tempo.rs`'s own `accented_grouping_exposes_a_measure`
        // test -- a "1 strong + 3 same-weak" pattern has real, genuine 2x
        // self-similarity too (confirmed live via the tracker's own
        // diagnostic accessors: `measure` locked onto period=2 beats, not
        // the intended 4, so the mechanism was answering the question
        // correctly for the structure that's actually there, just not the
        // structure this test assumed it had built). A clean 2-beat
        // accent pattern is unambiguous the same way `backbeat_envelope`
        // is in tempo.rs.
        let sample_rate = 44_100;
        let target_bpm = 120.0;
        let accent_every = 2;
        let period_samples = (60.0 / target_bpm * sample_rate as f32) as usize;
        let track = accented_click_track(target_bpm, 32, sample_rate, accent_every, 0.6);

        let mut tracker = DeterministicTracker::new();
        let mut checked_a_downbeat = false;
        let mut checked_a_non_downbeat = false;
        for (chunk_idx, chunk) in track.chunks(256).enumerate() {
            tracker.process_chunk(chunk, sample_rate);
            let est = tracker.current_estimate();
            if !est.has_rhythm || est.is_downbeat.is_none() {
                continue;
            }
            if est.beat_phase > 0.1 {
                continue; // not actually at a beat right now, skip
            }
            // Reconstruct which beat "now" (this chunk's end) actually is,
            // independent of the tracker's own internal state, to check
            // `is_downbeat` against ground truth.
            let now_samples = (chunk_idx + 1) * 256;
            let nearest_beat = (now_samples as f32 / period_samples as f32).round() as usize;
            if nearest_beat < 2 * accent_every {
                // Same class of cold-start unreliability already documented
                // and accepted for the tactus estimate itself (the bpm~304
                // boundary investigation) -- with less than two full
                // measure cycles of history, `measure`/`is_downbeat` can't
                // be expected to have converged yet. Mitigated at a higher
                // layer (debouncing), not by requiring perfection here.
                continue;
            }
            let expected_downbeat = nearest_beat % accent_every == 0;
            assert_eq!(
                est.is_downbeat,
                Some(expected_downbeat),
                "at beat {nearest_beat} (chunk {chunk_idx}, beat_phase={}), expected is_downbeat={expected_downbeat}",
                est.beat_phase
            );
            if expected_downbeat {
                checked_a_downbeat = true;
            } else {
                checked_a_non_downbeat = true;
            }
        }
        assert!(checked_a_downbeat, "never observed a near-zero beat_phase at a real downbeat");
        assert!(checked_a_non_downbeat, "never observed a near-zero beat_phase at a real non-downbeat");
    }

    #[test]
    fn offline_mode_detects_downbeat_on_accented_track() {
        // Offline counterpart to `streaming_detects_downbeat_on_accented_track`
        // -- same reuse chain, but sourcing beat positions directly from
        // `beat_frames` (via `beat_accent_sequence_at_frames`) instead of
        // reconstructing them from a causal phase estimate. Every detected
        // beat gets an `is_downbeat` classification in one pass, unlike
        // streaming's one-estimate-at-a-time.
        let sample_rate = 44_100;
        let target_bpm = 120.0;
        let accent_every = 2;
        let period_samples = (60.0 / target_bpm * sample_rate as f32) as usize;
        let track = accented_click_track(target_bpm, 32, sample_rate, accent_every, 0.6);

        let tracker = DeterministicTracker::new();
        let estimates = tracker.analyze_offline(&track, sample_rate);

        assert!(!estimates.is_empty());
        let with_downbeat: Vec<_> = estimates.iter().filter(|e| e.is_downbeat.is_some()).collect();
        assert!(
            !with_downbeat.is_empty(),
            "expected at least some beats to get a downbeat classification"
        );

        // Cross-check classified beats against the track's own known accent
        // structure (beat index nearest each estimate's timestamp), same
        // as the streaming test's independent-reconstruction approach --
        // skip the first two measure cycles for the same documented cold-
        // start reason.
        let mut checked_a_downbeat = false;
        let mut checked_a_non_downbeat = false;
        for e in &estimates {
            let nearest_beat = (e.next_beat_time_ms / 1000.0 * sample_rate as f64 / period_samples as f64).round() as usize;
            if nearest_beat < 2 * accent_every {
                continue;
            }
            let expected_downbeat = nearest_beat % accent_every == 0;
            assert_eq!(
                e.is_downbeat,
                Some(expected_downbeat),
                "beat {nearest_beat}: expected is_downbeat={expected_downbeat}, got {e:?}"
            );
            if expected_downbeat {
                checked_a_downbeat = true;
            } else {
                checked_a_non_downbeat = true;
            }
        }
        assert!(checked_a_downbeat, "never checked a real downbeat");
        assert!(checked_a_non_downbeat, "never checked a real non-downbeat");
    }
}
