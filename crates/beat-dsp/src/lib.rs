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

/// Mirrors the `BeatEstimate` shape from `docs/WASM_BEAT_MODULE_SPEC.md` §4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatEstimate {
    pub bpm: f32,
    pub beat_phase: f32,
    pub confidence: f32,
    /// Milliseconds since the tracker was constructed or last reset — not
    /// wall-clock time. Callers align this to their own audio clock using
    /// whatever offset they already track (e.g. `AudioContext.currentTime`
    /// at the moment `reset()` was called).
    pub next_beat_time_ms: f64,
    pub is_downbeat: Option<bool>,
}

impl Default for BeatEstimate {
    /// Confidence 0.0 signals "no reading yet" rather than a guess — honest
    /// uncertainty is the whole point (spec §6), including before any audio
    /// has arrived or while there isn't enough history to trust a reading.
    fn default() -> Self {
        Self {
            bpm: 0.0,
            beat_phase: 0.0,
            confidence: 0.0,
            next_beat_time_ms: 0.0,
            is_downbeat: None,
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
        }
    }

    pub fn reset(&mut self) {
        self.onset.reset();
        self.envelope.clear();
        self.total_samples = 0;
        self.current = BeatEstimate::default();
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
            return;
        };

        let now_ms = self.total_samples as f64 / self.sample_rate as f64 * 1000.0;
        let period = t.period_frames.round().max(1.0) as usize;

        self.current = match phase::estimate_phase(env, period, hop_ms) {
            Some(p) => BeatEstimate {
                bpm: t.bpm,
                beat_phase: p.beat_phase,
                confidence: t.confidence,
                next_beat_time_ms: now_ms + p.next_beat_offset_ms,
                is_downbeat: None,
            },
            None => BeatEstimate {
                bpm: t.bpm,
                beat_phase: 0.0,
                confidence: 0.0,
                next_beat_time_ms: now_ms,
                is_downbeat: None,
            },
        };
    }

    pub fn current_estimate(&self) -> BeatEstimate {
        self.current
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

        result
            .beat_frames
            .into_iter()
            .map(|frame_idx| BeatEstimate {
                bpm: result.bpm,
                beat_phase: 0.0,
                confidence: result.confidence,
                next_beat_time_ms: frame_idx as f64 * hop_ms,
                is_downbeat: None,
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
    fn fresh_tracker_reports_zero_confidence() {
        let tracker = DeterministicTracker::new();
        assert_eq!(tracker.current_estimate().confidence, 0.0);
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
}
