//! Deterministic (non-ML) beat/tempo tracking: onset detection, autocorrelation/
//! comb-filter tempo estimation, and causal phase tracking. Phase 1 of the build
//! plan in docs/ — everything here is reimplemented from published DSP
//! algorithms, so it carries no training-data licensing exposure.

/// Mirrors the `BeatEstimate` shape from `docs/WASM_BEAT_MODULE_SPEC.md` §4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeatEstimate {
    pub bpm: f32,
    pub beat_phase: f32,
    pub confidence: f32,
    pub next_beat_time_ms: f64,
    pub is_downbeat: Option<bool>,
}

impl Default for BeatEstimate {
    /// Confidence 0.0 signals "no reading yet" rather than a guess — honest
    /// uncertainty is the whole point (spec §6), including before any audio
    /// has arrived.
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

/// Streaming deterministic tracker. Onset detection, autocorrelation tempo
/// estimation, and the causal PLP-style phase tracker land here in Phase 1;
/// for now this is a structural stub so the WASM/TS integration path can be
/// built and tested before any tracking logic exists.
pub struct DeterministicTracker {
    current: BeatEstimate,
}

impl DeterministicTracker {
    pub fn new() -> Self {
        Self {
            current: BeatEstimate::default(),
        }
    }

    pub fn reset(&mut self) {
        self.current = BeatEstimate::default();
    }

    /// TODO(Phase 1): spectral-flux onset detection feeding the
    /// autocorrelation/comb-filter tempo estimator and PLP phase tracker.
    pub fn process_chunk(&mut self, _samples: &[f32], _sample_rate: u32) {}

    pub fn current_estimate(&self) -> BeatEstimate {
        self.current
    }

    /// TODO(Phase 1): full-buffer analysis via the from-scratch Ellis (2007)
    /// dynamic-programming beat tracker.
    pub fn analyze_offline(&self, _buffer: &[f32], _sample_rate: u32) -> Vec<BeatEstimate> {
        Vec::new()
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
}
