//! Causal beat-phase tracking: given a period estimate from `tempo.rs`,
//! finds which phase offset within that period the onset envelope actually
//! aligns to, using only present-and-past data (no lookahead, so this adds
//! zero latency beyond the onset detector's own frame hop) — the same
//! "only look backward" constraint that makes the zero-latency PLP approach
//! (TISMIR 2024) causal, though this is a simplified comb-filter-alignment
//! version rather than a literal port of that paper's windowed-kernel math.
//!
//! Recency-weighted so the estimate adapts to drift instead of being
//! dominated by the whole history equally.

const RECENCY_DECAY: f32 = 0.9;
const MIN_WEIGHT: f32 = 1e-4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseEstimate {
    /// Progress toward the next beat, in [0, 1).
    pub beat_phase: f32,
    /// Milliseconds from "now" (the most recent processed frame) until the
    /// next predicted beat.
    pub next_beat_offset_ms: f64,
}

/// `envelope` is the same onset-strength history `tempo::estimate_tempo`
/// was run on; `period_frames` is its `TempoEstimate::period_frames`
/// (rounded to whole hops).
pub fn estimate_phase(
    envelope: &[f32],
    period_frames: usize,
    hop_ms: f64,
) -> Option<PhaseEstimate> {
    let n = envelope.len();
    if period_frames == 0 || n == 0 || period_frames > n {
        return None;
    }

    let mut best_phase = 0usize;
    let mut best_score = f32::MIN;

    for phase in 0..period_frames {
        let score = weighted_backward_sum(envelope, phase, period_frames);
        if score > best_score {
            best_score = score;
            best_phase = phase;
        }
    }

    let last_beat_idx = last_index_at_phase(n, best_phase, period_frames);
    let frames_since_beat = (n - 1) - last_beat_idx;
    let frames_until_next = period_frames - frames_since_beat;

    Some(PhaseEstimate {
        beat_phase: frames_since_beat as f32 / period_frames as f32,
        next_beat_offset_ms: frames_until_next as f64 * hop_ms,
    })
}

fn last_index_at_phase(n: usize, phase: usize, period: usize) -> usize {
    phase + ((n - 1 - phase) / period) * period
}

fn weighted_backward_sum(envelope: &[f32], phase: usize, period: usize) -> f32 {
    let n = envelope.len();
    let mut idx = last_index_at_phase(n, phase, period);
    let mut weight = 1.0f32;
    let mut score = 0.0f32;
    loop {
        score += envelope[idx] * weight;
        weight *= RECENCY_DECAY;
        if weight < MIN_WEIGHT || idx < period {
            break;
        }
        idx -= period;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    fn click_envelope_with_offset(period: usize, offset: usize, n_beats: usize) -> Vec<f32> {
        let len = offset + period * n_beats;
        let mut env = vec![0.0f32; len];
        let mut idx = offset;
        while idx < len {
            env[idx] = 1.0;
            idx += period;
        }
        env
    }

    #[test]
    fn recovers_known_phase_offset() {
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        let period = 40;
        let offset = 13;
        let env = click_envelope_with_offset(period, offset, 15);

        let est = estimate_phase(&env, period, hop_ms).unwrap();

        let last_click_idx = offset + ((env.len() - 1 - offset) / period) * period;
        let expected_frames_since = (env.len() - 1) - last_click_idx;
        let expected_phase = expected_frames_since as f32 / period as f32;

        assert!(
            (est.beat_phase - expected_phase).abs() < 0.05,
            "expected phase ~{expected_phase}, got {}",
            est.beat_phase
        );
    }

    #[test]
    fn phase_zero_right_after_a_beat() {
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        let period = 40;
        let n_beats = 10;
        // Truncate right at the last click so the envelope's final sample
        // (i.e. "now") coincides with a beat.
        let mut env = click_envelope_with_offset(period, 0, n_beats);
        env.truncate(period * (n_beats - 1) + 1);

        let est = estimate_phase(&env, period, hop_ms).unwrap();
        assert!(est.beat_phase < 0.05, "got {}", est.beat_phase);
        // Next beat is a full period away.
        assert!((est.next_beat_offset_ms - period as f64 * hop_ms).abs() < 1.0);
    }
}
