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
    /// How strongly the recency-weighted backward sum favors `beat_phase`
    /// over every other phase candidate -- previously computed and then
    /// discarded (only the winning phase survived). Diagnostic-only so far
    /// (see production-readiness item 2's confidence-signal exploration):
    /// the intended magnitude half of a complex-vector phase/confidence
    /// representation, unvalidated until the Python prototype confirms it.
    pub best_score: f32,
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
        best_score,
    })
}

fn last_index_at_phase(n: usize, phase: usize, period: usize) -> usize {
    phase + ((n - 1 - phase) / period) * period
}

/// A detected beat's true onset transient rarely lands on the exact frame
/// `beat_phase` predicts -- real onsets spread across a few frames, and
/// streaming/hop-quantization effects can shift the predicted position by
/// a frame or two even for a clean synthetic click (confirmed live: an
/// exact-frame version of `beat_accent_sequence` misclassified downbeats
/// in `lib.rs`'s real-pipeline integration test, not just in noisy real
/// audio). Standard practice for accent-strength-at-a-beat is a local max
/// within a small tolerance window around the predicted position, not a
/// single exact sample -- so this takes one too.
const ACCENT_SAMPLE_WINDOW: usize = 2;

/// Builds a beat-indexed accent sequence from the raw onset envelope --
/// one value per beat (the local max of the envelope within
/// `ACCENT_SAMPLE_WINDOW` frames of the predicted beat position, not a
/// single exact-frame sample -- see `ACCENT_SAMPLE_WINDOW`'s doc comment),
/// oldest first. Phase 3.5's downbeat detection: feeding this straight
/// back into `estimate_phase` (with `beats_per_measure` standing in for
/// `period_frames`, and a per-beat duration standing in for `hop_ms`)
/// answers "which beat position is accented" the same way `estimate_phase`
/// already answers "which frame position is accented" -- same algorithm,
/// applied one metrical level up, not a new phase-tracking mechanism.
pub fn beat_accent_sequence(envelope: &[f32], period_frames: usize, beat_phase: f32) -> Vec<f32> {
    let n = envelope.len();
    if period_frames == 0 || n == 0 {
        return Vec::new();
    }
    let frames_since_beat = (beat_phase * period_frames as f32).round() as usize;
    let Some(last_beat_idx) = (n - 1).checked_sub(frames_since_beat) else {
        return Vec::new();
    };

    let mut idx = last_beat_idx;
    let mut frames = Vec::new();
    loop {
        frames.push(idx);
        if idx < period_frames {
            break;
        }
        idx -= period_frames;
    }
    frames.reverse(); // oldest first, matching estimate_phase's own convention
    beat_accent_sequence_at_frames(envelope, &frames)
}

/// Same accent-sampling logic as `beat_accent_sequence` (local max within
/// `ACCENT_SAMPLE_WINDOW` frames of each position, not a single exact
/// sample), but for already-known beat frame positions -- the offline
/// path's `OfflineResult::beat_frames` gives these explicitly, so there's
/// nothing to reconstruct from a `beat_phase` the way the streaming path
/// needs to.
pub fn beat_accent_sequence_at_frames(envelope: &[f32], beat_frames: &[usize]) -> Vec<f32> {
    let n = envelope.len();
    if n == 0 {
        return Vec::new();
    }
    beat_frames
        .iter()
        .map(|&center| {
            let lo = center.saturating_sub(ACCENT_SAMPLE_WINDOW);
            let hi = (center + ACCENT_SAMPLE_WINDOW).min(n - 1);
            envelope[lo..=hi].iter().copied().fold(f32::MIN, f32::max)
        })
        .collect()
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

    #[test]
    fn beat_accent_sequence_resamples_envelope_at_beat_positions() {
        let period = 10;
        // Distinct, ramping values at each beat position so extraction
        // order/alignment is unambiguous to check, unlike a uniform click
        // train where every beat looks the same. Truncated to end exactly
        // on the last beat (index 40, so len=41) so "now" truly coincides
        // with it -- an untruncated period*5=50-length buffer would leave
        // "now" nine frames *past* the last beat instead, at beat_phase
        // 0.9, not 0.0 (caught by the first attempt at this test panicking
        // with an all-zero result: index 49 is empty padding, not a beat).
        let mut env = vec![0.0f32; period * 5];
        for (beat, v) in [0.1f32, 0.2, 0.3, 0.4, 0.5].into_iter().enumerate() {
            env[beat * period] = v;
        }
        env.truncate(period * 4 + 1);

        let accents = beat_accent_sequence(&env, period, 0.0);
        assert_eq!(accents, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[test]
    fn downbeat_detection_via_reused_estimate_phase() {
        // End-to-end demonstration of Phase 3.5's core reuse: a tactus
        // phase estimate feeds `beat_accent_sequence`, whose output feeds
        // `estimate_phase` again -- same function, one metrical level up,
        // no new phase-tracking algorithm. 4 beats per measure, downbeat
        // (beat 0 of each group) accented at 1.0, others at 0.3.
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        let period = 40;
        let beats_per_measure = 4;
        let n_beats = 16;

        let mut env = vec![0.0f32; period * n_beats];
        for beat in 0..n_beats {
            env[beat * period] = if beat % beats_per_measure == 0 { 1.0 } else { 0.3 };
        }
        // Truncate to end exactly on beat 12 (the 4th downbeat, i.e. a
        // beat that genuinely *is* a downbeat) so the expected answer is
        // unambiguous.
        env.truncate(period * 12 + 1);

        let tactus_phase = estimate_phase(&env, period, hop_ms).unwrap();
        assert!(tactus_phase.beat_phase < 0.05, "expected phase ~0 right at a beat, got {}", tactus_phase.beat_phase);

        let accents = beat_accent_sequence(&env, period, tactus_phase.beat_phase);
        assert_eq!(accents.len(), 13); // beats 0..=12

        let beat_duration_ms = period as f64 * hop_ms;
        let measure_phase = estimate_phase(&accents, beats_per_measure, beat_duration_ms).unwrap();
        assert!(
            measure_phase.beat_phase < 0.5 / beats_per_measure as f32,
            "expected the most recent beat to be classified as the downbeat (phase ~0), got {}",
            measure_phase.beat_phase
        );
    }
}
