//! Offline/batch beat tracking: a from-scratch implementation of Ellis'
//! dynamic-programming beat tracker (D. Ellis, "Beat Tracking by Dynamic
//! Programming", ISMIR 2007) — a published, non-learned algorithm, so
//! reimplementing it here carries no training-data licensing exposure (see
//! docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md §4). Given full-buffer
//! context, this generally tracks tempo drift better than the causal
//! streaming path in `phase.rs`, which is the whole point of offering a
//! separate offline mode (spec §3).

use crate::tempo::{self, DEFAULT_BPM_MAX, DEFAULT_BPM_MIN};

/// Penalty weight on `log2(interval / target_period)`. Higher = more
/// rigidly locked to a constant tempo; lower = more tolerant of rubato.
pub const DEFAULT_ALPHA: f32 = 400.0;

/// Dynamic-programming beat sequence over an onset-strength envelope,
/// following Ellis (2007): each frame's cumulative score is its own onset
/// strength plus the best-scoring predecessor beat, where predecessors are
/// penalized quadratically (in log2 space) for straying from
/// `target_period_frames`. Search is bounded to predecessors within
/// [0.5x, 2x] the target period — well outside that range the penalty
/// dominates regardless, so there's nothing to gain from searching further.
pub fn track_beats_dp(envelope: &[f32], target_period_frames: f32, alpha: f32) -> Vec<usize> {
    let n = envelope.len();
    if n == 0 || target_period_frames < 1.0 {
        return Vec::new();
    }
    let tau = target_period_frames as f64;

    let mut cum = vec![0.0f32; n];
    let mut back = vec![usize::MAX; n];

    for t in 0..n {
        let mut best_prev = usize::MAX;
        let mut best_score = f32::MIN;

        if t >= 1 {
            let lo = ((t as f64) - tau * 2.0).ceil().max(0.0) as usize;
            let hi_f = (t as f64) - tau * 0.5;
            if hi_f >= 0.0 {
                let hi = (hi_f.floor() as usize).min(t - 1);
                if lo <= hi {
                    // `tp` is used both as the `cum` index and directly in
                    // the interval arithmetic below, so `.enumerate()`
                    // wouldn't simplify anything here.
                    #[allow(clippy::needless_range_loop)]
                    for tp in lo..=hi {
                        let interval = (t - tp) as f64;
                        let penalty = -(alpha as f64) * (interval / tau).log2().powi(2);
                        let score = cum[tp] as f64 + penalty;
                        if score as f32 > best_score {
                            best_score = score as f32;
                            best_prev = tp;
                        }
                    }
                }
            }
        }

        if best_prev == usize::MAX {
            cum[t] = envelope[t];
        } else {
            cum[t] = envelope[t] + best_score;
            back[t] = best_prev;
        }
    }

    let end = (0..n)
        .max_by(|&a, &b| cum[a].partial_cmp(&cum[b]).unwrap())
        .unwrap_or(0);

    let mut beats = Vec::new();
    let mut cur = end;
    loop {
        beats.push(cur);
        if back[cur] == usize::MAX {
            break;
        }
        cur = back[cur];
    }
    beats.reverse();
    beats
}

pub struct OfflineResult {
    pub bpm: f32,
    pub confidence: f32,
    /// Onset-envelope frame indices of each detected beat, chronological.
    pub beat_frames: Vec<usize>,
}

/// Full offline pipeline: estimate global tempo over the whole envelope
/// (non-causal — offline mode gets to use the full buffer), then run the DP
/// tracker against that period.
pub fn analyze(envelope: &[f32], hop_ms: f64) -> Option<OfflineResult> {
    let tempo_est = tempo::estimate_tempo(envelope, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX)?;
    let beat_frames = track_beats_dp(envelope, tempo_est.period_frames, DEFAULT_ALPHA);
    Some(OfflineResult {
        bpm: tempo_est.bpm,
        confidence: tempo_est.confidence,
        beat_frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn click_envelope(period: usize, n_beats: usize) -> Vec<f32> {
        let mut env = vec![0.0f32; period * n_beats];
        for beat in 0..n_beats {
            env[beat * period] = 1.0;
        }
        env
    }

    #[test]
    fn dp_recovers_exact_click_positions() {
        let period = 40;
        let env = click_envelope(period, 20);
        let beats = track_beats_dp(&env, period as f32, DEFAULT_ALPHA);

        let expected: Vec<usize> = (0..20).map(|b| b * period).collect();
        assert_eq!(beats, expected);
    }

    #[test]
    fn full_pipeline_matches_known_tempo() {
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        let period = tempo::period_frames_for_bpm(128.0, hop_ms).round() as usize;
        let env = click_envelope(period, 24);

        let result = analyze(&env, hop_ms).unwrap();
        assert!((result.bpm - 128.0).abs() < 2.0, "got {}", result.bpm);
        assert_eq!(result.beat_frames.len(), 24);
        for (i, &frame) in result.beat_frames.iter().enumerate() {
            assert_eq!(frame, i * period);
        }
    }

    #[test]
    fn tolerates_a_missed_click_without_derailing() {
        // A beat tracker should interpolate through a single dropped onset
        // rather than losing the tempo entirely.
        let period = 40;
        let mut env = click_envelope(period, 20);
        env[8 * period] = 0.0; // drop one click

        let beats = track_beats_dp(&env, period as f32, DEFAULT_ALPHA);
        // Expect ~20 beats still, roughly evenly spaced at `period`.
        assert!(
            beats.len() >= 18 && beats.len() <= 20,
            "got {} beats",
            beats.len()
        );
    }
}
