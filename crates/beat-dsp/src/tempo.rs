//! Comb-filter tempo estimation over an onset-strength envelope. Periodicity
//! salience doubles as the confidence signal (spec §6): a sharp,
//! harmonically-consistent peak means high confidence, a flat/noisy
//! autocorrelation means an honest low confidence rather than a guess.

pub const DEFAULT_BPM_MIN: f32 = 40.0;
pub const DEFAULT_BPM_MAX: f32 = 300.0;
const MAX_HARMONICS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoEstimate {
    pub period_frames: f32,
    pub bpm: f32,
    /// Normalized comb-filter peak salience, roughly in [0, 1].
    pub confidence: f32,
    /// The other tempo-octave candidate (2x or 0.5x `bpm`), populated only
    /// when it scores close enough to `confidence` to be a genuine
    /// ambiguity rather than a clearly-inferior runner-up. Real, common
    /// music-tempo phenomenon (is a track 76 or 152 BPM?), not an
    /// estimation failure — exposed rather than silently resolved so a
    /// consumer can pick whichever octave fits its own context.
    pub alternate: Option<AlternateTempo>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlternateTempo {
    pub period_frames: f32,
    pub bpm: f32,
    pub confidence: f32,
}

/// Estimates the dominant periodicity of `envelope` (one onset-strength
/// value per hop) within [bpm_min, bpm_max]. Returns `None` if there isn't
/// enough history yet to cover even the slowest candidate period, or if the
/// envelope has ~no energy (silence).
pub fn estimate_tempo(
    envelope: &[f32],
    hop_ms: f64,
    bpm_min: f32,
    bpm_max: f32,
) -> Option<TempoEstimate> {
    let n = envelope.len();
    if n < 2 {
        return None;
    }

    let mean = envelope.iter().sum::<f32>() / n as f32;
    let centered: Vec<f32> = envelope.iter().map(|v| v - mean).collect();

    let ac0 = autocorr(&centered, 0);
    if ac0 <= 1e-8 {
        return None; // silence / no variation to lock onto
    }

    let min_lag = period_frames_for_bpm(bpm_max, hop_ms).max(1.0).round() as usize;
    let max_lag = (period_frames_for_bpm(bpm_min, hop_ms).round() as usize).min(n - 1);
    if max_lag < min_lag {
        return None; // not enough history to see even one slow-tempo period
    }

    let scores: Vec<(usize, f32)> = (min_lag..=max_lag)
        .map(|lag| {
            let max_h = (max_lag / lag).clamp(1, MAX_HARMONICS);
            let mut weighted_sum = 0.0f32;
            let mut weight_total = 0.0f32;
            for h in 1..=max_h {
                // The true period essentially never lands on an exact whole
                // number of hops, so the raw autocorrelation value at the
                // nearest integer lag is a slightly-too-low sample of a
                // peak that actually sits between bins — and *how much*
                // too low varies with how close that particular multiple
                // happens to round, which biases raw bin comparison toward
                // whichever multiple got luckier with rounding (verified:
                // a 100 BPM click track's true period of 103.36 frames
                // scored lower at its own bin than 2x's 206.72 rounding to
                // 207, because 207 happened to land closer to its true
                // continuous peak than 103 did to its own). Parabolic
                // interpolation across the three bins around each peak
                // corrects for this directly instead of comparing raw,
                // differently-rounded bin heights.
                let r = interpolated_peak(&centered, h * lag) / ac0;
                let w = 1.0 / h as f32;
                weighted_sum += w * r;
                weight_total += w;
            }
            (lag, weighted_sum / weight_total)
        })
        .collect();

    let (raw_best_lag, raw_best_score) = *scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    // Classic tempo-octave ambiguity: a period P also scores highly at
    // every integer multiple of P, since those are harmonics of P too. On
    // a hop-quantized time grid, *which* multiple raw-scores highest can
    // come down to which one's true (non-integer) period happened to round
    // more cleanly to a whole number of hops — a real, unavoidable effect
    // of discretizing time, not just numerical noise, so it isn't fixable
    // by peak interpolation alone (confirmed during development: a 100 BPM
    // click track's true 103.36-frame period consistently scored lower
    // than its double, 206.72 -> 207, by a wide, non-tie margin). The
    // correction is the standard MIR approach: check whether a candidate's
    // sub-harmonics (period / 2, /3, /4) also score reasonably well, and if
    // so prefer the smallest one that does, on the prior that a shorter
    // period is the more likely true beat unless the evidence clearly rules
    // it out.
    const SUBHARMONIC_TOLERANCE: f32 = 0.85; // fraction of the raw best score
    let threshold = raw_best_score * SUBHARMONIC_TOLERANCE;
    let score_at = |lag: usize| -> Option<f32> {
        lag.checked_sub(min_lag)
            .and_then(|i| scores.get(i))
            .map(|&(_, s)| s)
    };

    let mut chosen = (raw_best_lag, raw_best_score);
    for divisor in [2usize, 3, 4] {
        let approx = raw_best_lag as f32 / divisor as f32;
        let window_lo = (approx.floor() as usize).saturating_sub(1);
        let window_hi = approx.ceil() as usize + 1;
        if let Some((cand_lag, cand_score)) = (window_lo..=window_hi)
            .filter_map(|lag| score_at(lag).map(|s| (lag, s)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            if cand_score >= threshold && cand_lag < chosen.0 {
                chosen = (cand_lag, cand_score);
            }
        }
    }

    // Octave-alternate: check both the double and half of the chosen period
    // (symmetric, since ambiguity can present with either octave winning
    // the primary slot) using the same scores already computed above — this
    // really is close to free, just reusing `score_at`, no new estimation
    // pass. Only exposed if it's a genuine near-tie with the chosen answer,
    // not merely "also plausible but clearly worse."
    const ALTERNATE_TOLERANCE: f32 = 0.85; // fraction of the chosen score
    let alt_threshold = chosen.1 * ALTERNATE_TOLERANCE;
    let alternate = [chosen.0 * 2, chosen.0 / 2]
        .into_iter()
        .filter(|&lag| lag != chosen.0)
        .filter_map(|lag| score_at(lag).map(|s| (lag, s)))
        .filter(|&(_, s)| s >= alt_threshold)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(lag, score)| AlternateTempo {
            period_frames: lag as f32,
            bpm: (60_000.0 / (lag as f64 * hop_ms)) as f32,
            confidence: score.clamp(0.0, 1.0),
        });

    Some(TempoEstimate {
        period_frames: chosen.0 as f32,
        bpm: (60_000.0 / (chosen.0 as f64 * hop_ms)) as f32,
        confidence: chosen.1.clamp(0.0, 1.0),
        alternate,
    })
}

pub fn period_frames_for_bpm(bpm: f32, hop_ms: f64) -> f32 {
    ((60_000.0 / bpm as f64) / hop_ms) as f32
}

/// Quadratic (parabolic) interpolation of the autocorrelation value at the
/// true continuous peak near integer lag `k`, using `k`'s two neighbors —
/// standard sub-bin peak correction (the same trick used for pitch/period
/// estimation generally), needed here so candidates aren't compared on raw
/// bin heights that differ purely by how luckily each one's true period
/// happened to round to an integer lag.
fn interpolated_peak(x: &[f32], k: usize) -> f32 {
    if k == 0 {
        return autocorr(x, 0);
    }
    let y_left = autocorr(x, k - 1);
    let y_mid = autocorr(x, k);
    let y_right = autocorr(x, k + 1);

    let denom = y_left - 2.0 * y_mid + y_right;
    if denom.abs() < 1e-9 {
        return y_mid;
    }
    let delta = (0.5 * (y_left - y_right) / denom).clamp(-0.5, 0.5);
    y_mid - 0.25 * (y_left - y_right) * delta
}

/// Mean product at lag `k`: `mean_i(x[i] * x[i+k])`.
fn autocorr(x: &[f32], k: usize) -> f32 {
    if k >= x.len() {
        return 0.0;
    }
    let n = x.len() - k;
    let sum: f32 = (0..n).map(|i| x[i] * x[i + k]).sum();
    sum / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic onset envelope: an impulse every `period_frames`, for
    /// `n_beats` beats, with light noise so it's not a degenerate all-zero
    /// signal between clicks.
    fn click_envelope(period_frames: usize, n_beats: usize) -> Vec<f32> {
        let mut env = vec![0.0f32; period_frames * n_beats];
        for beat in 0..n_beats {
            env[beat * period_frames] = 1.0;
        }
        env
    }

    #[test]
    fn recovers_known_period_from_clean_clicks() {
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        let true_period = period_frames_for_bpm(120.0, hop_ms).round() as usize;
        let env = click_envelope(true_period, 16);

        let est = estimate_tempo(&env, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).unwrap();
        assert!(
            (est.bpm - 120.0).abs() < 2.0,
            "expected ~120 BPM, got {}",
            est.bpm
        );
        assert!(
            est.confidence > 0.7,
            "expected high confidence, got {}",
            est.confidence
        );
    }

    #[test]
    fn recovers_a_range_of_tempi() {
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        for target_bpm in [60.0, 90.0, 140.0, 174.0, 200.0] {
            let period = period_frames_for_bpm(target_bpm, hop_ms).round() as usize;
            let env = click_envelope(period, 20);
            let est = estimate_tempo(&env, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).unwrap();
            assert!(
                (est.bpm - target_bpm).abs() / target_bpm < 0.03,
                "target {target_bpm}, got {}",
                est.bpm
            );
        }
    }

    #[test]
    fn white_noise_yields_low_confidence() {
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        // Deterministic pseudo-random envelope (no external RNG dependency).
        let mut state: u32 = 0x1234_5678;
        let env: Vec<f32> = (0..500)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 16) as f32 / 65535.0
            })
            .collect();

        let est = estimate_tempo(&env, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).unwrap();
        assert!(
            est.confidence < 0.4,
            "expected low confidence on noise, got {}",
            est.confidence
        );
    }

    #[test]
    fn silence_returns_none() {
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        let env = vec![0.0f32; 500];
        assert!(estimate_tempo(&env, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).is_none());
    }

    #[test]
    fn insufficient_history_returns_none() {
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        // Shorter than even the fastest candidate period (300 BPM) needs.
        let env = click_envelope(5, 2);
        assert!(estimate_tempo(&env, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).is_none());
    }

    /// A backbeat-style envelope: every other pulse at `weak_amplitude`
    /// instead of full strength — a real, common source of tempo-octave
    /// ambiguity in actual music (e.g. a snare landing on every beat but
    /// accented every second one). A perfectly uniform click train doesn't
    /// serve as an "unambiguous" baseline here: an idealized delta train is
    /// mathematically self-similar at every integer multiple of its period
    /// (autocorrelation ties at ~1.0 for both the fundamental and its
    /// octave), so it's actually the *most* ambiguous case, not the least —
    /// confirmed empirically during development, not assumed. Real
    /// unambiguity comes from the amplitude *pattern* itself favoring one
    /// period over its octave, which is what varying `weak_amplitude` does.
    fn backbeat_envelope(period: usize, n_beats: usize, weak_amplitude: f32) -> Vec<f32> {
        let mut env = vec![0.0f32; period * n_beats];
        for beat in 0..n_beats {
            env[beat * period] = if beat % 2 == 0 { 1.0 } else { weak_amplitude };
        }
        env
    }

    #[test]
    fn strong_amplitude_asymmetry_resolves_without_an_alternate() {
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        let period = period_frames_for_bpm(150.0, hop_ms).round() as usize;
        // Weak pulses at 40% amplitude: enough of a difference that the
        // octave with only the strong pulses is clearly the better
        // explanation, not just a near-tie.
        let env = backbeat_envelope(period, 24, 0.4);

        let est = estimate_tempo(&env, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).unwrap();
        assert!(
            est.alternate.is_none(),
            "expected no alternate once amplitude asymmetry clearly favors one octave, got {:?}",
            est.alternate
        );
    }

    #[test]
    fn genuinely_ambiguous_tempo_exposes_an_alternate() {
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        let period = period_frames_for_bpm(150.0, hop_ms).round() as usize;
        // Weak pulses at 90% amplitude: close enough to full strength that
        // both the fundamental and its octave remain plausible.
        let env = backbeat_envelope(period, 24, 0.9);

        let est = estimate_tempo(&env, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).unwrap();
        let alt = est
            .alternate
            .unwrap_or_else(|| panic!("expected an alternate hypothesis, got {est:?}"));

        let ratio = est.bpm / alt.bpm;
        assert!(
            (ratio - 2.0).abs() < 0.1 || (ratio - 0.5).abs() < 0.05,
            "expected the alternate to be the other octave, got ratio {ratio} ({} vs {})",
            est.bpm,
            alt.bpm
        );
        // `chosen` is picked by a smaller-period *preference* among
        // near-tied candidates, not a strict argmax, so the alternate can
        // legitimately score at or above the chosen answer's own
        // confidence — only its existence and rough magnitude matter here.
        assert!(
            alt.confidence > 0.5,
            "expected a real alternate score, got {}",
            alt.confidence
        );
    }
}
