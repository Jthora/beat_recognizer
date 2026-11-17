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
    /// The chosen candidate's score relative to the mean of every other
    /// candidate lag scored this call -- diagnostic-only so far (see
    /// production-readiness item 2's confidence-signal exploration in the
    /// build plan), not yet used by `has_rhythm`/`confidence` gating.
    pub prominence: f32,
    /// Normalized Shannon entropy (0-1) of the full candidate-lag score
    /// distribution -- diagnostic-only, see the computation site below for
    /// the full rationale. High = a real periodic signal (elevated score
    /// spread across harmonics/subharmonics of the true period); low = no
    /// dominant periodicity (mass concentrated on a few incidental bumps,
    /// the noise/no-rhythm case). Confirmed empirically, opposite of the
    /// naive "peakiness" intuition -- see the doc comment where it's
    /// computed.
    pub entropy: f32,
    /// The other tempo-octave candidate (2x or 0.5x `bpm`), populated only
    /// when it scores close enough to `confidence` to be a genuine
    /// ambiguity rather than a clearly-inferior runner-up. Real, common
    /// music-tempo phenomenon (is a track 76 or 152 BPM?), not an
    /// estimation failure — exposed rather than silently resolved so a
    /// consumer can pick whichever octave fits its own context.
    pub alternate: Option<AlternateTempo>,
    /// The finer metrical subdivision below the tactus (beat) -- typically
    /// a half, third, or quarter of the beat period. `None` when no
    /// candidate at those ratios clears a real-peak threshold, meaning the
    /// signal doesn't show clear sub-beat structure (common for sparse or
    /// ambiguous material) rather than a failure to look. See
    /// `find_metrical_level`'s doc comment for the full rationale.
    pub tatum: Option<MetricalLevel>,
    /// The coarser grouping above the tactus -- typically a musical
    /// measure (2, 3, or 4 beats). Same `None`-means-unclear semantics as
    /// `tatum`.
    pub measure: Option<MetricalLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlternateTempo {
    pub period_frames: f32,
    pub bpm: f32,
    pub confidence: f32,
}

/// A tatum or measure period candidate -- same shape as `AlternateTempo`
/// but a semantically distinct concept (metrical *hierarchy* around a
/// trusted tactus, not tactus-level octave *ambiguity*), so kept as its
/// own type rather than reusing `AlternateTempo` for both.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetricalLevel {
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

    // Known limitation, root-caused via the production-readiness bpm~304
    // investigation (build plan, Iteration 3 item 2): when `envelope` is
    // shorter than the true beat period -- normal during streaming startup,
    // before enough history has accumulated -- no candidate near the true
    // (often slower) tempo is even computable yet, so the estimator can only
    // choose among short lags near `min_lag` and may report a fast, wrong
    // BPM with a real but sometimes non-negligible confidence, most often
    // landing near min_lag's own boundary. Not a rounding bug: confirmed via
    // a synthetic sweep that plain broadband noise does *not* preferentially
    // land here (0/200 seeds), and that a short-history slice of a genuine
    // slow-tempo click train does, deterministically, until enough history
    // accumulates to see the true period. This is a structural cold-start
    // limit of any causal/windowed-autocorrelation estimator, not something
    // fixable in this function's own math -- it's mitigated one layer up, by
    // requiring several consecutive above-threshold frames before trusting
    // `has_rhythm` (the validated `debounced_25` result, Iteration 2),
    // rather than by changing candidate selection here.
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
    // by peak interpolation alone.
    //
    // The correction used to unconditionally prefer the *smallest* period
    // among near-tied candidates (faster tempo), on the theory that was the
    // more likely true beat. Measured against 664 real, annotated GiantSteps
    // tracks, that one-directional bias turned out to be net-harmful: 35.6%
    // of errors were "detected exactly 2x too fast" against only 20.3%
    // "detected 2x too slow" — a real, lopsided asymmetry directly caused by
    // always breaking ties toward the shorter period, not a wash.
    //
    // The fix is a bidirectional, principled tie-break instead of a
    // one-directional heuristic: among all near-tied candidates (both
    // sub-harmonics *and* harmonics of the raw best), prefer whichever is
    // closest in log-tempo space to ~120 BPM — the well-documented human
    // preferred-tempo center from tempo-perception research (Moelants 2002;
    // McKinney & Moelants 2004), not a value fit to this one benchmark.
    const NEAR_TIE_TOLERANCE: f32 = 0.85; // fraction of the raw best score
    const PREFERRED_BPM_CENTER: f64 = 120.0;
    let threshold = raw_best_score * NEAR_TIE_TOLERANCE;
    let score_at = |lag: usize| -> Option<f32> {
        lag.checked_sub(min_lag)
            .and_then(|i| scores.get(i))
            .map(|&(_, s)| s)
    };
    let bpm_of = |lag: usize| -> f64 { 60_000.0 / (lag as f64 * hop_ms) };
    let log_distance_from_center =
        |lag: usize| -> f64 { (bpm_of(lag).log2() - PREFERRED_BPM_CENTER.log2()).abs() };

    // Asymmetric near-tie tolerance for low-BPM candidates specifically
    // (BPM/octave accuracy improvement program, Stage 1a) -- directly
    // targeted at a real, mined finding (Stage 0a's analysis of
    // `data/ENSEMBLE.jsonl`'s 3,954 tracks): confidently-wrong octave
    // errors (no `alternate` ever flagged, meaning raw score never came
    // within `NEAR_TIE_TOLERANCE` of the true answer) are heavily
    // concentrated below ~80 BPM and skew "2x too slow" 85.2% of the time
    // (431/506) -- plausibly busy sub-rhythm content (hi-hats etc.) making
    // a genuinely slower, wrong candidate's raw score deceptively
    // dominant. A candidate at 2x/3x/4x a below-threshold chosen tempo
    // gets checked against a separate, deliberately looser tolerance
    // instead of the general near-tie bar, so it can be reconsidered by
    // the existing closer-to-preferred-center logic below even when it
    // isn't a genuine near-tie by raw score alone.
    //
    // An earlier version of this stage tried a general "cross-level
    // hierarchy" score (does a candidate have its own tatum/measure
    // support, via `find_metrical_level`) as tie-break evidence instead of
    // this targeted low-BPM check. Reverted after tracing a real
    // regression it introduced in `recovers_a_range_of_tempi`'s 160 BPM
    // case: a *correct*, fast chosen tempo near `DEFAULT_BPM_MAX`
    // structurally can't have a "tatum" (an even faster candidate) simply
    // because there's no room left in the valid BPM range to look for
    // one, while its own half-tempo octave gets to borrow the true tempo
    // itself as a free "tatum," inflating its hierarchy score above the
    // correct answer's. That bias runs backward from this stage's actual
    // goal (Stage 0a's problem is candidates resolving *too slow*, not
    // too fast), so it was dropped in favor of this narrower,
    // directly-evidenced fix instead.
    const LOW_BPM_OCTAVE_THRESHOLD: f64 = 80.0;
    // Calibrated directly against two real, opposite-answer cases traced
    // during this stage's development: Jamendo 232.mp3 (`ENSEMBLE.jsonl`'s
    // confidently-wrong 60->120 BPM case) needs its correct 120 BPM
    // candidate (score ratio 0.848 vs. the wrong chosen candidate) to
    // clear this bar, while the synthetic
    // `strong_amplitude_asymmetry_resolves_without_an_alternate` fixture's
    // deliberately-weaker 152 BPM candidate (ratio 0.769, intentionally
    // built to *not* be a real competing hypothesis) must not. 0.8 sits
    // cleanly between the two.
    const LOW_BPM_TIE_TOLERANCE: f32 = 0.8;

    // Local distributional shape (BPM/octave accuracy improvement program,
    // Stage 1b): Shannon entropy of just the small neighborhood of scores
    // immediately around a candidate lag, distinct from the `entropy`
    // field computed later (which characterizes the *whole* candidate-lag
    // range at once, and answers "is there real periodicity anywhere in
    // this signal" -- a global question). This is a local, per-candidate
    // question instead: does *this specific* lag sit under a sharp,
    // well-isolated peak, or in a smeared region where several nearby lags
    // score comparably? Low local entropy = a clean, confidently-isolated
    // peak; high local entropy = an ambiguous region where the raw winner
    // could easily be one of several near-equal near-neighbors.
    //
    // Traced directly against the same two known, opposite-answer cases
    // Stage 1a's own tolerance was calibrated against, before wiring this
    // into any decision: Jamendo 232.mp3's wrongly-chosen 60 BPM candidate
    // scores local_entropy=0.99 (a genuinely smeared, ambiguous region --
    // consistent with "confidently wrong" really meaning "several
    // plausible periods scored close together, and the wrong one edged
    // out"), while `strong_amplitude_asymmetry_resolves_without_an_alternate`'s
    // correctly-chosen 76 BPM candidate scores local_entropy=0.00 (a
    // completely isolated, unambiguous peak -- its immediate neighbors
    // have essentially no competing score at all). A real, strong,
    // opposite-direction separation on the only two calibration points
    // available, used here as a gate for a substantially looser score
    // tolerance than Stage 1a's own 0.8: only worth trusting a much
    // weaker-scoring candidate over the raw best when the raw best's own
    // peak already looks genuinely ambiguous, not just whenever a
    // below-80-BPM chosen candidate exists at all.
    let local_entropy = |lag: usize, half_window: usize| -> f32 {
        let lo = lag.saturating_sub(half_window);
        let hi = lag + half_window;
        let local: Vec<f32> = (lo..=hi).filter_map(&score_at).map(|s| s.max(0.0)).collect();
        let sum: f32 = local.iter().sum();
        if sum <= 1e-6 || local.len() < 2 {
            return 0.0;
        }
        let h: f32 = local
            .iter()
            .filter(|&&s| s > 0.0)
            .map(|&s| {
                let p = s / sum;
                -p * p.ln()
            })
            .sum();
        let max_h = (local.len() as f32).ln();
        if max_h > 1e-6 {
            (h / max_h).clamp(0.0, 1.0)
        } else {
            0.0
        }
    };
    const LOCAL_ENTROPY_AMBIGUITY_THRESHOLD: f32 = 0.5; // not yet empirically calibrated beyond the two traced cases above -- pending harness validation
    const SMEARED_LOW_BPM_TIE_TOLERANCE: f32 = 0.4; // substantially looser than LOW_BPM_TIE_TOLERANCE=0.8, gated behind genuine local ambiguity
    let mut chosen = (raw_best_lag, raw_best_score);
    for ratio in [1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 2.0, 3.0, 4.0] {
        let approx = raw_best_lag as f32 * ratio;
        let window_lo = (approx.floor() as usize).saturating_sub(1);
        let window_hi = approx.ceil() as usize + 1;
        if let Some((cand_lag, cand_score)) = (window_lo..=window_hi)
            .filter_map(|lag| score_at(lag).map(|s| (lag, s)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        {
            let low_bpm_octave_case =
                bpm_of(chosen.0) < LOW_BPM_OCTAVE_THRESHOLD && bpm_of(cand_lag) > bpm_of(chosen.0);
            let chosen_is_smeared =
                low_bpm_octave_case && local_entropy(chosen.0, 3) >= LOCAL_ENTROPY_AMBIGUITY_THRESHOLD;
            let clears_score_bar = cand_score >= threshold
                || (low_bpm_octave_case && cand_score >= raw_best_score * LOW_BPM_TIE_TOLERANCE)
                || (chosen_is_smeared && cand_score >= raw_best_score * SMEARED_LOW_BPM_TIE_TOLERANCE);
            if !clears_score_bar {
                continue;
            }
            if log_distance_from_center(cand_lag) < log_distance_from_center(chosen.0) {
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

    // Prominence: the winner vs. the rest of the candidate distribution,
    // not just its own raw height -- diagnostic-only for now (production-
    // readiness item 2's confidence-signal exploration), computed from
    // `scores`, which was already fully populated above; no new pass over
    // the envelope. A noise-spurious peak can be tall in absolute terms
    // but unremarkable next to its neighbors, which raw `confidence` alone
    // can't distinguish and this can.
    let score_mean = scores.iter().map(|&(_, s)| s).sum::<f32>() / scores.len() as f32;
    let prominence = if score_mean > 1e-6 {
        chosen.1 / score_mean
    } else {
        0.0
    };

    // Entropy: normalized Shannon entropy of the score distribution across
    // every candidate lag -- a different reduction of the same `scores`
    // data than prominence, so it's not redundant by construction: two
    // distributions can have the same winner-vs-mean ratio (prominence)
    // while differing in how concentrated the *rest* of the mass is (one
    // dominant peak vs. many middling ones). General "spectral entropy"
    // concept, close in spirit to MIRToolbox's "pulse entropy" pulse-
    // clarity feature (Lartillot, Eerola, Toiviainen & Fornari 2008,
    // "Multi-Feature Modeling of Pulse Clarity") though approximated here
    // over the whole candidate distribution rather than only its local
    // peaks.
    //
    // Direction confirmed empirically, and it's the *opposite* of the
    // naive "peakiness" intuition: a genuinely periodic signal's
    // autocorrelation is elevated not only at the true period but at
    // every harmonic/subharmonic of it too (the same tempo-octave
    // phenomenon documented above), spreading real positive score mass
    // across many candidate lags -- high entropy. Broadband noise has
    // near-zero autocorrelation at nearly every lag except a few
    // incidental bumps, so positive mass concentrates on far fewer bins --
    // low entropy. Measured on the no-rhythm eval set vs. a 100-track
    // Ballroom sample (production-readiness item 2, post-Iteration-3
    // continued research): median entropy 0.80 (no-rhythm) vs. 0.99
    // (Ballroom), a real, strong separation, also confirmed on Hainsworth
    // rubato (entropy correctly orders correct-vs-wrong there too, unlike
    // confidence/prominence, which invert). So: high entropy = more likely
    // a real, unambiguous pulse; low entropy = more likely no true
    // periodicity. Diagnostic-only, same discipline as prominence: not yet
    // used by has_rhythm/confidence gating.
    let entropy = {
        let sum: f32 = scores.iter().map(|&(_, s)| s.max(0.0)).sum();
        if sum > 1e-6 {
            let h: f32 = scores
                .iter()
                .map(|&(_, s)| s.max(0.0))
                .filter(|&s| s > 0.0)
                .map(|s| {
                    let p = s / sum;
                    -p * p.ln()
                })
                .sum();
            let max_h = (scores.len() as f32).ln();
            if max_h > 1e-6 {
                (h / max_h).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            1.0 // no usable energy at all -- maximally "unclear", consistent with near-silence
        }
    };

    // Tatum/measure: Phase 3.5's DSP capability expansion, reusing this
    // same `scores` array a third time (after `alternate` and
    // `prominence`/`entropy`) rather than building a second periodicity-
    // estimation mechanism -- see the build plan's Phase 3.5 for the full
    // rationale (Klapuri, Eronen & Astola, IEEE TASLP 2006, describes the
    // published reference method: a full HMM over jointly-estimated
    // tatum/tactus/measure periods with learned priors; this is a
    // deliberately simplified stand-in that captures the two ideas that
    // matter most for this project's actual need -- reuse already-
    // available periodicity evidence, and prefer small-integer ratios
    // between levels -- without the full statistical machinery). Modeled
    // as conditionally independent given the chosen tactus, matching the
    // paper's own stated assumption that "given the tactus period, the
    // tatum period does not give additional information regarding the
    // measure" (their Eq. 14) -- i.e. both searches start from `chosen`
    // directly, neither chains through the other.
    let tatum = find_metrical_level(chosen, &score_at, hop_ms, |r| chosen.0 / r);
    let measure = find_metrical_level(chosen, &score_at, hop_ms, |r| chosen.0 * r);

    Some(TempoEstimate {
        period_frames: chosen.0 as f32,
        bpm: (60_000.0 / (chosen.0 as f64 * hop_ms)) as f32,
        confidence: chosen.1.clamp(0.0, 1.0),
        prominence,
        entropy,
        alternate,
        tatum,
        measure,
    })
}

/// Fraction of the chosen tactus candidate's own score a tatum/measure
/// candidate needs to clear to count as real structure rather than noise.
/// Deliberately looser than `ALTERNATE_TOLERANCE` (0.85): tatum/measure
/// candidates are genuinely expected to score lower than the tactus peak
/// itself (they're a different metrical level, not a competing hypothesis
/// for the *same* level the way `alternate` is), so requiring near-parity
/// would under-detect real structure. Not yet empirically calibrated
/// against real audio -- placeholder pending the Ballroom/Hainsworth
/// downbeat validation harness (build plan Phase 3.5 go/no-go), same
/// discipline as `MIN_RHYTHM_CONFIDENCE`'s own calibration history.
const METRICAL_LEVEL_TOLERANCE: f32 = 0.5;

/// Searches `ratios` in order (2, 3, 4 -- binary before ternary before
/// quaternary) for the first candidate lag, derived from the chosen tactus
/// lag via `lag_for_ratio`, that clears `METRICAL_LEVEL_TOLERANCE` of the
/// tactus's own score. Takes the *first* qualifying ratio rather than the
/// best-scoring one deliberately: this encodes the well-documented musical
/// tendency toward small-integer metrical ratios (a measure is far more
/// often 2, 3, or 4 beats than some larger number, and a beat far more
/// often splits into 2 or 3 than more) as a preference order, the same
/// spirit as Klapuri et al.'s learned Gaussian-mixture weighting toward
/// small integer ratios (their Eq. 25), simplified to a fixed preference
/// order instead of a fitted distribution.
fn find_metrical_level(
    chosen: (usize, f32),
    score_at: &impl Fn(usize) -> Option<f32>,
    hop_ms: f64,
    lag_for_ratio: impl Fn(usize) -> usize,
) -> Option<MetricalLevel> {
    let threshold = chosen.1 * METRICAL_LEVEL_TOLERANCE;
    [2usize, 3, 4].into_iter().find_map(|r| {
        let lag = lag_for_ratio(r);
        if lag == 0 || lag == chosen.0 {
            return None;
        }
        let score = score_at(lag)?;
        if score < threshold {
            return None;
        }
        Some(MetricalLevel {
            period_frames: lag as f32,
            bpm: (60_000.0 / (lag as f64 * hop_ms)) as f32,
            confidence: score.clamp(0.0, 1.0),
        })
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

    /// Like `click_envelope`, but with small per-beat amplitude jitter
    /// (deterministic PRNG, no external dependency) instead of a uniform
    /// 1.0 at every beat. A perfectly uniform click train is mathematically
    /// self-similar at *every* integer multiple of its period (no unique
    /// answer to test against); a *regular alternating* amplitude pattern
    /// (see `backbeat_envelope`) fixes that but introduces its own
    /// competing sub-rhythm at 2x the period. Random per-beat jitter breaks
    /// the exact-tie degeneracy without constructing any alternative
    /// periodic hypothesis, so the true fundamental period stays the
    /// unique, unambiguous winner — closer to how real timing imprecision
    /// behaves than either of the other two fixtures.
    fn jittered_click_envelope(period_frames: usize, n_beats: usize) -> Vec<f32> {
        let mut env = vec![0.0f32; period_frames * n_beats];
        let mut state: u32 = 0xC0FF_EE11;
        for beat in 0..n_beats {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let jitter = (state >> 16) as f32 / 65535.0 * 0.15;
            env[beat * period_frames] = 1.0 - jitter;
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
        // Uses jittered_click_envelope, not a pure uniform click train: an
        // idealized delta train is mathematically self-similar at *every*
        // integer multiple of its period, so it has no uniquely correct
        // octave to assert against -- least of all once the tie-break
        // between near-tied octaves prefers whichever is perceptually more
        // plausible (see the preferred-tempo-center comment above), which
        // can legitimately differ from any single hardcoded target for a
        // signal that's ambiguous by construction. (A regular alternating
        // backbeat_envelope doesn't fix this either -- tried first, and it
        // introduces its own competing 2x sub-rhythm strong enough to
        // legitimately win instead, confirmed live: 90 BPM resolved to its
        // own strong-beats-only sub-rhythm at 45 BPM.) Small per-beat
        // jitter breaks the tie without adding a competing hypothesis.
        //
        // Capped at 160 BPM deliberately: swept 60-220 during development
        // and found *every* value above ~165 resolves to its own half
        // (170->86, 180->89, 200->99, ...), because at short periods (high
        // BPM) more harmonics fit inside the candidate window, so the
        // harmonic-averaged score genuinely converges toward the same
        // near-1.0 value at both a fast tempo and its half for a cleanly
        // periodic signal -- confirmed this is a real near-tie in the
        // score itself (tightening the tie tolerance from 0.85 to 0.95
        // changed nothing), not a loose-threshold artifact. A fixed 120 BPM
        // preference will always resolve that genuine tie toward the
        // slower octave once a tempo is more than a factor of sqrt(2) above
        // it (~170 BPM) -- correct, principled behavior for a signal that
        // really is ambiguous, but it means high-BPM octave resolution on
        // *idealized* periodic test signals isn't a meaningful thing to
        // assert against; real audio's natural asymmetry (dynamics,
        // syncopation, non-uniform onset strength) is what actually
        // resolves those cases, which is exactly why Phase 2.5a's
        // real-audio harness against GiantSteps -- not this synthetic
        // test -- is the authoritative check for fast-tempo accuracy (see
        // training/eval/RESULTS.md: Accuracy1 28.2% -> 52.3% after this
        // same change, measured on 664 real annotated tracks spanning up
        // to 190+ BPM).
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        for target_bpm in [60.0, 90.0, 120.0, 140.0, 160.0] {
            let period = period_frames_for_bpm(target_bpm, hop_ms).round() as usize;
            let env = jittered_click_envelope(period, 20);
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

    /// Confirms the bpm~304 boundary mechanism root-caused in the doc
    /// comment above `min_lag`: a short-history slice of a genuine
    /// slow-tempo click train is forced toward fast candidates near
    /// `min_lag` until enough history accumulates to see the true period,
    /// deterministically (not a rounding fluke) -- while pure broadband
    /// noise does *not* preferentially land there, confirmed by a 200-seed
    /// sweep during the original investigation (not kept as a permanent
    /// test; a single seed here is enough to guard the mechanism itself).
    #[test]
    fn short_history_biases_toward_fast_boundary_lag() {
        let hop_ms = 256.0 / 44_100.0 * 1000.0;
        let true_period = period_frames_for_bpm(80.0, hop_ms).round() as usize; // ~129 frames
        let mut env = vec![0.0f32; true_period * 6];
        for beat in 0..6 {
            env[beat * true_period] = 1.0;
        }

        // Too short to contain even one true-period cycle: forced toward a
        // fast, wrong answer with near-zero confidence (not a confident
        // wrong answer -- `has_rhythm` gating gets a real, low signal here).
        let short = &env[..60];
        let est_short = estimate_tempo(short, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).unwrap();
        assert!(
            est_short.bpm > 250.0,
            "expected a fast boundary-biased guess from insufficient history, got {}",
            est_short.bpm
        );
        assert!(
            est_short.confidence < 0.1,
            "expected low confidence while forced toward the boundary, got {}",
            est_short.confidence
        );

        // Once enough history exists to see the true period, it resolves
        // correctly and confidently.
        let full = &env[..200];
        let est_full = estimate_tempo(full, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).unwrap();
        assert!(
            (est_full.bpm - 80.0).abs() < 2.0,
            "expected ~80 BPM once enough history accumulated, got {}",
            est_full.bpm
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

    #[test]
    fn layered_subdivision_exposes_a_tatum() {
        // A clean beat, plus a genuinely weaker pulse inserted exactly
        // halfway between each beat (a plausible 8th-note hi-hat pattern
        // under a kick). `period` is forced even so the halfway point is
        // exact -- with an odd period, a first attempt at this fixture
        // (period/2 rounding down, then stepping by the *full* period)
        // accidentally produced a phase-shifted copy of the same beat-rate
        // rhythm instead of real double-rate content, caught by inspecting
        // the actual failure rather than assuming the production logic was
        // wrong. Stepping by `sub_period` here, not `period`, is what
        // actually places a pulse twice per beat.
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        let mut period = period_frames_for_bpm(120.0, hop_ms).round() as usize;
        if period % 2 == 1 {
            period += 1;
        }
        let sub_period = period / 2;
        let n_beats = 24;
        let target_bpm = 60_000.0 / (period as f64 * hop_ms);

        let mut env = vec![0.0f32; period * n_beats];
        for beat in 0..n_beats {
            env[beat * period] = 1.0;
        }
        let mut i = sub_period;
        while i < env.len() {
            env[i] = (env[i] + 0.35).min(1.0);
            i += sub_period;
        }

        let est = estimate_tempo(&env, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).unwrap();
        assert!(
            (est.bpm as f64 - target_bpm).abs() < 2.0,
            "layering a weak subdivision shouldn't disturb the tactus itself, expected ~{target_bpm}, got {}",
            est.bpm
        );
        let tatum = est
            .tatum
            .unwrap_or_else(|| panic!("expected a tatum given a real half-period subdivision, got {est:?}"));
        assert!(
            (tatum.period_frames as usize).abs_diff(sub_period) <= 1,
            "expected tatum period ~{sub_period}, got {}",
            tatum.period_frames
        );
    }

    #[test]
    fn accented_grouping_exposes_a_measure() {
        // Reuses `backbeat_envelope` -- a genuine, common 2-beat grouping (a
        // snare landing on every beat but accented every 2nd), not an
        // artifact of harmonic self-similarity. Chosen over a from-scratch
        // 3-beat fixture after an actual empirical sweep (not guesswork)
        // found that a synthetic waltz-style 1-strong/2-weak pattern
        // structurally favors an r=2 self-similarity artifact almost
        // regardless of amplitude contrast, for the same reason a plain
        // click train does -- 2 doesn't evenly divide the r=2 comparison
        // the way it would need to for genuine suppression.
        //
        // weak=0.6, not the 0.4 used by
        // `strong_amplitude_asymmetry_resolves_without_an_alternate`
        // below: a separate empirical sweep at *this* fixture's 120 BPM
        // (that other test uses 150 BPM) found 0.3-0.4 actually flips the
        // *chosen tactus itself* to the accented-beats-only half-rate
        // (60 BPM) -- the weak=0.4 choice there doesn't transfer to a
        // different base tempo, caught by checking `est.bpm` directly
        // here rather than assuming success from the absence of an
        // alternate the way that other test does. 0.5-0.9 all keep the
        // tactus correct at this tempo; 0.6 has real margin either side.
        //
        // METRICAL_LEVEL_TOLERANCE (0.5) is deliberately looser than
        // ALTERNATE_TOLERANCE (0.85): this same 2x candidate is real
        // metrical structure worth surfacing as `measure` even when it
        // doesn't clear the much stricter bar for being treated as a
        // competing *tactus* hypothesis.
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        let period = period_frames_for_bpm(120.0, hop_ms).round() as usize;
        let env = backbeat_envelope(period, 24, 0.6);

        let est = estimate_tempo(&env, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).unwrap();
        assert!(
            (est.bpm - 120.0).abs() < 2.0,
            "the measure-level accent shouldn't disturb the tactus itself, got {}",
            est.bpm
        );
        let measure = est
            .measure
            .unwrap_or_else(|| panic!("expected a measure given a real 2-beat accent pattern, got {est:?}"));
        let expected = period * 2;
        assert!(
            (measure.period_frames as usize).abs_diff(expected) <= 1,
            "expected measure period ~{expected}, got {}",
            measure.period_frames
        );
    }

    #[test]
    fn tatum_and_measure_are_none_on_broadband_noise() {
        // A plain *regular* click train isn't the right "no structure"
        // fixture here -- a first attempt used one and it failed loudly,
        // for a real reason: an idealized periodic click train is
        // mathematically self-similar at *every* integer multiple of its
        // period (the same documented property `genuinely_ambiguous_
        // tempo_exposes_an_alternate` above relies on), so a plain click
        // train genuinely *does* show a real r=2 candidate -- that's
        // correct autocorrelation behavior, not a bug to paper over.
        // Broadband noise is the honest "no periodic structure at all"
        // fixture, matching `white_noise_yields_low_confidence` above.
        let hop_ms = 512.0 / 44_100.0 * 1000.0;
        let mut state: u32 = 0x1234_5678;
        let env: Vec<f32> = (0..500)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 16) as f32 / 65535.0
            })
            .collect();

        let est = estimate_tempo(&env, hop_ms, DEFAULT_BPM_MIN, DEFAULT_BPM_MAX).unwrap();
        assert!(
            est.tatum.is_none(),
            "expected no tatum on broadband noise, got {:?}",
            est.tatum
        );
        assert!(
            est.measure.is_none(),
            "expected no measure on broadband noise, got {:?}",
            est.measure
        );
    }
}

