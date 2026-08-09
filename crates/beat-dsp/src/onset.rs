//! Spectral-flux onset detection: turns a raw audio stream into a scalar
//! "onset strength" envelope, sampled once per hop. This is the input the
//! tempo/phase estimators in `tempo.rs` and the offline tracker in
//! `offline.rs` operate on — deliberately not any kind of learned feature,
//! just a windowed FFT and a half-wave-rectified spectral difference.

use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::collections::VecDeque;
use std::sync::Arc;

pub const FRAME_SIZE: usize = 1024;
/// 256 rather than the more common 512: finer time resolution reduces
/// frame-quantization error in the tempo/phase estimators (matters more at
/// the high end of the BPM range, where a period is only a few dozen
/// frames to begin with) at a still-cheap 2x onset-detector frame rate.
pub const HOP_SIZE: usize = 256;

/// Sampling rate of the onset envelope itself, once `sample_rate` is fixed:
/// `sample_rate / HOP_SIZE` frames per second (e.g. ~86Hz at 44.1kHz).
pub fn hop_duration_ms(sample_rate: u32) -> f64 {
    HOP_SIZE as f64 / sample_rate as f64 * 1000.0
}

pub struct OnsetDetector {
    fft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    /// Raw samples not yet consumed into a frame.
    pending: VecDeque<f32>,
    /// Magnitude spectrum from the previous frame, for spectral flux.
    prev_magnitude: Vec<f32>,
    scratch: Vec<Complex32>,
}

impl OnsetDetector {
    pub fn new() -> Self {
        let mut planner = FftPlanner::new();
        Self {
            fft: planner.plan_fft_forward(FRAME_SIZE),
            window: hann_window(FRAME_SIZE),
            pending: VecDeque::new(),
            prev_magnitude: vec![0.0; FRAME_SIZE / 2 + 1],
            scratch: vec![Complex32::new(0.0, 0.0); FRAME_SIZE],
        }
    }

    pub fn reset(&mut self) {
        self.pending.clear();
        self.prev_magnitude.iter_mut().for_each(|m| *m = 0.0);
    }

    /// Feed new samples in; returns one onset-strength value per hop that
    /// became available (zero, one, or several, depending on chunk size
    /// versus HOP_SIZE).
    pub fn push_samples(&mut self, samples: &[f32]) -> Vec<f32> {
        self.pending.extend(samples.iter().copied());

        let mut onsets = Vec::new();
        while self.pending.len() >= FRAME_SIZE {
            // Windowed frame = the most recent FRAME_SIZE pending samples.
            for (i, s) in self.pending.iter().take(FRAME_SIZE).enumerate() {
                self.scratch[i] = Complex32::new(s * self.window[i], 0.0);
            }
            self.fft.process(&mut self.scratch);

            let mut flux = 0.0f32;
            for bin in 0..(FRAME_SIZE / 2 + 1) {
                let mag = self.scratch[bin].norm();
                let diff = mag - self.prev_magnitude[bin];
                flux += diff.max(0.0); // half-wave rectify
                self.prev_magnitude[bin] = mag;
            }
            onsets.push(flux);

            // Advance by one hop.
            for _ in 0..HOP_SIZE {
                self.pending.pop_front();
            }
        }
        onsets
    }
}

impl Default for OnsetDetector {
    fn default() -> Self {
        Self::new()
    }
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / (size as f32 - 1.0)).cos()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_produces_zero_flux() {
        let mut det = OnsetDetector::new();
        let onsets = det.push_samples(&vec![0.0; FRAME_SIZE * 3]);
        assert!(!onsets.is_empty());
        assert!(onsets.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn sudden_impulse_produces_a_flux_spike() {
        let mut det = OnsetDetector::new();
        let mut samples = vec![0.0f32; FRAME_SIZE * 2];
        // A sharp click partway through the second frame's window.
        samples[FRAME_SIZE + 100] = 1.0;
        let onsets = det.push_samples(&samples);
        assert!(onsets.iter().any(|&v| v > 0.01));
    }

    #[test]
    fn chunked_input_matches_single_call() {
        let mut whole = OnsetDetector::new();
        let mut chunked = OnsetDetector::new();

        let mut samples = vec![0.0f32; FRAME_SIZE * 4];
        samples[500] = 1.0;
        samples[3000] = 0.8;

        let onsets_whole = whole.push_samples(&samples);

        let mut onsets_chunked = Vec::new();
        for chunk in samples.chunks(37) {
            onsets_chunked.extend(chunked.push_samples(chunk));
        }

        assert_eq!(onsets_whole.len(), onsets_chunked.len());
        for (a, b) in onsets_whole.iter().zip(onsets_chunked.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }
}
