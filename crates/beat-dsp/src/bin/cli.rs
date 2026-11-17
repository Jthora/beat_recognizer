//! Phase 2.5a: a thin CLI wrapper around `beat-dsp`, used by
//! `training/eval/harness.py` to score the *actual* Rust implementation
//! against real annotated audio (SMC, GiantSteps) via mir_eval, rather than
//! only ever validating against synthetic click tracks in `cargo test`, or
//! risking a second, separately-maintained Python reimplementation that
//! could silently drift from what actually ships.
//!
//! Hand-rolled JSON output rather than pulling in `serde_json` — the output
//! shape is small and fixed, not worth a new dependency for.

use std::env;
use std::process::ExitCode;

use beat_dsp::{BeatEstimate, DeterministicTracker};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let Some(path) = args.get(1) else {
        eprintln!("usage: beat-dsp-cli <wav-file> [--mode offline|streaming|trace]");
        return ExitCode::FAILURE;
    };
    let mode = args
        .iter()
        .position(|a| a == "--mode")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
        .unwrap_or("offline");

    let mono = match read_wav_mono(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("failed to read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut tracker = DeterministicTracker::new();
    match mode {
        "offline" => {
            let estimates = tracker.analyze_offline(&mono.samples, mono.sample_rate);
            // Cheap to compute twice for a diagnostic CLI (not the shipped
            // hot path) -- keeps analyze_offline's actual WASM-facing
            // signature untouched. See analyze_offline_raw's doc comment.
            let raw = tracker.analyze_offline_raw(&mono.samples, mono.sample_rate);
            let prominence = raw.as_ref().map(|r| r.prominence).unwrap_or(0.0);
            let entropy = raw.map(|r| r.entropy).unwrap_or(0.0);
            println!("{}", offline_json(&estimates, prominence, entropy));
        }
        "streaming" => {
            for chunk in mono.samples.chunks(512) {
                tracker.process_chunk(chunk, mono.sample_rate);
            }
            println!("{}", streaming_json(&tracker.current_estimate()));
        }
        "trace" => {
            // Production-readiness item 2's confidence-signal exploration:
            // emits one JSON-lines record per chunk (not just the final
            // estimate) so the Python prototype can test whether
            // prominence/stability-over-time separates real rhythm from
            // noise better than the instantaneous confidence value does,
            // before any of this feeds into shipped tracker behavior.
            const CHUNK_SIZE: usize = 512;
            for (i, chunk) in mono.samples.chunks(CHUNK_SIZE).enumerate() {
                tracker.process_chunk(chunk, mono.sample_rate);
                let t_ms = (i * CHUNK_SIZE) as f64 / mono.sample_rate as f64 * 1000.0;
                if let Some(line) = trace_json(t_ms, &tracker) {
                    println!("{line}");
                }
            }
        }
        other => {
            eprintln!("unknown --mode {other:?}, expected offline, streaming, or trace");
            return ExitCode::FAILURE;
        }
    }

    ExitCode::SUCCESS
}

struct MonoAudio {
    samples: Vec<f32>,
    sample_rate: u32,
}

fn read_wav_mono(path: &str) -> Result<MonoAudio, String> {
    let mut reader = hound::WavReader::open(path).map_err(|e| e.to_string())?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
    };

    let mono = if channels > 1 {
        samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };

    Ok(MonoAudio {
        samples: mono,
        sample_rate: spec.sample_rate,
    })
}

fn opt_f32_json(v: Option<f32>) -> String {
    v.map(|x| x.to_string())
        .unwrap_or_else(|| "null".to_string())
}

fn offline_json(estimates: &[BeatEstimate], prominence: f32, entropy: f32) -> String {
    let Some(first) = estimates.first() else {
        return r#"{"bpm":null,"confidence":0.0,"prominence":0.0,"entropy":1.0,"alternate_bpm":null,"alternate_confidence":null,"beats_s":[],"downbeats_s":[]}"#
            .to_string();
    };
    let beats_s: Vec<String> = estimates
        .iter()
        .map(|e| (e.next_beat_time_ms / 1000.0).to_string())
        .collect();
    // Mirrors Ballroom/Hainsworth's own ground-truth index format
    // (`beat_times_s`/`downbeat_times_s`) directly, so the Phase 3.5
    // downbeat validation harness can compare like-for-like without
    // reshaping either side -- see the build plan's Phase 3.5.
    let downbeats_s: Vec<String> = estimates
        .iter()
        .filter(|e| e.is_downbeat == Some(true))
        .map(|e| (e.next_beat_time_ms / 1000.0).to_string())
        .collect();
    format!(
        r#"{{"bpm":{},"confidence":{},"prominence":{},"entropy":{},"alternate_bpm":{},"alternate_confidence":{},"beats_s":[{}],"downbeats_s":[{}]}}"#,
        first.bpm,
        first.confidence,
        prominence,
        entropy,
        opt_f32_json(first.alternate_bpm),
        opt_f32_json(first.alternate_confidence),
        beats_s.join(","),
        downbeats_s.join(","),
    )
}

fn trace_json(t_ms: f64, tracker: &DeterministicTracker) -> Option<String> {
    let tempo = tracker.last_tempo_estimate()?;
    let phase = tracker.last_phase_estimate();
    Some(format!(
        r#"{{"t_ms":{},"has_rhythm":{},"bpm":{},"confidence":{},"prominence":{},"entropy":{},"beat_phase":{},"best_score":{},"has_alternate":{}}}"#,
        t_ms,
        tracker.current_estimate().has_rhythm,
        tempo.bpm,
        tempo.confidence,
        tempo.prominence,
        tempo.entropy,
        phase.map(|p| p.beat_phase).unwrap_or(0.0),
        phase.map(|p| p.best_score).unwrap_or(0.0),
        tempo.alternate.is_some(),
    ))
}

fn streaming_json(e: &BeatEstimate) -> String {
    format!(
        r#"{{"has_rhythm":{},"bpm":{},"beat_phase":{},"confidence":{},"next_beat_time_ms":{},"alternate_bpm":{},"alternate_confidence":{}}}"#,
        e.has_rhythm,
        e.bpm,
        e.beat_phase,
        e.confidence,
        e.next_beat_time_ms,
        opt_f32_json(e.alternate_bpm),
        opt_f32_json(e.alternate_confidence),
    )
}
