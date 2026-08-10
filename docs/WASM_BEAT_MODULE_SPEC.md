# Musical Beat Recognizer — WASM Module Spec

> Requirements document for a standalone project: a custom-trained, license-clean,
> real-time music beat/tempo detector, compiled to a **TypeScript-callable WASM
> module**, for integration into ProtoFusionGirl (a Phaser 3 + TypeScript browser
> game). Self-contained — this doc assumes no prior context on ProtoFusionGirl.
> Written 2026-08-09.

---

## 1. Why this exists

ProtoFusionGirl is building a "Reality Engine" — a system where live music (and,
separately, real astrology) continuously reshapes an ongoing game simulation:
animation timing, companion-character behavior, environmental effects, and a
weighted-selection "Director" all react to what's actually playing. None of that
works without a real, live tempo/beat signal extracted from whatever audio the
player has going.

Two existing options were evaluated and both have real problems for this project:

- **BeatNet** (the most relevant published research system — CRNN + particle
  filtering, ISMIR 2021) is Python/PyTorch-only with no browser or WASM path.
  Confirmed, not assumed.
- **TempoCNN** (via `essentia.js-model`) is a real, already browser-deployable CNN
  tempo model (TF.js/ONNX, 0.1–4.7MB, 12-second receptive field) — but its
  pretrained weights are distributed under CC BY-NC-ND 4.0 (non-commercial,
  no-derivatives), which is a real problem if ProtoFusionGirl ships commercially.

Rather than depend on someone else's licensing terms, this project is building an
owned, license-clean equivalent from scratch. This document is the requirements
spec for that build, written from the consuming game's side so the model/module
gets built against real integration needs, not guessed at.

---

## 2. Hard platform requirement: TypeScript-callable WASM

This is non-negotiable, not a preference: **the deliverable must be a WASM module
callable directly from TypeScript**, with clean, idiomatic bindings — not raw
C/C++ FFI requiring hand-written glue, not a Python runtime, not a server-side
dependency of any kind. ProtoFusionGirl is a Vite-built, purely client-side
Phaser 3 game with no backend. Whatever inference runtime the model needs
(`wasm-bindgen`/Rust, Emscripten from C++, ONNX Runtime Web, TensorFlow.js — any of
these are acceptable *implementation choices*) must resolve to: **something
`npm install`-able (or drop-in loadable) that exposes a typed TS API and runs
entirely in-browser.**

If ONNX or TF.js ends up being part of the internal implementation, that's fine —
the requirement is about the *external contract* (TS-callable, browser-native,
no server), not about ruling out those runtimes internally.

---

## 3. Functional requirements — what it must do

### Must have

- **Real-time streaming tempo + beat-phase detection.** Feed it a continuously
  arriving audio stream (not a complete pre-loaded file) and get back a
  continuously-updated estimate: current BPM, phase (0–1 progress to the next
  beat), and a confidence score. This is the core capability.
- **Offline/batch mode too.** Given a complete audio buffer up front (a locally
  uploaded file, for example), it should be able to analyze the whole thing in one
  pass — likely with higher accuracy than the streaming mode, since full context is
  available. Both modes are required, not just one.
- **Confidence, not just a number.** Every estimate must come with a confidence/
  quality score. This matters more than it might seem — see §6.
- **Genre-agnostic robustness.** Real players will play arbitrary music — not just
  clean 4/4 electronic tracks. The model needs to handle syncopation, rubato,
  ambient/sparse material, and non-Western rhythmic structures reasonably, not just
  the easy cases. See §7 for the evaluation material this should be checked against.

### Nice to have (not blocking)

- **Downbeat detection** — distinguishing the first beat of a bar from the others.
  Real value (BeatNet does this jointly), but not required for a first version.
- **Meter/time-signature estimation.** Same — valuable later, not blocking.

### Explicitly out of scope for this module

Don't build these — they're handled elsewhere in the consuming system and adding
them here would be scope creep:

- **Onset/harmonicity/spectral-shape features** — handled separately via Meyda.js
  (cheap, non-ML, already integrated). This module is specifically about tempo/beat,
  not general audio feature extraction.
- **Key/chord/harmonic analysis** — separate system, separate library.
- **Lyrics/vocal transcription** — separate concern entirely, not audio-rhythm
  related.

---

## 4. Proposed API contract

Mirror this shape as closely as practical — it's designed to plug directly into an
existing `IBeatSource` interface already defined on the consuming side, so
integration is a drop-in, not a translation layer:

```typescript
export interface BeatEstimate {
  hasRhythm: boolean;        // false = no discernible rhythm (silence, noise,
                              // not enough history yet) — distinct from a
                              // detected-but-low confidence; ignore the rest
                              // of the fields when false. Added per
                              // WASM_BEAT_MODULE_ENHANCEMENTS.md §2.
  bpm: number;
  beatPhase: number;        // 0–1, progress toward the next beat
  confidence: number;       // 0–1
  nextBeatTimeMs: number;   // ms since construction/reset, not wall-clock —
                              // caller adds its own AudioContext.currentTime
                              // offset (see ENHANCEMENTS.md §5)
  isDownbeat?: boolean;     // optional, if downbeat detection is implemented
  alternateBpm?: number;         // the other tempo-octave candidate (e.g. 76
  alternateConfidence?: number;  // vs 152 BPM), present only when genuinely
                                   // ambiguous. Added per ENHANCEMENTS.md §2.
}

export interface BeatRecognizer {
  // Streaming mode
  processAudioChunk(samples: Float32Array, sampleRate: number): void;
  getCurrentEstimate(): BeatEstimate;
  onBeat?(callback: (estimate: BeatEstimate) => void): void;  // event-driven alternative/addition
  reset(): void;             // call when switching tracks/sources

  // Offline/batch mode
  analyzeOffline(buffer: Float32Array, sampleRate: number): BeatEstimate[]; // time-stamped sequence
}
```

Exact naming can flex — the shape (streaming feed + pull-or-callback read, a
distinct offline path, confidence on every estimate, explicit reset) is what
matters for a clean integration. `hasRhythm` and the `alternate*` fields were
added after the original v1 pass, per `WASM_BEAT_MODULE_ENHANCEMENTS.md` §2 —
both additive, neither changes the meaning of the original fields.

---

## 5. Performance and size budget

- **Size**: this ships inside a downloadable browser game. Use TempoCNN's
  0.1–4.7MB range as a reference ceiling, not a floor to hit exactly — smaller is
  better, but don't sacrifice the accuracy/robustness requirements in §3 to hit an
  arbitrary size target.
- **Runtime cost**: inference must not cause perceptible frame drops in a running
  game. If per-chunk inference cost is non-trivial, run it off the main thread
  (Web Worker) — don't assume it's cheap enough to run inline in the render loop
  without checking.
- **Latency is expected and fine, instant is not required.** No real beat tracker
  — this one included — can produce a meaningful tempo estimate with zero
  observed context; periodicity only reveals itself over a window of time.
  Document whatever your model's actual context window/settling time is (the way
  TempoCNN documents its 12-second receptive field) so the consuming side can
  design around a known number rather than an unknown one.

---

## 6. Why confidence calibration matters as much as raw accuracy

Concrete product context, since this should genuinely influence design tradeoffs,
not just be a spec line item: the consuming system feeds beat estimates into
several places simultaneously —

- **Animation retiming** (a sprite's loop `timeScale` continuously adjusted to
  match tempo) — needs a smooth, continuously-updated phase, not just discrete
  beat ticks.
- **A weighted-selection "Director"** that blends this signal with other data —
  it fuses on confidence explicitly, easing toward new estimates rather than
  snapping, and *needs to know when to distrust a reading*, not just receive a
  number.
- **Companion-character behavior** — a character deciding whether to attempt a
  visible trick/action synced to a beat. A wrong-but-confident estimate here is
  worse than a correctly-low-confidence "unclear" signal, because the downstream
  system will visibly commit to acting on it.

**Net implication for training/tuning priorities**: if there's ever a tradeoff
between squeezing out slightly higher peak accuracy versus better-calibrated
confidence (a model that reliably *knows* when it's uncertain, e.g. on ambient or
rhythmically ambiguous material), prefer the latter. A model that fails loudly
(low confidence) on hard material is more useful here than one that fails
silently (confident but wrong).

---

## 7. Evaluation material and target metrics

Use the same evaluation material already identified on the consuming side, so both
projects are measuring against the same yardstick:

- **[SMC dataset](https://essentia.upf.edu)** — 217 manually annotated 40-second
  clips specifically compiled to be hard for beat trackers (rhythmically ambiguous
  material). This is the "does it degrade gracefully on hard cases" check.
- **Osu2MIR** — beat/downbeat annotations derived from *Osu!* rhythm-game
  community beatmaps, spanning genres underrepresented in typical MIR datasets
  (anime, Vocaloid, game music). Good for genre-diversity coverage.
- Standard MIR beat-tracking metrics apply: F-measure, Cemgil, Goto, and
  continuity-based (P-score) metrics are the field's normal vocabulary for
  reporting this — use whichever standard evaluation toolkit is convenient
  (e.g. `mir_eval`'s beat-tracking module) rather than inventing a new metric.

**Target BPM range**: cover the realistic range real music actually falls into —
roughly 40–220 BPM as the common case, up to ~300 BPM for extreme/edge material
(matching TempoCNN's stated 30–286 range as a reasonable reference point).

---

## 8. Licensing requirement (the actual reason this project exists)

This is the one non-negotiable business requirement: **every input to the trained
model — architecture, training data, any pretrained/transfer-learning
components — must be clear for unrestricted commercial use in a shipped,
potentially monetized product.** No CC BY-NC, no CC BY-ND, no "research use only"
licensed datasets or backbone weights anywhere in the training pipeline. If a
public dataset or pretrained component looks useful but carries a restrictive
license, treat that as a hard blocker on using it, not a detail to resolve later —
verify licensing before training on something, not after.

---

## 9. Reference material (to avoid duplicating research already done)

- **BeatNet** (`mjhydri/BeatNet`, ISMIR 2021) — the most relevant published
  architecture: CRNN for frame-wise beat/downbeat activation, particle filtering
  for causal tracking over time. Worth studying the architecture and the
  train/eval methodology even though the code itself can't be used directly
  (Python/PyTorch, and its own license/training data would need separate
  verification against §8 regardless).
- **TempoCNN** (via Essentia's model repository) — real, converted TF.js/ONNX
  tempo model; useful as an architecture and size/latency reference point (mel-band
  input patches, 12-second receptive field, 0.1–4.7MB depending on variant) even
  though its specific pretrained weights can't be used per §8.
- **Datasets worth evaluating for training-data licensing**: Ballroom, Extended
  Ballroom, GTZAN rhythm annotations, Hainsworth, GiantSteps tempo — standard MIR
  beat/tempo research datasets. Check each one's actual license before committing
  to it, per §8 — don't assume "commonly used in papers" means "commercially
  clear."

---

## 10. Open questions left to this project's judgment

These are genuinely this project's design decisions, not requirements from the
consuming side:

- Final packaging choice — hand-written WASM (Rust/`wasm-bindgen`, C++/Emscripten)
  vs. ONNX Runtime Web vs. TensorFlow.js as the actual inference runtime. Any of
  these satisfy §2 as long as the external contract is TS-callable and
  browser-native.
- Model architecture specifics (CNN vs. CRNN vs. something else) — informed by
  §3/§6/§7's requirements, not dictated here.
- Training data sourcing and augmentation strategy, subject to §8's licensing
  constraint.

---

## 11. Summary checklist

- [ ] TypeScript-callable, browser-native WASM module — no server, no Python runtime
- [ ] Streaming (chunk-fed) tempo/beat/phase detection with confidence
- [ ] Offline/batch mode for complete pre-loaded audio
- [ ] Confidence well-calibrated, not just accuracy-optimized (§6)
- [ ] Evaluated against SMC dataset + Osu2MIR, standard MIR metrics reported
- [ ] Genre-robust across the ~40–220 (up to ~300) BPM range
- [ ] Size budget respected (reference: 0.1–4.7MB), off-main-thread if inference
      cost requires it
- [ ] 100% commercially-clear licensing on every training input — verified, not assumed
