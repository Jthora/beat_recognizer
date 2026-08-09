# Architecture & Licensing Research — Recommendation

> Companion research doc to `WASM_BEAT_MODULE_SPEC.md`. Answers the two open
> questions that document deliberately left to this project's judgment (§10):
> which inference/postprocessing architecture to build, and where license-clean
> training data actually comes from. Written 2026-08-09.

---

## 1. Bottom-line recommendation

**Architecture**: a small causal CRNN (or causal TCN) *learned front-end*
producing a frame-wise beat/downbeat activation curve, paired with a
**hand-written, deterministic** comb-filter/autocorrelation tempo-and-phase
estimator as the postprocessing/tracking layer — not a learned DBN, not a
particle filter borrowed from existing code. This is exactly the "hybrid of
probabilistic/deterministic DSP + a lightweight ML model" shape you described,
and it sidesteps the single biggest licensing trap found below (madmom's DBN
and pretrained weights are NC-licensed; reimplementing the *algorithm* — DBN
or comb-filtering, both are standard published DSP techniques, not
copyrightable in the abstract — in your own Rust code is clean; using madmom's
actual weights or a package that bundles madmom is not).

**Runtime**: hand-rolled Rust (`wasm-bindgen`) for both the small learned
front-end and the deterministic tracking layer, rather than pulling in ONNX
Runtime Web or TensorFlow.js. At the model sizes this task needs (likely
well under 1MB), a generic inference runtime's own baseline WASM payload
(often several MB before your model even loads) is the thing that blows the
size budget in §5 of the spec, not the model itself.

**Training data**: none of the standard MIR beat-tracking datasets have
full-length, commercially-licensed *audio* — this is the main finding below,
and it changes the practical plan. The fix is a two-tier split: source
license-clean full tracks from curated CC-BY/CC0 subsets of FMA and
MTG-Jamendo (training audio), bootstrap beat labels for them via a
teacher-model pipeline (see §4), and keep SMC/Osu2MIR/GiantSteps/Ballroom/
Hainsworth/GTZAN exactly where the spec already puts them — **evaluation
only**, never training, which turns out to resolve what otherwise looks like
a direct conflict between spec §7 and §8.

---

## 2. Architecture comparison

| Approach | Causal / streaming? | Model type | Postprocessing | License risk | Verdict |
|---|---|---|---|---|---|
| **BeatNet** (Heydari et al., ISMIR 2021) | Yes (online mode) | Causal CRNN, frame-wise beat/downbeat activation | Sequential Monte Carlo particle filter (online) / DBN (offline) | Architecture pattern is publishable/reproducible from the paper; the *reference implementation and its trained weights* are not something to depend on directly — verify BeatNet's own repo license and training data before reusing anything but the architectural idea | **Use as the architectural template**, reimplement from scratch |
| **TCN-based beat tracking** (Eurasip 2019 and follow-ups) | Yes, with causal dilated convolutions | Causal TCN | Varies | Architecture description is public/reimplementable | Solid alternative to CRNN for the learned front-end — TCNs parallelize better than RNNs and are often easier to keep small and fast in WASM (no recurrent state to manage across calls) |
| **Real-Time Zero-Latency (PLP-based)** (TISMIR 2024) | Yes, genuinely zero added latency | Non-learned: windowed-sinusoid kernel extrapolation over an activation function | None needed — the kernel-splitting trick *is* the tracker | The activation function it runs on top of, in the paper, is madmom's pretrained LSTM (Ballroom-trained → **CC BY-NC-SA**, not usable) — but the **PLP-splitting technique itself** is DSP, license-free | **Use the postprocessing technique**, feed it your own license-clean learned activation function instead of madmom's |
| **"Beat this!"** (2024, no-DBN) | Not stated as causal in the abstract — treat as offline-oriented (conv+transformer over frequency/time, needs fuller context) | CNN+Transformer hybrid | None — removes DBN entirely, uses a shift-tolerant loss instead | Architecture idea reusable; specific trained weights unverified for license | Good input for the **offline/batch mode** (spec §3 requires this as a separate path from streaming) — the "skip the DBN, use a shift-tolerant loss" idea is directly applicable to your offline analyzer even though the model itself isn't causal |

**Recommended shape, concretely**:

- **Learned front-end** (small, causal): mel-spectrogram or similar low-level
  features → 2-4 causal conv layers → a small causal GRU (BeatNet-style) or a
  causal dilated TCN stack (TCN-style; likely easier to keep deterministic
  and fast frame-by-frame in Rust since there's no hidden recurrent state to
  thread through the streaming API). Output: a beat-likelihood and
  downbeat-likelihood curve at ~50-100Hz. Target size: tens to low hundreds
  of KB of weights — this is a much smaller task than full audio
  classification, and BeatNet's own network is already lightweight.
- **Deterministic tracking layer** (hand-written DSP, not learned): an
  autocorrelation or comb-filter bank over the activation curve for tempo
  periodicity + salience (this doubles as your confidence signal — a sharp,
  unambiguous periodicity peak is high confidence, a flat/noisy
  autocorrelation is honestly low confidence, which is exactly what §6 of
  the spec asks for), plus the PLP-style causal kernel-splitting trick for
  zero-added-latency phase tracking in streaming mode.
- **Offline mode**: rerun the same learned front-end over the full buffer
  (already non-causal-friendly since there's no streaming constraint), then
  a full (non-causal) DBN-equivalent or Viterbi decode over the
  autocorrelation/tempogram for the higher-accuracy batch path the spec
  requires as a separate mode. Write this yourself rather than depend on
  madmom's DBN implementation — the DBN algorithm for beat tracking is
  well-documented in the literature (Böck et al.) and is a standard HMM/DBN
  formulation, not something owned by any one codebase; only madmom's
  *specific trained model files* are NC-licensed, not the technique.
- **Confidence output**: combine (a) the learned activation's own
  peakiness/entropy at the detected beat instants, (b) autocorrelation peak
  salience (how dominant the winning tempo hypothesis is vs. runner-ups),
  and (c) agreement/variance across a short sliding window of estimates.
  Consider a small calibration pass (temperature scaling or isotonic
  regression against held-out data) specifically tuned so the model reports
  low confidence on SMC-style hard material — this is the single
  highest-leverage thing to get right per both spec docs' shared emphasis
  on honest uncertainty over raw accuracy.

---

## 3. WASM runtime comparison

| Option | Baseline WASM payload | Fits a hand-rolled small CRNN/TCN? | License | Verdict |
|---|---|---|---|---|
| **Hand-rolled Rust + `wasm-bindgen`** (no ML runtime dependency, forward pass coded directly) | ~0 beyond your own code | Yes — this is the natural fit for a small, fixed architecture | Fully your own code | **Recommended.** At this model size, a generic runtime is overhead you don't need, and full control over the forward pass makes it straightforward to interleave with the deterministic DSP layer in the same streaming state machine |
| **`tract`** (pure-Rust ONNX inference engine, MIT/Apache-2.0, WASM-compilable) | Small relative to ORT Web/TF.js, but still a general-purpose loader | Yes, if you'd rather train in PyTorch/TF and export ONNX than hand-write the forward pass | Permissive (MIT/Apache-2.0) | Good middle ground if you want a standard training→export workflow without pulling in a heavyweight browser runtime |
| **ONNX Runtime Web** | Multi-MB WASM binary baseline before any model loads | Technically yes, but the runtime itself likely exceeds your entire size budget (§5: reference ceiling 0.1–4.7MB) | Runtime is MIT; fine license-wise, wrong size-wise here | Not recommended for this project's size budget, even though it's a legitimate choice in general |
| **TensorFlow.js (WASM backend)** | Similar multi-MB baseline concern as ORT Web | Same issue | Apache-2.0 | Same verdict as ORT Web |

---

## 4. Training data — the real finding

The spec's §8 licensing requirement and §7's evaluation-dataset list
(SMC, Osu2MIR, plus the "reference material" list in §9: Ballroom, Extended
Ballroom, GTZAN, Hainsworth, GiantSteps) look like they could be used for
both at first glance. **They can't** — and that's not a minor caveat, it
changes the data-sourcing plan. Per-dataset findings:

| Dataset | What's actually licensed | Audio provenance | Usable for training? | Usable for eval? |
|---|---|---|---|---|
| **Ballroom** (ISMIR 2004 tempo contest) | No clear open license found on the audio | Original contest audio, provenance unclear | No | Yes (standard benchmark; verify before any commercial claim rests on it) |
| **Extended Ballroom** | Conflicting claims across sources (one paper cites CC BY 4.0; the dataset's own site was unreachable during this research to confirm directly) | 30s previews | **Unresolved — verify directly at the source before using at all**, per spec §8's own instruction not to assume | Eval-only pending verification |
| **GTZAN** | Copyright status explicitly ambiguous — the creator has stated redistribution permission was never obtained from the original copyright holders | Real commercial recordings | No | Use cautiously, for comparability with published papers only — its provenance issues technically extend to eval use too, so don't lean on it alone |
| **Hainsworth** | No explicit license found; academic-thesis provenance | Real recordings, mixed genre | No | Yes, with the same "verify before relying on it" caveat |
| **GiantSteps Tempo** | Annotations: CC BY-SA 4.0. **Audio: 2-minute previews sourced from Beatport**, governed by Beatport's own ToS, not the CC license | Commercial EDM previews | No | Yes (annotations are fine for scoring; you're not redistributing the audio) |
| **SMC** | No machine-usable license found; it's a purpose-built "hard cases" benchmark | Mixed, adversarially selected for difficulty | No | Yes — this is literally its intended use |
| **Osu2MIR** | Annotation/pipeline code is separately licensed on GitHub (check that repo's own license); the dataset explicitly does **not** bundle or own the audio | Real anime/Vocaloid/game-music tracks, individually copyrighted, fetched by the user separately | No | Yes — same reasoning as GiantSteps, you're scoring against audio you're not redistributing |
| **madmom** (code + pretrained models) | Code: BSD. **Pretrained models/weights (including the LSTM used in the zero-latency PLP paper): CC BY-NC-SA 4.0** | N/A | **No** — cannot ship these weights or anything derived by fine-tuning them in a commercial product | N/A (it's a tool, not a corpus) |

**The pattern**: essentially every standard MIR beat-tracking dataset either
has no clear license on the audio itself, or explicitly restricts it to
non-commercial/research use, or only distributes short previews under a
third party's separate ToS. This is normal for the field — these datasets
were built for research comparability, not commercial redistribution — but
it means treating them as a training-data source would violate spec §8
regardless of how useful they look. **This is exactly the kind of thing §8
says to verify before training on, not after** — confirmed here before any
training work starts, not discovered mid-project.

**Where license-clean training audio actually is**:

- **FMA (Free Music Archive) dataset** — 106,574 full-length tracks total,
  but licensed per-track; a curated commercial-use subset (public domain +
  CC BY + CC BY-SA + Open Audio License + Free Art License tracks) already
  exists as ~8,800 clips / ~531 hours in an existing HuggingFace repackaging.
  This is real volume of genuinely commercially-clean, full-length audio.
- **MTG-Jamendo** — 55,000+ full tracks, Jamendo's whole model is
  commercial-use-oriented royalty-free music, but the dataset itself mixes
  CC license variants (including NC and ND tracks) — **must be filtered to
  CC BY / CC0 / CC BY-SA only** per-track before use, don't take the whole
  dataset as clean by default.
- **Self-commissioned or procedurally-generated tracks** — worth budgeting
  for directly, especially for filling in edge cases the CC catalogs
  under-represent (extreme BPM, non-Western meters, deliberately ambiguous/
  ambient material to specifically train confidence calibration on
  known-hard cases) — 100% clean by construction and you control the ground
  truth exactly.

**None of this audio comes with beat annotations.** The practical path is a
teacher/pseudo-labeling pipeline: run an existing strong offline tracker
(even an NC-licensed one, like madmom's DBN+RNN, or an architecture like
"Beat this!") purely as an **internal annotation tool** over your
license-clean audio, spot-check and hand-correct a sample for quality, then
train your own from-scratch model on (license-clean audio, teacher-generated
labels) pairs. This is standard practice (teacher/student distillation,
pseudo-labeling) and the resulting model and weights are your own — but flag
it for an actual legal sanity check before committing, rather than treating
"the training audio is clean and the labels are just numbers, not the
teacher's weights" as automatically settled. It's the standard reading, not
a guaranteed one.

**The apparent §7/§8 conflict, resolved**: using SMC/Osu2MIR/GiantSteps/etc.
for *evaluation* — running your already-trained model against their audio
locally to report accuracy — doesn't require the same redistribution rights
that *training* on that audio would. You're not shipping their audio or
anything derived from it inside your product; you're just measuring against
a published benchmark, which is how every paper cited in §9 already uses
them. Keep the spec's evaluation plan exactly as written; just don't let
those same datasets bleed into the training set.

---

## 5. Open items worth resolving before training starts

- Confirm Extended Ballroom's actual license directly at the source (site
  was unreachable during this research) before citing it anywhere, even for
  eval.
- Get a real legal read on the teacher/pseudo-labeling approach in §4 before
  relying on it at scale — the reasoning above is the standard industry
  practice, not a substitute for actually checking.
- Decide TCN vs. GRU for the causal learned front-end with a small
  prototype on FMA/Jamendo data before committing — both are reasonable per
  §2, and the real deciding factor (ease of state management in a streaming
  WASM API, actual latency/size on real hardware) is easier to settle
  empirically than to argue from the literature alone.
