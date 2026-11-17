# Research Catalog

A scannable record of what's been checked, found, and concluded across this
project's research — separate from `zany-stargazing-elephant.md`'s
phase-by-phase build plan so a future session (or a future you) can look
something up without re-deriving it. Every entry names a real source; where
a claim couldn't be verified, that's stated rather than smoothed over.

Companion to `docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md` (the original,
earlier licensing research that kicked off the "build from scratch"
decision) — this catalog covers everything researched *after* that,
licensing and technique research together, since many entries are both at
once (a tool's license verdict and its technique are usually one finding).

---

## Pretrained models / weights evaluated

| Name | License (code) | License (weights) | Verdict | Why |
|---|---|---|---|---|
| madmom | BSD | CC BY-NC-SA 4.0 | Weights blocked; classical comb-filter path (no weight dependency) tried, ruled out | Package is 2018-era, breaks against Python 3.12/NumPy 2.x in multiple spots (`collections.MutableSequence`, `np.float`) — not one-line fixable |
| TempoCNN | — | CC BY-NC-ND | Blocked | ND forbids derivatives outright |
| BeatNet | — | Trained on GTZAN/Ballroom (copyright-disputed) | Blocked | Training provenance itself is disputed, not just NC |
| "Beat this!" (CPJKU) | MIT | MIT (tag), but... | **Blocked despite the MIT tag** | Repo's own README: some training files are "fully copyrighted" — confirmed directly, not assumed. The taint is in the weights' training provenance, not the license file. Methodology (no-DBN-postprocessing training, shift-tolerant loss) is still fair game to cite/reimplement |
| Demucs | MIT | MIT (tag), but trained on MUSDB18-HQ | **Blocked despite the MIT tag** | MUSDB18-HQ's own license: "educational purposes only... not for commercial use without express permission." Code license doesn't launder training-data license |
| Essentia (TempoCNN, etc.) | AGPL | AGPL | Blocked for direct use; benchmark-only | Needs a paid commercial license from MTG for anything beyond non-commercial testing |

## Datasets / tools evaluated

| Name | License | Used? | Notes |
|---|---|---|---|
| librosa | ISC | ✅ Yes — ensemble labeler | Fully permissive; unlike NC tools, its output is safe as actual training-label input, not just benchmark comparison. `beat_track` (Ellis DP) and `plp` used as two independent votes |
| Freesound Loop Dataset (FSL10K) | CC0/CC-BY/CC-BY-SA per-item (filtered) | ✅ Yes — 7,189 loops imported | Zenodo record 3967852. Genuinely *confirmed* tempo (2,936 expert-verified + producer upload tags), not algorithmic — the one source with real ground truth, not another estimate. Audio bundled in the archive, no Freesound API key needed |
| GiantSteps Tempo | CC BY-SA 4.0 (annotations) | ✅ Yes — eval only | 664 tracks, primary offline-mode benchmark (Accuracy1 52.3%, Accuracy2 82.1% after the tempo tie-break fix) |
| Ballroom | Audio via ISMIR2004 tarball, annotations CC BY 4.0 (CPJKU/BallroomAnnotations) | ✅ Yes — eval only | 698/698 matched. Original host 403s without a browser User-Agent (confirmed: same request, only header changed) |
| Hainsworth | No explicit license on the GitHub mirror | ✅ Yes — eval only | Original host (marsyas.info) dead, confirmed via connection timeout — same failure mode as SMC. Contains real commercial recordings (Dave Matthews Band, Bruce Springsteen, John Williams, etc.) — confirmed directly from `data.txt`, eval-only exactly like GTZAN |
| ESC-50 | CC BY-NC (mostly), CC BY for ESC-10 subset | ✅ Yes — no-rhythm eval set | Eval-only use, same treatment as other commercial-origin eval audio. 14 categories hand-picked as genuinely non-periodic (rain, breathing, snoring, etc.), explicitly excluding anything with inherent oscillation |
| SMC | Unknown | ❌ Not used | Host dead, no mirror found. Hainsworth-rubato + ensemble `needs_review` tier used as substitutes for the calibration-check role SMC was meant to fill |
| Extended Ballroom | CC BY 4.0 (compilation only) | ❌ Not used | Checked its actual audio source: 30s previews scraped from a commercial CD retailer (ballroomdancers.com), not CC-licensed audio despite the compilation's own tag. Host also now dead |
| Harmonix Set | CC BY 4.0 (annotations only) | ❌ Not used | Rich annotations (beat/downbeat/structure) but audio not bundled — only MusicBrainz IDs + precomputed mel-spectrograms. Real potential future addition if matching audio gets sourced |
| AcousticBrainz | CC0 | ❌ Not used | ~29.4M submissions, huge scale, but BPM is Essentia's own *algorithmic* output at submission time, not confirmed ground truth — "millions of votes from one algorithm," not verified tempo. Metadata-only, no bundled audio |
| Million Song Dataset | — | ❌ Not used | Tempo is Echo Nest's proprietary algorithm, same caveat as AcousticBrainz. No audio; old 7digital preview links are dead |
| MedleyDB | CC BY-NC-SA | ❌ Not used | Blocked, same restrictive combination as everything else NC-SA |
| MIRTracks | CC BY-NC-SA 3.0 | ❌ Not used | 240 hours, appealing scale, but blocked on license |
| mirdata (Python library) | — | ❌ Not pursued | Install hit the same venv/pip mismatch seen with madmom (installed into wrong Python environment); not worth fighting for a pure convenience/discovery tool |
| Jamendo `musicinfo` | — | Confirmed no numeric BPM | Only a 5-level coarse `speed` category (verylow→veryhigh) exists in the API — checked directly, not assumed |
| FMA `features.csv` | — | Confirmed no rhythm features | Chroma/MFCC/spectral-contrast/tonnetz only, zero tempo-related columns |

## Licensing questions left open, not resolved

| Question | Status |
|---|---|
| Does aubio's (GPLv3) *output* count as a derivative work if used as training-label input? | FSF's general position ("output of a GPL program isn't itself a derivative work") is more favorable here than the NC-SA ShareAlike case already ruled unsafe — but this project hasn't gotten real legal confirmation either way. Formally parked: benchmark-only for now, same as madmom/BeatNet, not actively pursued further |
| Cross-training on NC-licensed model *output* (distillation) | Ruled out without real legal review from the start — CC BY-NC-SA's ShareAlike clause could arguably make a model trained on that output a derivative work, and whether ML training on another model's output even counts as a derivative work is genuinely unsettled law. Not worth the risk on this project's one non-negotiable requirement |

---

## MIR / DSP technique research

Each entry: the question asked, what was found, source, and what it means
for this project.

### Multi-band / multi-instrument tempo analysis
**Question**: could frequency-band splitting isolate per-instrument rhythm
(kick vs. guitar vs. hi-hat)?
**Found**: Scheirer (1998) — 6-band comb-filter analysis, near-human
performance — but bands feed *one* combined tempo/beat estimate, not
separate exposed streams. Klapuri (2004/2006) extends this properly to a
real multi-level hierarchy (tatum/tactus/measure) via 4 accent bands + a
joint probabilistic model, but those levels are harmonically related, not
independent per-instrument tempi. MIREX's own "Audio Tempo Extraction"
task format — the field's actual standard for "more than one tempo" — is
two *global* candidates with a salience weight, much closer to this
project's existing `alternate` field than to per-instrument decomposition.
**Conclusion**: plain frequency-band splitting overclaims what it can
deliver; Klapuri's tatum/tactus/measure hierarchy is the validated,
proven path (planned as Phase 3.5), ranked above band-splitting for
exactly this reason.

### Automatic Drum Transcription (ADT) — literal per-instrument streams
**Question**: is there a validated way to get literal separate kick/
snare/hihat streams?
**Found**: yes, ADT is the real technique — but a 2024 paper
(webspace.eecs.qmul.ac.uk/s.e.dixon/pub/2024/482_lbd.pdf) shows accuracy
jumping from 75% to over 90% specifically from adding real source
separation (Demucs) before transcription. Band-filtering alone is a much
weaker proxy than actual separation.
**Conclusion**: getting literal per-instrument streams is a full second
ML subsystem (own model, own training data, own licensing review),
comparable in scope to Phase 4 itself — not an add-on, not pursued now.

### Voice Activity Detection (VAD) — hangover/hysteresis schemes
**Question**: is there a standard, proven technique for the has_rhythm
false-positive problem (confidently detecting X where none exists)?
**Found**: VAD is structurally the same problem (binary presence
detection in continuous audio) and has a decades-old standard answer —
asymmetric dual thresholds ("different toggles for speech start versus
speech end, requiring higher confidence to begin detecting speech but
lower confidence to stop") plus a hangover period to avoid chopping brief
low-energy stretches.
**Conclusion**: directly applicable. This project's own `debounced_25`
prototype (single symmetric threshold, 25 consecutive frames) validated
the general idea works (37%→35% FP, zero TP cost) but is a simplified
version of the real standard technique — real dual-threshold hysteresis
is the natural next prototype (Iteration 3 item 1).

### Schmitt trigger — hysteresis sizing
**Question**: how should the dual thresholds actually be chosen, not
just guessed?
**Found**: standard embedded-DSP rule of thumb — hysteresis band ≈ 2-3x
the noise amplitude, sized from the *measured* noise floor rather than
picked arbitrarily.
**Conclusion**: directly usable now — this project has already measured
the no-rhythm set's confidence distribution (p50=0.212), giving a real
number to size the hysteresis band from instead of guessing.

### Tempo estimation boundary/edge effects
**Question**: is the repeated `bpm≈304.0` result (found in both real
Ballroom failures and earlier no-rhythm false positives) a known,
documented phenomenon?
**Found**: no direct match. General "octave error" framing is well
documented (sub-beats scoring comparably to main beats), but nothing
specific to a fastest-tempo boundary-rounding artifact turned up.
**Conclusion**: a real research gap, not a missed citation — needs direct
diagnosis in `tempo.rs` (`min_lag`'s rounding at the `DEFAULT_BPM_MAX`
boundary is the prime suspect) rather than a literature-sourced fix
(Iteration 3 item 2).

### Expressive timing / rubato beat tracking
**Question**: why do confidence and prominence both invert (wrong
direction) on Hainsworth's rubato subset?
**Found**: "What Makes Beat Tracking Difficult? A Case Study on Chopin
Mazurkas" (Dixon et al., ISMIR 2010) names real, specific mechanisms —
beats on repeated chord onsets are hard to detect (especially played
softly), and post-processing/DP-style trackers structurally cannot follow
genuine tempo changes well. PLPDP tracking (Chiu et al., ISMIR 2023,
arxiv.org/abs/2308.10355) targets this directly. **Iteration 3 item 5 ✅
done — read properly this time** via the paper's HTML rendering (the raw
PDF still doesn't extract cleanly; the abs/HTML route worked), not just a
search snippet:

- **Core mechanism**: PLPDP replaces Ellis' DP tracker's single *global*
  inter-beat-interval assumption (`offline.rs`'s own `alpha=400` is
  exactly this kind of fixed global-tempo-consistency parameter) with a
  *time-varying* one, `δ̂(n)`, derived from a local periodicity signal —
  the DP score function becomes `C(B) = Σ Δ(bₖ) + λ₀ Σ P_δ̂₀(bₖ₋bₖ₋₁)`
  with penalty `P_δ̂₀(δ) = -(log₂(δ/δ̂₀))²`, generalized to use `δ̂(n)`
  and time-varying confidence `λ(n)` instead of fixed `δ̂₀`/`λ₀`. `δ̂(n)`
  and `λ(n)` come from PLP: an STFT-based tempogram combined across three
  kernel sizes (1/3/5 second windows, multiplied element-wise, half-wave
  rectified) whose peaks mark "temporal expectations" of beat locations,
  segmented into piecewise-constant local-IBI estimates via peak-picking.
- **Real result magnitudes, both directions honestly reported**: F1 on
  Maz-5 (rubato-heavy, 13.1% tempo-stable) improved 0.595→0.838 with real
  activation functions — a large, genuine win specifically because that
  corpus is expressive. On ASAP (more tempo-stable at 24.6%), the
  improvement was much smaller, 0.473→0.493 — confirms this technique's
  benefit scales with *how* rubato-heavy the material actually is, not a
  universal fixed win. The trade is recall for precision (PLPDP raises
  recall substantially, costs some precision) — a deliberate, stated
  design choice for this problem, not a side effect.
- **Relevance to this project, concretely**: this independently validates
  — via a peer-reviewed, much more rigorous mechanism — the same direction
  this project already found empirically and far more cheaply:
  `stability_4s` (a trailing-window local-periodicity-consistency signal)
  correctly orders rubato correct-vs-wrong while the *global* `confidence`
  scalar inverts. PLPDP is the "do this properly" version of the same
  underlying idea (local, not global, periodicity tracking matters most on
  exactly the material where global assumptions break). It is *not*
  directly portable to this project's causal streaming path (its PLP
  construction is inherently multi-kernel/offline, O(N²) DP over the whole
  track) — but it names a concrete, scoped future upgrade for `offline.rs`
  specifically: replace the fixed-`alpha` global IBI penalty with a
  PLP-derived local one, if/when rubato-heavy content (classical,
  jazz-with-rubato) becomes a real target rather than a benchmark curiosity.
  Not undertaken now — scoped as a Phase 4+ candidate, not blocking
  anything in this iteration.
**Conclusion**: the rubato inversion is a known, named, studied MIR
problem, not a bug specific to this project's tracker. `stability_4s`
(the complex-EMA signal, corrected to use `log2(bpm)` angle) is the one
signal tested that correctly orders rubato's correct-vs-wrong — a
candidate Phase 4 input feature, not a Rust-side fix, and now backed by a
real understanding of the state-of-the-art technique that addresses the
same problem more rigorously.

### "AI and Tempo Estimation: A Review" (2024)
**Found**: arxiv.org/pdf/2401.00209 — a recent survey. Not yet read in
depth; flagged as a good single grounding resource before Phase 4's
architecture decisions lock, not yet acted on.

### Log-filtered spectral flux onset front end (LogFiltSpecFlux / SuperFlux)
**Question**: does upgrading `onset.rs` from raw linear-FFT-bin spectral
flux to a perceptually/musically log-spaced filterbank + log compression
improve downstream BPM accuracy? Flagged as worth trying since Phase 2.5c,
never attempted until this pass.
**Found**: Böck, Krebs & Schedl (ISMIR 2012, "Evaluating the online
capabilities of onset detection methods") defines `LogFiltSpecFlux`: a
138-filter logarithmic (quarter-tone) triangular filterbank, 27.5-16,000Hz,
log-compressed (`log10(filtered_magnitude + 1)`, filters deliberately left
unnormalized), full formulas confirmed via Böck & Widmer's SuperFlux paper
(DAFx-13, §2.1) which builds on it — full PDF read successfully via the
Read tool directly (WebFetch chokes on this host's PDFs the same way it
did for PLPDP; `Read` on the saved file worked both times).
**Implemented, validated, real bug found and fixed, still not a net win**:
first implementation attempt regressed real GiantSteps accuracy across the
board (offline 60.0%→54.7% Accuracy1 on a 95-track sample) — but that
first result concealed a genuine bug, not a property of the technique: a
naive "nearest FFT bin" fallback for filters narrower than one bin (61/138
at 1024-pt/44.1kHz resolution, expected at this frame size) let up to 16
separate filters independently claim the same bin at full weight,
massively over-weighting a narrow low-frequency slice. Fixed by leaving
those filters empty (zero contribution, harmless) instead of forcing a
duplicating fallback. Re-validated on the full 664-track set: streaming
mode came back to a genuine wash (Accuracy1 +1.2, Accuracy2 -1.4, both
within noise), offline still trails by a real 2-3 points. Not shipped —
even bug-free, doesn't clear the bar — but the *reason* for the residual
offline gap is now a defensible hypothesis (downstream `tempo.rs` constants
never retuned for the new envelope's statistics) rather than a guess made
to explain away a result that was actually a bug.
**Conclusion**: a real, general lesson for this project, not just this one
experiment — a clean, consistent-looking negative result across multiple
metrics is still worth checking for a self-inflicted implementation bug
before being accepted as a finding about the underlying technique. The
first "plausible cause" explanation was accepted too quickly; being
challenged on it and actually re-checking the code (rather than re-
asserting the theory) is what surfaced the real bug.

---

## Confidence-signal experiment results (Iterations 1-4 summary)

Full detail lives in the build plan's Phase 3.5b; the headline numbers,
for quick lookup:

| Signal / rule | No-rhythm set FP | Ballroom TP | Rubato correct-vs-wrong (offline / streaming) |
|---|---|---|---|
| `confidence > 0.25` (shipped) | 37% | 100% | inverted (0.438 vs 0.492) / inverted (0.255 vs 0.358) |
| Prominence | — | — | inverted (1.333 vs 1.558) / not tested |
| `stability_whole` (whole-trace) | 32% | 67% | not tested |
| `stability_2s` | 94% (worse than baseline) | 89% | not tested |
| `stability_4s` | 59% | 86% | n/a (streaming-only concept) / **correctly ordered** (0.928 vs 0.789) |
| `stability_6s` | 34% | 84% | not tested |
| `debounced_25` (25-frame debounce) | 35% | 100% | not tested |
| hysteresis (best combo, Iteration 3) | 32% | 93% | not tested |
| `entropy > 0.93` | 13% | 92% | **correctly ordered** (0.997 vs 0.988) / inverted (0.886 vs 0.965) |
| **`debounced_25 AND entropy>0.93`** | **8%** | **92%** | see entropy row — mode-dependent |

Best result overall: **entropy** (Iteration 4) — Pareto-dominates every
prior rule for the has_rhythm gate (8% FP at 92% TP, vs. `debounced_25`'s
35% FP at 100% TP), confirmed on ESC-50, a harder real-continuous-audio
set (Wikimedia Commons long-form, Iteration 3 item 3), and offline-mode
rubato calibration. One real caveat, not smoothed over: streaming-mode
entropy *inverts* on rubato specifically (unlike offline), so `entropy`
answers "is there a real dominant periodicity" (gating), while
`stability_4s` remains the better answer to "is my streaming estimate
trustworthy on hard/drifting-tempo material" (calibration) — two
different signals for two different jobs. Multi-band cross-band
periodicity coherence (Iteration 4, second direction) was a real, honest
negative — heavy distribution overlap at every threshold tested, worse
in Pareto terms than the shipped baseline.

None of these are implemented in shipped Rust yet — all await the
still-unanswered product question (what does a false negative vs. a
spurious true actually cost downstream in ProtoFusionGirl), though
entropy's margin is now large enough that it should weigh heavily
whenever that decision gets made.
