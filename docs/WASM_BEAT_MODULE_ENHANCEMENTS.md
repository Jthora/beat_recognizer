# Musical Beat Recognizer — Enhancement Proposals (v1 Addendum)

> Companion document to **`WASM_BEAT_MODULE_SPEC.md`** (the actual requirements
> contract) and **`MUSIC_VISION.md`** (the creative rationale). Read those first —
> this doc assumes them. Written 2026-08-10, after `WASM_BEAT_MODULE_SPEC.md` was
> already handed off and work was underway.
>
> **This is a wishlist to triage, not a scope change.** Nothing here modifies the
> v1 contract (§4/§11 of the spec doc) you're already building against. Ship v1 as
> specified. Everything below is "if any of this is cheap given the architecture
> you land on, here's what it would unlock" — ranked by our best guess at cost, so
> you can make the actual call once you know your real training-data and
> architecture constraints better than we do from the outside.

---

## 1. Why this doc exists

Once v1 lands we'll be integrating a genuinely custom-trained model instead of a
generic library, which means — for the first time — the shape of what it outputs
isn't fixed by someone else's pretrained API. That's worth a second look at what
would actually help, beyond the baseline `BeatEstimate` shape in the original spec.
This pass tries to be honest about cost, not just describe what would be nice.

---

## 2. Tier 1 — plausibly near-free, given the architecture v1 already needs

These are things a beat/downbeat-tracking model plausibly produces as a byproduct
of its own internal representation, not a new training objective. Flagging them
as candidates; you're the ones who'll actually know if this is true for whatever
architecture you land on.

- **Downbeat detection, promoted from "nice to have" to "actually wanted."**
  Already in v1 §3 as non-blocking, and BeatNet (your own §9 reference
  architecture) already does this jointly with beat tracking. Real consumer:
  the Director (a weighted-selection system, see `MUSIC_VISION.md` #4) wants to
  land "big" moments on bar boundaries, not just any beat.
- **Per-beat salience, not just overall tempo confidence.** `BeatEstimate.confidence`
  answers "how sure are we of the tempo." A separate "how strong was *this*
  beat" score would compose directly with an existing real consumer:
  `CompanionActionDirector.onBeat()` already rolls `P(action) = baseProbabilityPerBeat
  * matchScore` per beat — a salience term slots into that multiplication cleanly.
  If your architecture already produces a beat-activation curve internally (most
  CRNN beat trackers do), exposing its peak value at each detected beat may be
  literally free.
- **Explicit "no discernible rhythm" state, distinct from low confidence.**
  Not "confidence: 0.05" (a number that could mean "barely detected" or
  "genuinely nothing periodic here") — an actual third state. Ties directly to
  `MUSIC_VISION.md`'s honesty principle (§"Why 'good enough most of the time'
  isn't good enough") — silence, talk radio, and pure noise are a materially
  different case from "hard but real music," and downstream consumers may want
  to react differently (fall back to a default ambiance rather than sit on a
  stale number).
- **Tempo-octave alternate hypothesis, exposed rather than resolved internally.**
  We already hand-flag real tracks as ambiguous (`musicBpmTable.ts`'s
  `octaveAmbiguous` field, e.g. a track we measured as either 76 or 152 BPM
  depending on which beat you count). If your model internally resolves this
  ambiguity to pick one answer, exposing *both* candidates + relative confidence
  instead costs little and lets each consumer choose per-context (fast flight vs.
  a walk cycle might want opposite octaves off the same audio).

---

## 3. Tier 2 — real new capability, real cost — propose only if v1's timeline and licensing discipline have headroom

These likely need their own training signal/data, not just a different read of
what beat-tracking already computes. Don't let any of these threaten v1's actual
deliverable or its commercially-clean-licensing requirement (spec §8) — that
requirement is the entire reason this project exists; scope creep here would be
self-defeating.

- **Continuous energy/intensity envelope** (loudness/dynamics over time, distinct
  from tempo confidence). Real consumer: `RealityWeights.computeMusicEnergy()`
  currently *approximates* this from bpm + harmonicity as a heuristic stand-in,
  because nothing measures it directly today.
- **Lightweight structural-boundary event** — not full verse/chorus/drop
  labeling (that's a separate XL research spike on our side, explicitly not
  committed scope), just "something changed significantly right here." Cheaper
  than real structure analysis, still enough to gate a Director "big moment"
  trigger per `MUSIC_VISION.md` #4.
- **Groove/swing estimate** (straight subdivision vs. swung/triplet feel).
  Grounded consumer: Ra "galloping to the rhythm" (`MUSIC_VISION.md` #3) reads
  as mechanical if the underlying subdivision doesn't match the genre's actual
  feel.
- **Look-ahead beat buffer — offline/batch mode only.** Since batch mode already
  sees the complete file, exposing "next N beats' predicted timestamps" for
  pre-loaded/uploaded tracks lets a scripted flourish pre-stage instead of
  reacting after the fact. Explicitly does **not** apply to streaming/live
  input — there's no future to look ahead into on genuinely live audio, so this
  only helps the file-upload path (`FileUploadSource.ts`, already built on our
  side), not the general case.

---

## 4. Deliberately not asking you for

Stating these explicitly so they don't get silently re-proposed later:

- **Onset/transient (general-purpose "how percussive is this") detection.**
  v1 §3 already scopes this out correctly — a beat-tracker's internal onset
  detection is tuned for periodicity, not the same thing as a general spectral-
  flux feature. ✅ Fixed on our side (2026-08-10) — Meyda's own `spectralFlux`
  extractor turned out to have two real bugs, not one (the confirmed
  upstream-bug plumbing issue, plus a separate time-domain/negative-index bug
  in its actual formula) — replaced with a hand-rolled, correct half-wave-
  rectified spectral flux over Meyda's own working `amplitudeSpectrum`
  feature. No ML, no training data, no licensing exposure, confirmed live
  with zero regressions. Was never your problem to solve, and now isn't a gap
  either.
- **Key/chord/mode detection.** ✅ Also done on our side (2026-08-10) —
  `keyDetection.ts`/`harmonicAnalysis.ts`, Krumhansl-Schmuckler over chroma.
  Already out of scope per v1 §3, staying that
  way — it's a separate system on our side (chroma features +
  Krumhansl-Schmuckler), independent of tempo/beat entirely.
- **Tempo-stability / "locked-in groove" signal.** This is a rolling-window
  statistic over ordinary `BeatEstimate` history (how steady has tempo been the
  last N seconds) — fully computable on our side from data v1 already provides.
  No reason to spend your model budget on it.
- **Frequency-band-weighted beat source** (bass/kick vs. higher percussion).
  Speculative — invented while brainstorming, not grounded in any existing
  design doc or code today. Cut until there's an actual consumer for it.

---

## 5. Integration-contract details worth confirming now, independent of any Tier 2 item

- **Clock domain**: `BeatEstimate.nextBeatTimeMs` should be in the same clock
  domain this side already uses (`AudioContext.currentTime`/`performance.now()`),
  or the translation needs to be explicit. Worth pinning down before it becomes
  a hard-to-diagnose drift bug later.
- Everything else in v1's API contract (§4) stands as-is — `reset()` on
  track-switch, streaming + offline as two distinct required paths, confidence
  on every estimate — no changes proposed here.

---

## 6. How to use this doc

Triage Tier 1 against whatever architecture you actually land on — if any of it
falls out for free, great; if not, it's not required. Tier 2 is genuinely
optional and only worth considering if it doesn't compromise v1's timeline or
§8's licensing discipline. §4 is us pre-committing to *not* asking for those,
so effort doesn't get spent on them by mistake. The actual contract remains
`WASM_BEAT_MODULE_SPEC.md` §11's checklist.
