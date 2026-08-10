# Corpus characterization (Phase 2.5b)

Real findings from sweeping and sampling the acquired Jamendo corpus
(3,975 tracks / 287.8 hours, see `PROVENANCE.jsonl`) — before Phase 3
bootstraps training labels from it. Nothing here is inferred or assumed;
every number was produced by actually running the tools against the real
files.

## Health sweep (`training/data/verify_corpus.py`)

3,954 / 3,975 files (99.5%) passed decode/duration/silence checks.
21 flagged and quarantined into `data/CORPUS_HEALTH.jsonl` (excluded from
`load_active_corpus`, not deleted):

- 6 undecodable/corrupted files
- 1 near-silent file
- 14 files with implausible duration (33-94 minutes — DJ sets/mixes/podcasts
  uploaded as a single "track", not usable as single-song training examples)

## Genre/tag distribution (`training/data/corpus_genre_report.py`)

Pulled via Jamendo's `musicinfo` API field for the already-downloaded track
IDs (0.7% no longer queryable — likely delisted since acquisition, not an
error). Top genres by tag count (tracks can carry multiple tags):

rock 846, experimental 540, electronic 498, pop 408, punk 92, ambient 89,
metal 83, techno 63, reggae 51, classical 47, jazz 47, hardcore 41,
chansonfrancaise 40, triphop 34, folk 33, hiphop 33, indie 30, dub 28,
industrial 25, ... world 18, funk 17.

**Real gap, now documented rather than assumed**: rock/experimental/
electronic/pop dominate (846/540/498/408); genres more associated with
syncopation/rubato/complex meter (classical, jazz, world) sit at 47/47/18 —
a small fraction of the corpus. No explicit non-Western genre categories
appear at all. Spec §3's genre-agnostic robustness requirement will likely
need supplemental sourcing or synthetic augmentation to cover this, per the
build plan's existing risk-register mitigation.

Coarse tempo-feel ("speed") tag distribution: medium 2800 (70%), high 364,
verylow 264, veryhigh 183, low 124 — skews heavily toward "medium."

## Tempo distribution + confidence (300-track sample, `training/data/tempo_distribution.py`)

Run with the Phase 2.5c-retuned tracker (not the original), via the same
`beat-dsp-cli` used in the real-audio harness:

| BPM bucket | count |
|---|---|
| <70 | 51 |
| 70-90 | 23 |
| 90-110 | 63 |
| 110-130 | 59 |
| 130-150 | 53 |
| 150-170 | 26 |
| 170-200 | 8 |
| >200 | 17 |

Detected tempo is genuinely spread across the range, not narrowly
clustered — a better result than the "everything at 120-130" risk flagged
earlier in planning.

Two findings worth carrying into Phase 3's design directly:

- **Only 53% of sampled tracks (159/300) have confidence > 0.5.** Roughly
  half this corpus is hard for a DSP-only tracker — plausibly explained by
  the genre mix (experimental/ambient are the 2nd/6th most-tagged genres).
  Phase 3's confidence-filter should expect to drop or flag close to half
  the corpus, not a small minority.
- **73.3% of sampled tracks had an octave-alternate hypothesis exposed.**
  Notably higher than what showed up on GiantSteps (EDM, mostly clean 4/4).
  Either real, varied music is genuinely this often octave-ambiguous, or
  `ALTERNATE_TOLERANCE` (0.85) is somewhat loose for non-EDM material —
  worth Phase 3's uncertainty-driven review specifically checking a sample
  of these flagged cases, since it'll encounter a lot of them.
