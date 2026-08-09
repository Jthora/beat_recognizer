# Why Music Matters in ProtoFusionGirl

> Vision and motivation document, written for the team building the custom beat-
> recognizer WASM module (see the companion `WASM_BEAT_MODULE_SPEC.md` for the
> technical requirements). This doc is about *why* — what the beat detector you're
> building actually powers, and why it needs to be good. Self-contained; no prior
> context on ProtoFusionGirl assumed. Written 2026-08-09.

---

## The one-sentence version

Music isn't a soundtrack in this game — it's cosmology. The pitch, in the team's
own words: as if music is the active psionic frequency of the entire domain,
defining how everything moves and interacts. Not a mood-setter playing in the
background. An input the simulated world actually listens to, continuously, in
real time.

## The premise this sits inside

ProtoFusionGirl isn't built around a win condition. There's no score to optimize,
no completion state — closer to SimCity, RimWorld, or Stellaris than a
level-by-level game. The player is an ASI (an artificial intelligence, referred to
in-fiction as PsiSys) observing and lightly guiding an autonomous protagonist,
Jane, through an ongoing simulation she's training inside. The game's own pitch
already says "you can literally just let the game run and watch like a movie" —
the whole point is that the world keeps being alive and worth watching on its own,
not that you're racing toward an ending.

Music is one of the two real, live signals (the other is actual astrology — real
planetary positions, computed from an ephemeris) that drive that ongoing-aliveness.
Astrology is the slow "climate" — what kind of day the world is having, computed
once and changing gradually. Music is the fast "weather" — whatever's actually
playing right now, reshaping the moment-to-moment texture of the simulation live.
This document is about the music half specifically.

---

## Every place music actually touches the game

### 1. The world itself pulses with it

Ley-line energy pulses, ambient particle density, the ASI's own HUD glow, terrain
motion — these are meant to sync directly and continuously to the beat, with no
filtering or interpretation in between. Not "the environment reacts to music" as a
mechanic — the environment *is* partly made of the music, the same way it's made
of weather or terrain. This is the most literal expression of the "psionic
frequency" pitch: turn music on, and the physical fabric of the simulated world
visibly moves with it.

### 2. Jane moves through a world that has rhythm — she isn't puppeted by it

This distinction matters and was deliberately designed in: Jane, the protagonist,
is explicitly *not* a puppet — one of the strongest, most repeated ideas in this
game's design is that she has her own will and her own AI, and the game is worse
if she ever stops feeling like she's choosing things. So her animation timing
(footstep cadence, an attack's exact trigger moment) gets subtly retimed to match
the music — cosmetic polish, not a decision — but *what she does and when she
decides to do it* stays entirely on her own logic. She's a person moving through a
world with a pulse, not a marionette on a beat.

### 3. Companion robots express personality *through* music — differently, each one

This is where music becomes characterization, not just atmosphere. The game has a
roster of companion AI robots — Ra, Aero, Zoe, Thor — and each one is meant to have
a genuinely different relationship with whatever's playing, derived from who they
already are:

- **Ra** — a fire-elemental, self-expressive, "happy-go-lucky" companion who
  pilots/embodies the Magneto Speeder. She *loves* fast, high-energy, aggressive
  music, and she shows it: pulsing faster through the air in time with the beat,
  literally "galloping" to the rhythm like a steed, timing her booster-rocket bursts
  and aerial tricks to land on the beat — showing off, for the fun of it, when the
  music is the kind she loves. When the music doesn't match her energy, she still
  moves with it, just visibly more subdued — going through the motions instead of
  genuinely into it.
- **Aero** — quick, air-elemental, event-oriented. Reacts sharply to sudden
  onsets and drops with quick discrete maneuvers rather than sustained grooving.
- **Zoe** — water-elemental, other-focused. Her engagement with music is
  supportive rather than showy — steadying the *team's* mood and rhythm rather
  than performing for herself.
- **Thor** — earth-elemental, steady and grounded. Responds to a song's overall
  sustained character rather than instant hits — slow to shift, powerful and
  immovable once he's locked into a groove.

The deeper reason this matters more than it might first seem: these companion
robots are explicitly "nodes" of PsiSys — the same distributed AI consciousness
the player *is*. A robot grooving to the music isn't an NPC being animated by an
external system. It's arguably the player's own attention/consciousness,
distributed into a body, resonating with what it's actually tuned into. That's the
emotional core this feature is really serving — it's supposed to feel like *you*,
not like scenery.

### 4. The story itself can respond to what's playing

A weighted-selection system (not a scripted trigger list) reads both the music and
the astrology signal and can surface narrative beats, mission-flavor shifts, or
world-mood changes that land naturally with what's happening musically — a moment
timed to arrive with a song's chorus, an encounter's intensity tracking a track's
energy. Think of it as a lightweight "director" AI in the sense RimWorld or Left 4
Dead use that word — pacing and selection, not scripted cutscenes.

### 5. Where the music actually comes from — the real ambition

The long-term goal isn't "the game plays its own soundtrack and detects its own
beat" (though that's the safe, real starting point). It's **whatever the player is
actually listening to becomes part of their session** — their own local music
files, eventually their Spotify session, eventually literally whatever's playing
on their computer. There's even a version of this inverted entirely: the player
taps out a rhythm themselves, and the game's own procedural score follows *them* —
the player becomes the metronome instead of the other way around. The throughline
across every version: the player's actual musical world, whatever form it takes,
becomes part of the simulation's pulse. That's the ambition your beat detector is
in service of — not a fixed, curated playlist, but genuinely arbitrary,
player-supplied audio, reliably.

---

## This isn't a rhythm-game gimmick bolted onto the world — it's made of the same material as the rest of it

This is worth knowing because it changes how much this feature actually matters to
the game's identity, not just its polish. The game has its own detailed cosmology
— a symbolic/mathematical framework (built on real geometry: a tetrahedron,
octahedron, and icosahedron realizing progressively richer structure) that
underlies its magic system, its elemental combat, and its astrology. Independent
research into that framework turned up something the team didn't expect going in:
**the mathematical structure underneath real acoustic sound — the physical
axes that describe any sound at all (its onset sharpness, its tonal-vs-noisy
character, its spectral spread) — turns out to be the *exact same* three-axis
structure as the game's magic-system primitives and its elemental-combination
system.** Not a loose metaphor. The same axes, independently confirmed, showing up
in acoustics research the same way they show up in the game's own invented
cosmology. There's even a real, table-backed correspondence between the musical
circle of fifths and the game's zodiac-based astrology system — a song's actual
musical key genuinely maps onto the same structure as a real planetary position.

Practically, this means: getting music detection right isn't decorating an
otherwise-unrelated game with a music feature. It's completing a structure the
rest of the game already half-describes. That's the actual stakes here.

---

## Why "good enough most of the time" isn't good enough — the honesty point

The companion spec document goes into the technical requirements (confidence
scores, evaluation datasets, licensing), but the *reason* those requirements exist
is worth stating plainly: **a wrong-but-confident beat estimate is worse than an
honest "this is unclear."** Concretely — if Ra commits visibly to a booster-rocket
burst timed to a beat that isn't actually there, in a moment that's supposed to
feel alive and responsive, it reads as broken, not charming. If the detector
instead recognizes "this section is ambiguous" and eases off, nothing breaks —
the moment just quietly doesn't happen, which is invisible rather than jarring.
That's why the spec asks for well-calibrated confidence over squeezing out the
last percent of raw accuracy: the failure mode that actually hurts this game is
confident wrongness, not honest uncertainty.

---

## The short version, if you only read one paragraph

Music in this game is meant to be felt as the world's actual pulse — something the
environment, the player's companions, and the unfolding story all genuinely listen
to, not a track playing underneath a game that would be identical on mute. It has
to work on real, arbitrary, player-supplied audio, not a curated demo playlist,
because the whole point is that *your* music becomes part of *your* session. And
it has to fail honestly when it's unsure, because in a game built around a world
that's supposed to feel alive and responsive, a confident wrong answer is more
damaging than an honest shrug.
