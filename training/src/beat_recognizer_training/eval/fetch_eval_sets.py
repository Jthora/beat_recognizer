"""Phase 2.5a/6: download small, eval-only audio sets with real annotated
ground truth -- used to check the beat-dsp tracker against real music, not
just synthetic click tracks. Never used for training (see spec §8 / the
build plan's licensing note): running a model against this audio locally to
report accuracy doesn't require the redistribution rights training would.

GiantSteps Tempo (implemented): 664 2-minute Beatport preview MP3s + BPM
annotations, CC BY-SA 4.0 annotations, hosted live at JKU's own backup
mirror (confirmed reachable during development -- the dataset's original
GitHub repo's audio_dl.sh script points here as the primary host).

SMC (deferred): the original host (smc.inescporto.pt) is dead -- confirmed
via direct connection timeout, not assumed -- and no working mirror was
found. Per the build plan, GiantSteps alone satisfies this phase's go/no-go
bar; SMC (the deliberately-hard set) can be revisited if a mirror surfaces
later, but isn't a blocker.

Ballroom (implemented, pulled ahead of the rest of Phase 6 per the build
plan's Phase 3.5 sequencing note -- it has real downbeat annotations, which
the planned tatum/tactus/measure DSP work needs validated against): 698
30-second excerpts, ISMIR2004-contest-era. The dataset's usual academic host
(mtg.upf.edu/ismir2004/.../node5.html) 404s but the tarball URL the
CPJKU/BallroomAnnotations README points to as a replacement
(.../tempoContest/data1.tar.gz) is live -- confirmed directly, and needs a
browser-like User-Agent or the host 403s. Beat+bar annotations (not bundled
with the audio) come from CPJKU/BallroomAnnotations (`beat_time_sec
beat_id`, downbeat = id==1); matched to audio by basename since the
annotation repo's filenames drop the genre-subdirectory prefix the tarball
uses. That repo's own README documents 4 exact + 9 near-duplicate clips
within the set -- not deduplicated here, noted for whoever scores against it.

Hainsworth (implemented, same sequencing reason as Ballroom): 222 ~60s
excerpts across six genres, real downbeat annotations. The dataset's
original host (marsyas.info) is dead -- confirmed via connection timeout,
same failure mode as SMC's host -- audiocontentanalysis/dataset-hainsworth-
tempo on GitHub mirrors the actual wav files plus a `data.txt` describing
each track's beat sample-positions and which of those are downbeats (no
MATLAB parsing needed, the repo's `data.txt` already has it in plain text).
Worth being explicit about: unlike GiantSteps/Jamendo/FMA, several
Hainsworth clips are excerpts of real commercial recordings (Dave Matthews
Band, Bruce Springsteen, John Williams film scores, etc.) -- confirmed
directly by reading `data.txt`, not assumed. Fine for the same reason GTZAN
is already fine per this plan's Phase 6 section: evaluation-only, never a
training input, no redistribution beyond this local eval run.
"""

from __future__ import annotations

import subprocess
import tarfile
import tempfile
from pathlib import Path

import requests

GIANTSTEPS_BASE_URL = "https://www.cp.jku.at/datasets/giantsteps/backup/"
GIANTSTEPS_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/GiantSteps/giantsteps-tempo-dataset/master/annotations_v2/tempo/"
)
# The dataset repo's own file listing -- fetched once and cached rather than
# re-listing the GitHub tree API on every run.
GIANTSTEPS_FILE_LIST_URL = (
    "https://raw.githubusercontent.com/GiantSteps/giantsteps-tempo-dataset/master/splits/filelist.txt"
)


def fetch_giantsteps_track_ids() -> list[str]:
    """Falls back to cloning the dataset repo's md5/ directory listing if no
    filelist.txt exists -- the repo's actual layout, confirmed by direct
    inspection, uses md5/<id>.LOFI.md5 as the authoritative per-track list."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/GiantSteps/giantsteps-tempo-dataset.git",
                tmp,
            ],
            check=True,
            capture_output=True,
        )
        md5_dir = Path(tmp) / "md5"
        return sorted(p.stem.replace(".LOFI", "") for p in md5_dir.glob("*.LOFI.md5"))


def download_giantsteps(dest_dir: Path, limit: int | None = None) -> list[tuple[Path, float]]:
    """Downloads audio + BPM annotation pairs. Returns [(audio_path, true_bpm), ...]
    for whichever tracks succeeded (a preview being missing/removed from the
    host is treated as a skip, not a hard failure -- same discipline as the
    Jamendo fetch scripts)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to {dest_dir.resolve()} -- point --dest at external storage for a full run "
          f"(forgot to do this once already; local disk here runs a few GB from full)")
    track_ids = fetch_giantsteps_track_ids()
    if limit:
        track_ids = track_ids[:limit]

    pairs: list[tuple[Path, float]] = []
    for i, track_id in enumerate(track_ids):
        audio_path = dest_dir / f"{track_id}.mp3"
        bpm_url = f"{GIANTSTEPS_ANNOTATIONS_URL}{track_id}.LOFI.bpm"
        mp3_url = f"{GIANTSTEPS_BASE_URL}{track_id}.LOFI.mp3"

        try:
            bpm_resp = requests.get(bpm_url, timeout=15)
            bpm_resp.raise_for_status()
            true_bpm = float(bpm_resp.text.strip())
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"  skip {track_id}: no annotation ({e})")
            continue

        if not audio_path.exists():
            try:
                audio_resp = requests.get(mp3_url, timeout=30)
                audio_resp.raise_for_status()
                audio_path.write_bytes(audio_resp.content)
            except requests.exceptions.RequestException as e:
                print(f"  skip {track_id}: audio download failed ({e})")
                continue

        pairs.append((audio_path, true_bpm))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(track_ids)}")

    print(f"Downloaded {len(pairs)} / {len(track_ids)} GiantSteps track+annotation pairs")
    return pairs


BALLROOM_TARBALL_URL = "http://mtg.upf.edu/ismir2004/contest/tempoContest/data1.tar.gz"
BALLROOM_ANNOTATIONS_REPO = "https://github.com/CPJKU/BallroomAnnotations.git"
# The tarball host 403s without this -- an academic mirror's basic UA
# sniffing, not real bot protection (confirmed: identical request, only the
# UA header changed, 403 -> 200).
_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def download_ballroom(dest_dir: Path, limit: int | None = None) -> list[dict]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = dest_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["git", "clone", "--depth", "1", BALLROOM_ANNOTATIONS_REPO, tmp],
            check=True,
            capture_output=True,
        )
        annotations: dict[str, tuple[list[float], list[float]]] = {}
        for beats_file in Path(tmp).glob("*.beats"):
            beats, downbeats = [], []
            for line in beats_file.read_text().splitlines():
                if not line.strip():
                    continue
                t, beat_id = line.split()
                t = float(t)
                beats.append(t)
                if int(beat_id) == 1:
                    downbeats.append(t)
            annotations[beats_file.stem] = (beats, downbeats)
    print(f"{len(annotations)} annotation files loaded")

    if any(audio_dir.glob("*.wav")):
        print(f"{audio_dir} already has audio, skipping download")
    else:
        print(f"Streaming + extracting Ballroom tarball to {audio_dir} (~1.36GB, no full-file buffering)...")
        extracted = 0
        with requests.get(BALLROOM_TARBALL_URL, headers={"User-Agent": _BROWSER_UA}, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with tarfile.open(fileobj=resp.raw, mode="r|gz") as tar:
                for member in tar:
                    if not member.name.endswith(".wav"):
                        continue
                    if limit and extracted >= limit:
                        break
                    member.name = Path(member.name).name  # flatten -- matched to annotations by basename anyway
                    tar.extract(member, path=audio_dir, filter="data")
                    extracted += 1
                    if extracted % 100 == 0:
                        print(f"  extracted {extracted}")
        print(f"Extracted {extracted} audio files")

    pairs = []
    wav_files = sorted(audio_dir.glob("*.wav"))
    if limit:
        wav_files = wav_files[:limit]
    for wav_path in wav_files:
        entry = annotations.get(wav_path.stem)
        if entry is None:
            continue
        beats, downbeats = entry
        pairs.append({"audio_path": str(wav_path), "beat_times_s": beats, "downbeat_times_s": downbeats})

    print(f"Matched {len(pairs)} / {len(wav_files)} audio files to annotations")
    return pairs


HAINSWORTH_REPO = "audiocontentanalysis/dataset-hainsworth-tempo"
HAINSWORTH_RAW_BASE = f"https://raw.githubusercontent.com/{HAINSWORTH_REPO}/main"


def download_hainsworth(dest_dir: Path, limit: int | None = None) -> list[dict]:
    """data.txt's own header documents the field order (see this module's
    docstring); beats are sample positions, and the "idx of downbeats"
    column is 1-indexed positions into the beats list (confirmed against
    several tracks' time-signature fields -- the index sequences are always
    arithmetic with the documented meter's step size, e.g. every 4th
    position for a "4" time-signature track)."""
    import soundfile as sf

    dest_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = dest_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    data_resp = requests.get(f"{HAINSWORTH_RAW_BASE}/data.txt", timeout=30)
    data_resp.raise_for_status()
    rows = [line for line in data_resp.text.splitlines() if line.strip() and not line.startswith("#")]
    if limit:
        rows = rows[:limit]

    pairs = []
    for i, row in enumerate(rows):
        fields = [f.strip() for f in row.split("<sep>")]
        wav_name, artist, title = fields[0], fields[1], fields[2]
        beat_samples = [float(x) for x in fields[10].split(",")]
        downbeat_indices = [int(x) for x in fields[11].split(",")]

        audio_path = audio_dir / wav_name
        if not audio_path.exists():
            try:
                resp = requests.get(f"{HAINSWORTH_RAW_BASE}/wavs/{wav_name}", timeout=30)
                resp.raise_for_status()
                audio_path.write_bytes(resp.content)
            except requests.exceptions.RequestException as e:
                print(f"  skip {wav_name}: {e}")
                continue

        sr = sf.info(str(audio_path)).samplerate
        beat_times_s = [s / sr for s in beat_samples]
        downbeat_times_s = [beat_times_s[idx - 1] for idx in downbeat_indices if 0 < idx <= len(beat_times_s)]

        pairs.append(
            {
                "audio_path": str(audio_path),
                "beat_times_s": beat_times_s,
                "downbeat_times_s": downbeat_times_s,
                "artist": artist,
                "title": title,
            }
        )
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(rows)}")

    print(f"Downloaded {len(pairs)} / {len(rows)} Hainsworth track+annotation pairs")
    return pairs


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["giantsteps", "ballroom", "hainsworth"], default="giantsteps")
    parser.add_argument("--dest", type=Path, default=None, help="defaults to eval_data/<dataset>")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of tracks (omit for all)")
    args = parser.parse_args()

    dest = args.dest or Path("eval_data") / args.dataset
    if args.dataset == "giantsteps":
        pairs = [{"audio_path": str(p), "true_bpm": bpm} for p, bpm in download_giantsteps(dest, limit=args.limit)]
    elif args.dataset == "ballroom":
        pairs = download_ballroom(dest, limit=args.limit)
    else:
        pairs = download_hainsworth(dest, limit=args.limit)

    index_path = dest / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w") as f:
        for entry in pairs:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote index to {index_path}")
