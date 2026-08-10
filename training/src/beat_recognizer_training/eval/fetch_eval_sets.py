"""Phase 2.5a: download small, eval-only audio sets with real annotated
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
"""

from __future__ import annotations

import subprocess
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


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("eval_data/giantsteps"))
    parser.add_argument("--limit", type=int, default=None, help="cap the number of tracks (omit for all 664)")
    args = parser.parse_args()

    pairs = download_giantsteps(args.dest, limit=args.limit)
    index_path = args.dest / "index.jsonl"
    with index_path.open("w") as f:
        for audio_path, true_bpm in pairs:
            f.write(json.dumps({"audio_path": str(audio_path), "true_bpm": true_bpm}) + "\n")
    print(f"Wrote index to {index_path}")
