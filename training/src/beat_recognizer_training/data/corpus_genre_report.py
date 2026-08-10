"""Phase 2.5b: characterize the acquired Jamendo corpus's genre/tag
diversity using the track IDs already in PROVENANCE.jsonl -- no new
download needed, just richer metadata for tracks already on disk.

Note: some tracks may no longer be queryable via the API even though our
local copy (and its provenance entry, recorded at download time) is fine --
confirmed live during development that at least one already-downloaded
track ID returns zero results, most likely delisted from Jamendo's catalog
since acquisition. Handled as a normal, expected gap, not an error.

Separately -- and this is the more surprising finding -- Jamendo's batched
`id=X+Y+Z` filter is itself flaky at their API level: the *exact same*
request, repeated a couple seconds apart, was observed live to return 3
results one time and 0 the next. Not a bug in this script, not a rate
limit (no 429), just an inconsistent backend. Retrying a suspiciously-empty
batch a few times resolves it in practice.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import requests

API_URL = "https://api.jamendo.com/v3.0/tracks/"
BATCH_SIZE = 25
BATCH_DELAY_S = 0.3
MAX_RETRIES = 4


def load_jamendo_track_ids(manifest_path: Path) -> list[str]:
    ids = []
    with manifest_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["source_dataset"] == "jamendo":
                ids.append(Path(entry["file_path"]).stem)
    return ids


def fetch_batch(client_id: str, batch: list[str]) -> dict[str, dict]:
    """A single batch query, retried if the response is suspiciously empty
    -- Jamendo's id=X+Y+Z batch filter was observed live to be flaky
    (identical requests returning 0 results one call and correct results
    the next), not a rate limit and not a real "these are all delisted"
    signal."""
    for attempt in range(MAX_RETRIES):
        resp = requests.get(
            API_URL,
            params={
                "client_id": client_id,
                "format": "json",
                "id": "+".join(batch),
                "include": "musicinfo",
                "limit": BATCH_SIZE,
            },
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            return {track["id"]: track.get("musicinfo", {}) for track in results}
        time.sleep(0.5 * (attempt + 1))
    return {}  # genuinely empty after retries -- accept it, likely all delisted


def fetch_musicinfo(client_id: str, track_ids: list[str]) -> dict[str, dict]:
    """Returns {track_id: musicinfo dict} for whichever of track_ids the API
    still has metadata for."""
    info: dict[str, dict] = {}
    for i in range(0, len(track_ids), BATCH_SIZE):
        batch = track_ids[i : i + BATCH_SIZE]
        info.update(fetch_batch(client_id, batch))
        print(f"  {min(i + BATCH_SIZE, len(track_ids))}/{len(track_ids)} queried")
        time.sleep(BATCH_DELAY_S)
    return info


def report(track_ids: list[str], info: dict[str, dict]) -> None:
    genre_counts: Counter[str] = Counter()
    speed_counts: Counter[str] = Counter()
    missing = 0

    for tid in track_ids:
        musicinfo = info.get(tid)
        if musicinfo is None:
            missing += 1
            continue
        for genre in musicinfo.get("tags", {}).get("genres", []):
            genre_counts[genre] += 1
        speed = musicinfo.get("speed", "")
        if speed:
            speed_counts[speed] += 1

    print(f"\n{len(track_ids)} total tracks, {missing} with no longer queryable metadata "
          f"({missing / len(track_ids):.1%})\n")

    print("Genre distribution (tracks can have multiple genre tags):")
    for genre, count in genre_counts.most_common(30):
        print(f"  {genre:<20} {count:>5}")

    print("\nSpeed category distribution:")
    for speed, count in speed_counts.most_common():
        print(f"  {speed:<20} {count:>5}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-id", default=os.environ.get("JAMENDO_CLIENT_ID"))
    parser.add_argument("--manifest", type=Path, default=Path("data/PROVENANCE.jsonl"))
    args = parser.parse_args()

    if not args.client_id:
        raise SystemExit("No Jamendo API client_id. Pass --client-id or set JAMENDO_CLIENT_ID.")

    track_ids = load_jamendo_track_ids(args.manifest)
    print(f"Found {len(track_ids)} Jamendo track IDs in {args.manifest}")
    info = fetch_musicinfo(args.client_id, track_ids)
    report(track_ids, info)
