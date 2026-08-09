"""Phase 2: pull commercial-use-clean tracks from Jamendo via its public API
(https://developer.jamendo.com/v3.0/tracks) — not the MTG-Jamendo research
dataset's bulk shards, deliberately: MTG-Jamendo's own metadata files
(raw.meta.tsv etc.) don't carry per-track license info, and reverse-
engineering an undocumented file layout is exactly the kind of "probably
fine" assumption docs/ARCHITECTURE_AND_LICENSING_RESEARCH.md §4 exists to
avoid. Jamendo's own API returns `license_ccurl` directly from the platform
that actually hosts the licensing terms — the authoritative source.

Requires a free Jamendo API client_id (https://devportal.jamendo.com) —
this is a real external credential only the user can provide, passed via
--client-id or the JAMENDO_CLIENT_ID environment variable. Nothing in this
module can run without it.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import requests

from . import provenance

API_URL = "https://api.jamendo.com/v3.0/tracks/"
PAGE_LIMIT = 200  # Jamendo's documented max per request


@dataclass(frozen=True)
class JamendoTrack:
    track_id: str
    name: str
    license_ccurl: str
    audiodownload: str
    audiodownload_allowed: bool


def fetch_tracks(client_id: str, max_tracks: int = 2000) -> list[JamendoTrack]:
    """Pages through the Jamendo API collecting track metadata. Metadata-only
    — no audio downloaded here — so this is cheap and safe to run freely."""
    tracks: list[JamendoTrack] = []
    offset = 0
    while len(tracks) < max_tracks:
        resp = requests.get(
            API_URL,
            params={
                "client_id": client_id,
                "format": "json",
                "limit": min(PAGE_LIMIT, max_tracks - len(tracks)),
                "offset": offset,
                "include": "licenses",
                "audioformat": "mp32",
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()

        if payload.get("headers", {}).get("status") == "failed":
            raise RuntimeError(f"Jamendo API error: {payload['headers']}")

        results = payload.get("results", [])
        if not results:
            break

        for t in results:
            tracks.append(
                JamendoTrack(
                    track_id=str(t["id"]),
                    name=t.get("name", ""),
                    license_ccurl=t.get("license_ccurl", ""),
                    audiodownload=t.get("audiodownload", ""),
                    audiodownload_allowed=bool(t.get("audiodownload_allowed", False)),
                )
            )
        offset += len(results)
        if len(results) < PAGE_LIMIT:
            break  # last page

    return tracks


def classify_license(track: JamendoTrack) -> str | None:
    return provenance.classify_cc_url(track.license_ccurl)


def report_license_distribution(tracks: list[JamendoTrack]) -> None:
    counts: Counter[str] = Counter()
    outcomes: dict[str, str] = {}
    downloadable: Counter[str] = Counter()

    for t in tracks:
        counts[t.license_ccurl] += 1
        result = classify_license(t)
        outcomes[t.license_ccurl] = result if result else "REJECTED"
        if result and t.audiodownload_allowed and t.audiodownload:
            downloadable[t.license_ccurl] += 1

    print(f"{'license_ccurl':<55} {'count':>7}  {'downloadable':>12}  outcome")
    print("-" * 95)
    for url, count in counts.most_common():
        print(f"{url[:55]:<55} {count:>7}  {downloadable[url]:>12}  {outcomes[url]}")

    clean = [t for t in tracks if classify_license(t)]
    clean_and_downloadable = [t for t in clean if t.audiodownload_allowed and t.audiodownload]
    print(
        f"\n{len(clean)} / {len(tracks)} tracks are commercial-use-clean; "
        f"{len(clean_and_downloadable)} of those have audiodownload_allowed "
        "and an actual download URL (license and download permission are "
        "separate things — a clean license doesn't guarantee the API can "
        "hand you the audio)."
    )


def download_clean_tracks(
    tracks: list[JamendoTrack],
    dest_dir: Path,
    manifest_path: Path,
) -> None:
    """Downloads audio for every track that is both license-clean and has
    audiodownload_allowed. This is the step with real bandwidth/time cost —
    call it only after reviewing report_license_distribution's output."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for t in tracks:
        license_tag = classify_license(t)
        if not license_tag or not t.audiodownload_allowed or not t.audiodownload:
            continue

        out_path = dest_dir / f"{t.track_id}.mp3"
        if not out_path.exists():
            resp = requests.get(t.audiodownload, timeout=60)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)

        provenance.append_entry(
            manifest_path,
            file_path=out_path,
            source_url=t.audiodownload,
            license=license_tag,
            source_dataset="jamendo",
        )
        downloaded += 1

    print(f"Downloaded {downloaded} clean tracks into {dest_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client-id", default=os.environ.get("JAMENDO_CLIENT_ID"))
    parser.add_argument("--dest", type=Path, default=Path("data/jamendo"))
    parser.add_argument("--max-tracks", type=int, default=2000)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Also download audio for clean tracks (real bandwidth cost). "
        "Omit to only fetch metadata and print the license report.",
    )
    args = parser.parse_args()

    if not args.client_id:
        raise SystemExit(
            "No Jamendo API client_id. Get a free one at "
            "https://devportal.jamendo.com and pass --client-id or set "
            "JAMENDO_CLIENT_ID."
        )

    tracks = fetch_tracks(args.client_id, max_tracks=args.max_tracks)
    report_license_distribution(tracks)

    if args.download:
        download_clean_tracks(tracks, args.dest, Path("data/PROVENANCE.jsonl"))
