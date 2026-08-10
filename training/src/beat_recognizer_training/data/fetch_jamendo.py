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
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import requests

from . import provenance

API_URL = "https://api.jamendo.com/v3.0/tracks/"
PAGE_LIMIT = 200  # Jamendo's documented max per request
PAGE_DELAY_S = 0.3  # pacing between pagination requests, to avoid tripping
# the rate limit in the first place rather than just reacting to it
MAX_RETRIES = 5


def _get_with_retry(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """GET with retry/backoff on 429 — pulling ~25,000 tracks across ~125
    pages tripped Jamendo's rate limit on a real run with no retry logic at
    all; confirmed live, not a hypothetical edge case."""
    delay = 1.0
    for attempt in range(MAX_RETRIES + 1):
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 429 and attempt < MAX_RETRIES:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else delay
            print(f"  rate limited, waiting {wait:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(wait)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("unreachable")  # loop always returns or raises above


@dataclass(frozen=True)
class JamendoTrack:
    track_id: str
    name: str
    license_ccurl: str
    audiodownload: str
    audiodownload_allowed: bool
    duration_s: int


def fetch_tracks(client_id: str, max_tracks: int = 2000) -> list[JamendoTrack]:
    """Pages through the Jamendo API collecting track metadata. Metadata-only
    — no audio downloaded here — so this is cheap and safe to run freely."""
    tracks: list[JamendoTrack] = []
    offset = 0
    while len(tracks) < max_tracks:
        resp = _get_with_retry(
            API_URL,
            params={
                "client_id": client_id,
                "format": "json",
                "limit": min(PAGE_LIMIT, max_tracks - len(tracks)),
                "offset": offset,
                "include": "licenses",
                "audioformat": "mp32",
            },
        )
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
                    duration_s=int(t.get("duration", 0)),
                )
            )
        offset += len(results)
        if len(results) < PAGE_LIMIT:
            break  # last page
        time.sleep(PAGE_DELAY_S)

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
    clean_hours = sum(t.duration_s for t in clean_and_downloadable) / 3600
    yield_rate = len(clean_and_downloadable) / len(tracks) if tracks else 0
    print(
        f"\n{len(clean)} / {len(tracks)} tracks are commercial-use-clean; "
        f"{len(clean_and_downloadable)} of those have audiodownload_allowed "
        "and an actual download URL (license and download permission are "
        "separate things — a clean license doesn't guarantee the API can "
        f"hand you the audio).\n{clean_hours:.1f} hours of downloadable clean "
        f"audio in this sample ({yield_rate:.1%} yield) — at this rate, "
        f"reaching N hours needs roughly querying ~{1 / yield_rate:.0f}x as "
        "many tracks as hours-equivalent needed, given Jamendo's catalog "
        "is far larger than this sample."
    )


def download_clean_tracks(
    tracks: list[JamendoTrack],
    dest_dir: Path,
    manifest_path: Path,
) -> None:
    """Downloads audio for every track that is both license-clean and has
    audiodownload_allowed. This is the step with real bandwidth/time cost —
    call it only after reviewing report_license_distribution's output.

    Resumable and fault-tolerant: skips files already recorded in the
    manifest (so re-running after an interruption doesn't re-download or
    double-log), and a single track's download failure is logged and
    skipped rather than aborting the whole batch — a real 500 from
    Jamendo's storage backend on one track (out of ~1000) killed an entire
    run before this existed."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    already_logged = provenance.load_logged_paths(manifest_path)
    downloaded = 0
    skipped_existing = 0
    failed = 0

    for t in tracks:
        license_tag = classify_license(t)
        if not license_tag or not t.audiodownload_allowed or not t.audiodownload:
            continue

        out_path = dest_dir / f"{t.track_id}.mp3"
        if str(out_path) in already_logged:
            skipped_existing += 1
            continue

        try:
            if not out_path.exists():
                resp = _get_with_retry(t.audiodownload, params={}, timeout=60)
                out_path.write_bytes(resp.content)

            provenance.append_entry(
                manifest_path,
                file_path=out_path,
                source_url=t.audiodownload,
                license=license_tag,
                source_dataset="jamendo",
            )
            downloaded += 1
        except (requests.exceptions.RequestException, ValueError) as e:
            print(f"  skipping track {t.track_id} after error: {e}")
            failed += 1

    print(
        f"Downloaded {downloaded} clean tracks into {dest_dir} "
        f"({skipped_existing} already done, {failed} failed and were skipped)"
    )


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
