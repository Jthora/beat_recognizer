"""Phase 2.5b: sweep the acquired corpus for decode validity, duration/
sample-rate sanity, and near-silence — catching corrupted or unsuitable
files before Phase 3 bootstraps training labels from them. Never deletes
anything; flagged files are quarantined out of the *active* corpus list
(see `load_active_corpus`) while staying on disk and in PROVENANCE.jsonl,
since the provenance manifest is the license audit trail and shouldn't be
edited after the fact.

Uses `ffprobe`/`ffmpeg` (metadata + a 30s volume-detect pass, not a full
decode of every file — full-length analysis of ~4,000 tracks would take
hours for no extra signal versus the first 30 seconds).
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

# Below this, a track is flagged as likely silent/near-silent rather than
# real musical content. -50dB mean volume is well below any audible mix.
SILENCE_MEAN_VOLUME_DB = -50.0
MIN_PLAUSIBLE_DURATION_S = 5.0
MAX_PLAUSIBLE_DURATION_S = 30 * 60  # 30 minutes — generous, catches mislabeled non-music


@dataclass(frozen=True)
class HealthResult:
    file_path: str
    ok: bool
    duration_s: float | None
    sample_rate: int | None
    mean_volume_db: float | None
    reason: str | None  # populated only when ok=False


def _run(cmd: list[str], timeout: int = 60) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.stdout + result.stderr


def probe_file(path: Path) -> HealthResult:
    path_str = str(path)

    probe_out = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=sample_rate",
            "-of",
            "json",
            path_str,
        ]
    )
    try:
        probe_json = json.loads(probe_out)
        duration = float(probe_json["format"]["duration"])
        sample_rate = int(probe_json["streams"][0]["sample_rate"])
    except (json.JSONDecodeError, KeyError, IndexError, ValueError):
        return HealthResult(path_str, False, None, None, None, "ffprobe: undecodable or missing streams")

    if duration < MIN_PLAUSIBLE_DURATION_S:
        return HealthResult(path_str, False, duration, sample_rate, None, f"duration too short ({duration:.1f}s)")
    if duration > MAX_PLAUSIBLE_DURATION_S:
        return HealthResult(path_str, False, duration, sample_rate, None, f"duration implausibly long ({duration:.1f}s)")

    # volumedetect's stats are logged at "info" level -- -v error would
    # suppress them entirely (confirmed live: returned empty output before
    # this fix, silently making every mean_volume_db None).
    volume_out = _run(
        ["ffmpeg", "-v", "info", "-t", "30", "-i", path_str, "-af", "volumedetect", "-f", "null", "-"]
    )
    mean_volume = None
    for line in volume_out.splitlines():
        if "mean_volume:" in line:
            try:
                mean_volume = float(line.split("mean_volume:")[1].strip().split(" ")[0])
            except (IndexError, ValueError):
                pass

    if mean_volume is not None and mean_volume < SILENCE_MEAN_VOLUME_DB:
        return HealthResult(
            path_str, False, duration, sample_rate, mean_volume, f"near-silent (mean {mean_volume:.1f}dB)"
        )

    return HealthResult(path_str, True, duration, sample_rate, mean_volume, None)


def load_provenance_paths(manifest_path: Path, exclude_source_datasets: frozenset[str] = frozenset()) -> list[Path]:
    paths = []
    with manifest_path.open() as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry.get("source_dataset") in exclude_source_datasets:
                    continue
                paths.append(Path(entry["file_path"]))
    return paths


# Freesound Loop Dataset tracks carry confirmed ground-truth BPM already
# (see fetch_freesound_loops.py / data/FREESOUND_LOOPS_LABELS.jsonl) even
# though they're logged into the same shared PROVENANCE.jsonl as every
# other source. Phase 3's DSP-estimate-and-cross-check tiering pipeline is
# for tracks *without* independent ground truth -- running it on these too
# would waste real time re-estimating BPM this project already has a
# better answer for, and could even silently tier a confirmed-BPM loop
# into needs_review if DSP+ensemble happened to disagree with it.
DEFAULT_EXCLUDED_SOURCE_DATASETS = frozenset({"freesound-loops"})


def load_active_corpus(
    manifest_path: Path,
    health_path: Path,
    exclude_source_datasets: frozenset[str] = DEFAULT_EXCLUDED_SOURCE_DATASETS,
) -> list[Path]:
    """The list Phase 3's DSP-bootstrap/ensemble/tiering pipeline should
    actually use: everything in PROVENANCE.jsonl minus anything this sweep
    flagged in CORPUS_HEALTH.jsonl, minus sources with their own confirmed
    ground truth (see DEFAULT_EXCLUDED_SOURCE_DATASETS)."""
    all_paths = load_provenance_paths(manifest_path, exclude_source_datasets)
    if not health_path.exists():
        return all_paths
    flagged = set()
    with health_path.open() as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if not entry["ok"]:
                    flagged.add(entry["file_path"])
    return [p for p in all_paths if str(p) not in flagged]


def sweep(manifest_path: Path, health_path: Path, max_workers: int = 12) -> None:
    paths = load_provenance_paths(manifest_path)
    print(f"Sweeping {len(paths)} files...")

    results: list[HealthResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(probe_file, p): p for p in paths}
        done = 0
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(paths)}")

    health_path.parent.mkdir(parents=True, exist_ok=True)
    with health_path.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    ok_count = sum(1 for r in results if r.ok)
    print(f"\n{ok_count} / {len(results)} files passed ({len(results) - ok_count} flagged)")
    for r in results:
        if not r.ok:
            print(f"  FLAGGED: {r.file_path} -- {r.reason}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/PROVENANCE.jsonl"))
    parser.add_argument("--health", type=Path, default=Path("data/CORPUS_HEALTH.jsonl"))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    sweep(args.manifest, args.health, max_workers=args.workers)
