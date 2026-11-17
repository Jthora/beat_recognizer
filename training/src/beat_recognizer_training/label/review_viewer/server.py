"""Local review UI: a human can't mentally "math" a BPM number into a
verdict, but is very good at noticing whether a visual pulse lines up with
what they're hearing -- this serves a page that plays a track's original
(unmodified) audio next to a waveform with the detected beat times overlaid
as a marker that flashes in sync with playback. Complements (doesn't
replace) the audio-only click_track.py overlay -- some misalignments are
easier to catch by eye, some by ear, so both stay available per track.

Stdlib-only (http.server), no new dependency, and deliberately not meant to
be exposed beyond localhost -- there's no auth on the /verdict write path.
"""

from __future__ import annotations

import csv
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from beat_recognizer_training.label.dsp_teacher import load_labels

STATIC_DIR = Path(__file__).parent / "static"
# .../training/src/beat_recognizer_training/label/review_viewer/server.py -> repo root,
# so defaults resolve the same way whether this is launched from the repo root or from
# training/ (both are common -- everything else in this package is run from training/).
REPO_ROOT = Path(__file__).resolve().parents[5]
VERDICT_COLUMN = "verdict (fill in: ok/bad/unsure)"


def load_manifest(checklist_path: Path, labels_path: Path) -> list[dict]:
    labels_by_path = {r.audio_path: r for r in load_labels(labels_path)}
    tracks = []
    with checklist_path.open(newline="") as f:
        for row in csv.DictReader(f):
            record = labels_by_path.get(row["audio_path"])
            tracks.append(
                {
                    "id": row["click_track_file"],
                    "group": row["group"],
                    "audio_path": row["audio_path"],
                    "bpm": float(row["bpm"]) if row["bpm"] else None,
                    "confidence": float(row["confidence"]) if row["confidence"] else None,
                    "alternate_bpm": float(row["alternate_bpm"]) if row["alternate_bpm"] else None,
                    "librosa_bt_bpm": float(row["librosa_bt_bpm"]) if row.get("librosa_bt_bpm") else None,
                    "librosa_plp_bpm": float(row["librosa_plp_bpm"]) if row.get("librosa_plp_bpm") else None,
                    "agree_count": row.get("agree_count") or None,
                    "ensemble_agrees": row.get("ensemble_agrees") == "True" if row.get("ensemble_agrees") else None,
                    "beat_times_s": record.beat_times_s if record else [],
                    "verdict": row.get(VERDICT_COLUMN, ""),
                }
            )
    return tracks


def save_verdict(checklist_path: Path, track_id: str, verdict: str) -> None:
    with checklist_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        if row["click_track_file"] == track_id:
            row[VERDICT_COLUMN] = verdict

    with checklist_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_handler(checklist_path: Path, labels_path: Path, review_dir: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            pass  # keep stdout quiet; this runs on every audio byte range request

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                self._serve_file(STATIC_DIR / "index.html", "text/html")
            elif parsed.path == "/manifest.json":
                self._send_json(load_manifest(checklist_path, labels_path))
            elif parsed.path.startswith("/audio/"):
                self._serve_audio(unquote(parsed.path[len("/audio/") :]), parse_qs(parsed.query))
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if urlparse(self.path).path == "/verdict":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                save_verdict(checklist_path, body["id"], body["verdict"])
                self._send_json({"ok": True})
            else:
                self.send_error(404)

        def _serve_audio(self, track_id: str, query: dict) -> None:
            mode = query.get("mode", ["original"])[0]
            if mode == "click":
                path = review_dir / track_id
            else:
                manifest = {t["id"]: t for t in load_manifest(checklist_path, labels_path)}
                entry = manifest.get(track_id)
                if entry is None:
                    self.send_error(404)
                    return
                path = Path(entry["audio_path"])

            if not path.exists():
                self.send_error(404)
                return
            self._serve_file(path, mimetypes.guess_type(str(path))[0] or "application/octet-stream", allow_range=True)

        def _serve_file(self, path: Path, content_type: str, allow_range: bool = False) -> None:
            data = path.read_bytes()
            total = len(data)
            range_header = self.headers.get("Range") if allow_range else None

            if range_header and range_header.startswith("bytes="):
                start_s, _, end_s = range_header[len("bytes=") :].partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else total - 1
                end = min(end, total - 1)
                chunk = data[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
                return

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(total))
            if allow_range:
                self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def run(checklist_path: Path, labels_path: Path, review_dir: Path, port: int, open_browser: bool = True) -> None:
    for path, label in [(checklist_path, "--checklist"), (labels_path, "--labels"), (review_dir, "--review-dir")]:
        if not path.exists():
            raise SystemExit(f"{label} path does not exist: {path}")

    handler = make_handler(checklist_path, labels_path, review_dir)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as e:
        raise SystemExit(
            f"Couldn't bind to port {port} ({e}). Another review viewer is probably "
            f"already running -- reuse that tab, or pass --port to run a second one."
        ) from e

    url = f"http://127.0.0.1:{port}"
    print(f"Review viewer at {url}  (Ctrl+C to stop)")
    if open_browser:
        import threading
        import webbrowser

        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checklist", type=Path, default=REPO_ROOT / "data/review_batch/checklist.csv")
    parser.add_argument("--labels", type=Path, default=REPO_ROOT / "data/LABELS.jsonl")
    parser.add_argument("--review-dir", type=Path, default=REPO_ROOT / "data/review_batch")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't auto-open a browser tab")
    args = parser.parse_args()

    run(args.checklist, args.labels, args.review_dir, args.port, open_browser=not args.no_browser)
