from __future__ import annotations

import csv
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from ..live_buffer import read_live_status


def read_history_csv(run_dir: Path) -> dict[str, np.ndarray]:
    """Full-run series from the CSVs, keyed like the live payload.

    New-format runs already use per-GPU columns (gpu1_voltage_v, ...). Legacy
    single-GPU runs (voltage_v / current1_a / current2_a) are mapped onto one
    synthetic "gpu1" so old runs replay in the multi-GPU dashboard.
    """
    paths = sorted(Path(run_dir).glob("*.csv"))
    if not paths:
        return {"time_s": np.empty(0, dtype=float)}
    with open(paths[0], newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), [])
    legacy = "voltage_v" in header
    if legacy:
        columns = ["time_s", "voltage_v", "current1_a", "current2_a", "total_power_w"]
    else:
        columns = [name for name in header if name == "time_s" or name.endswith(("_voltage_v", "_current_a", "_power_w"))]
    values: dict[str, list[float]] = {name: [] for name in columns}
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    parsed = [float(row[name]) for name in columns]
                except (KeyError, TypeError, ValueError):
                    continue
                for name, value in zip(columns, parsed):
                    values[name].append(value)
    arrays = {key: np.asarray(value, dtype=float) for key, value in values.items()}
    if legacy:
        arrays = {
            "time_s": arrays["time_s"],
            "gpu1_voltage_v": arrays["voltage_v"],
            "gpu1_current_a": arrays["current1_a"] + arrays["current2_a"],
            "gpu1_power_w": arrays["total_power_w"],
            "total_power_w": arrays["total_power_w"],
        }
    return arrays

WEB_DIR = Path(__file__).resolve().parent
INDEX_PATH = WEB_DIR / "index.html"

# Default display settings; overridden by whatever the CLI passes to serve().
DEFAULT_DISPLAY = {
    "window_sec": 60.0,
    "refresh_ms": 50,
    "stale_after_sec": 2.0,
    "fullscreen": False,
    "replay": False,
    "y_limits": {
        "voltage": [11.5, 13.0],
        "current": [-1.0, 15.0],
        "power": [-10.0, 180.0],
    },
}

# Cap on points served by /api/history. Sized so 1x replay of a long run still
# has visibly smooth traces (~50k points ≈ a few MB of JSON, loads in <1 s
# locally); the full-resolution record stays in the CSVs.
HISTORY_MAX_POINTS = 50000


def _downsample(values: np.ndarray, max_points: int) -> np.ndarray:
    if values.size <= max_points:
        return values
    idx = np.linspace(0, values.size - 1, max_points).astype(int)
    return values[idx]


def build_latest_payload(run_dir: Path, stale_after_sec: float) -> dict:
    """JSON-able snapshot of the most recent live window for the dashboard."""
    run_dir = Path(run_dir)
    status = read_live_status(run_dir, stale_after_sec)
    payload: dict = {
        "state": status.state,
        "generated_at_epoch": status.generated_at_epoch,
        "server_epoch": time.time(),
        "latest_sample_time_s": status.latest_sample_time_s,
        "elapsed_s": status.latest_sample_time_s,
        "sample_count": status.sample_count,
        "run_avg_power_w": status.run_avg_power_w,
        "run_peak_power_w": status.run_peak_power_w,
        "energy_j": status.energy_j,
    }
    npz = run_dir / "latest.npz"
    if npz.exists():
        with np.load(npz) as data:
            for key in data.files:
                payload[key] = np.asarray(data[key], dtype=float).tolist()
    return payload


def build_config_payload(run_dir: Path, display: dict) -> dict:
    """Static run/display info the page fetches once on load."""
    run_dir = Path(run_dir)
    title = run_dir.name
    sample_rate_hz = None
    started_at = None
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            title = data.get("measurement_name") or title
            sample_rate_hz = data.get("sample_rate_hz")
            started_at = data.get("started_at")
        except (ValueError, OSError):
            pass
    return {
        "run_id": run_dir.name,
        "title": title,
        "sample_rate_hz": sample_rate_hz,
        "started_at": started_at,
        "window_sec": display.get("window_sec", DEFAULT_DISPLAY["window_sec"]),
        "refresh_ms": display.get("refresh_ms", DEFAULT_DISPLAY["refresh_ms"]),
        "stale_after_sec": display.get("stale_after_sec", DEFAULT_DISPLAY["stale_after_sec"]),
        "fullscreen": display.get("fullscreen", DEFAULT_DISPLAY["fullscreen"]),
        "replay": display.get("replay", DEFAULT_DISPLAY["replay"]),
        "y_limits": display.get("y_limits", DEFAULT_DISPLAY["y_limits"]),
    }


def build_history_payload(run_dir: Path, max_points: int = HISTORY_MAX_POINTS) -> dict:
    """Downsampled full run from the CSVs (for REPLAY scrubbing — fast follow)."""
    arrays = read_history_csv(Path(run_dir))
    return {key: _downsample(value, max_points).tolist() for key, value in arrays.items()}


def _make_handler(run_dir: Path, display: dict, activity: dict | None = None):
    stale_after_sec = float(display.get("stale_after_sec", DEFAULT_DISPLAY["stale_after_sec"]))
    activity = activity if activity is not None else {"last_poll": 0.0}

    class DashboardHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, path: Path, content_type: str) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # The page must never be cached: a stale index.html means a tab
            # left open keeps running old dashboard code against a new run.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            route = urlparse(self.path).path
            try:
                if route in ("/", "/index.html"):
                    self._send_file(INDEX_PATH, "text/html; charset=utf-8")
                elif route == "/api/latest":
                    activity["last_poll"] = time.time()
                    self._send_json(build_latest_payload(run_dir, stale_after_sec))
                elif route == "/api/config":
                    activity["last_poll"] = time.time()
                    self._send_json(build_config_payload(run_dir, display))
                elif route == "/api/history":
                    self._send_json(build_history_payload(run_dir))
                else:
                    self._send_json({"error": "not found"}, status=404)
            except ConnectionError:
                # The browser tab closed mid-response (Windows: WinError
                # 10053/10054, POSIX: broken pipe). Routine, not an error —
                # and nothing can be sent back on a dead socket.
                pass
            except Exception as exc:  # keep the server alive on a bad read
                try:
                    self._send_json({"error": str(exc)}, status=500)
                except ConnectionError:
                    pass

        def log_message(self, *args) -> None:  # silence per-request console spam
            pass

    return DashboardHandler


class _ExclusiveHTTPServer(ThreadingHTTPServer):
    """HTTP server that refuses to share its port.

    On Windows, SO_REUSEADDR (stdlib default) lets a second server bind an
    already-serving port with no error — the old process keeps receiving all
    traffic, so every dashboard silently shows whatever run that stale server
    was opened on. Binding exclusively turns that into a loud OSError, which
    serve() answers by hopping to the next free port.

    On POSIX, SO_REUSEADDR is kept: there it does not allow double-binding a
    listening port, and without it a restart within TIME_WAIT (~60 s) of the
    previous run would fail for no user-visible reason.
    """

    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request, client_address) -> None:
        # Backstop for disconnects that happen outside the handler's own
        # try/except (e.g. while the request line is still being read): a
        # closing browser tab is routine and must not spew tracebacks.
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, ConnectionError):
            return
        super().handle_error(request, client_address)


def create_server(host: str, port: int, handler, max_tries: int = 10):
    """Bind the dashboard server to `port`, or the next free port after it.

    Returns (httpd, bound_port). A busy port usually means a dashboard from an
    unfinished (possibly another user's) session is still running there.
    """
    last_err: OSError | None = None
    for candidate in range(port, port + max_tries):
        try:
            httpd = _ExclusiveHTTPServer((host, candidate), handler)
        except OSError as err:
            last_err = err
            continue
        if candidate != port:
            print(f"[WARN] Port {port} is in use (a dashboard from another session?) - "
                  f"using port {candidate} instead.")
        return httpd, candidate
    raise OSError(
        f"No free dashboard port in {port}-{port + max_tries - 1} "
        f"(is a stale dashboard hogging them? last error: {last_err})"
    )


def _wait_for_existing_client(activity: dict, wait_sec: float = 1.5) -> bool:
    """True if a dashboard tab is already polling this server.

    A tab left open from a previous run keeps polling port 8000, so it reaches
    the new server within one refresh interval of it binding. That tab reloads
    itself into the new run (the page watches run_id), so opening another tab
    would just duplicate it.
    """
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if activity["last_poll"]:
            return True
        time.sleep(0.1)
    return False


def serve(
    run_dir: Path,
    display: dict | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:
    run_dir = Path(run_dir)
    display = {**DEFAULT_DISPLAY, **(display or {})}
    activity = {"last_poll": 0.0}
    handler = _make_handler(run_dir, display, activity)
    httpd, port = create_server(host, port, handler)
    url = f"http://{host}:{port}/"
    print(f"[INFO] Dashboard for {run_dir.name} at {url}")
    print("[INFO] Press Ctrl+C to stop.")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    if open_browser:
        if _wait_for_existing_client(activity):
            print("[INFO] Reusing the already-open dashboard tab.")
        else:
            import webbrowser

            webbrowser.open(url)
    try:
        while thread.is_alive():
            thread.join(0.5)
    except KeyboardInterrupt:
        print("\n[INFO] Dashboard stopped.")
    finally:
        httpd.shutdown()
        httpd.server_close()
