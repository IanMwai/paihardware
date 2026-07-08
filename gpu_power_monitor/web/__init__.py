"""Local web dashboard.

Serves a single-page dashboard that streams from the live snapshot files the
acquisition already writes (``latest.npz`` / ``latest_status.json``). Uses only
the Python standard library — no extra dependencies — which is plenty for the
handful of viewers on the operator PC and the video wall.
"""

from .server import build_latest_payload, build_config_payload, serve

__all__ = ["serve", "build_latest_payload", "build_config_payload"]
