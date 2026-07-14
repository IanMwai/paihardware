from __future__ import annotations

import datetime as dt
import os
import re
import time
from pathlib import Path

# Some managed Windows devices (e.g. with Defender controlled by Group Policy)
# intermittently lock the destination of an os.replace() while scanning the
# freshly written temp file, surfacing as PermissionError / WinError 5. We
# retry the rename a few times to ride out a transient scan, then fall back to
# an in-place write so acquisition never dies on a locked rename.
_REPLACE_RETRIES = 5
_REPLACE_BACKOFF_SEC = 0.05


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


# Test runs live in their own subfolder so anyone browsing output/ (or a run
# selector) sees at a glance what is real data and what is disposable scratch.
# The manifest's run_kind stays the source of truth; the folder is derived.
TEST_RUNS_SUBDIR = "test"


def runs_root(output_root: Path, *, test: bool) -> Path:
    """Where runs of this kind live: ``<output_root>`` or ``<output_root>/test``."""
    root = Path(output_root)
    return root / TEST_RUNS_SUBDIR if test else root


def sanitize_folder_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)
    name = name.rstrip(" .")
    return name if name else "measurement"


def unique_run_dir(output_root: Path, measurement_name: str) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base = sanitize_folder_name(measurement_name)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{base}_{ts}"
    counter = 1
    while run_dir.exists():
        run_dir = output_root / f"{base}_{ts}_{counter}"
        counter += 1
    run_dir.mkdir(parents=True)
    return run_dir


def _replace_with_fallback(tmp: Path, path: Path, write_direct) -> None:
    """Atomically swap ``tmp`` into ``path``; fall back to a direct write.

    Tries ``os.replace`` with a short bounded retry to ride out transient
    antivirus locks. If the rename is still blocked (e.g. Defender Group Policy
    on a managed device), writes ``path`` in place and removes the temp file.
    """
    last_err: OSError | None = None
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except (PermissionError, OSError) as err:
            last_err = err
            if attempt < _REPLACE_RETRIES - 1:
                time.sleep(_REPLACE_BACKOFF_SEC)
    # Rename is persistently blocked: write directly to the final path.
    try:
        write_direct(path)
    except OSError:
        # Re-raise the original rename failure for a clearer diagnosis.
        raise last_err if last_err is not None else None
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def rename_with_fallback(src: Path, dst: Path) -> None:
    """Rename ``src`` to ``dst``; on a blocked rename, copy then remove.

    Unlike :func:`_replace_with_fallback` this preserves a *meaningful* rename
    (the destination name encodes data, not a temp swap). Bounded retry rides
    out transient antivirus locks; if the rename is persistently blocked, the
    bytes are copied to ``dst`` and the source is removed best-effort.
    """
    src = Path(src)
    dst = Path(dst)
    last_err: OSError | None = None
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except (PermissionError, OSError) as err:
            last_err = err
            if attempt < _REPLACE_RETRIES - 1:
                time.sleep(_REPLACE_BACKOFF_SEC)
    # Rename persistently blocked: copy the bytes across, then drop the source.
    try:
        dst.write_bytes(src.read_bytes())
    except OSError:
        raise last_err if last_err is not None else None
    try:
        src.unlink()
    except OSError:
        pass


def atomic_replace_text(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    _replace_with_fallback(tmp, path, lambda p: p.write_text(text, encoding="utf-8"))


def atomic_replace_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    _replace_with_fallback(tmp, path, lambda p: p.write_bytes(data))


def fmt_hhmmss_ms(value: dt.datetime) -> str:
    return value.strftime("%H%M%S") + f"{int(value.microsecond / 1000):03d}"
