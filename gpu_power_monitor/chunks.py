"""Chunk publication and canonical selection, shared by readers and export.

A new writer advertises immutable files in .chunks only after compression is
settled. Old runs remain readable, including validated CSV/gzip duplicates.
"""
from __future__ import annotations

from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import warnings
import zlib

from .utils import atomic_replace_text

REGISTRY = ".chunks"


class ChunkIntegrityError(RuntimeError):
    pass


def is_chunk(path: Path) -> bool:
    return path.name.endswith((".csv", ".csv.gz"))


def logical_name(path: Path) -> str:
    return path.name[:-3] if path.name.endswith(".csv.gz") else path.name


def fingerprint(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path.absolute()), stat.st_size, stat.st_mtime_ns


@lru_cache(maxsize=4096)
def digest_file(stamp: tuple[str, int, int]) -> str:
    h = hashlib.sha256()
    with open(stamp[0], "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(path: Path) -> str:
    return digest_file(fingerprint(path))


@lru_cache(maxsize=4096)
def decoded_digest(stamp: tuple[str, int, int]) -> str:
    h = hashlib.sha256()
    with gzip.open(stamp[0], "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_pair(raw: Path, zipped: Path) -> bool:
    """False means corrupt gzip with an intact CSV fallback; conflicts raise."""
    raw_hash = digest(raw)  # unreadable original is never an acceptable fallback
    try:
        gzip_hash = decoded_digest(fingerprint(zipped))  # consumes footer/CRC
    except (OSError, EOFError, zlib.error):
        warnings.warn(f"Invalid gzip retained; using original CSV: {zipped.name}", RuntimeWarning)
        return False
    if raw_hash != gzip_hash:
        raise ChunkIntegrityError(f"Conflicting CSV/gzip contents: {raw.name}")
    return True


def begin_publication(run_dir: Path) -> None:
    registry = run_dir / REGISTRY
    registry.mkdir(parents=True, exist_ok=True)
    atomic_replace_text(registry / "protocol.json", json.dumps({"version": 1}))


def publish(path: Path, notices: list[str], *, source_digest: str) -> None:
    record = {"path": path.name, "logical_name": logical_name(path),
              "size_bytes": path.stat().st_size, "sha256": digest(path),
              "source_sha256": source_digest, "warnings": notices}
    atomic_replace_text(path.parent / REGISTRY / (logical_name(path) + ".json"), json.dumps(record))


def ready_records(run_dir: Path) -> list[dict]:
    registry = run_dir / REGISTRY
    records = []
    for path in sorted(registry.glob("*.csv.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ChunkIntegrityError(f"Unreadable chunk publication: {path.name}") from exc
        if not isinstance(record, dict):
            raise ChunkIntegrityError(f"Invalid chunk publication: {path.name}")
        name = record.get("path", "")
        if Path(name).name != name or not name.endswith((".csv", ".csv.gz")):
            raise ChunkIntegrityError(f"Invalid chunk publication: {path.name}")
        if record.get("logical_name") != logical_name(Path(name)):
            raise ChunkIntegrityError(f"Mismatched chunk publication: {path.name}")
        if (path.name != logical_name(Path(name)) + ".json"
                or not isinstance(record.get("size_bytes"), int) or record["size_bytes"] < 0
                or not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in record["sha256"])):
            raise ChunkIntegrityError(f"Invalid chunk publication metadata: {path.name}")
        records.append(record)
    return records


def canonical_chunks(run_dir: Path) -> list[Path]:
    run_dir = Path(run_dir)
    if (run_dir / REGISTRY / "protocol.json").exists():
        result = []
        for record in ready_records(run_dir):
            path = run_dir / record["path"]
            if path.stat().st_size != record["size_bytes"] or digest(path) != record["sha256"]:
                raise ChunkIntegrityError(f"Published chunk checksum mismatch: {path.name}")
            other = (run_dir / logical_name(path)) if path.name.endswith(".gz") else path.with_name(path.name + ".gz")
            if other.exists():
                raw, zipped = (other, path) if path.name.endswith(".gz") else (path, other)
                validate_pair(raw, zipped)
            result.append(path)
        return sorted(result, key=lambda p: logical_name(p))
    groups: dict[str, list[Path]] = {}
    for path in run_dir.iterdir():
        if path.is_file() and is_chunk(path) and "_to_PENDING" not in path.name:
            groups.setdefault(logical_name(path), []).append(path)
    result = []
    for name, variants in sorted(groups.items()):
        raw, zipped = run_dir / name, run_dir / (name + ".gz")
        if len(variants) == 2:
            result.append(zipped if validate_pair(raw, zipped) else raw)
        else:
            if variants[0].name.endswith(".gz"):
                try:
                    decoded_digest(fingerprint(variants[0]))
                except (OSError, EOFError, zlib.error) as exc:
                    raise ChunkIntegrityError(f"Corrupt gzip with no original CSV: {variants[0].name}") from exc
            result.append(variants[0])
    return result


def inventory_paths(run_dir: Path) -> list[Path]:
    """Keep control files and redundant/intermediate root chunks out of inventory."""
    canonical = set(canonical_chunks(run_dir))
    result = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir)
        if rel.parts[0] == REGISTRY or ".before-reconcile-" in path.name or path.name.endswith((".copying", ".tmp")):
            continue
        if len(rel.parts) == 1:
            if path.name in {"manifest.json", "latest.npz", "latest_status.json", "supervisor_status.json", "STOP"}:
                continue
            if is_chunk(path) and path not in canonical:
                continue
        result.append(path)
    return result
