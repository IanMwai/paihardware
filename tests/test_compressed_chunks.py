import csv
import datetime as dt
import gzip

import numpy as np

from gpu_power_monitor.logging_writer import ChunkWriter
from gpu_power_monitor.processing import ProcessedBlock
from gpu_power_monitor.utils import chunk_paths, open_chunk


def _block(t0, n, rate=1000.0):
    t = t0 + np.arange(n) / rate
    v = np.full(n, 12.0); i = np.full(n, 2.0); p = v * i
    return ProcessedBlock(time_s=t, labels=["GPU1"], voltage_v=[v], current_a=[i], power_w=[p], total_power_w=p)


def test_compressed_chunks_roundtrip(tmp_path):
    w = ChunkWriter(tmp_path, "nidaq", chunk_len_sec=1.0, start_wall_dt=dt.datetime(2026, 1, 1),
                    labels=["GPU1"], compress=True)
    for k in range(3):
        w.write_block(_block(k * 1.0, 1000))
    paths = w.finalize(3.0)
    assert paths and all(p.name.endswith(".csv.gz") for p in paths)
    assert not list(tmp_path.glob("*.csv"))  # plain files removed after gzip
    found = chunk_paths(tmp_path)
    assert [p.name for p in found] == sorted(p.name for p in paths)
    with open_chunk(found[0]) as fh:
        rows = list(csv.DictReader(fh))
    assert rows and float(rows[0]["gpu1_power_w"]) == 24.0
    with gzip.open(found[0], "rt") as fh:  # really gzip
        assert fh.readline().startswith("time_s")


def test_plain_chunks_unchanged(tmp_path):
    w = ChunkWriter(tmp_path, "nidaq", chunk_len_sec=1.0, start_wall_dt=dt.datetime(2026, 1, 1),
                    labels=["GPU1"])
    w.write_block(_block(0.0, 1000))
    paths = w.finalize(1.0)
    assert paths[0].suffix == ".csv" and chunk_paths(tmp_path) == paths
