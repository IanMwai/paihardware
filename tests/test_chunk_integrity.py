import csv
import datetime as dt
import gzip
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from gpu_power_monitor import chunks, logging_writer as lw
from gpu_power_monitor.manifest import build_file_inventory, compute_run_stats
from gpu_power_monitor.processing import ProcessedBlock
from gpu_power_monitor.utils import chunk_paths


def raw_chunk(folder, name='chunk.csv', text='time_s,total_power_w\n0,24\n1,24\n'):
    folder.mkdir(parents=True, exist_ok=True)
    raw=folder/name;raw.write_text(text)
    return raw


def zipped(raw):
    path=raw.with_name(raw.name+'.gz');path.write_bytes(gzip.compress(raw.read_bytes()));return path


def block():
    times=np.array([0.,.1,.2]);v=np.ones(3)*12;i=np.ones(3)*2
    return ProcessedBlock(time_s=times,labels=['GPU1'],voltage_v=[v],current_a=[i],power_w=[v*i],total_power_w=v*i)


@pytest.mark.parametrize('temporary',[True,False])
def test_source_cleanup_lock_is_nonfatal_and_samples_count_once(tmp_path,monkeypatch,temporary):
    original=Path.unlink;attempts=[]
    def locked(path,*a,**kw):
        if path.name.endswith('.csv') and 'PENDING' not in path.name:
            attempts.append(path)
            if not temporary or len(attempts)<3:raise PermissionError('WinError 32 sharing violation')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'unlink',locked)
    writer=lw.ChunkWriter(tmp_path,'nidaq',1,dt.datetime.now(),['GPU1'],compress=True)
    writer.write_block(block());paths=writer.finalize(.3)
    assert writer.error is None and len(paths)==1 and paths[0].name.endswith('.csv.gz')
    assert chunk_paths(tmp_path)==paths
    assert compute_run_stats(tmp_path)['sample_count']==3
    assert len([r for r in build_file_inventory(tmp_path) if r['path'].endswith(('.csv','.gz'))])==1
    notices=chunks.ready_records(tmp_path)[0]['warnings']
    assert bool(notices)==(not temporary)
    assert len(attempts)==(3 if temporary else 5)


def test_intermediate_csv_not_visible_before_compression_settles(tmp_path,monkeypatch):
    entered,release=threading.Event(),threading.Event();original=lw._gzip_chunk
    def delayed(*a):entered.set();release.wait(3);return original(*a)
    monkeypatch.setattr(lw,'_gzip_chunk',delayed)
    writer=lw.ChunkWriter(tmp_path,'nidaq',1,dt.datetime.now(),['GPU1'],compress=True)
    writer.write_block(block());writer.stop_event.set()
    assert entered.wait(1)
    try:
        assert list(tmp_path.glob('*.csv'))
        assert chunk_paths(tmp_path)==[]
    finally:
        release.set()
    assert writer.finalize(.3)[0].suffix=='.gz'


def test_compression_failure_publishes_intact_plain_csv(tmp_path,monkeypatch):
    def broken(*a,**kw):raise OSError('compression unavailable')
    monkeypatch.setattr(gzip,'open',broken)
    writer=lw.ChunkWriter(tmp_path,'nidaq',1,dt.datetime.now(),['GPU1'],compress=True)
    writer.write_block(block());paths=writer.finalize(.3)
    assert len(paths)==1 and paths[0].suffix=='.csv' and writer.error is None
    assert chunks.ready_records(tmp_path)[0]['warnings'][0].startswith('compression_failed')
    assert compute_run_stats(tmp_path)['sample_count']==3


def test_duplicate_legacy_chunks_are_verified_and_counted_once(tmp_path):
    raw=raw_chunk(tmp_path);gz=zipped(raw)
    assert chunk_paths(tmp_path)==[gz]
    assert compute_run_stats(tmp_path)['sample_count']==2
    assert [item['path'] for item in build_file_inventory(tmp_path)]==[gz.name]


def test_corrupt_gzip_falls_back_to_original(tmp_path):
    raw=raw_chunk(tmp_path);gz=zipped(raw);gz.write_bytes(gz.read_bytes()[:-5])
    with pytest.warns(RuntimeWarning,match='Invalid gzip'):
        assert chunk_paths(tmp_path)==[raw]


def test_corrupt_gzip_without_original_is_integrity_failure(tmp_path):
    raw=raw_chunk(tmp_path);gz=zipped(raw);raw.unlink();gz.write_bytes(b'broken gzip')
    with pytest.raises(chunks.ChunkIntegrityError,match='no original'):chunk_paths(tmp_path)


def test_valid_but_conflicting_csv_and_gzip_fail(tmp_path):
    raw=raw_chunk(tmp_path);zipped(raw);raw.write_text('time_s,total_power_w\n0,99\n')
    with pytest.raises(chunks.ChunkIntegrityError,match='Conflicting'):chunk_paths(tmp_path)


def test_corrupted_published_file_is_not_exportable(tmp_path):
    chunks.begin_publication(tmp_path);raw=raw_chunk(tmp_path)
    chunks.publish(raw,[],source_digest=chunks.digest(raw));raw.write_text('changed data')
    with pytest.raises(chunks.ChunkIntegrityError,match='checksum'):chunk_paths(tmp_path)

