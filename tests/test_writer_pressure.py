"""Reproduce transient and persistent writer pressure without NI hardware."""
import csv
import dataclasses
import datetime as dt
import io
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from gpu_power_monitor import acquisition as acq, daq, logging_writer as lw
from gpu_power_monitor.config import AcquisitionConfig, LoggingConfig, RemoteConfig, GpuChannel
from gpu_power_monitor.manifest import load_manifest
from gpu_power_monitor.processing import ProcessedBlock, PowerProcessor, moving_average_causal_ignore_nan
from gpu_power_monitor.utils import chunk_paths, open_chunk


def block(start=0, n=1000, rate=10000):
    times=np.arange(start,start+n)/rate;ones=np.ones(n)
    return ProcessedBlock(times,['GPU1'],[ones*12],[ones*2],[ones*24],ones*24)


def wait_for(predicate, timeout=2):
    deadline=time.monotonic()+timeout
    while not predicate():
        if time.monotonic()>=deadline: raise AssertionError('Timed out waiting for writer')
        time.sleep(.005)


def gated_writer(tmp_path, monkeypatch, timeout=.5):
    entered,release=threading.Event(),threading.Event()
    writer=lw.ChunkWriter(tmp_path,'test',60,dt.datetime.now(),['GPU1'],queue_blocks=1,enqueue_timeout_sec=timeout)
    original=writer._write_block_locked
    def delayed(value):
        entered.set()
        if not release.wait(3):raise TimeoutError('test gate')
        original(value)
    monkeypatch.setattr(writer,'_write_block_locked',delayed)
    writer.write_block(block());assert entered.wait(1)
    writer.write_block(block(1000))
    return writer,release


def test_temporary_full_queue_recovers_without_losing_or_duplicating_samples(tmp_path,monkeypatch):
    writer,release=gated_writer(tmp_path,monkeypatch)
    errors=[]
    def send():
        try:writer.write_block(block(2000))
        except BaseException as exc:errors.append(exc)
    sender=threading.Thread(target=send);sender.start()
    try:
        wait_for(lambda:writer.queue_full_events==1)
        assert sender.is_alive()
        before=writer.diagnostics()
        assert before['queued_samples']==1000 and before['inflight_samples']==1000
        assert before['queued_seconds']==pytest.approx(.1)
        release.set();sender.join(2)
        assert not sender.is_alive() and not errors
        paths=writer.finalize(.3)
        assert writer.samples_written==writer.samples_enqueued==3000
        with open_chunk(paths[0]) as f:times=[float(r['time_s']) for r in csv.DictReader(f)]
        np.testing.assert_allclose(times,np.arange(3000)/10000)
        after=writer.diagnostics()
        assert after['queue_depth']==0 and after['enqueue_wait_max_sec']>0
    finally:
        release.set();writer.finalize(.3)


def test_sustained_overload_is_bounded_and_rejected_block_is_not_counted_as_written(tmp_path,monkeypatch):
    writer,release=gated_writer(tmp_path,monkeypatch,timeout=.03)
    try:
        started=time.monotonic()
        with pytest.raises(RuntimeError,match='1000 samples queued'):
            writer.write_block(block(2000))
        assert .02 <= time.monotonic()-started < .5
        assert writer.samples_enqueued==2000
    finally:
        release.set();writer.finalize(.3)
    assert writer.samples_written==2000 and writer.diagnostics()['queue_full_events']==1


def test_backpressure_checks_daq_health_instead_of_waiting_blindly(tmp_path,monkeypatch):
    writer,release=gated_writer(tmp_path,monkeypatch)
    try:
        with pytest.raises(RuntimeError,match='driver nearly full'):
            writer.write_block(block(2000),on_wait=lambda:(_ for _ in ()).throw(RuntimeError('driver nearly full')))
    finally:
        release.set();writer.finalize(.3)
    assert writer.samples_written==2000


def test_actual_writer_failure_interrupts_backpressure_promptly(tmp_path,monkeypatch):
    entered,release=threading.Event(),threading.Event()
    writer=lw.ChunkWriter(tmp_path,'test',60,dt.datetime.now(),['GPU1'],queue_blocks=1,enqueue_timeout_sec=5)
    def broken(value):
        entered.set();release.wait(2);raise OSError('disk write failed')
    monkeypatch.setattr(writer,'_write_block_locked',broken)
    writer.write_block(block());assert entered.wait(1);writer.write_block(block(1000))
    timer=threading.Timer(.03,release.set);timer.start()
    started=time.monotonic()
    with pytest.raises(RuntimeError,match='disk write failed'):writer.write_block(block(2000))
    assert time.monotonic()-started<1
    with pytest.raises(RuntimeError,match='disk write failed'):writer.finalize(.3)
    timer.join()


def test_compression_is_serial_and_full_compression_queue_publishes_csv(tmp_path,monkeypatch):
    entered,release=threading.Event(),threading.Event();original=lw._gzip_chunk
    active=maximum=0;lock=threading.Lock()
    def delayed(*args):
        nonlocal active,maximum
        with lock:active+=1;maximum=max(maximum,active)
        entered.set()
        assert release.wait(3)
        try:return original(*args)
        finally:
            with lock:active-=1
    monkeypatch.setattr(lw,'_gzip_chunk',delayed)
    writer=lw.ChunkWriter(tmp_path,'test',1,dt.datetime.now(),['GPU1'],compress=True,compression_queue_chunks=1)
    try:
        writer.write_block(block(0,4,rate=2));assert entered.wait(1)
        wait_for(lambda:writer.samples_written==4)
        writer.write_block(block(4,2,rate=2));wait_for(lambda:writer.samples_written==6)
        writer.write_block(block(6,2,rate=2));wait_for(lambda:writer.samples_written==8)
        assert writer.diagnostics()['compression_deferred']>=1
        assert any(p.suffix=='.csv' for p in chunk_paths(tmp_path))
    finally:
        release.set();paths=writer.finalize(4)
    assert maximum==1 and len(paths)==4 and writer.samples_written==8
    assert len({lw.logical_name(p) for p in paths})==4
    assert any(p.suffix=='.csv' for p in paths)


@pytest.mark.parametrize('window',[1,2,10,500])
def test_vectorized_smoothing_matches_the_nan_window_reference(window):
    values=np.random.default_rng(17).normal(size=257)
    values[5:30]=np.nan;values[90]=np.inf;values[130]=-np.inf
    expected=[]
    for i in range(len(values)):
        part=values[max(0,i-window+1):i+1];part=part[~np.isnan(part)]
        with np.errstate(invalid='ignore'):
            expected.append(float(np.mean(part)) if len(part) else np.nan)
    if window == 1: expected = values.copy()  # Unfiltered mode preserves infinities too.
    np.testing.assert_allclose(moving_average_causal_ignore_nan(values,window),expected,atol=1e-13,equal_nan=True)


def test_smoothing_keeps_history_across_irregular_block_boundaries():
    values=np.random.default_rng(29).uniform(1,5,113);values[50:55]=np.nan
    p=PowerProcessor(10000,[GpuChannel('GPU1','ai0','ai1',1)],1,power_average_samples=10)
    parts=[]
    for start,end in [(0,1),(1,4),(4,33),(33,113)]:
        parts.extend(p.process([np.ones(end-start)],[values[start:end]]).power_w[0])
    expected=[np.nanmean(values[max(0,i-9):i+1]) for i in range(len(values))]
    np.testing.assert_allclose(parts,expected,equal_nan=True,atol=1e-13)


def test_bulk_csv_preserves_column_order_precision_and_special_values(tmp_path):
    rng=np.random.default_rng(41);n=15
    arrays=[rng.normal(size=n) for _ in range(12)];arrays[0][0]=-0.;arrays[1][1]=np.nan
    times=np.arange(n)/10000
    value=ProcessedBlock(times,['GPU1','GPU2','GPU3','GPU4'],arrays[::3],arrays[1::3],arrays[2::3],sum(arrays[2::3]))
    writer=lw.ChunkWriter(tmp_path,'test',60,dt.datetime.now(),value.labels)
    writer.write_block(value);paths=writer.finalize(n/10000)
    reference=io.StringIO(newline='');expected=csv.writer(reference);expected.writerow(lw.csv_header(value.labels))
    for i in range(n):
        row=[times[i]]+[a[i] for a in arrays]+[value.total_power_w[i]]
        expected.writerow([f'{float(v):.9f}' for v in row])
    assert paths[0].read_bytes()==reference.getvalue().encode()


def cfg(tmp_path,**logging):
    return dataclasses.replace(AcquisitionConfig(),logging=LoggingConfig(compress=False,**logging),remote=RemoteConfig(enabled=False))


def test_hardware_batches_small_arrivals_and_drains_the_partial_stop_tail(tmp_path,monkeypatch):
    reads=[];available=iter([7,850,1000,17])
    folder=tmp_path/'run'
    class Source:
        def __init__(self,*args):pass
        def start(self):pass
        def stop(self):pass
        def available_samples(self):return next(available)
        def read(self,n):
            reads.append(n)
            if len(reads)==1:(folder/'STOP').write_text('{}')
            return [[2.5]*n for _ in range(4)],[[.3]*n for _ in range(4)]
    monkeypatch.setattr(acq,'NIDaqSource',Source);monkeypatch.setattr(acq,'STOP_FILE_POLL_SEC',0)
    acq.acquire(cfg(tmp_path),run_dir=folder)
    assert reads==[1000,17]
    manifest=load_manifest(folder);diagnostics=manifest['recording_diagnostics']
    assert manifest['status']=='completed' and diagnostics['written_samples']==1017
    assert diagnostics['daq']['min_block_samples']==17 and diagnostics['daq']['max_block_samples']==1000
    history=[json.loads(s) for s in (folder/'diagnostics/writer.jsonl').read_text().splitlines()]
    assert history[-1]['writer']['written_samples']==1017


def test_failed_acquisition_retains_diagnostics_after_the_writer_drains(tmp_path,monkeypatch):
    entered,release=threading.Event(),threading.Event();folder=tmp_path/'run'
    original=lw.ChunkWriter._write_block_locked
    def delayed(writer,value):
        entered.set()
        assert release.wait(2)
        original(writer,value)
    class Source:
        def __init__(self,*args):self.reads=0
        def start(self):pass
        def stop(self):release.set()
        def available_samples(self):return 1000
        def read(self,n):
            self.reads+=1
            if self.reads==2:assert entered.wait(1)
            return [[2.5]*n for _ in range(4)],[[.3]*n for _ in range(4)]
    monkeypatch.setattr(acq,'NIDaqSource',Source);monkeypatch.setattr(lw.ChunkWriter,'_write_block_locked',delayed)
    with pytest.raises(RuntimeError,match='did not recover'):
        acq.acquire(cfg(tmp_path,queue_blocks=1,enqueue_timeout_sec=.03),run_dir=folder)
    manifest=load_manifest(folder);d=manifest['recording_diagnostics']
    assert manifest['status']=='error' and d['acquired_samples']==3000 and d['written_samples']==2000
    assert d['queue_depth']==0 and d['writer']['queue_full_events']==1
    history=[json.loads(s) for s in (folder/'diagnostics/writer.jsonl').read_text().splitlines()]
    assert any(s['phase']=='error' and s['writer']['pending_block_samples']==0 for s in history)


def test_daq_backlog_guard_stops_before_the_driver_buffer_is_full(tmp_path,monkeypatch):
    class Source:
        buffer_capacity_samples=600000
        def __init__(self,*args):pass
        def start(self):pass
        def stop(self):pass
        def available_samples(self):return 540000
    monkeypatch.setattr(acq,'NIDaqSource',Source)
    with pytest.raises(RuntimeError,match='before hardware buffer overflow'):
        acq.acquire(cfg(tmp_path),run_dir=tmp_path/'run')
    manifest=load_manifest(tmp_path/'run')
    assert manifest['status']=='error' and manifest['recording_diagnostics']['daq']['max_available_samples']==540000


def test_daq_accepts_exact_one_sample_per_channel_but_never_truncates_mismatches():
    source=object.__new__(daq.NIDaqSource);source.channels=AcquisitionConfig().channels
    source.task=SimpleNamespace(read=lambda **kw:list(range(8)))
    volts,amps=source.read(1)
    assert volts==[[0],[2],[4],[6]] and amps==[[1],[3],[5],[7]]
    source.task=SimpleNamespace(read=lambda **kw:[[1,2] for _ in range(7)]+[[1]])
    with pytest.raises(RuntimeError,match='refusing to truncate'):source.read(2)


def test_diagnostic_file_lock_does_not_mask_or_stop_valid_sample_writes(tmp_path,monkeypatch):
    original=Path.open
    def locked(path,*args,**kwargs):
        if path.name=='writer.jsonl':raise PermissionError('diagnostic file locked')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',locked)
    acq.acquire(cfg(tmp_path),simulate=True,duration_sec=.02,run_dir=tmp_path/'run')
    manifest=load_manifest(tmp_path/'run')
    assert manifest['status']=='completed'
    assert 'diagnostic file locked' in manifest['recording_diagnostics']['diagnostic_error']


def test_four_gpu_burst_spans_many_rotations_with_bounded_queues_and_no_gaps(tmp_path):
    from gpu_power_monitor.chunks import canonical_chunks
    labels=['GPU1','GPU2','GPU3','GPU4']
    writer=lw.ChunkWriter(tmp_path,'stress',.25,dt.datetime.now(),labels,
                         queue_blocks=4,compress=True,compression_queue_chunks=2)
    try:
        for start in range(0,100000,1000):
            value=block(start)
            value.labels=labels
            value.voltage_v=[value.voltage_v[0]]*4
            value.current_a=[value.current_a[0]]*4
            value.power_w=[value.power_w[0]]*4
            value.total_power_w=value.total_power_w*4
            writer.write_block(value)
    finally:paths=writer.finalize(10)
    assert writer.samples_enqueued==writer.samples_written==100000
    assert len(paths)==40 and canonical_chunks(tmp_path)==paths
    times=[]
    for path in paths:
        with open_chunk(path) as handle:
            reader=csv.DictReader(handle)
            assert len(reader.fieldnames)==14
            times.extend(float(row['time_s']) for row in reader)
    np.testing.assert_allclose(times,np.arange(100000)/10000)
    assert writer.diagnostics()['max_queue_depth']<=4
    assert writer.diagnostics()['compression_pending_chunks']==0
