import dataclasses
import json
from pathlib import Path
import pytest
from gpu_power_monitor import acquisition as acq, cli, remote
from gpu_power_monitor.config import AcquisitionConfig, StorageConfig, RemoteConfig, load_config, select_gpus, select_nvml_gpus
from gpu_power_monitor.manifest import load_manifest

def test_required_nvml_filters_to_physical_uuid(monkeypatch):
    commands=[]
    class Client:
        def close(self):pass
    monkeypatch.setattr(remote,'connect',lambda cfg:Client())
    def command(client,cmd,**kwargs):
        commands.append(cmd)
        return (0,'123\n','') if 'nohup' in cmd else (0,'','')
    monkeypatch.setattr(remote,'run_command',command)
    remote.NvmlLogger(RemoteConfig(gpu_ids=('GPU-abc',),required=True),'run').start()
    assert 'nvidia-smi -i GPU-abc --query-gpu=' in commands[0]
    assert 'timestamp,index,uuid,name' in commands[0]


def test_required_nvml_rejects_missing_and_unselected_gpu_samples(tmp_path):
    path=tmp_path/'nvml.csv'
    path.write_text('timestamp, index, uuid, power.draw [W]\n2026/10/24 03:00:00.000, 1, GPU-b, 42\n2026/10/24 03:00:01.000, 1, GPU-b, 43\n')
    data=remote.validate_nvml(path,('GPU-b',))
    assert data['observed_span_s']==1 and data['sample_counts']=={'GPU-b':2}
    with pytest.raises(remote.RemoteUnavailable):remote.validate_nvml(path,('GPU-a',))
    with pytest.raises(remote.RemoteUnavailable):remote.validate_nvml(path,('GPU-b','GPU-c'))
    path.write_text('timestamp, index, uuid, power.draw [W]\n')
    with pytest.raises(remote.RemoteUnavailable):remote.validate_nvml(path,('GPU-b',))



def test_lab_calibration_and_gzip_default(monkeypatch):
    monkeypatch.delenv('PAI_OUTPUT_ROOT', raising=False)
    cfg=load_config(Path(__file__).resolve().parents[1]/"configs/default.yaml")
    assert [g.current_sign for g in cfg.channels.gpus]==[1,-1,-1,-1]
    assert [g.voltage_scale for g in cfg.channels.gpus]==[4.741144,4.611890,4.611890,4.606657]
    assert cfg.logging.compress is True
    assert AcquisitionConfig().logging.compress is True
    assert cfg.storage.output_root.is_absolute()


def test_subset_maps_only_selected_ids_in_config_order():
    cfg=AcquisitionConfig()
    cfg=dataclasses.replace(cfg,remote=RemoteConfig(gpu_ids=('GPU-a','GPU-b','GPU-c','GPU-d')))
    subset=select_nvml_gpus(select_gpus(cfg,'3,1'),None)
    assert subset.channels.labels==['GPU1','GPU3']
    assert subset.remote.gpu_ids==('GPU-a','GPU-c') and subset.remote.required
    with pytest.raises(ValueError):select_nvml_gpus(subset,'1')
    with pytest.raises(ValueError):select_nvml_gpus(subset,'1,1')
    with pytest.raises(ValueError):select_nvml_gpus(subset,'1;evil,2')


def test_nvml_ids_on_channels_are_selected():
    cfg=AcquisitionConfig()
    channels=dataclasses.replace(cfg.channels,gpus=tuple(dataclasses.replace(g,nvml_id=f'GPU-{i}') for i,g in enumerate(cfg.channels.gpus)))
    subset=select_gpus(dataclasses.replace(cfg,channels=channels),'GPU2')
    assert subset.remote.gpu_ids==('GPU-1',) and subset.remote.required


def test_start_failure_keeps_original_error_even_when_stop_fails(tmp_path,monkeypatch):
    class Source:
        def start(self):raise RuntimeError('original start error')
        def stop(self):raise OSError('secondary stop error')
    monkeypatch.setattr(acq,'NIDaqSource',lambda *a:Source())
    cfg=dataclasses.replace(AcquisitionConfig(),remote=RemoteConfig(enabled=False),storage=StorageConfig(output_root=tmp_path))
    run=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='original start error'):acq.acquire(cfg,run_dir=run)
    manifest=load_manifest(run)
    assert manifest['status']=='error' and manifest['error']=='original start error'
    assert manifest['cleanup_error']=='secondary stop error'


def test_failed_headless_recording_never_archives(tmp_path,monkeypatch):
    cfg=dataclasses.replace(AcquisitionConfig(),storage=StorageConfig(output_root=tmp_path))
    monkeypatch.setattr('gpu_power_monitor.config.load_config',lambda *a:cfg)
    monkeypatch.setattr(acq,'acquire',lambda *a,**k:(_ for _ in ()).throw(OSError('write failed')))
    monkeypatch.setattr(cli,'_auto_push',lambda *a:pytest.fail('must not archive failed recording'))
    assert cli.run_hardware(simulate=True,display=False)==1


def test_failed_dashboard_recording_never_archives(tmp_path,monkeypatch):
    import gpu_power_monitor.web as web
    cfg=dataclasses.replace(AcquisitionConfig(),storage=StorageConfig(output_root=tmp_path))
    monkeypatch.setattr('gpu_power_monitor.config.load_config',lambda *a:cfg)
    monkeypatch.setattr(acq,'acquire',lambda *a,**k:(_ for _ in ()).throw(OSError('write failed')))
    monkeypatch.setattr(web,'serve',lambda *a,**k:None)
    monkeypatch.setattr(cli,'_auto_push',lambda *a:pytest.fail('must not archive failed recording'))
    assert cli.run_hardware(simulate=True,display=True)==1


def test_unconfirmed_manifest_never_archives(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'_auto_push',lambda *a:pytest.fail('must not archive unconfirmed recording'))
    (tmp_path/'manifest.json').write_text(json.dumps({'status':'error','error':'real error'}))
    assert cli._finish_hardware_run(tmp_path,AcquisitionConfig())==1


def test_live_preview_is_bounded_but_stats_keep_every_sample(tmp_path,monkeypatch):
    import numpy as np
    from gpu_power_monitor.live_buffer import LiveBuffer
    from gpu_power_monitor.processing import ProcessedBlock
    live=LiveBuffer(tmp_path,10000,60,['GPU1'],max_points=25)
    writes=[]
    monkeypatch.setattr(live,'write_latest',lambda state:writes.append(state))
    monkeypatch.setattr('gpu_power_monitor.live_buffer.time.monotonic',lambda:1.)
    t=np.arange(100000)/10000; v=np.full(t.size,12.);i=np.full(t.size,2.);p=v*i
    block=ProcessedBlock(t,['GPU1'],[v],[i],[p],p)
    live.append(block);live.append(block)
    assert len(live.time_s)<=25 and len(writes)==1
    assert live._run_count==200000 and live._run_sum==4800000


def test_replay_reads_one_verified_gzip_copy(tmp_path):
    import gzip
    from gpu_power_monitor.web.server import read_history_csv
    raw=tmp_path/'chunk.csv'
    raw.write_text('time_s,gpu2_voltage_v,gpu2_current_a,gpu2_power_w,total_power_w\n0,12,2,24,24\n1,12,2,24,24\n')
    raw.with_suffix('.csv.gz').write_bytes(gzip.compress(raw.read_bytes()))
    data=read_history_csv(tmp_path)
    assert data['time_s'].tolist()==[0,1]
    assert data['gpu2_power_w'].tolist()==[24,24]
