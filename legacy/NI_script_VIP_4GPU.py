# NI_script_VIP_4GPU.py
# Updated 4-GPU version of NI_script_VIP_1GPU_v20260307.py, by Bin.
# Kept as the reference for the 4-GPU wiring/calibration now used by gpu_power_monitor.
# Realtime NI-DAQmx acquisition (8 inputs / 4 GPUs) + scrolling plots + chunked CSV logging
# Keys: p = pause/resume DISPLAY only (acquisition & logging continue), q = quit

import matplotlib
matplotlib.use("TkAgg")  # MUST be before importing pyplot

import os
import re
import csv
import queue
import threading
import datetime as dt
from pathlib import Path
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import nidaqmx
from nidaqmx.constants import AcquisitionType, TerminalConfiguration, Edge


# =========================
# Scaling (apply AFTER read)
# =========================
VOLT_SCALES = [
    12.18 / 2.569,  # GPU1 / ai0
    12.18 / 2.641,  # GPU2 / ai2
    12.18 / 2.641,  # GPU3 / ai4
    12.18 / 2.644,  # GPU4 / ai6
]
CURR_SCALE = 1.0/0.08   # scaled_current = raw_current * CURR_SCALE
CURR_SIGNS = [1.0, -1.0, -1.0, -1.0]  # invert GPU2 and GPU3 to match GPU1 direction


# =========================
# Plot Y limits (scaled units)
# =========================
# POWER_YMAX = Prated * 1.44
# CURR_YMAX = Irated * 1.44
VOLT_YMIN, VOLT_YMAX = 11.5, 13
CURR_YMIN, CURR_YMAX = -1, 15
POWER_YMIN, POWER_YMAX = -10, 180.0


# =========================
# Measurement Task Naming
# =========================
MEASUREMENT_NAME = "GPU Power Measurement Test"   # folder name (if exists, suffix with YYYYMMDD_HHMM)


# =========================
# NI / Acquisition Settings
# =========================
DEVICE = "Dev1"
# Each tuple is: (GPU label, voltage channel, current channel)
GPU_CHANNELS = [
    ("GPU1", "ai0", "ai1"),
    ("GPU2", "ai2", "ai3"),
    ("GPU3", "ai4", "ai5"),
    ("GPU4", "ai6", "ai7"),
]
TERMINAL = "RSE"          # "RSE" / "DIFF" / "NRSE"

# Raw input range for both channels (DAQ read range; keep broad enough for both signals)
MIN_V = -0.5
MAX_V = 3.5

SAMPLE_RATE = 10000       # Hz
WINDOW_SEC = 5         # seconds shown on screen (oscilloscope window)

CHUNK_SIZE = 1000         # max samples per UI update read (1000 @ 10kHz = 100 ms)
SAVE_EVERY_MIN = 1        # chunk duration written to CSV (minutes)
FILE_PREFIX = "nidaq"     # filename prefix


# =========================
# Internal State
# =========================
paused = False
quit_flag = False

window_points = int(SAMPLE_RATE * WINDOW_SEC)
v_buffers = [deque(maxlen=window_points) for _ in GPU_CHANNELS]
i_buffers = [deque(maxlen=window_points) for _ in GPU_CHANNELS]
p_buffers = [deque(maxlen=window_points) for _ in GPU_CHANNELS]

sample_index = 0                 # total samples acquired since start
start_wall = dt.datetime.now()   # wall-clock time at script start (for filenames)


def sanitize_folder_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", name)  # Windows-illegal + control chars
    name = name.rstrip(" .")
    return name if name else "measurement"


def get_output_dir() -> Path:
    """
    Create:
      <script_dir>/output/<MEASUREMENT_NAME>_YYYYMMDD_HHMM/

    Always appends a timestamp (year, month, day, hour, minute) to the folder name.
    If a rare collision occurs (same minute), adds _<counter>.
    """
    script_dir = Path(__file__).resolve().parent
    root = script_dir / "output"
    root.mkdir(parents=True, exist_ok=True)

    base = sanitize_folder_name(MEASUREMENT_NAME)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    task_dir = root / f"{base}_{ts}"

    # Rare collision: add counter
    counter = 1
    while task_dir.exists():
        task_dir = root / f"{base}_{ts}_{counter}"
        counter += 1

    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def terminal_config_from_string(mode: str):
    mode = mode.strip().upper()
    if mode == "RSE":
        return TerminalConfiguration.RSE
    if mode == "DIFF":
        return TerminalConfiguration.DIFFERENTIAL
    if mode == "NRSE":
        return TerminalConfiguration.NRSE
    raise ValueError("TERMINAL must be 'RSE', 'DIFF', or 'NRSE'")


def fmt_hhmmss_ms(t: dt.datetime) -> str:
    # HHMMSSmmm (milliseconds)
    return t.strftime("%H%M%S") + f"{int(t.microsecond/1000):03d}"


def ensure_channel_list(raw):
    """
    NI-DAQmx task.read() returns:
      - list[float] for 1 channel
      - list[list[float]] for N channels (one list per channel)
    This function normalizes to: list[list[float]] with one list per channel.
    """
    if raw is None:
        return []
    # 1-channel numeric scalar
    if isinstance(raw, (float, int, np.floating, np.integer)):
        return [[float(raw)]]
    # 1-channel list of numerics
    if isinstance(raw, list) and (len(raw) == 0 or isinstance(raw[0], (float, int, np.floating, np.integer))):
        return [list(raw)]
    # multi-channel list-of-lists / tuple-of-lists
    return [list(ch) for ch in raw]


class ChunkCsvWriterBG:
    """
    Background CSV writer:
      - main thread enqueues blocks: (t0_sec, voltages, currents, powers, fs)
      - writer thread writes blocks to CSV and rotates by fixed chunk_len_sec
    Notes:
      - queue is bounded; if disk can't keep up, oldest blocks are dropped (warning printed).
      - CSV at 20 kHz is very large; consider TDMS/binary for long runs.
    """
    def __init__(self, out_dir: Path, prefix: str, chunk_len_sec: float, start_wall_dt: dt.datetime):
        self.out_dir = Path(out_dir)
        self.prefix = prefix
        self.chunk_len = float(chunk_len_sec)
        self.start_wall = start_wall_dt

        self.chunk_start_sec = 0.0
        self.chunk_start_wall = None

        self.f = None
        self.writer = None
        self.tmp_path = None

        self.queue = queue.Queue(maxsize=200)  # tune for your disk speed
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self._last_written_t_end = 0.0

        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.thread.start()

    def _open_new(self, chunk_start_sec: float):
        self.chunk_start_sec = float(chunk_start_sec)
        self.chunk_start_wall = self.start_wall + dt.timedelta(seconds=self.chunk_start_sec)

        date_str = self.chunk_start_wall.strftime("%Y%m%d")
        start_str = fmt_hhmmss_ms(self.chunk_start_wall)

        tmp_name = f"{self.prefix}_{date_str}_{start_str}_to_PENDING.csv"
        self.tmp_path = self.out_dir / tmp_name

        os.makedirs(self.out_dir, exist_ok=True)
        self.f = open(self.tmp_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.f)
        header = ["time_s"]
        for gpu_label, _, _ in GPU_CHANNELS:
            gpu = gpu_label.lower()
            header.extend([f"{gpu}_voltage_v", f"{gpu}_current_a", f"{gpu}_power_w"])
        self.writer.writerow(header)

    def _close_and_rename(self, chunk_end_sec: float):
        if self.f is None:
            return

        self.f.flush()
        self.f.close()
        self.f = None
        self.writer = None

        chunk_end_wall = self.start_wall + dt.timedelta(seconds=float(chunk_end_sec))
        date_str = self.chunk_start_wall.strftime("%Y%m%d")
        start_str = fmt_hhmmss_ms(self.chunk_start_wall)
        end_str = fmt_hhmmss_ms(chunk_end_wall)

        final_name = f"{self.prefix}_{date_str}_{start_str}_to_{end_str}.csv"
        final_path = self.out_dir / final_name

        try:
            os.replace(self.tmp_path, final_path)
            print(f"[INFO] Saved chunk: {final_path}")
        except Exception as e:
            print(f"[WARNING] Could not rename file: {e}. Temp kept at: {self.tmp_path}")

        self.tmp_path = None
        self.chunk_start_wall = None

    def _ensure_chunk_for_time_locked(self, t_now_sec: float):
        """
        Must be called with self.lock held.
        """
        if self.f is None:
            self._open_new(0.0)

        # Roll chunks until t_now is within current chunk
        while t_now_sec >= (self.chunk_start_sec + self.chunk_len):
            end_sec = self.chunk_start_sec + self.chunk_len
            self._close_and_rename(end_sec)
            self._open_new(end_sec)

    def write_samples(self, t0_sec: float, voltages: list, currents: list, powers: list, fs: float):
        """
        Enqueue a block for writing. Non-blocking; drops oldest if queue full.
        """
        signal_lists = list(voltages) + list(currents) + list(powers)
        if len(voltages) != len(GPU_CHANNELS) or len(currents) != len(GPU_CHANNELS) or len(powers) != len(GPU_CHANNELS):
            raise ValueError(f"Expected data for {len(GPU_CHANNELS)} GPUs")
        n = min((len(values) for values in signal_lists), default=0)
        if n <= 0:
            return

        # Convert to plain python floats once to keep writer loop fast
        v = [[float(x) for x in values[:n]] for values in voltages]
        i = [[float(x) for x in values[:n]] for values in currents]
        p = [[float(x) for x in values[:n]] for values in powers]

        item = (float(t0_sec), v, i, p, float(fs))

        try:
            self.queue.put(item, block=False)
        except queue.Full:
            # Drop oldest to make room (keep latest data)
            try:
                _ = self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                pass
            try:
                self.queue.put(item, block=False)
            except Exception:
                print("[WARNING] Writer queue full; dropping data block")

    def _writer_loop(self):
        """
        Writer thread: drains queue and writes CSV. Rotates files by chunk_len.
        """
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                t0_sec, v, i, p, fs = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            n = min((len(values) for values in (v + i + p)), default=0)
            if n == 0:
                self.queue.task_done()
                continue

            t_end_sec = t0_sec + (n / fs)
            self._last_written_t_end = max(self._last_written_t_end, t_end_sec)

            with self.lock:
                # Ensure correct chunk open for this block (might roll if needed)
                self._ensure_chunk_for_time_locked(t0_sec)

                # If this block crosses chunk boundary, split it
                idx = 0
                while idx < n:
                    chunk_end = self.chunk_start_sec + self.chunk_len
                    # how many samples fit before chunk_end?
                    remaining_sec = chunk_end - (t0_sec + idx / fs)
                    if remaining_sec <= 0:
                        # roll to next chunk
                        self._close_and_rename(chunk_end)
                        self._open_new(chunk_end)
                        continue

                    take = min(n - idx, int(remaining_sec * fs))
                    if take <= 0:
                        # roll and continue
                        self._close_and_rename(chunk_end)
                        self._open_new(chunk_end)
                        continue

                    inv_fs = 1.0 / fs
                    def make_row(k):
                        row = [f"{t0_sec + (idx + k) * inv_fs:.9f}"]
                        for gpu_idx in range(len(GPU_CHANNELS)):
                            row.extend([
                                f"{v[gpu_idx][idx + k]:.9f}",
                                f"{i[gpu_idx][idx + k]:.9f}",
                                f"{p[gpu_idx][idx + k]:.9f}",
                            ])
                        return row

                    rows = (make_row(k) for k in range(take))
                    try:
                        self.writer.writerows(rows)
                    except Exception as e:
                        print(f"[ERROR] CSV writer error: {e}")

                    idx += take

                    # If we exactly hit boundary, roll for next write
                    if (t0_sec + idx / fs) >= chunk_end:
                        self._close_and_rename(chunk_end)
                        self._open_new(chunk_end)

            self.queue.task_done()

        # Thread exit: close current file with the last written end time
        with self.lock:
            if self.f is not None:
                self._close_and_rename(self._last_written_t_end)

    def finalize(self, t_end_sec: float):
        """
        Stop writer thread and finalize the last file name.
        """
        self.stop_event.set()
        try:
            self.queue.join()
        except Exception:
            pass
        self.thread.join(timeout=10.0)

        with self.lock:
            if self.f is not None:
                self._close_and_rename(float(t_end_sec))


def on_key_press(event):
    global paused, quit_flag
    if event.key == "p":
        paused = not paused
        if paused:
            print("[INFO] Paused display (acquisition/logging continues).")
        else:
            # Clear display buffers so they repopulate quickly after resume
            for buffer_group in (v_buffers, i_buffers, p_buffers):
                for signal_buffer in buffer_group:
                    signal_buffer.clear()
            print("[INFO] Resumed display.")
    elif event.key == "q":
        quit_flag = True
        print("[INFO] Exiting...")


def main():
    global quit_flag, sample_index

    fs = float(SAMPLE_RATE)
    chunk_size = int(CHUNK_SIZE)
    chunk_len_sec = float(SAVE_EVERY_MIN) * 60.0

    physical_channels = [
        (label, f"{DEVICE}/{voltage_channel}", f"{DEVICE}/{current_channel}")
        for label, voltage_channel, current_channel in GPU_CHANNELS
    ]
    terminal_cfg = terminal_config_from_string(TERMINAL)

    out_dir = get_output_dir()
    writer = ChunkCsvWriterBG(
        out_dir=out_dir,
        prefix=f"{FILE_PREFIX}_{DEVICE}_ai0_ai7_4GPU",
        chunk_len_sec=chunk_len_sec,
        start_wall_dt=start_wall,
    )

    with nidaqmx.Task() as task:
        # Add channels in interleaved order: GPU1 V/I, GPU2 V/I, GPU3 V/I, GPU4 V/I.
        for _, volt_phys, curr_phys in physical_channels:
            for physical_channel in (volt_phys, curr_phys):
                task.ai_channels.add_ai_voltage_chan(
                    physical_channel,
                    min_val=float(MIN_V),
                    max_val=float(MAX_V),
                    terminal_config=terminal_cfg,
                )

        # Larger DAQ buffer for high sample rates
        task.timing.cfg_samp_clk_timing(
            rate=fs,
            source="",
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=int(max(chunk_size * 200, int(fs) * 2)),  # >= ~2 seconds buffer
        )

        task.start()

        # Plot setup (3 stacked axes in one figure)
        fig, (ax_v, ax_i, ax_p) = plt.subplots(3, 1, sharex=True)
        fig.canvas.mpl_connect("key_press_event", on_key_press)

        # fig.suptitle(f"{volt_phys} (V) + {curr_phys} (A)  (p=Pause/Resume display, q=Quit)")
        fig.suptitle(MEASUREMENT_NAME)

        for ax in (ax_v, ax_i, ax_p):
            ax.grid(True)

        ax_v.set_ylabel("Voltage (V)")
        ax_i.set_ylabel("Current (A)")
        ax_p.set_ylabel("Power (W)")
        ax_p.set_xlabel("Time (s)")

        lines_v = [ax_v.plot([], [], lw=1, label=label)[0] for label, _, _ in GPU_CHANNELS]
        lines_i = [ax_i.plot([], [], lw=1, label=label)[0] for label, _, _ in GPU_CHANNELS]
        lines_p = [ax_p.plot([], [], lw=1, label=label)[0] for label, _, _ in GPU_CHANNELS]
        for ax in (ax_v, ax_i, ax_p):
            ax.legend(loc="upper left")

        ax_v.set_ylim(float(VOLT_YMIN), float(VOLT_YMAX))
        ax_i.set_ylim(float(CURR_YMIN), float(CURR_YMAX))
        ax_p.set_ylim(float(POWER_YMIN), float(POWER_YMAX))

        ax_p.set_xlim(0.0, float(WINDOW_SEC))

        # Corner time text (always updates)
        time_text = ax_v.text(
            0.02, 0.98, "",
            transform=ax_v.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )


        # Stats text (avg/std over the current on-screen window)
        stats_text_v = ax_v.text(
            0.98, 0.98, "",
            transform=ax_v.transAxes,
            va="top", ha="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )
        stats_text_i = ax_i.text(
            0.98, 0.98, "",
            transform=ax_i.transAxes,
            va="top", ha="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )
        stats_text_p = ax_p.text(
            0.98, 0.98, "",
            transform=ax_p.transAxes,
            va="top", ha="right",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

        # Relative time offsets (seconds) for up to window_points samples
        x_rel = np.arange(window_points, dtype=float) / fs
        last_window_right = None  # used to reduce x-axis tick jitter

        def update(_frame):
            nonlocal last_window_right
            global quit_flag, sample_index

            if quit_flag:
                plt.close(fig)
                return (*lines_v, *lines_i, *lines_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            available = int(task.in_stream.avail_samp_per_chan)

            # Update time text even if nothing to read
            t_current = sample_index / fs
            time_text.set_text(f"t = {t_current:.6f} s")

            if available <= 0:
                return (*lines_v, *lines_i, *lines_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            # If backlog is large, read more to catch up
            if available > 5 * chunk_size:
                n_to_read = min(available, chunk_size * 10)
            else:
                n_to_read = min(chunk_size, available)

            try:
                raw = task.read(number_of_samples_per_channel=int(n_to_read), timeout=0.0)
                ch = ensure_channel_list(raw)
                expected_channels = 2 * len(GPU_CHANNELS)
                if len(ch) < expected_channels:
                    raise RuntimeError(f"Expected {expected_channels} channels from task.read(), got {len(ch)}")
                raw_v = [ch[2 * gpu_idx] for gpu_idx in range(len(GPU_CHANNELS))]
                raw_i = [ch[2 * gpu_idx + 1] for gpu_idx in range(len(GPU_CHANNELS))]
            except Exception as e:
                print(f"[ERROR] DAQ read failed: {e}")
                quit_flag = True
                plt.close(fig)
                return (*lines_v, *lines_i, *lines_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            n = min((len(values) for values in (raw_v + raw_i)), default=0)
            if n <= 0:
                return (*lines_v, *lines_i, *lines_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            raw_v = [values[:n] for values in raw_v]
            raw_i = [values[:n] for values in raw_i]

            # Apply scaling
            v_scaled = [
                [float(value) * float(VOLT_SCALES[gpu_idx]) for value in raw_v[gpu_idx]]
                for gpu_idx in range(len(GPU_CHANNELS))
            ]
            i_scaled = [
                [
                    float(value) * float(CURR_SCALE) * float(CURR_SIGNS[gpu_idx])
                    for value in raw_i[gpu_idx]
                ]
                for gpu_idx in range(len(GPU_CHANNELS))
            ]

            # Compute instantaneous power separately for each GPU.
            p_scaled = [
                [v_scaled[gpu_idx][k] * i_scaled[gpu_idx][k] for k in range(n)]
                for gpu_idx in range(len(GPU_CHANNELS))
            ]

            # Time for first sample in this block
            block_start_idx = sample_index
            block_start_sec = block_start_idx / fs

            # Advance time
            sample_index += n
            t_current = sample_index / fs

            # Update time text
            time_text.set_text(f"t = {t_current:.6f} s")

            # Enqueue for background logging (always, even when paused)
            writer.write_samples(block_start_sec, v_scaled, i_scaled, p_scaled, fs)

            # Freeze display when paused
            if paused:
                return (*lines_v, *lines_i, *lines_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            # Update display buffers
            for gpu_idx in range(len(GPU_CHANNELS)):
                v_buffers[gpu_idx].extend(v_scaled[gpu_idx])
                i_buffers[gpu_idx].extend(i_scaled[gpu_idx])
                p_buffers[gpu_idx].extend(p_scaled[gpu_idx])

            if len(v_buffers[0]) < 1:
                stats_text_v.set_text("")
                stats_text_i.set_text("")
                stats_text_p.set_text("")
                return (*lines_v, *lines_i, *lines_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            # Window always ends at the current (present) time
            window_right = t_current
            window_left = window_right - WINDOW_SEC
            if window_left < 0.0:
                window_left = 0.0

            nbuf = len(v_buffers[0])

            # Align the newest sample to window_right so the curve always reaches the right edge
            x0 = window_right - ((nbuf - 1) / fs) if nbuf > 0 else window_right
            x = x0 + x_rel[:nbuf]

            for gpu_idx in range(len(GPU_CHANNELS)):
                lines_v[gpu_idx].set_data(x, np.asarray(v_buffers[gpu_idx], dtype=float))
                lines_i[gpu_idx].set_data(x, np.asarray(i_buffers[gpu_idx], dtype=float))
                lines_p[gpu_idx].set_data(x, np.asarray(p_buffers[gpu_idx], dtype=float))
            # Update stats text (avg/std over the current window)
            def stats_lines(buffers, unit):
                result = []
                for gpu_idx, (label, _, _) in enumerate(GPU_CHANNELS):
                    values = np.asarray(buffers[gpu_idx], dtype=float)
                    if values.size:
                        result.append(
                            f"{label}: avg={np.mean(values):.3f}, std={np.std(values):.3f} {unit}"
                        )
                return "\n".join(result)

            stats_text_v.set_text(stats_lines(v_buffers, "V"))
            stats_text_i.set_text(stats_lines(i_buffers, "A"))
            stats_text_p.set_text(stats_lines(p_buffers, "W"))


            # Keep the x-axis window anchored to "now" (window_right)
            # and update limits only when time advances enough to matter visually
            if (last_window_right is None) or (abs(window_right - last_window_right) > (10.0 / fs)):
                for ax in (ax_v, ax_i, ax_p):
                    ax.set_xlim(window_left, window_right)
                last_window_right = window_right

            return (*lines_v, *lines_i, *lines_p, time_text, stats_text_v, stats_text_i, stats_text_p)

        _ani = FuncAnimation(
            fig,
            update,
            interval=30,
            blit=False,
            cache_frame_data=False,
        )

        plt.show(block=True)

        try:
            task.stop()
        except Exception:
            pass

    # Finalize writer with final time
    try:
        writer.finalize(sample_index / float(SAMPLE_RATE))
    except Exception:
        pass

    print(f"[INFO] Output folder: {out_dir}")
    print("[INFO] Program terminated.")


if __name__ == "__main__":
    main()