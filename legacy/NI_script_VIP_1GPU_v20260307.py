# NI_script_2sig.py
# Realtime NI-DAQmx acquisition (3 inputs) + oscilloscope-like scrolling plots + chunked CSV logging (background thread)
# Keys: p = pause/resume DISPLAY only (acquisition & logging continue), y = auto-scale y, f = reset y, q = quit

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
VOLT_SCALE = 4.768   # scaled_voltage = raw_voltage * VOLT_SCALE
CURR_SCALE = 1.0/0.08   # scaled_current = raw_current * CURR_SCALE


# =========================
# Power moving average
# =========================
# Instantaneous power is computed sample-by-sample as V*I, then smoothed with a causal moving average
# over the most recent N samples (set to 5 or 10 as desired).
POWER_AVG_SAMPLES = 10  # set to 5 or 10; 1 disables averaging

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
VOLTAGE_CHANNEL = "ai0"   # voltage
CURRENT_CHANNEL = "ai1"   # current 1 (ai1)
CURRENT2_CHANNEL = "ai2"  # current 2 (ai2)
TERMINAL = "RSE"          # "RSE" / "DIFF" / "NRSE"

# Raw input range for both channels (DAQ read range; keep broad enough for both signals)
MIN_V = -0.5
MAX_V = 3.5

SAMPLE_RATE = 10000       # Hz

# =========================
# Delay compensation (integer sample delays)
# =========================
# These delays are expressed in integer multiples of the sampling period (1/SAMPLE_RATE).
# They are applied as *timestamp offsets*:
#   t_v = t_nominal + VOLT_DELAY_SAMPLES / SAMPLE_RATE
#   t_i = t_nominal + CURR_DELAY_SAMPLES / SAMPLE_RATE
#
# Power is computed using delay-compensated V and I samples that share the same timestamp
# (i.e., the streams are aligned by an integer-sample delay line based on the difference
# between these two delays). Adjust these values in the future to compensate measured phase lag.
VOLT_DELAY_SAMPLES = 0   # e.g., +1 means shift voltage timestamps later by 1 sample period
CURR_DELAY_SAMPLES = 0   # e.g., +1 means shift current timestamps later by 1 sample period

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
v_buffer = deque(maxlen=window_points)
i_buffer = deque(maxlen=window_points)
p_buffer = deque(maxlen=window_points)


# Alignment delay-line state (initialized in main once SAMPLE_RATE is known)
# NOTE: these deques hold *scaled* samples (post VOLT_SCALE / CURR_SCALE) for alignment.
v_align_line = None  # deque
i_align_line = None  # deque (total current)
i1_align_line = None  # deque
i2_align_line = None  # deque
align_offset = 0     # VOLT_DELAY_SAMPLES - CURR_DELAY_SAMPLES


# Matplotlib axes handles (set in main)
ax_v = None
ax_i = None
ax_p = None


# sample_index
# Holds last (POWER_AVG_SAMPLES-1) instantaneous power samples to ensure the moving average
# is continuous across DAQ read blocks.
p_inst_prev = deque(maxlen=max(int(POWER_AVG_SAMPLES) - 1, 0))

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


def moving_average_causal(x: np.ndarray, n: int) -> np.ndarray:
    """
    Causal moving average of length n.
    For the first n-1 samples, uses the average of available samples (window grows from 1..n).
    """
    n = int(n)
    if n <= 1 or x.size == 0:
        return x.astype(float, copy=False)

    x = x.astype(float, copy=False)
    c = np.cumsum(x)
    k = np.arange(x.size)

    start = k - (n - 1)
    start_clip = np.clip(start, 0, None)

    # cumsum at index (start_clip-1); 0 when start_clip == 0
    c_prev = np.concatenate(([0.0], c[:-1]))
    sum_window = c - np.where(start_clip > 0, c_prev[start_clip], 0.0)
    denom = k - start_clip + 1
    return sum_window / denom


class ChunkCsvWriterBG:
    """
    Background CSV writer:
      - main thread enqueues blocks: (t0_sec, v_list, i_list, p_list, fs)
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
        self.writer.writerow(["time_s", "voltage_v", "current1_a", "current2_a", "total_power_w"])

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

    def write_samples(self, t0_sec: float, v_list: list, i1_list: list, i2_list: list, p_list: list, fs: float):
        """
        Enqueue a block for writing. Non-blocking; drops oldest if queue full.
        """
        n = min(len(v_list), len(i1_list), len(i2_list), len(p_list))
        if n <= 0:
            return

        # Convert to plain python floats once to keep writer loop fast
        v = [float(x) for x in v_list[:n]]
        i1 = [float(x) for x in i1_list[:n]]
        i2 = [float(x) for x in i2_list[:n]]
        p = [float(x) for x in p_list[:n]]

        item = (float(t0_sec), v, i1, i2, p, float(fs))

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
                t0_sec, v, i1, i2, p, fs = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            n = min(len(v), len(i1), len(i2), len(p))
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
                    rows = (
                        (f"{t0_sec + (idx + k) * inv_fs:.9f}",
                         f"{v[idx + k]:.9f}",
                         f"{i1[idx + k]:.9f}",
                         f"{i2[idx + k]:.9f}",
                         f"{p[idx + k]:.9f}")
                        for k in range(take)
                    )
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



def _autoscale_axis(ax, data, usage=0.90):
    """Auto-set y-limits so the data uses ~`usage` fraction of the y-axis.
    Uses symmetric margins above/below based on current data range.
    """
    if ax is None:
        return
    if not data:
        return
    dmin = min(data)
    dmax = max(data)
    if dmin == dmax:
        # Avoid zero range; give a small span
        span = abs(dmin) * 0.10 if dmin != 0 else 1.0
        dmin -= span
        dmax += span

    rng = dmax - dmin
    # Choose margin so rng / (rng + 2*margin) ~= usage
    margin = ((1.0 / usage) - 1.0) * 0.5 * rng
    ax.set_ylim(dmin - margin, dmax + margin)


def on_key_press(event):
    global paused, quit_flag, v_buffer, i_buffer, p_buffer, ax_v, ax_i, ax_p
    if event.key == "p":
        paused = not paused
        if paused:
            print("[INFO] Paused display (acquisition/logging continues).")
        else:
            # Clear display buffers so they repopulate quickly after resume
            v_buffer.clear()
            i_buffer.clear()
            p_buffer.clear()
            print("[INFO] Resumed display.")
    elif event.key == "y":
        # Auto-scale y-limits for all three plots so traces use ~90% of the y-axis
        _autoscale_axis(ax_v, list(v_buffer), usage=0.90)
        _autoscale_axis(ax_i, list(i_buffer), usage=0.90)
        _autoscale_axis(ax_p, list(p_buffer), usage=0.90)
        # Redraw immediately
        try:
            event.canvas.draw_idle()
        except Exception:
            pass
    elif event.key == "f":
        # Reset y-limits to original hard-coded values
        try:
            ax_v.set_ylim(float(VOLT_YMIN), float(VOLT_YMAX))
            ax_i.set_ylim(float(CURR_YMIN), float(CURR_YMAX))
            ax_p.set_ylim(float(POWER_YMIN), float(POWER_YMAX))
            event.canvas.draw_idle()
        except Exception:
            pass
    
    elif event.key == "q":
        quit_flag = True
        print("[INFO] Exiting...")


def main():
    global quit_flag, sample_index, ax_v, ax_i, ax_p

    fs = float(SAMPLE_RATE)
    chunk_size = int(CHUNK_SIZE)

    # --- Delay alignment setup (integer-sample delay lines) ---
    # Offset > 0 means voltage timestamps are shifted later than current timestamps by `offset` samples.
    # To compute power at common timestamps, we delay the *earlier* stream by |offset|.
    global v_align_line, i_align_line, align_offset
    align_offset = int(VOLT_DELAY_SAMPLES) - int(CURR_DELAY_SAMPLES)

    if align_offset > 0:
        # Need voltage from `align_offset` samples earlier to match current timestamps
        from collections import deque as _dq
        v_align_line = _dq(maxlen=align_offset + 1)
        i_align_line = None
        i1_align_line = None
        i2_align_line = None
    elif align_offset < 0:
        # Need current from `-align_offset` samples earlier to match voltage timestamps
        from collections import deque as _dq
        need = (-align_offset) + 1
        i_align_line = _dq(maxlen=need)   # total current
        i1_align_line = _dq(maxlen=need)  # current 1
        i2_align_line = _dq(maxlen=need)  # current 2
        v_align_line = None
    else:
        v_align_line = None
        i_align_line = None
        i1_align_line = None
        i2_align_line = None
    chunk_len_sec = float(SAVE_EVERY_MIN) * 60.0

    volt_phys = f"{DEVICE}/{VOLTAGE_CHANNEL}"
    curr1_phys = f"{DEVICE}/{CURRENT_CHANNEL}"
    curr2_phys = f"{DEVICE}/{CURRENT2_CHANNEL}"
    terminal_cfg = terminal_config_from_string(TERMINAL)

    out_dir = get_output_dir()
    writer = ChunkCsvWriterBG(
        out_dir=out_dir,
        prefix=f"{FILE_PREFIX}_{DEVICE}_{VOLTAGE_CHANNEL}_{CURRENT_CHANNEL}_{CURRENT2_CHANNEL}",
        chunk_len_sec=chunk_len_sec,
        start_wall_dt=start_wall,
    )

    with nidaqmx.Task() as task:
        # IMPORTANT: add channels in a known order so we can unpack reads:
        #   channel 0: voltage (ai0)
        #   channel 1: current (ai1)
        task.ai_channels.add_ai_voltage_chan(
            volt_phys,
            min_val=float(MIN_V),
            max_val=float(MAX_V),
            terminal_config=terminal_cfg,
        )
        task.ai_channels.add_ai_voltage_chan(
            curr1_phys,
            min_val=float(MIN_V),
            max_val=float(MAX_V),
            terminal_config=terminal_cfg,
        )
        task.ai_channels.add_ai_voltage_chan(
            curr2_phys,
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

        # fig.suptitle(f"{volt_phys} (V) + {curr1_phys} (A) + {curr2_phys} (A)  (p=Pause/Resume display, y=Auto-scale, f=Reset y, q=Quit)")
        fig.suptitle(MEASUREMENT_NAME)

        for ax in (ax_v, ax_i, ax_p):
            ax.grid(True)

        ax_v.set_ylabel("Voltage (V)")
        ax_i.set_ylabel("Total Current (A)")
        ax_p.set_ylabel("Power (W)")
        ax_p.set_xlabel("Time (s)")

        line_v, = ax_v.plot([], [], lw=1)
        line_i, = ax_i.plot([], [], lw=1)
        line_p, = ax_p.plot([], [], lw=1)

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

        
        def moving_average_causal_ignore_nan(x: np.ndarray, n: int) -> np.ndarray:
            """Causal moving average over last n samples, ignoring NaNs."""
            n = int(n)
            if n <= 1:
                return x.astype(float, copy=False)

            out = np.empty_like(x, dtype=float)
            for k in range(len(x)):
                start = max(0, k - n + 1)
                window = x[start:k+1]
                valid = window[~np.isnan(window)]
                out[k] = np.mean(valid) if valid.size > 0 else np.nan
            return out


        def update(_frame):
            nonlocal last_window_right
            global quit_flag, sample_index

            if quit_flag:
                plt.close(fig)
                return (line_v, line_i, line_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            available = int(task.in_stream.avail_samp_per_chan)

            # Update time text even if nothing to read
            t_current = sample_index / fs
            time_text.set_text(f"t = {t_current:.6f} s")

            if available <= 0:
                return (line_v, line_i, line_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            # If backlog is large, read more to catch up
            if available > 5 * chunk_size:
                n_to_read = min(available, chunk_size * 10)
            else:
                n_to_read = min(chunk_size, available)

            try:
                raw = task.read(number_of_samples_per_channel=int(n_to_read), timeout=0.0)
                ch = ensure_channel_list(raw)
                if len(ch) < 3:
                    raise RuntimeError("Expected 3 channels from task.read()")
                raw_v = ch[0]
                raw_i1 = ch[1]
                raw_i2 = ch[2]
            except Exception as e:
                print(f"[ERROR] DAQ read failed: {e}")
                quit_flag = True
                plt.close(fig)
                return (line_v, line_i, line_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            n = min(len(raw_v), len(raw_i1), len(raw_i2))
            if n <= 0:
                return (line_v, line_i, line_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            raw_v = raw_v[:n]
            raw_i1 = raw_i1[:n]
            raw_i2 = raw_i2[:n]

            # Apply scaling
            v_scaled = [float(v) * float(VOLT_SCALE) for v in raw_v]
            i1_scaled = [float(i) * float(CURR_SCALE) for i in raw_i1]
            i2_scaled = [float(i) * float(CURR_SCALE) for i in raw_i2]
            i_total_scaled = [a + b for a, b in zip(i1_scaled, i2_scaled)]

            # --- Delay-compensated timestamps (for future calibration / debugging) ---
            # Nominal timestamps are based on the DAQ sample clock index.
            # We then add per-channel integer-sample delays.
            block_start_idx = sample_index
            idx_arr = np.arange(block_start_idx, block_start_idx + n, dtype=np.int64)
            t_nom = idx_arr.astype(float) / fs
            t_v = t_nom + (float(VOLT_DELAY_SAMPLES) / fs)
            t_i = t_nom + (float(CURR_DELAY_SAMPLES) / fs)

            # --- Align V and I onto a common timestamp grid for power computation ---
            # Because delays are integer multiples of Ts, the streams can be aligned by an
            # integer-sample delay line. We compute V_aligned and I_aligned such that each
            # power sample uses V and I with the same timestamp.
            #
            # Common timestamp choice:
            #   align_offset > 0  -> output timestamps follow current: t_common = t_i
            #   align_offset < 0  -> output timestamps follow voltage: t_common = t_v
            #   align_offset == 0 -> t_common = t_nom (either)
            v_aligned = np.full(n, np.nan, dtype=float)
            i_total_aligned = np.full(n, np.nan, dtype=float)
            i1_aligned = np.full(n, np.nan, dtype=float)
            i2_aligned = np.full(n, np.nan, dtype=float)

            if align_offset == 0:
                v_aligned[:] = np.asarray(v_scaled, dtype=float)
                i1_aligned[:] = np.asarray(i1_scaled, dtype=float)
                i2_aligned[:] = np.asarray(i2_scaled, dtype=float)
                i_total_aligned[:] = i1_aligned + i2_aligned
                t_common = t_nom
            elif align_offset > 0:
                # Voltage timestamps are later; delay voltage stream by align_offset samples to match current timestamps.
                # For each current sample at index k, use voltage sample from k-align_offset (available via delay line).
                t_common = t_i
                for k in range(n):
                    v_align_line.append(float(v_scaled[k]))
                    if len(v_align_line) > align_offset:
                        v_aligned[k] = float(v_align_line[0])
                        i1_aligned[k] = float(i1_scaled[k])
                        i2_aligned[k] = float(i2_scaled[k])
                        i_total_aligned[k] = i1_aligned[k] + i2_aligned[k]
            else:
                # Current timestamps are later; delay current streams by -align_offset samples to match voltage timestamps.
                t_common = t_v
                need = -align_offset
                for k in range(n):
                    i1_align_line.append(float(i1_scaled[k]))
                    i2_align_line.append(float(i2_scaled[k]))
                    i_align_line.append(float(i_total_scaled[k]))
                    if len(i_align_line) > need:
                        i1_aligned[k] = float(i1_align_line[0])
                        i2_aligned[k] = float(i2_align_line[0])
                        i_total_aligned[k] = float(i_align_line[0])
                        v_aligned[k] = float(v_scaled[k])

            # Compute instantaneous power using aligned samples only
            # total_power = voltage * (current1 + current2)
            p_inst = v_aligned * i_total_aligned
            # Apply causal moving average over the most recent POWER_AVG_SAMPLES samples.
            # We ignore NaNs (which appear during initial delay-line warm-up).
            if int(POWER_AVG_SAMPLES) <= 1:
                p_scaled = p_inst.tolist()
            else:
                prev = np.fromiter(p_inst_prev, dtype=float) if len(p_inst_prev) else np.empty(0, dtype=float)
                p_concat = np.concatenate([prev, p_inst]) if prev.size else p_inst
                p_ma = moving_average_causal_ignore_nan(p_concat, int(POWER_AVG_SAMPLES))
                p_scaled = p_ma[-n:].tolist()

                # Update history with last (N-1) instantaneous samples (including NaN warm-up)
                keep = int(POWER_AVG_SAMPLES) - 1
                if keep > 0:
                    tail = p_concat[-keep:] if p_concat.size >= keep else p_concat
                    p_inst_prev.clear()
                    p_inst_prev.extend(tail.tolist())

            # Use aligned V/I for display and logging as well (so each row corresponds to the same timestamp)
            v_out = v_aligned.tolist()
            i1_out = i1_aligned.tolist()
            i2_out = i2_aligned.tolist()
            i_total_out = i_total_aligned.tolist()
            # Time for first sample in this block
            block_start_idx = sample_index
            block_start_sec = block_start_idx / fs

            # Advance time
            sample_index += n
            t_current = sample_index / fs

            # Update time text
            time_text.set_text(f"t = {t_current:.6f} s")

            # Enqueue for background logging (always, even when paused)
            writer.write_samples(block_start_sec, v_out, i1_out, i2_out, p_scaled, fs)

            # Freeze display when paused
            if paused:
                return (line_v, line_i, line_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            # Update display buffers
            v_buffer.extend(v_out)
            i_buffer.extend(i_total_out)
            p_buffer.extend(p_scaled)

            if len(v_buffer) < 1:
                stats_text_v.set_text("")
                stats_text_i.set_text("")
                stats_text_p.set_text("")
                return (line_v, line_i, line_p, time_text, stats_text_v, stats_text_i, stats_text_p)

            # Window always ends at the current (present) time
            window_right = t_current
            window_left = window_right - WINDOW_SEC
            if window_left < 0.0:
                window_left = 0.0

            nbuf = len(v_buffer)

            # Align the newest sample to window_right so the curve always reaches the right edge
            x0 = window_right - ((nbuf - 1) / fs) if nbuf > 0 else window_right
            x = x0 + x_rel[:nbuf]

            line_v.set_data(x, np.asarray(v_buffer, dtype=float))
            line_i.set_data(x, np.asarray(i_buffer, dtype=float))
            line_p.set_data(x, np.asarray(p_buffer, dtype=float))
            # Update stats text (avg/std over the current window)
            v_arr = np.asarray(v_buffer, dtype=float)
            i_arr = np.asarray(i_buffer, dtype=float)
            p_arr = np.asarray(p_buffer, dtype=float)

            if v_arr.size > 0:
                v_avg = float(np.mean(v_arr))
                v_std = float(np.std(v_arr))
                stats_text_v.set_text(f"avg = {v_avg:.3f} V\nstd = {v_std:.3f} V")
            else:
                stats_text_v.set_text("")

            if i_arr.size > 0:
                i_avg = float(np.mean(i_arr))
                i_std = float(np.std(i_arr))
                stats_text_i.set_text(f"avg = {i_avg:.3f} A\nstd = {i_std:.3f} A")
            else:
                stats_text_i.set_text("")

            if p_arr.size > 0:
                p_avg = float(np.mean(p_arr))
                p_std = float(np.std(p_arr))
                stats_text_p.set_text(f"avg = {p_avg:.3f} W\nstd = {p_std:.3f} W")
            else:
                stats_text_p.set_text("")


            # Keep the x-axis window anchored to "now" (window_right)
            # and update limits only when time advances enough to matter visually
            if (last_window_right is None) or (abs(window_right - last_window_right) > (10.0 / fs)):
                for ax in (ax_v, ax_i, ax_p):
                    ax.set_xlim(window_left, window_right)
                last_window_right = window_right

            return (line_v, line_i, line_p, time_text, stats_text_v, stats_text_i, stats_text_p)

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
