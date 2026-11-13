"""Live audio input: microphone in, waves on the model, take on disk.

Three jobs, in the order you use them:

1.  Drive the generated object's shape keys from an input device in real
    time, so the mesh reacts to the room while you watch.
2.  Record the same audio to a WAV, because a live performance only
    happens once.
3.  Hand that WAV to the normal Generate & Bake, which turns the take into
    keyframes that render and export like any other track.

The live view and the bake read the audio through the same log-spaced bands
(analysis.log_band_masks) and drive the same shape keys with the same
travelling-wave maths as baking.bake_spectral_ripple, so what you see live
is what the bake gives back.

Blender does not ship sounddevice, and its own aud module cannot record, so
capture needs that one wheel installed into Blender's own Python:

    python -m pip install sounddevice

Without it the operators say so instead of failing quietly.
"""

import os
import queue
import threading
import time
import wave

import bpy
import numpy as np
from bpy.types import Operator

from . import analysis
from .constants import (OBJ_NAME, SK_BASS, SK_BIN, SK_MID, SK_RIPPLE_COS,
                        SK_RIPPLE_SIN, SK_TREBLE)

try:
    import sounddevice as sd
except Exception:  # no wheel, or PortAudio missing on the system
    sd = None

# Capture block: small enough to stay responsive, big enough that the
# callback is not called thousands of times a second.
BLOCK = 1024
# Analysis window, independent of BLOCK so the band edges stay put even when
# the device hands over short blocks.
FFT = 2048

# The running capture, so the panel can show its state and the stop button
# can reach it. One stream at a time.
ACTIVE = [None]
_STOP = [False]

# Where the last recording landed, for "Bake Last Take" after the fact.
LAST_TAKE = {"path": "", "seconds": 0.0, "rate": 0}

_DEVICES = {"stamp": 0.0, "items": []}


def available():
    return sd is not None


def device_list(force=False):
    """(identifier, name, description) for every input device.

    Cached: this is an EnumProperty callback, so it runs on every redraw,
    and query_devices() talks to the audio backend each time.
    """
    if sd is None:
        return []

    now = time.time()
    if not force and _DEVICES["items"] and now - _DEVICES["stamp"] < 5.0:
        return _DEVICES["items"]

    items = []
    try:
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] < 1:
                continue
            items.append((
                str(i), dev["name"][:64],
                "%d in, %.0f Hz" % (dev["max_input_channels"],
                                    dev["default_samplerate"])))
    except Exception:
        items = []

    # Blender keeps no reference to the strings an items callback returns, so
    # this cache is also what stops them being collected mid-draw.
    _DEVICES.update(stamp=now, items=items)
    return items


def device_default(props):
    """(index or None, rate, channels) for the chosen device."""
    index = None
    if props.live_device:
        try:
            index = int(props.live_device)
        except ValueError:
            index = None

    rate, channels = int(props.live_rate), 1
    if sd is not None:
        try:
            info = sd.query_devices(index, "input")
            channels = max(1, min(2, int(info["max_input_channels"])))
            if not rate:
                rate = int(info["default_samplerate"])
        except Exception:
            pass
    return index, rate, channels


def _rate_candidates(device, rate):
    """The asked-for rate first, then whatever the device prefers."""
    rates = [int(rate)]
    if sd is not None:
        try:
            default = int(sd.query_devices(device, "input")["default_samplerate"])
            if default and default not in rates:
                rates.append(default)
        except Exception:
            pass
    for fallback in (48000, 44100):
        if fallback not in rates:
            rates.append(fallback)
    return rates


class Capture:
    """Owns the input stream.

    The audio thread only appends; the UI thread only reads. Nothing here
    touches bpy: the callback runs off the main thread, and the Blender API
    is not safe there.
    """

    def __init__(self, device, rate, channels, record=True, max_seconds=300):
        self.rate = int(rate)
        self.channels = int(channels)
        self.record = bool(record)
        self.max_samples = int(max(1.0, max_seconds) * self.rate)

        self.monitor = queue.Queue(maxsize=4)
        self._blocks = []
        self._lock = threading.Lock()
        self.recorded = 0
        self.captured = 0
        self.dropped = 0
        self.full = False
        self.started = 0.0

        self.stream = sd.InputStream(
            device=device, channels=self.channels, samplerate=self.rate,
            blocksize=BLOCK, dtype="float32", callback=self._callback)

    def _callback(self, indata, _frames, _time_info, status):
        if status:
            # Overruns are routine on a busy machine and cost one block.
            self.dropped += 1

        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = indata.mean(axis=1)
        else:
            mono = indata.reshape(-1)
        mono = np.asarray(mono, dtype=np.float32).copy()
        self.captured += mono.size

        if self.record:
            with self._lock:
                if self.recorded < self.max_samples:
                    self._blocks.append(mono)
                    self.recorded += mono.size
                else:
                    self.full = True

        try:
            self.monitor.put_nowait(mono)
        except queue.Full:
            # The viewport wants the newest audio, not a backlog of it.
            pass

    def latest(self):
        """Every block waiting, oldest first, or None."""
        blocks = []
        while True:
            try:
                blocks.append(self.monitor.get_nowait())
            except queue.Empty:
                break
        return blocks or None

    def start(self):
        self.stream.start()
        self.started = time.time()

    def close(self):
        try:
            self.stream.stop()
        finally:
            self.stream.close()

    @property
    def seconds(self):
        count = self.recorded if self.record else self.captured
        return count / float(self.rate)

    def take(self):
        """The recording as one float32 array."""
        with self._lock:
            blocks = list(self._blocks)
        if not blocks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(blocks).astype(np.float32)


class Analyser:
    """One audio block in, levels out.

    Levels are normalised the way analysis.spectrogram normalises a baked
    spectrogram -- dB against a reference -- so a loud passage sits near 1.0
    whatever the input gain is. Here the reference decays instead of being
    the whole file's maximum, since the loudest moment has not happened yet.
    Its floor, plus the noise gate, is what stops a quiet room from being
    amplified into full-scale hiss.
    """

    REF_FLOOR_DB = -24.0   # a reference quieter than this means silence
    REF_DECAY_DB = 0.6     # per update, so levels follow a fade down

    def __init__(self, rate, bins, props):
        self.rate = float(rate)
        self.bins = int(max(1, bins))
        self.window = np.hanning(FFT).astype(np.float32)
        self.buffer = np.zeros(FFT, dtype=np.float32)

        freqs = np.fft.rfftfreq(FFT, 1.0 / self.rate)
        nyquist = self.rate * 0.5
        self.masks = analysis.log_band_masks(
            freqs, 30.0, min(16000.0, nyquist), self.bins)
        self.band_masks = [
            (freqs >= 20.0) & (freqs < props.bass_max),
            (freqs >= props.bass_max) & (freqs < props.mid_max),
            (freqs >= props.mid_max) & (freqs < nyquist),
        ]

        self.ref_db = self.REF_FLOOR_DB
        self.levels = np.zeros(self.bins, dtype=np.float32)
        self.bands = np.zeros(3, dtype=np.float32)
        self.rms = 0.0
        self.onset = 0.0
        self._previous = np.zeros(self.bins, dtype=np.float32)

    def push(self, block):
        """Slide a block into the rolling analysis window."""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size >= FFT:
            self.buffer[:] = block[-FFT:]
        elif block.size:
            self.buffer = np.roll(self.buffer, -block.size)
            self.buffer[-block.size:] = block

    def _rms_per_mask(self, mag, masks):
        return np.array(
            [float(np.sqrt(np.mean(mag[m] ** 2))) if m.any() else 0.0
             for m in masks], dtype=np.float32)

    def update(self, props):
        raw_rms = float(np.sqrt(np.mean(self.buffer ** 2)))
        mag = np.abs(np.fft.rfft(self.buffer * self.window))
        gain = float(props.live_gain)

        bins = self._rms_per_mask(mag, self.masks) * gain
        bands = self._rms_per_mask(mag, self.band_masks) * gain

        db = 20.0 * np.log10(np.maximum(bins, 1e-9))
        self.ref_db = max(float(db.max()), self.ref_db - self.REF_DECAY_DB,
                          self.REF_FLOOR_DB)
        span = max(float(props.dynamic_range), 1e-3)
        floor_db = self.ref_db - span
        norm = np.clip((db - floor_db) / span, 0.0, 1.0)

        # Noise gate, measured in dB above the threshold, so a quiet room
        # reads as still rather than as a shimmer of microphone hiss.
        rms_db = 20.0 * np.log10(max(raw_rms * gain, 1e-9))
        gate = float(np.clip((rms_db - float(props.live_gate_db)) / 6.0,
                             0.0, 1.0))
        gate = gate * gate * (3.0 - 2.0 * gate)   # smoothstep
        norm = (norm * gate).astype(np.float32)

        # Spectral flux: new energy only, which is what reads as a hit.
        flux = float(np.maximum(norm - self._previous, 0.0).sum()
                     / max(self.bins, 1))
        self._previous = norm

        keep = float(np.clip(props.live_smoothing, 0.0, 0.99))
        blend = 1.0 - keep
        band_norm = np.clip(
            (20.0 * np.log10(np.maximum(bands, 1e-9)) - floor_db) / span,
            0.0, 1.0).astype(np.float32) * gate

        self.levels = self.levels * keep + norm * blend
        self.bands = self.bands * keep + band_norm * blend
        self.rms = self.rms * keep + min(raw_rms * gain * 4.0, 2.0) * blend
        # Onsets have to stay sharp, so they attack instantly and only the
        # decay is smoothed.
        self.onset = max(flux * 6.0, self.onset * 0.82)
        return self.levels, self.bands, self.rms, self.onset


class Driver:
    """Writes live levels onto whatever shape keys the object happens to
    have, and puts back the values it found when the stream stops.

    Ripple pairs get the travelling wave from bake_spectral_ripple --
    SIN = amp*cos(phase), COS = -amp*sin(phase) -- with the phase advanced
    by real elapsed time instead of by frame number.
    """

    def __init__(self, obj, props):
        self.obj = obj
        keys = obj.data.shape_keys.key_blocks

        self.pairs = []
        while True:
            i = len(self.pairs)
            sin = keys.get(SK_RIPPLE_SIN % i)
            cos = keys.get(SK_RIPPLE_COS % i)
            if sin is None or cos is None:
                break
            self.pairs.append((sin, cos))

        self.bins = []
        while True:
            key = keys.get(SK_BIN % len(self.bins))
            if key is None:
                break
            self.bins.append(key)

        self.bands = [keys.get(SK_BASS), keys.get(SK_MID), keys.get(SK_TREBLE)]
        self.gains = (props.bass_gain, props.mid_gain, props.treble_gain)

        n = max(len(self.pairs), len(self.bins), 1)
        self.components = n
        self.phase = np.zeros(n, dtype=np.float64)
        ratio = (np.linspace(0.0, 1.0, n, dtype=np.float64) if n > 1
                 else np.zeros(1))
        # Same spread as the baker: higher components travel faster.
        self.speeds = props.ripple_speed * (
            2.0 ** (ratio * props.ripple_spread * 0.5))

        self._restore = [(k, k.value) for k in self._touched()]
        self._transform = (tuple(obj.scale), tuple(obj.rotation_euler))

    def _touched(self):
        for sin, cos in self.pairs:
            yield sin
            yield cos
        for key in self.bins:
            yield key
        for key in self.bands:
            if key is not None:
                yield key

    def update(self, levels, bands, rms, onset, dt, props):
        gain = float(props.master_gain)

        for i, (sin, cos) in enumerate(self.pairs):
            amp = float(np.clip(levels[i] * gain, 0.0, 2.0))
            self.phase[i] += 2.0 * np.pi * dt * (
                self.speeds[i] + onset * float(props.ripple_beat_gain))
            sin.value = amp * float(np.cos(self.phase[i]))
            cos.value = -amp * float(np.sin(self.phase[i]))

        for i, key in enumerate(self.bins):
            key.value = float(np.clip(levels[i] * gain, 0.0, 2.0))

        for key, value, band_gain in zip(self.bands, bands, self.gains):
            if key is not None:
                key.value = float(np.clip(value * band_gain * gain, 0.0, 2.0))

        scale = float(props.base_scale) * (
            1.0 + rms * float(props.scale_gain)
            + onset * float(props.pulse_gain))
        self.obj.scale = (scale, scale, scale)

        spin = float(props.spin_speed) + onset * float(props.spin_gain)
        if abs(spin) > 1e-6:
            self.obj.rotation_euler.z += spin * dt

    def restore(self):
        try:
            for key, value in self._restore:
                key.value = value
            self.obj.scale = self._transform[0]
            self.obj.rotation_euler = self._transform[1]
        except ReferenceError:
            pass  # the object was deleted mid-session


def write_wav(path, samples, rate):
    """Write mono 16-bit PCM and return (path, seconds, peak).

    The take is normalised on the way out: microphone input usually peaks
    tens of dB below full scale, and 16-bit would keep only a handful of
    bits of it. The bake peak-normalises on load anyway, so this changes
    nothing downstream except the resolution it starts from. The returned
    peak is the level before that, which is the number worth reporting.
    """
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise RuntimeError("Nothing was recorded.")

    peak = float(np.max(np.abs(samples)))
    if peak > 1e-6:
        samples = samples * (0.98 / peak)
    pcm = np.clip(samples * 32767.0, -32768.0, 32767.0).astype("<i2")

    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with wave.open(path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(int(rate))
        out.writeframes(pcm.tobytes())

    return path, samples.size / float(rate), peak


def take_path(props):
    """Where this take should be written.

    A directory -- or a blend-relative path with nothing to be relative to
    yet -- becomes a timestamped file, so one session's takes do not
    overwrite each other.
    """
    raw = (props.live_record_path or "").strip()
    if raw.startswith("//") and not bpy.data.is_saved:
        raw = ""
    path = bpy.path.abspath(raw) if raw else ""
    if not path or os.path.isdir(path) or path.endswith(("/", "\\")):
        folder = path if path and os.path.isdir(path) else bpy.app.tempdir
        path = os.path.join(folder, time.strftime("live_%Y%m%d_%H%M%S.wav"))
    if not path.lower().endswith(".wav"):
        path += ".wav"
    return path


def stop_stream():
    """Close the stream if one is open. Safe to call from anywhere."""
    capture = ACTIVE[0]
    ACTIVE[0] = None
    _STOP[0] = False
    if capture is not None:
        try:
            capture.close()
        except Exception:
            pass
    return capture


def status_line():
    """One line for the panel while a stream is open."""
    capture = ACTIVE[0]
    if capture is None:
        return ""
    mark = "recording" if capture.record else "monitoring"
    return "%s %0.1f s" % (mark, capture.seconds)


class AUDIOGEO_OT_live(Operator):
    bl_idname = "audiogeo.live"
    bl_label = "Start Live Input"
    bl_description = ("Drive the generated object from a microphone or a "
                      "loopback device in real time, and record the take")
    bl_options = {"REGISTER"}

    _timer = None
    _driver = None
    _capture = None
    _analyser = None
    _last = 0.0

    @classmethod
    def poll(cls, context):
        obj = bpy.data.objects.get(OBJ_NAME)
        return (ACTIVE[0] is None and obj is not None
                and obj.data.shape_keys is not None)

    def invoke(self, context, event):
        if not available():
            self.report(
                {"ERROR"},
                "Live input needs the 'sounddevice' module: run "
                "'python -m pip install sounddevice' with Blender's own "
                "Python, then restart Blender.")
            return {"CANCELLED"}

        props = context.scene.audiogeo
        obj = bpy.data.objects.get(OBJ_NAME)
        if obj is None or obj.data.shape_keys is None:
            self.report({"ERROR"}, "Generate the object first.")
            return {"CANCELLED"}

        device, rate, channels = device_default(props)
        capture = None
        for attempt_rate in _rate_candidates(device, rate):
            try:
                capture = Capture(
                    device, attempt_rate, channels, record=props.live_record,
                    max_seconds=props.live_record_max)
                capture.start()
                break
            except Exception as exc:
                capture, error = None, exc
        if capture is None:
            self.report({"ERROR"},
                        "Could not open the input device: %s" % error)
            return {"CANCELLED"}
        if capture.rate != rate:
            # Plenty of devices are locked to one rate, and refusing to
            # record because the box says 44100 would be unhelpful.
            self.report({"INFO"},
                        "Device runs at %d Hz, not %d." % (capture.rate, rate))

        self._capture = capture
        self._driver = Driver(obj, props)
        self._analyser = Analyser(capture.rate, self._driver.components, props)
        self._last = time.perf_counter()

        ACTIVE[0] = capture
        _STOP[0] = False
        self._timer = context.window_manager.event_timer_add(
            1.0 / max(10.0, float(props.live_fps)), window=context.window)
        context.window_manager.modal_handler_add(self)

        props.last_report = "Live input on -- Esc or Stop to finish"
        context.workspace.status_text_set("Spectroform live input -- Esc to stop")
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if _STOP[0] or event.type in {"ESC", "RIGHTMOUSE"}:
            return self._finish(context)
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        capture = self._capture
        if capture is None:
            return self._finish(context)

        props = context.scene.audiogeo
        blocks = capture.latest()
        if blocks is None:
            return {"RUNNING_MODAL"}
        for block in blocks:
            self._analyser.push(block)

        now = time.perf_counter()
        dt = min(max(now - self._last, 1e-4), 0.25)
        self._last = now

        levels, bands, rms, onset = self._analyser.update(props)
        self._driver.update(levels, bands, rms, onset, dt, props)
        _redraw(context)

        if capture.full:
            self.report({"WARNING"},
                        "Reached the recording limit of %d s."
                        % props.live_record_max)
            return self._finish(context)
        return {"RUNNING_MODAL"}

    def _finish(self, context):
        props = context.scene.audiogeo
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.workspace.status_text_set(None)

        capture = self._capture
        self._capture = None
        stop_stream()

        if self._driver is not None:
            self._driver.restore()
            self._driver = None
        _redraw(context)

        if capture is None:
            return {"CANCELLED"}

        if not capture.record or capture.recorded == 0:
            props.last_report = "Live input off"
            return {"FINISHED"}

        try:
            path, seconds, peak = write_wav(
                take_path(props), capture.take(), capture.rate)
        except Exception as exc:
            self.report({"ERROR"}, "Could not write the recording: %s" % exc)
            props.last_report = "Live input off -- recording failed"
            return {"CANCELLED"}

        LAST_TAKE.update(path=path, seconds=seconds, rate=capture.rate)
        # Point the normal pipeline at the take, so Generate & Bake and every
        # exporter behave exactly as they do for a file chosen by hand.
        props.audio_path = path
        props.last_report = "Recorded %.1f s, peak %.0f dB -- %s" % (
            seconds, 20.0 * np.log10(max(peak, 1e-9)), os.path.basename(path))
        self.report({"INFO"}, props.last_report)

        if props.live_auto_bake:
            # Out of the modal first: generate() is long and blocking, and
            # running it while this operator unwinds invites a crash.
            bpy.app.timers.register(_bake_later, first_interval=0.1)
        return {"FINISHED"}

    def cancel(self, context):
        self._finish(context)


def _bake_later():
    try:
        bpy.ops.audiogeo.generate()
    except Exception:
        pass
    return None  # one shot


def _redraw(context):
    screen = getattr(context, "screen", None)
    if screen is None:
        return
    for area in screen.areas:
        if area.type in {"VIEW_3D", "PROPERTIES"}:
            area.tag_redraw()


class AUDIOGEO_OT_live_stop(Operator):
    bl_idname = "audiogeo.live_stop"
    bl_label = "Stop"
    bl_description = "Stop live input and write the recording"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return ACTIVE[0] is not None

    def execute(self, context):
        # The modal owns the stream; it picks this up on its next tick.
        _STOP[0] = True
        return {"FINISHED"}


class AUDIOGEO_OT_live_bake(Operator):
    bl_idname = "audiogeo.live_bake"
    bl_label = "Bake Last Take"
    bl_description = ("Analyse the recorded take and bake it to keyframes, "
                      "so it can be played back, rendered and exported")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(LAST_TAKE["path"]) and ACTIVE[0] is None

    def execute(self, context):
        props = context.scene.audiogeo
        path = LAST_TAKE["path"]
        if not os.path.isfile(path):
            self.report({"ERROR"}, "The recording is no longer there: %s"
                        % path)
            return {"CANCELLED"}

        props.audio_path = path
        return bpy.ops.audiogeo.generate()


classes = (AUDIOGEO_OT_live, AUDIOGEO_OT_live_stop, AUDIOGEO_OT_live_bake)
