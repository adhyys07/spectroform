import queue

import bpy
import numpy as np
from bpy.types import Operator

try:
    import sounddevice as sd
except Exception:
    sd = None

BLOCK = 1024
_QUEUE = queue.Queue(maxsize=8)
_STREAM = [None]
_STATE = {"bands": np.zeros(3, dtype=np.float32), "peak": 1e-6}


def available():
    return sd is not None


def device_list():
    if sd is None:
        return []
    try:
        return [(str(i), d["name"], "")
                for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0]
    except Exception:
        return []


def _callback(indata, _frames, _time, _status):
    try:
        _QUEUE.put_nowait(indata[:, 0].copy())
    except queue.Full:
        pass


def _analyse_block(block, rate, bass_max, mid_max, smoothing=0.6):
    window = np.hanning(block.size).astype(np.float32)
    mag = np.abs(np.fft.rfft(block * window))
    freqs = np.fft.rfftfreq(block.size, 1.0 / rate)

    out = np.zeros(3, dtype=np.float32)
    for i, (lo, hi) in enumerate(((20.0, bass_max), (bass_max, mid_max),
                                  (mid_max, rate * 0.5))):
        m = (freqs >= lo) & (freqs < hi)
        if m.any():
            out[i] = float(np.sqrt((mag[m] ** 2).mean()))

    _STATE["peak"] = max(float(out.max()), _STATE["peak"] * 0.995, 1e-6)
    out = np.clip(out / _STATE["peak"], 0.0, 1.0)

    _STATE["bands"] = (_STATE["bands"] * smoothing + out * (1.0 - smoothing))
    return _STATE["bands"]


class AUDIOGEO_OT_live(Operator):
    bl_idname = "audiogeo.live"
    bl_label = "Live Input"
    bl_description = ("Drive the shape keys from the microphone or a loopback "
                      "device in real time")
    bl_options = {"REGISTER"}

    _timer = None

    @classmethod
    def poll(cls, context):
        from .constants import OBJ_NAME
        obj = bpy.data.objects.get(OBJ_NAME)
        return obj is not None and obj.data.shape_keys is not None

    def invoke(self, context, event):
        if not available():
            self.report({"ERROR"},
                        "Live input needs the 'sounddevice' module. Install it "
                        "into Blender's Python, or use a file instead.")
            return {"CANCELLED"}

        props = context.scene.audiogeo
        try:
            device = int(props.live_device) if props.live_device else None
            _STREAM[0] = sd.InputStream(
                device=device, channels=1, samplerate=int(props.live_rate),
                blocksize=BLOCK, callback=_callback)
            _STREAM[0].start()
        except Exception as exc:
            self.report({"ERROR"}, "Could not open the input device: %s" % exc)
            return {"CANCELLED"}

        self._timer = context.window_manager.event_timer_add(
            1.0 / 60.0, window=context.window)
        context.window_manager.modal_handler_add(self)
        props.last_report = "Live input ON -- Esc to stop"
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type in {"ESC", "RIGHTMOUSE"}:
            return self._stop(context)
        if event.type != "TIMER":
            return {"PASS_THROUGH"}

        props = context.scene.audiogeo
        block = None
        while True:
            try:
                block = _QUEUE.get_nowait()
            except queue.Empty:
                break
        if block is None:
            return {"RUNNING_MODAL"}

        bands = _analyse_block(block, float(props.live_rate),
                               props.bass_max, props.mid_max,
                               smoothing=props.live_smoothing)
        self._drive(context, bands)
        return {"RUNNING_MODAL"}

    def _drive(self, context, bands):
        from .constants import OBJ_NAME, SK_BASS, SK_MID, SK_TREBLE

        obj = bpy.data.objects.get(OBJ_NAME)
        if obj is None or obj.data.shape_keys is None:
            return
        keys = obj.data.shape_keys.key_blocks
        props = context.scene.audiogeo

        for name, value in ((SK_BASS, bands[0]), (SK_MID, bands[1]),
                            (SK_TREBLE, bands[2])):
            key = keys.get(name)
            if key is not None:
                key.value = float(value)

        sin_keys = [k for k in keys if k.name.endswith("_SIN")]
        for i, key in enumerate(sin_keys):
            band = bands[min(2, int(3 * i / max(len(sin_keys), 1)))]
            key.value = float(band)

        scale = props.base_scale * (1.0 + float(bands.mean()) * props.scale_gain)
        obj.scale = (scale, scale, scale)

    def _stop(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if _STREAM[0] is not None:
            _STREAM[0].stop()
            _STREAM[0].close()
            _STREAM[0] = None
        while not _QUEUE.empty():
            _QUEUE.get_nowait()
        context.scene.audiogeo.last_report = "Live input OFF"
        return {"FINISHED"}


classes = (AUDIOGEO_OT_live,)
