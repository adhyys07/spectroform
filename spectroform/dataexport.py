import json
import os
import bpy
import numpy as np
from bpy.types import Operator

_CURVE_KEYS = ("bass", "mid", "treble", "rms", "onset", "section_gain",
               "correlation", "pitch_hz", "pitch_conf", "local_bpm")
_EVENT_KEYS = ("beats", "bars", "phrases", "sections")

LAST_RUN = {}

def _clean(value, places=5):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return round(float(arr.item()), places)
    return [round(float(x), places) for x in arr.ravel()]

def build_payload(data, props, spec=None, bin_hz=None, chroma=None,
                  stems=None, n_frames=None):
    n = int(n_frames or data.get("n_frames", 0))
    payload = {
        "format": "spectroform-analysis",
        "version": 1,
        "source": os.path.basename(bpy.path.abspath(props.audio_path)),
        "fps": float(props.bake_fps),
        "frame_start": int(props.frame_start),
        "frame_count": n,
        "duration": round(float(data.get("duration", 0.0)), 4),
        "tempo": {
            "bpm": round(float(data.get("bpm", 0.0)), 3),
            "beats_per_bar": int(getattr(props, "beats_per_bar", 4)),
            "downbeat_phase": int(data.get("downbeat_phase", 0)),
        },
        "key": data.get("key", ""),
        "curves": {},
        "events": {},
    }

    for name in _CURVE_KEYS:
        value = data.get(name)
        if value is not None and len(np.asarray(value).ravel()):
            payload["curves"][name] = _clean(np.asarray(value)[:n] if n else value)

    for name in _EVENT_KEYS:
        value = data.get(name)
        if value is not None and len(np.asarray(value).ravel()):
            frames = np.asarray(value, dtype=np.int64)
            payload["events"][name] = {
                "frames": [int(f) for f in frames],
                "seconds": [round(float(f - props.frame_start)
                                  / float(props.bake_fps), 4) for f in frames],
            }

    if stems:
        payload["stems"] = {k: _clean(v) for k, v in stems.items()
                            if not k.startswith("_")}
    if chroma is not None and len(chroma):
        payload["chroma"] = {
            "notes": ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A",
                      "A#", "B"],
            "frames": [[round(float(x), 4) for x in row]
                       for row in np.asarray(chroma)[:n]],
        }
    if spec is not None:
                payload["spectrogram"] = {
            "bins": int(spec.shape[1]),
            "bin_hz": _clean(bin_hz, 2) if bin_hz is not None else [],
            "frames": [[round(float(x), 4) for x in row] for row in spec[:n]],
        }
    return payload

def write_csv(path, data, props, n_frames):
    n = int(n_frames)
    columns = [("frame", np.arange(n, dtype=np.int64) + int(props.frame_start)),
               ("seconds", np.arange(n, dtype=np.float64) / float(props.bake_fps))]

    for name in _CURVE_KEYS:
        value = data.get(name)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float64).ravel()
        if arr.size < n:
            arr = np.pad(arr, (0, n - arr.size))
        columns.append((name, arr[:n]))

    for name in _EVENT_KEYS:
        value = data.get(name)
        if value is None or not len(np.asarray(value).ravel()):
            continue
        flags = np.zeros(n, dtype=np.int64)
        idx = np.asarray(value, dtype=np.int64) - int(props.frame_start)
        flags[idx[(idx >= 0) & (idx < n)]] = 1
        columns.append((name, flags))

    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(name for name, _ in columns) + "\n")
        for i in range(n):
            fh.write(",".join(
                ("%d" % col[i]) if col.dtype.kind == "i" else ("%.5f" % col[i])
                for _name, col in columns) + "\n")
    return os.path.getsize(path)

class AUDIOGEO_OT_export_data(Operator):
    bl_idname = "audiogeo.export_data"
    bl_label = "Export Analysis"
    bl_description = ("Write the analysis curves and events for another tool "
                      "-- Unity, Three.js, TouchDesigner, a notebook")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return LAST_RUN.get("data") is not None

    def execute(self, context):
        props = context.scene.audiogeo
        data = LAST_RUN.get("data")
        if data is None:
            self.report({"ERROR"}, "Generate first -- nothing analysed yet.")
            return {"CANCELLED"}

        root = os.path.splitext(bpy.path.abspath(props.data_export_path))[0]
        folder = os.path.dirname(root)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)

        n = int(data.get("n_frames", 0))
        size = 0
        try:
            if props.data_export_format in {"JSON", "BOTH"}:
                payload = build_payload(
                    data, props,
                    spec=LAST_RUN.get("spec") if props.data_include_spectrogram else None,
                    bin_hz=LAST_RUN.get("bin_hz"),
                    chroma=LAST_RUN.get("chroma") if props.data_include_chroma else None,
                    stems=LAST_RUN.get("stems"), n_frames=n)
                with open(root + ".json", "w", encoding="utf-8") as fh:
                    if props.data_minify:
                        json.dump(payload, fh, separators=(",", ":"))
                    else:
                        json.dump(payload, fh, indent=2)
                size = os.path.getsize(root + ".json")
            if props.data_export_format in {"CSV", "BOTH"}:
                size = write_csv(root + ".csv", data, props, n)
        except Exception as exc:
            self.report({"ERROR"}, "Analysis export failed: %s" % exc)
            return {"CANCELLED"}

        props.last_report = "Analysis: %s (%.0f KB)" % (root, size / 1024.0)
        self.report({"INFO"}, props.last_report)
        return {"FINISHED"}


classes = (AUDIOGEO_OT_export_data,)

