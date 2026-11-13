import bpy
import numpy as np
from bpy.types import Operator

def loop_window(bars, bar_count, fps, bpm, beats_per_bar=4, prefer_frame=None):
    bars = np.asarray(bars, dtype=np.int64)
    want = max(1, int(bar_count))
    if bars.size >= want+1:
        start_i = (0 if prefer_frame is None
                   else int(np.argmin(np.abs(bars - int(prefer_frame)))))
        start_i = min(start_i, bars.size - want - 1)
        return int(bars[start_i]), int(bars[start_i + want])

    if bpm <= 0.0:
        return None
    period = 60.0 * float(fps) / float(bpm) * float(beats_per_bar)
    start = int(prefer_frame or 0)
    return start, int(round(start + period * want))


def pick_loudest_window(rms, bars, bar_count, start_frame=0):
    bars = np.asarray(bars, dtype=np.int64)
    want = max(1, int(bar_count))
    if bars.size < want + 1:
        return None
    rms = np.asarray(rms, dtype=np.float64)
    best_i, best = 0, -1.0
    for i in range(bars.size - want):
        a = max(0, int(bars[i]) - int(start_frame))
        b = min(rms.size, int(bars[i + want]) - int(start_frame))
        if b <= a:
            continue
        score = float(rms[a:b].mean())
        if score > best:
            best, best_i = score, i
    return int(bars[best_i])


def trim_scene_to_loop(scene, start, end):
    scene.frame_start = int(start)
    scene.frame_end = int(end) - 1
    return scene.frame_end - scene.frame_start + 1

def verify_loop(obj, start, end):
    key = obj.data.shape_keys
    if key is None or key.animation_data is None or key.animation_data.action is None:
        return 0.0
    return max((abs(float(fc.evaluate(int(start))) - float(fc.evaluate(int(end))))
                for fc in key.animation_data.action.fcurves), default=0.0)


def suggest_speed(bpm, bar_count, fps, beats_per_bar=4, cycles=1):
    if bpm <= 0.0:
        return None
    seconds = 60.0 / float(bpm) * float(beats_per_bar) * float(bar_count)
    return float(cycles) / max(seconds, 1e-6)


class AUDIOGEO_OT_make_loop(Operator):
    bl_idname = "audiogeo.make_loop"
    bl_label = "Trim To Loop"
    bl_description = ("Set the frame range to a whole number of bars so the "
                      "clip retriggers cleanly, with no crossfade")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        from . import dataexport
        return bool(dataexport.LAST_RUN.get("data"))

    def execute(self, context):
        from . import dataexport
        from .constants import OBJ_NAME

        props = context.scene.audiogeo
        data = dataexport.LAST_RUN.get("data") or {}
        bars = data.get("bars")
        if bars is None or not len(bars):
            self.report({"ERROR"},
                        "No bars detected -- enable Detect Bars and re-generate.")
            return {"CANCELLED"}

        prefer = (pick_loudest_window(data.get("rms", []), bars,
                                      props.loop_bars, props.frame_start)
                  if props.loop_pick == "LOUDEST" else None)
        window = loop_window(bars, props.loop_bars, props.bake_fps,
                             data.get("bpm", 0.0), props.beats_per_bar, prefer)
        if window is None:
            self.report({"ERROR"}, "Could not place a loop window.")
            return {"CANCELLED"}

        start, end = window
        count = trim_scene_to_loop(context.scene, start, end)
        obj = bpy.data.objects.get(OBJ_NAME)
        drift = verify_loop(obj, start, end) if obj else 0.0

        msg = "Loop: frames %d-%d (%d frames, %d bars)" % (
            context.scene.frame_start, context.scene.frame_end, count,
            props.loop_bars)
        if drift > 0.02:
            speed = suggest_speed(data.get("bpm", 0.0), props.loop_bars,
                                  props.bake_fps, props.beats_per_bar)
            msg += " | seam %.3f -- try Wave Speed %.2f" % (drift, speed or 0.0)

        props.last_report = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}

classes = (AUDIOGEO_OT_make_loop, )