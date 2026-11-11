import os
import bpy
from bpy.types import Operator

from.constants import SUPPORTED_EXTENSIONS

def find_tracks(folder, recursive=True):
    folder = bpy.path.abspath(folder)
    if not os.path.isdir(folder):
        return []
    out = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            out += [os.path.join(root, f) for f in sorted(files)
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
    else:
        for f in sorted(os.listdir(folder)):
            full = os.path.join(folder, f)
            if (os.path.isfile(full)
                    and os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS):
                out.append(full)
    return out

class AUDIOGEO_OT_batch(Operator):
    bl_idname = "audiogeo.batch"
    bl_label = "Batch Render Folder"
    bl_description = ("Generate and render one video per audio file in the "
                      "folder, using the current settings")
    bl_options = {"REGISTER"}

    _timer = None

    @classmethod
    def poll(cls, context):
        return bool(context.scene.audiogeo.batch_folder)

    def invoke(self, context, event):
        props = context.scene.audiogeo
        self._queue = find_tracks(props.batch_folder, props.batch_recursive)
        if not self._queue:
            self.report({"ERROR"}, "No supported audio files in that folder.")
            return {"CANCELLED"}

        out = bpy.path.abspath(props.batch_output)
        if not os.path.isdir(out):
            try:
                os.makedirs(out, exist_ok=True)
            except OSError as exc:
                self.report({"ERROR"}, "Cannot create output folder: %s" % exc)
                return {"CANCELLED"}

        self._done, self._failed = [], []
        self._timer = context.window_manager.event_timer_add(
            0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            return self._finish(context, cancelled=True)
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if not self._queue:
            return self._finish(context)

        props = context.scene.audiogeo
        track = self._queue.pop(0)
        stem = os.path.splitext(os.path.basename(track))[0]

        props.audio_path = track
        props.render_path = os.path.join(
            bpy.path.abspath(props.batch_output), stem + ".mp4")
        try:
            if bpy.ops.audiogeo.generate() != {"FINISHED"}:
                raise RuntimeError("generate failed")
            if bpy.ops.audiogeo.render(preview=False,
                                       blocking=True) != {"FINISHED"}:
                raise RuntimeError("render failed")
            self._done.append(track)
        except Exception as exc:
            self._failed.append(track)
            self.report({"WARNING"}, "%s: %s" % (stem, exc))

        props.last_report = "Batch: %d done, %d failed, %d left" % (
            len(self._done), len(self._failed), len(self._queue))
        return {"RUNNING_MODAL"}

    def _finish(self, context, cancelled=False):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        msg = "Batch %s: %d rendered, %d failed" % (
            "cancelled" if cancelled else "done",
            len(self._done), len(self._failed))
        context.scene.audiogeo.last_report = msg
        self.report({"INFO"}, msg)
        return {"CANCELLED"} if cancelled else {"FINISHED"}

classes = (AUDIOGEO_OT_batch,)