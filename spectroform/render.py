import os

import bpy

from bpy.types import Operator

PRESETS = {
    "SQUARE_1080": (1080, 1080),
    "LANDSCAPE_1080": (1920, 1080),
    "VERTICAL_1080": (1080, 1920),
    "LANDSCAPE_4K": (3840, 2160),
}

class AUDIOGEO_OT_render(Operator):
    bl_idname = "audiogeo.render"
    bl_label = "Render Animation"
    bl_description = "Set up output settings and render the baked animation"
    bl_options = {"REGISTER"}

    preview: bpy.props.BoolProperty(default=False)
    blocking: bpy.props.BoolProperty(
        default=False,
        description="Wait for the render to finish instead of opening the "
                    "render window. Needed when a script renders a queue")

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get("Spectroform_Mesh") is not None

    def execute(self, context):
        scene = context.scene
        props = scene.audiogeo
        w, h = PRESETS.get(props.render_preset, (1920, 1080))

        scene.render.resolution_x = w
        scene.render.resolution_y = h
        scene.render.resolution_percentage = 50 if self.preview else 100
        scene.render.fps = int(props.bake_fps)
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"

        scene.render.ffmpeg.audio_codec = "AAC"

        path = bpy.path.abspath(props.render_path or "//spectroform.mp4")
        folder = os.path.dirname(path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        scene.render.filepath = path

        if scene.camera is None:
            self.report({"ERROR"}, "No camera. Enable Add Camera Rig and re-bake.")
            return {"CANCELLED"}

        if self.blocking:
            # EXEC context: returns only once the whole animation is written.
            # INVOKE_DEFAULT returns immediately, so a caller rendering a queue
            # would overwrite the scene while this render was still running.
            bpy.ops.render.render(animation=True)
        else:
            bpy.ops.render.render("INVOKE_DEFAULT", animation=True)
        props.last_report = "Rendering to %s" % path
        return {"FINISHED"}

classes = (AUDIOGEO_OT_render,)
