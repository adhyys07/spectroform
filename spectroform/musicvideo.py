import bpy
from bpy.types import Operator

PRESET_FLAGS = {
    "add_camera": True,
    "camera_cuts": True,
    "mark_sections": True,
    "create_material": True,
    "use_glare": True,
    "light_rig": True,
    "sync_sound": True,
    "vertex_colour": True,
}


class AUDIOGEO_OT_music_video(Operator):
    bl_idname = "audiogeo.music_video"
    bl_label = "Generate Music Video"
    bl_description = ("Build the geometry, cut cameras on the sections, "
                      "animate the lyrics, light it and render")
    bl_options = {"REGISTER", "UNDO"}

    render: bpy.props.BoolProperty(
        name="Render Now", default=False,
        description="Render straight away. Off just sets the scene up so you "
                    "can look before committing the time")

    @classmethod
    def poll(cls, context):
        return bool(context.scene.audiogeo.audio_path)

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        props = context.scene.audiogeo
        col = self.layout.column(align=True)
        col.prop(props, "render_preset")
        col.prop(props, "render_path")
        col.prop(props, "lyrics_path")
        col.prop(self, "render")
        if self.render:
            col.label(text="Rendering blocks Blender until it finishes",
                      icon="INFO")

    def execute(self, context):
        props = context.scene.audiogeo
        scene = context.scene

        previous = {k: getattr(props, k) for k in PRESET_FLAGS}
        for key, value in PRESET_FLAGS.items():
            setattr(props, key, value)

        try:
            if bpy.ops.audiogeo.generate() != {"FINISHED"}:
                raise RuntimeError("generate failed -- see the report above")

            if scene.camera is None:
                raise RuntimeError("No camera was created; enable Add Camera Rig.")

            if "EEVEE" not in scene.render.engine:
                scene.render.engine = (
                    "BLENDER_EEVEE_NEXT" if bpy.app.version >= (4, 2, 0)
                    else "BLENDER_EEVEE")

            scene.render.film_transparent = False
            scene.render.image_settings.file_format = "FFMPEG"
            scene.render.ffmpeg.format = "MPEG4"
            scene.render.ffmpeg.codec = "H264"
            scene.render.ffmpeg.constant_rate_factor = "HIGH"
            scene.render.ffmpeg.audio_codec = "AAC"
            scene.render.ffmpeg.audio_bitrate = 192

            if self.render:
                if bpy.ops.audiogeo.render(preview=False) != {"FINISHED"}:
                    raise RuntimeError("render failed")
                msg = "Music video rendered: %s" % props.render_path
            else:
                msg = "Scene ready: %d frames, press Render" % (
                    scene.frame_end - scene.frame_start + 1)
        except Exception as exc:
            for key, value in previous.items():
                setattr(props, key, value)
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        props.last_report = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


classes = (AUDIOGEO_OT_music_video,)
