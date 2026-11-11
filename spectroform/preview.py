import bpy
import numpy as np

_CACHE = {}        
_ENABLED = [False]


def set_cache(spec, data, cycles, fps):
    _CACHE.clear()
    _CACHE.update(spec=spec, data=data, cycles=cycles, fps=float(fps))


def _apply(scene):
    obj = bpy.data.objects.get("Spectroform_Mesh")
    if obj is None or not _CACHE or obj.data.shape_keys is None:
        return
    from . import analysis, physics

    props = scene.audiogeo
    spec = _CACHE["spec"]
    fps = _CACHE["fps"]
    i = int(np.clip(scene.frame_current - props.frame_start, 0, spec.shape[0] - 1))

    speeds = physics.dispersion_speeds(_CACHE["cycles"], props.size,
                                       scale=props.ripple_speed)
    t = i / fps
    keys = obj.data.shape_keys.key_blocks
    for c in range(spec.shape[1]):
        sin_k = keys.get("RIPPLE_%d_SIN" % c)
        cos_k = keys.get("RIPPLE_%d_COS" % c)
        if sin_k is None or cos_k is None:
            continue
        col = spec[:, c]
        peak = float(col.max())
        amp = float(col[i] / peak if (props.per_band_level and peak > 1e-6)
                    else col[i]) * props.master_gain
        ph = 2.0 * np.pi * float(speeds[c]) * t
        sin_k.value = amp * np.cos(ph)
        cos_k.value = -amp * np.sin(ph)


def _handler(scene, _depsgraph=None):
    if _ENABLED[0]:
        _apply(scene)


class AUDIOGEO_OT_preview(bpy.types.Operator):
    bl_idname = "audiogeo.preview"
    bl_label = "Live Preview"
    bl_description = "Drive the shape keys live while scrubbing, without baking"

    def execute(self, context):
        _ENABLED[0] = not _ENABLED[0]
        if _ENABLED[0]:
            if _handler not in bpy.app.handlers.frame_change_post:
                bpy.app.handlers.frame_change_post.append(_handler)
            _apply(context.scene)
        else:
            if _handler in bpy.app.handlers.frame_change_post:
                bpy.app.handlers.frame_change_post.remove(_handler)
        context.scene.audiogeo.last_report = (
            "Live preview ON" if _ENABLED[0] else "Live preview OFF")
        return {"FINISHED"}


def unregister_handler():
    """Call from the add-on's unregister(), or the handler outlives it."""
    _ENABLED[0] = False
    if _handler in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_handler)


classes = (AUDIOGEO_OT_preview,)
