bl_info = {
    "name": "Spectroform",
    "author": "Adhdhyan",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > Audio Geo",
    "description": "Generate audio-reactive 3D geometry with baked keyframe animation",
    "doc_url": "",
    "category": "Object",
}

if "bpy" in locals():
    import importlib

    for _mod in (
        constants,
        audio_io,
        cache,
        analysis,
        physics,
        geometry,
        baking,
        scene_build,
        waveform,
        midi_io,
        lyrics,
        webexport,
        texture,
        stamp,
        properties,
        presets,
        render,
        preview,
        batch,
        operators,
        ui,
    ):
        importlib.reload(_mod)
else:
    from . import (
        analysis,
        audio_io,
        baking,
        batch,
        cache,
        constants,
        geometry,
        lyrics,
        midi_io,
        operators,
        physics,
        presets,
        preview,
        properties,
        render,
        scene_build,
        stamp,
        texture,
        ui,
        waveform,
        webexport,
    )

import bpy
from bpy.props import PointerProperty

from .constants import MIN_BLENDER

CLASSES = (properties.classes + operators.classes + preview.classes
           + presets.classes + render.classes + batch.classes + ui.classes)


def register():
    if bpy.app.version < MIN_BLENDER:
        raise RuntimeError(
            "Audio Reactive Geometry requires Blender %d.%d or newer."
            % (MIN_BLENDER[0], MIN_BLENDER[1])
        )

    for cls in CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.audiogeo = PointerProperty(type=properties.AUDIOGEO_Props)


def unregister():
    # The frame_change handler holds a reference into this module, so it
    # keeps firing after an unregister unless it is removed explicitly.
    preview.unregister_handler()

    if hasattr(bpy.types.Scene, "audiogeo"):
        del bpy.types.Scene.audiogeo

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()