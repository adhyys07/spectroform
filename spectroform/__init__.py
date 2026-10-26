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

# Reload-safe imports: on "Reload Scripts" the package globals survive, so the
# presence of `bpy` tells us this is a re-run rather than a cold import.
if "bpy" in locals():
    import importlib

    for _mod in (
        constants,     
        audio_io,      
        analysis,      
        geometry,      
        baking,        
        scene_build,   
        properties,    
        operators,     
        ui,            
    ):
        importlib.reload(_mod)
else:
    from . import (
        analysis,
        audio_io,
        baking,
        constants,
        geometry,
        operators,
        properties,
        scene_build,
        ui,
    )

import bpy
from bpy.props import PointerProperty

from .constants import MIN_BLENDER

CLASSES = properties.classes + operators.classes + ui.classes


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
    if hasattr(bpy.types.Scene, "audiogeo"):
        del bpy.types.Scene.audiogeo

    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()