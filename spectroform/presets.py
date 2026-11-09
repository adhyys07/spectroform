import bpy
from bpy.types import Operator

PRESETS = {
    "CALM_OCEAN": {
        "geometry_type": "RIPPLE", "ripple_pattern": "LINEAR",
        "ripple_profile": "GERSTNER", "ripple_steepness": 0.45,
        "ripple_cycles": 2.0, "ripple_speed": 0.25, "ripple_beat_gain": 0.5,
        "terrain_strength": 0.3, "wave_strength": 0.12, "spike_strength": 0.05,
        "resolution": 128, "attack_ms": 60.0, "release_ms": 400.0,
    },
    "CLUB_VISUALIZER": {
        "geometry_type": "RIPPLE", "ripple_pattern": "RADIAL",
        "ripple_profile": "GERSTNER", "ripple_steepness": 0.3,
        "ripple_cycles": 5.0, "ripple_speed": 1.2, "ripple_beat_gain": 6.0,
        "terrain_strength": 0.35, "wave_strength": 0.25, "spike_strength": 0.3,
        "resolution": 96, "attack_ms": 10.0, "release_ms": 90.0,
        "pulse_gain": 0.4, "spin_speed": 0.35,
    },
    "GLITCH": {
        "geometry_type": "RIPPLE", "ripple_pattern": "TURBULENT",
        "ripple_profile": "SINE", "ripple_turbulence": 1.2,
        "ripple_cycles": 9.0, "ripple_speed": -1.8, "ripple_beat_gain": 10.0,
        "terrain_strength": 0.2, "wave_strength": 0.3, "spike_strength": 0.5,
        "resolution": 128, "attack_ms": 4.0, "release_ms": 40.0,
    },
    "SPECTRUM_RING": {
        "geometry_type": "BARS_RADIAL", "spectrum_bins": 32,
        "spec_height": 0.6, "master_gain": 2.0,
        "attack_ms": 8.0, "release_ms": 110.0, "spin_speed": 0.2,
    },
    "TRACK_LANDSCAPE": {
        "geometry_type": "SPECTROGRAM", "spectrum_bins": 64,
        "time_steps": 384, "spec_height": 0.4, "scroll_terrain": True,
    },
}

class AUDIOGEO_OT_preset(Operator):
    bl_idname = "audiogeo.preset"
    bl_label = "Apply Preset"
    bl_description = "Load a named starting point"
    bl_options = {"REGISTER", "UNDO"}

    preset : bpy.props.StringProperty()

    def execute(self, context):
        values = PRESETS.get(self.preset)
        if values is None:
            self.report({"ERROR"}, "Unknown preset %r" % self.preset)
            return {"CANCELLED"}
        props = context.scene.audiogeo
        for name, value in values.items():
            if hasattr(props, name):
                setattr(props, name, value)
        props.last_report = "Preset: %s" % self.preset.replace("_", " ").title()
        return {"FINISHED"}

classes = (AUDIOGEO_OT_preset,)