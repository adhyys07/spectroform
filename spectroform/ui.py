# SPDX-License-Identifier: GPL-3.0-or-later
"""Sidebar UI — View3D ▸ N ▸ Audio Geo."""

from bpy.types import Panel

from .constants import SPECTRAL_BARS, SPECTRAL_STATIC


class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Audio Geo"


class AUDIOGEO_PT_main(_Base, Panel):
    bl_label = "Audio Reactive Geometry"
    bl_idname = "AUDIOGEO_PT_main"

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo

        col = layout.column(align=True)
        col.prop(props, "audio_path", text="")
        row = col.row(align=True)
        row.prop(props, "bake_fps")
        row.prop(props, "frame_start")

        layout.separator()
        layout.operator("audiogeo.generate", icon="PLAY")
        row = layout.row(align=True)
        row.prop(props, "replace_existing", toggle=True)
        row.operator("audiogeo.clear", text="", icon="TRASH")

        if props.last_report:
            box = layout.box()
            box.label(text=props.last_report, icon="INFO")


class AUDIOGEO_PT_frequency(_Base, Panel):
    bl_label = "Frequency Response"
    bl_parent_id = "AUDIOGEO_PT_main"

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True

        col = layout.column(align=True)
        col.prop(props, "bass_gain")
        col.prop(props, "mid_gain")
        col.prop(props, "treble_gain")
        col.prop(props, "master_gain")

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "bass_max")
        col.prop(props, "mid_max")
        col.prop(props, "dynamic_range")
        col.prop(props, "per_band_level")

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "attack_ms")
        col.prop(props, "release_ms")


class AUDIOGEO_PT_geometry(_Base, Panel):
    bl_label = "Geometry"
    bl_parent_id = "AUDIOGEO_PT_main"

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True

        kind = props.geometry_type
        col = layout.column(align=True)
        col.prop(props, "geometry_type")
        col.prop(props, "size")
        col.prop(props, "base_scale")
        col.prop(props, "shade_smooth")

        if kind == "RIPPLE":
            col.prop(props, "resolution")
            col.prop(props, "ripple_pattern")
            col.prop(props, "ripple_source")
            if props.ripple_source == "SPECTRUM":
                col.prop(props, "ripple_components")
                col.prop(props, "ripple_spread")
            col.prop(props, "ripple_cycles")
            col.prop(props, "ripple_speed")
            col.prop(props, "ripple_beat_gain")

            box = layout.box()
            box.label(text="Wave Height Per Band", icon="MOD_WAVE")
            c = box.column(align=True)
            c.prop(props, "terrain_strength", text="Bass")
            c.prop(props, "wave_strength", text="Mid")
            c.prop(props, "spike_strength", text="Treble")
            return

        if kind == "SPECTROGRAM":
            col.prop(props, "scroll_terrain")

        if kind in SPECTRAL_BARS or kind == "SPECTROGRAM":
            col.prop(props, "spectrum_bins")
            col.prop(props, "spec_height")
        if kind in SPECTRAL_STATIC:
            col.prop(props, "time_steps")
        if kind in {"WAVE_TUBE", "WAVE_RING"}:
            col.prop(props, "tube_radius")
            col.prop(props, "tube_segments")

        if kind in SPECTRAL_STATIC or kind in SPECTRAL_BARS:
            return

        # Noise modes only: the three bands drive separate shape keys.
        col.prop(props, "resolution")

        box = layout.box()
        box.label(text="Bass > Terrain", icon="RNDCURVE")
        c = box.column(align=True)
        c.prop(props, "terrain_strength")
        c.prop(props, "noise_scale")
        c.prop(props, "noise_octaves")

        box = layout.box()
        box.label(text="Mid > Wave", icon="MOD_WAVE")
        c = box.column(align=True)
        c.prop(props, "wave_strength")
        c.prop(props, "wave_count")

        box = layout.box()
        box.label(text="Treble > Spikes", icon="MOD_SIMPLEDEFORM")
        c = box.column(align=True)
        c.prop(props, "spike_strength")
        c.prop(props, "spike_density")
        c.prop(props, "spike_sharpness")

        layout.prop(props, "seed")


class AUDIOGEO_PT_motion(_Base, Panel):
    bl_label = "Motion & Look"
    bl_parent_id = "AUDIOGEO_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True

        col = layout.column(align=True)
        col.prop(props, "scale_gain")
        col.prop(props, "pulse_gain")
        col.prop(props, "spin_speed")
        col.prop(props, "spin_gain")

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "create_material")
        col.prop(props, "sync_sound")


class AUDIOGEO_PT_export(_Base, Panel):
    bl_label = "Export"
    bl_parent_id = "AUDIOGEO_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True

        col = layout.column(align=True)
        col.prop(props, "export_format")
        col.prop(props, "export_path")
        col.prop(props, "export_selected_only")
        layout.operator("audiogeo.export", icon="EXPORT")


classes = (
    AUDIOGEO_PT_main,
    AUDIOGEO_PT_frequency,
    AUDIOGEO_PT_geometry,
    AUDIOGEO_PT_motion,
    AUDIOGEO_PT_export,
)