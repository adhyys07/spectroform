from bpy.types import Panel

from .constants import RIPPLE_MODES, SPECTRAL_BARS, SPECTRAL_STATIC


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
        col.prop(props, "use_stereo")

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
            if props.ripple_pattern == "SPIRAL":
                col.prop(props, "ripple_arms")
            if props.ripple_pattern == "TURBULENT":
                col.prop(props, "ripple_turbulence")
            col.prop(props, "ripple_profile")
            if props.ripple_profile == "GERSTNER":
                col.prop(props, "ripple_steepness")

            col.prop(props, "ripple_source")
            if props.ripple_source == "SPECTRUM":
                col.prop(props, "ripple_components")
                col.prop(props, "ripple_spread")
            elif props.ripple_source == "EMITTER":
                col.prop(props, "emitter_slots")
                col.prop(props, "emitter_decay")
                col.prop(props, "drum_response")
            elif props.ripple_source == "HPSS":
                col.prop(props, "ripple_components")
                col.prop(props, "hpss_kernel")
            elif props.ripple_source == "CHLADNI":
                col.prop(props, "chladni_modes")
            elif props.ripple_source == "INTERFERENCE":
                col.prop(props, "ripple_components")
                col.prop(props, "ripple_spread")
                col.prop(props, "source_separation")
            elif props.ripple_source == "FARADAY":
                col.prop(props, "ripple_components")
                col.prop(props, "ripple_spread")
            elif props.ripple_source == "HARMONIC":
                col.prop(props, "ripple_components")
            elif props.ripple_source == "MIDI":
                col.prop(props, "midi_path")
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
        col.prop(props, "beat_lock")
        col.prop(props, "cycles_per_beat")
        col.prop(props, "loop_fade")
        col.prop(props, "vertex_colour")

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "mark_sections")
        col.prop(props, "section_variation")
        col.prop(props, "colour_source")
        col.prop(props, "add_camera")
        if props.add_camera:
            col.prop(props, "camera_orbit")


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

class AUDIOGEO_PT_presets(_Base, Panel):
    bl_label = "Presets"
    bl_parent_id = "AUDIOGEO_PT_main"

    def draw(self, context):
        from .presets import PRESETS
        col = self.layout.column(align=True)
        for key in PRESETS:
            col.operator("audiogeo.preset",
                         text=key.replace("_", " ").title()).preset = key


class AUDIOGEO_PT_render(_Base, Panel):
    bl_label = "Render"
    bl_parent_id = "AUDIOGEO_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True

        col = layout.column(align=True)
        col.prop(props, "render_preset")
        col.prop(props, "render_path")

        row = layout.row(align=True)
        row.operator("audiogeo.render", text="Render", icon="RENDER_ANIMATION")
        row.operator("audiogeo.render", text="Preview",
                     icon="SEQ_PREVIEW").preview = True

        if context.scene.camera is None:
            layout.label(text="No camera: enable Add Camera Rig", icon="ERROR")

class AUDIOGEO_PT_physics(_Base, Panel):
    bl_label = "Physics"
    bl_parent_id = "AUDIOGEO_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        kind = context.scene.audiogeo.geometry_type
        return kind in RIPPLE_MODES or kind == "SPECTOGRAM"

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True
        kind = props.geometry_type

        if kind in RIPPLE_MODES:
            col = layout.column(align=True)
            col.prop(props, "use_dispersion")
            if props.use_dispersion:
                col.label(text="Speed ramp derived from wavelength", icon="INFO")
            col.prop(props, "use_resonance")
            sub = col.column(align=True)
            sub.enabled = props.use_resonance
            sub.prop(props, "resonance_q")

        if kind == "SPECTROGRAM":
            layout.separator()
            box = layout.box()
            box.label(text="Terrain Erosion", icon="MOD_FLUIDSIM")
            c = box.column(align=True)
            c.prop(props, "erosion_iterations")
            sub = c.column(align=True)
            sub.enabled = props.erosion_iterations > 0
            sub.prop(props, "erosion_talus")

class AUDIOGEO_PT_lighting(_Base, Panel):
    bl_label = "Lightning and Effects"
    bl_parent_id = "AUDIOGEO_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True

        col = layout.column(align=True)
        col.prop(props, "use_glare")
        sub = col.column(align=True)
        sub.enabled = props.use_glare and props.create_material
        sub.prop(props, "glare_threshold")
        if props.use_glare and not props.create_material:
            col.label(text="Glare needs Create Material", icon="INFO")

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "light_rig")
        sub = col.column(align=True)
        sub.enabled = props.light_rig
        sub.prop(props, "light_power")

        layout.separator()
        layout.column(align=True).prop(props, "spark_count")

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "camera_cuts")
        if props.camera_cuts:
            col.label(text="Overrides Add Camera Rig", icon="INFO")


class AUDIOGEO_PT_extra(_Base, Panel):
    bl_label = "Extra Input"
    bl_parent_id = "AUDIOGEO_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True

        col = layout.column(align=True)
        col.prop(props, "midi_path")
        if props.midi_path and props.ripple_source != "MIDI":
            col.label(text="Set Wave Source to MIDI Channels", icon="INFO")
        col.prop(props, "lyrics_path")

        layout.separator()
        box = layout.box()
        box.label(text="WebGL Page", icon="WORLD")
        c = box.column(align=True)
        c.prop(props, "web_export_path")
        c.operator("audiogeo.export_web", icon="EXPORT")


classes = (
    AUDIOGEO_PT_main,
    AUDIOGEO_PT_frequency,
    AUDIOGEO_PT_geometry,
    AUDIOGEO_PT_motion,
    AUDIOGEO_PT_export,
    AUDIOGEO_PT_presets,
    AUDIOGEO_PT_render,
)