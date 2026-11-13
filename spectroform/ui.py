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
        layout.operator("audiogeo.preview", icon="PLAY_SOUND")

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

        # E17: deform an existing object instead of a generated primitive.
        box = layout.box()
        box.prop(props, "mesh_source", text="")
        if props.mesh_source == "CUSTOM":
            c = box.column(align=True)
            c.prop(props, "custom_mesh")
            c.prop(props, "custom_apply_mods")
            c.prop(props, "custom_auto_size")
            if props.custom_mesh is None:
                c.label(text="Pick a mesh to deform", icon="ERROR")
            else:
                c.label(text="The original is left untouched", icon="INFO")

        col = layout.column(align=True)
        if props.mesh_source == "CUSTOM":
            # geometry_type still selects which deformer runs, but the base
            # mesh no longer comes from it.
            col.prop(props, "geometry_type", text="Deformer")
        else:
            col.prop(props, "geometry_type")
        col.prop(props, "size")
        col.prop(props, "base_scale")
        col.prop(props, "shade_smooth")

        if kind == "VECTORSCOPE":
            col.prop(props, "scope_rotate")
            col.prop(props, "scope_spiral")
            col.prop(props, "tube_radius")
            col.prop(props, "tube_segments")
            if not props.use_stereo:
                col.label(text="Needs Stereo Split in the main panel",
                          icon="ERROR")

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

        if kind == "MELODY":
            col.prop(props, "pitch_min")
            col.prop(props, "pitch_max")
            col.prop(props, "pitch_gap")
            col.prop(props, "spec_height", text="Height")

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

        layout.separator()
        box = layout.box()
        box.label(text="Analysis Data", icon="GRAPH")
        c = box.column(align=True)
        c.prop(props, "data_export_format")
        c.prop(props, "data_export_path")
        c.prop(props, "data_include_spectrogram")
        c.prop(props, "data_include_chroma")
        c.prop(props, "data_minify")
        c.operator("audiogeo.export_data", icon="EXPORT")

class AUDIOGEO_PT_presets(_Base, Panel):
    bl_label = "Presets"
    bl_parent_id = "AUDIOGEO_PT_main"

    def draw(self, context):
        from .presets import PRESETS
        layout = self.layout
        col = layout.column(align=True)
        for key in PRESETS:
            col.operator("audiogeo.preset",
                         text=key.replace("_", " ").title()).preset = key

        layout.separator()
        col = layout.column(align=True)
        col.operator("audiogeo.copy_preset", icon="COPYDOWN")
        col.operator("audiogeo.load_stamp", icon="PASTEDOWN")
        col.operator("audiogeo.rebake", icon="FILE_REFRESH")


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

        layout.separator()
        box = layout.box()
        box.label(text="VJ Loop", icon="FILE_REFRESH")
        c = box.column(align=True)
        c.prop(props, "loop_bars")
        c.prop(props, "loop_pick")
        c.operator("audiogeo.make_loop", icon="TRIA_RIGHT_BAR")
        if props.loop_fade > 0:
            # The two mechanisms fight: a crossfade over a seam that is
            # already exact only softens frames that were correct.
            c.label(text="Set Loop Fade to 0 for an exact loop", icon="INFO")

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


class AUDIOGEO_PT_timing(_Base, Panel):
    bl_label = "Timing"
    bl_parent_id = "AUDIOGEO_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True

        col = layout.column(align=True)
        col.prop(props, "beat_tracker")
        sub = col.column(align=True)
        sub.enabled = props.beat_tracker == "DP"
        sub.prop(props, "beat_tightness")

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "use_bars")
        sub = col.column(align=True)
        sub.enabled = props.use_bars
        sub.prop(props, "beats_per_bar")
        sub.prop(props, "bars_per_phrase")
        sub.prop(props, "bar_accent_gain")
        sub.prop(props, "phrase_accent_gain")
        if props.use_bars and props.geometry_type not in RIPPLE_MODES:
            # bar_accent is applied by the ripple bakers only.
            col.label(text="Accents apply to Water Ripple", icon="INFO")


class AUDIOGEO_PT_batch(_Base, Panel):
    bl_label = "Batch"
    bl_parent_id = "AUDIOGEO_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True

        col = layout.column(align=True)
        col.prop(props, "batch_folder")
        col.prop(props, "batch_output")
        col.prop(props, "batch_recursive")

        # No track count here on purpose: draw() runs on every redraw, and a
        # directory scan per redraw stalls the UI on a network path.
        layout.operator("audiogeo.batch", icon="RENDER_ANIMATION")
        layout.label(text="Blender is blocked while each track renders",
                     icon="INFO")


class AUDIOGEO_PT_engine(_Base, Panel):
    bl_label = "Engine"
    bl_parent_id = "AUDIOGEO_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True

        col = layout.column(align=True)
        col.prop(props, "reactive_mode")
        if props.reactive_mode == "TEXTURE":
            sub = col.column(align=True)
            sub.prop(props, "texture_strength")
            sub.prop(props, "texture_displace")
            if not props.create_material:
                sub.label(text="Needs Create Material", icon="ERROR")
            else:
                sub.label(text="Any bin count; no keyframe bloat", icon="INFO")

        layout.separator()
        col = layout.column(align=True)
        enabled = (props.geometry_type in RIPPLE_MODES
                   and props.ripple_source == "SPECTRUM")
        sub = col.column(align=True)
        sub.enabled = enabled
        sub.prop(props, "wave_backend")
        if not enabled:
            col.label(text="Geometry Nodes needs Water Ripple + Spectrum",
                      icon="INFO")
        elif props.wave_backend == "NODES":
            # Both read SIN/COS shape keys by name, which this path does not
            # create.
            col.label(text="No web export or glTF morphs", icon="ERROR")


class AUDIOGEO_PT_modulation(_Base, Panel):
    bl_label = "Modulation"
    bl_parent_id = "AUDIOGEO_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.prop(props, "use_mod_matrix")

        col = layout.column()
        col.enabled = props.use_mod_matrix
        row = col.row()
        row.template_list("AUDIOGEO_UL_routes", "", props, "mod_routes",
                          props, "mod_index", rows=4)
        side = row.column(align=True)
        side.operator("audiogeo.route_add", text="", icon="ADD")
        side.operator("audiogeo.route_remove", text="", icon="REMOVE")

        if props.mod_routes and 0 <= props.mod_index < len(props.mod_routes):
            route = props.mod_routes[props.mod_index]
            box = col.box()
            box.use_property_split = True
            c = box.column(align=True)
            c.prop(route, "source")
            c.prop(route, "target")
            c.prop(route, "gain")
            c.prop(route, "offset")
            c.prop(route, "curve")
            c.prop(route, "attack_ms")
            c.prop(route, "release_ms")
            c.prop(route, "invert")

        col.operator("audiogeo.route_defaults", icon="PRESET")


class AUDIOGEO_PT_arrangement(_Base, Panel):
    bl_label = "Arrangement"
    bl_parent_id = "AUDIOGEO_PT_main"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        props = context.scene.audiogeo
        layout.use_property_split = True

        col = layout.column(align=True)
        col.prop(props, "mark_sections")
        col.prop(props, "section_variation")

        layout.separator()
        col = layout.column(align=True)
        col.prop(props, "use_section_presets")
        sub = col.column(align=True)
        sub.enabled = props.use_section_presets
        sub.prop(props, "section_blend")
        sub.operator("audiogeo.auto_sections", icon="PRESET")

        if props.use_section_presets:
            if len(props.section_slots):
                box = layout.box()
                box.label(text="Look per section", icon="SEQUENCE")
                c = box.column(align=True)
                for i, slot in enumerate(props.section_slots):
                    c.prop(slot, "preset", text=slot.label or "Section %d" % (i + 1))
            else:
                col.label(text="Generate once, then Auto-Assign", icon="INFO")
            # Honest about the limit rather than letting it look broken.
            col.label(text="Only wave heights vary; shape is fixed",
                      icon="INFO")


classes = (
    AUDIOGEO_PT_main,
    AUDIOGEO_PT_frequency,
    AUDIOGEO_PT_geometry,
    AUDIOGEO_PT_physics,
    AUDIOGEO_PT_timing,
    AUDIOGEO_PT_engine,
    AUDIOGEO_PT_motion,
    AUDIOGEO_PT_modulation,
    AUDIOGEO_PT_arrangement,
    AUDIOGEO_PT_lighting,
    AUDIOGEO_PT_extra,
    AUDIOGEO_PT_export,
    AUDIOGEO_PT_presets,
    AUDIOGEO_PT_render,
    AUDIOGEO_PT_batch,
)