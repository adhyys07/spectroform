import os
import time

import bpy
import numpy as np
from bpy.types import Operator

from . import (analysis, audio_io, baking, geometry, physics, scene_build,
               waveform)
from .constants import (MAT_NAME, OBJ_NAME, RIPPLE_MODES,
                        SPECTRAL_BARS, SPECTRAL_STATIC)

# Wave sources that need a spectrogram at ripple_components resolution.
SPECTRAL_RIPPLE = {"SPECTRUM", "EMITTER", "HPSS", "FARADAY", "INTERFERENCE",
                   "HARMONIC"}



class AUDIOGEO_OT_generate(Operator):
    bl_idname = "audiogeo.generate"
    bl_label = "Generate & Bake"
    bl_description = "Analyse the audio file, build the geometry and bake the animation"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bool(context.scene.audiogeo.audio_path)

    def execute(self, context):
        props = context.scene.audiogeo
        scene = context.scene
        t0 = time.time()

        wm = context.window_manager
        wm.progress_begin(0, 100)
        try:
            wm.progress_update(5)
            if props.use_stereo:
                samples, left, right, rate = audio_io.load_audio_channels(
                    props.audio_path)
            else:
                samples, rate = audio_io.load_audio(props.audio_path)
                left = right = samples

            wm.progress_update(20)
            data = analysis.analyse_audio(
                samples, rate, props.bake_fps,
                props.bass_max, props.mid_max, props.dynamic_range,
                per_band=props.per_band_level,
            )
            bpm, _lag = analysis.detect_tempo(data["onset"], props.bake_fps)
            data["bpm"] = bpm
            data["beats"] = analysis.beat_frames(
                data["onset"], props.bake_fps, bpm,
                start_frame=props.frame_start)

            n_frames = data["n_frames"]
            if n_frames > 30000:
                self.report({"WARNING"}, "Very long clip (%d frames) - bake may be slow." % n_frames)

            wm.progress_update(45)
            if props.replace_existing:
                scene_build.remove_previous(context)

            kind = props.geometry_type
            frames = np.arange(n_frames, dtype=np.float32) + float(props.frame_start)
            spec = None

            if kind in SPECTRAL_STATIC or kind in SPECTRAL_BARS:
                spec = analysis.spectrogram(
                    samples, rate, props.bake_fps,
                    bins=props.spectrum_bins,
                    dynamic_range=props.dynamic_range,
                )
            elif kind in RIPPLE_MODES and props.ripple_source in SPECTRAL_RIPPLE:
                if props.use_stereo and props.ripple_source == "SPECTRUM":
                    spec_l = analysis.spectrogram(
                        left, rate, props.bake_fps,
                        bins=props.ripple_components,
                        dynamic_range=props.dynamic_range)
                    spec_r = analysis.spectrogram(
                        right, rate, props.bake_fps,
                        bins=props.ripple_components,
                        dynamic_range=props.dynamic_range)
                    spec = spec_l.copy()
                    spec[:, 1::2] = spec_r[:, 1::2]
                else:
                    spec = analysis.spectrogram(
                        samples, rate, props.bake_fps,
                        bins=props.ripple_components,
                        dynamic_range=props.dynamic_range,
                    )
            elif kind in RIPPLE_MODES and props.ripple_source == "CHLADNI":
                # One band per plate mode, so each eigenmode has its own drive.
                spec = analysis.spectrogram(
                    samples, rate, props.bake_fps,
                    bins=props.chladni_modes,
                    dynamic_range=props.dynamic_range,
                )
            elif kind in RIPPLE_MODES and props.ripple_source == "MIDI":
                if not props.midi_path:
                    raise RuntimeError(
                        "MIDI wave source needs a MIDI file.")
                from . import midi_io
                notes, tpb, tempo = midi_io.read_midi(
                    bpy.path.abspath(props.midi_path))
                if not notes:
                    raise RuntimeError("No notes found in the MIDI file.")
                fr_notes = midi_io.midi_to_frames(
                    notes, tpb, tempo, props.bake_fps)
                envs = midi_io.midi_envelopes(fr_notes, n_frames, props.bake_fps)
                # Each channel becomes a column, so the rest of the pipeline
                # treats MIDI exactly like a spectrogram.
                chans = sorted(envs)[: props.ripple_components]
                spec = np.stack([envs[c] for c in chans], axis=1).astype(np.float32)
                data["midi_channels"] = len(chans)
                data["bpm"] = 60.0e6 / float(tempo)

            if spec is not None:
                bin_hz = analysis.bin_frequencies(spec.shape[1])
                data["sections"] = analysis.detect_sections(
                    spec, props.bake_fps, min_seconds=4.0)
                if props.section_variation > 0.0:
                    data["section_gain"] = baking.section_gain(
                        n_frames, data["sections"], props.section_variation,
                        seed=props.seed)
            else:
                bin_hz = None

            if kind == "SPECTROGRAM":
                geo = waveform.build_spectrogram(
                    spec, props.size, props.spec_height, props.time_steps)
            elif kind == "WAVE_TUBE":
                geo = waveform.build_waveform_tube(
                    samples, props.size, props.tube_radius,
                    props.tube_segments, props.time_steps)
            elif kind == "WAVE_RING":
                geo = waveform.build_waveform_ring(
                    samples, props.size, props.tube_radius,
                    props.tube_segments, props.time_steps)
            elif kind == "RIPPLE":
                geo = waveform.build_ripple(
                    props.resolution, props.size, props.ripple_pattern,
                    arms=props.ripple_arms, warp=props.ripple_turbulence,
                    seed=props.seed)
            elif kind in SPECTRAL_BARS:
                geo = waveform.build_spectrum_bars(
                    props.spectrum_bins, props.size, radial=(kind == "BARS_RADIAL"))
            else:
                geo = geometry.build_geometry(props)

            if kind == "SPECTROGRAM" and props.erosion_iterations > 0:
                rows = geo["verts"].shape[0] // spec.shape[1]
                geo = waveform.erode_spectrogram(
                    geo, rows, spec.shape[1],
                    props.erosion_iterations, props.erosion_talus)

            obj = scene_build.create_object(context, geo, props.shade_smooth)

            wm.progress_update(65)
            if kind in RIPPLE_MODES:
                if props.ripple_source == "HPSS":
                    harm, perc = analysis.hpss(
                        spec, props.hpss_kernel, props.hpss_kernel)
                    # Alternate tonal/percussive, matching the builder's
                    # alternating wavelengths.
                    spec = harm.copy()
                    spec[:, 1::2] = perc[:, 1::2]
                    pairs = waveform.make_hpss_ripple_offsets(
                        geo, props, spec.shape[1])
                    scene_build.add_ripple_shape_keys(obj, pairs)
                    wm.progress_update(80)
                    baked = baking.bake_spectral_ripple(
                        obj, spec, data, props, frames)
                elif props.ripple_source == "EMITTER":
                    pairs = waveform.make_emitter_offsets(geo, props)
                    scene_build.add_ripple_shape_keys(obj, pairs)
                    wm.progress_update(80)
                    if props.drum_response and spec is not None:
                        baking.bake_drum_emitters(
                            obj, props, frames, spec, bin_hz)
                        baked = baking.bake_object_motion(
                            obj, data, props, frames)
                    else:
                        baked = baking.bake_emitters(
                            obj, data, props, frames, data["beats"])
                elif props.ripple_source == "SPECTRUM":
                    pairs = waveform.make_spectral_ripple_offsets(
                        geo, props, spec.shape[1], stereo=props.use_stereo)
                    scene_build.add_ripple_shape_keys(obj, pairs)
                    wm.progress_update(80)
                    baked = baking.bake_spectral_ripple(obj, spec, data, props, frames)
                elif props.ripple_source == "CHLADNI":
                    modes = physics.mode_table()[: props.chladni_modes]
                    pairs = waveform.chladni_offsets(geo, props, modes)
                    scene_build.add_ripple_shape_keys(obj, pairs)
                    wm.progress_update(80)
                    baked = baking.bake_modal(
                        obj, spec, data, props, frames, modes)
                    data["modes"] = len(modes)
                elif props.ripple_source == "FARADAY":
                    pairs = waveform.make_spectral_ripple_offsets(
                        geo, props, spec.shape[1])
                    scene_build.add_ripple_shape_keys(obj, pairs)
                    wm.progress_update(80)
                    baked = baking.bake_faraday(obj, spec, data, props, frames)
                elif props.ripple_source == "INTERFERENCE":
                    pairs = waveform.interference_offsets(
                        geo, props, spec.shape[1])
                    scene_build.add_ripple_shape_keys(obj, pairs)
                    wm.progress_update(80)
                    baked = baking.bake_spectral_ripple(
                        obj, spec, data, props, frames)
                elif props.ripple_source == "HARMONIC":
                    chroma = analysis.chroma(samples, rate, props.bake_fps)
                    tonic, _maj, key_name, _c = analysis.detect_key(chroma)
                    data["key"] = key_name
                    tonic_hz = 440.0 * (2.0 ** ((tonic - 9) / 12.0))
                    count = min(spec.shape[1], props.ripple_components)
                    pairs = waveform.harmonic_ring_offsets(
                        geo, props, tonic_hz, count)
                    scene_build.add_ripple_shape_keys(obj, pairs)
                    wm.progress_update(80)
                    baked = baking.bake_harmonic_rings(
                        obj, spec, data, props, frames, count)
                elif props.ripple_source == "MIDI":
                    pairs = waveform.make_spectral_ripple_offsets(
                        geo, props, spec.shape[1])
                    scene_build.add_ripple_shape_keys(obj, pairs)
                    wm.progress_update(80)
                    baked = baking.bake_spectral_ripple(
                        obj, spec, data, props, frames)
                else:
                    pairs = waveform.make_ripple_offsets(geo, props)
                    scene_build.add_ripple_shape_keys(obj, pairs)
                    wm.progress_update(80)
                    baked = baking.bake_ripple(obj, data, props, frames)
            elif kind in SPECTRAL_BARS:
                offsets = waveform.make_bin_offsets(geo, props.spec_height, props.size)
                scene_build.add_bin_shape_keys(obj, offsets)
                wm.progress_update(80)
                baking.bake_spectrum(obj, spec, props, frames)
                baked = baking.bake_object_motion(obj, data, props, frames)
            elif kind in SPECTRAL_STATIC:
                # The mesh already *is* the audio; only the object moves.
                wm.progress_update(80)
                baked = baking.bake_object_motion(obj, data, props, frames)
                if kind == "SPECTROGRAM" and props.scroll_terrain:
                    baking.bake_scroll(obj, props, frames)
            else:
                bass_co, mid_co, treble_co = geometry.make_shape_offsets(geo, props)
                scene_build.add_shape_keys(obj, bass_co, mid_co, treble_co)
                wm.progress_update(80)
                baked = baking.bake_animation(obj, data, props, frames)

            scene.render.fps = int(props.bake_fps)
            scene.render.fps_base = 1.0
            scene.frame_start = int(props.frame_start)
            scene.frame_end = int(props.frame_start + n_frames - 1)
            scene.frame_set(scene.frame_start)

            if props.create_material:
                if "energy" in baked:
                    energy = baked["energy"]
                else:
                    energy = np.clip(
                        baked["bass"] * 0.4 + baked["mid"] * 0.3 + baked["treble"] * 0.3,
                        0.0, 2.0,
                    )
                scene_build.setup_material(obj, frames, energy)

                mat = bpy.data.materials.get(MAT_NAME)
                if mat is not None and props.colour_source == "HARMONY":
                    chroma = analysis.chroma(samples, rate, props.bake_fps)
                    _t, _maj, key_name, _c = analysis.detect_key(chroma)
                    data["key"] = key_name
                    hue, sat = baking.harmony_hue(chroma[:n_frames])
                    if hue.size >= n_frames:
                        scene_build.bake_emission_hue(
                            mat, frames, hue[:n_frames], sat[:n_frames])
                elif mat is not None and props.colour_source == "TIMBRE"                         and spec is not None:
                    centroid = analysis.spectral_centroid(spec, bin_hz)
                    scene_build.bake_emission_hue(mat, frames, centroid)

            if props.vertex_colour:
                rgba = waveform.frequency_colours(geo, props, spec)
                attr = scene_build.add_colour_attribute(obj, rgba)
                if attr is not None and props.create_material:
                    mat = bpy.data.materials.get(MAT_NAME)
                    if mat is not None:
                        scene_build.wire_colour_into_material(mat)

            if props.mark_sections and len(data.get("sections", ())):
                scene_build.add_section_markers(
                    scene, data["sections"], props.frame_start)

            if props.camera_cuts:
                # One camera per section, each bound to its own marker, so
                # Blender cuts between them on playback.
                scene_build.add_camera_cuts(
                    context, props, data.get("sections", ()), props.frame_start)
            elif props.add_camera:
                scene_build.add_camera_rig(context, obj, frames, props)

            if props.use_glare and props.create_material:
                # EEVEE has no bloom toggle any more; without this the
                # emission is bright but does not glow.
                scene_build.setup_glare(scene, props.glare_threshold)

            if props.light_rig:
                scene_build.add_light_rig(context, props, frames, (
                    ("Bass", data["bass"]),
                    ("Mid", data["mid"]),
                    ("Treble", data["treble"]),
                ))

            if props.spark_count > 0:
                try:
                    scene_build.add_onset_sparks(
                        context, obj, props, data["onset"], frames)
                except Exception as exc:
                    self.report({"WARNING"}, "Sparks skipped: %s" % exc)

            if props.lyrics_path:
                from . import lyrics as lyrics_mod
                try:
                    lines = lyrics_mod.read_lrc(
                        bpy.path.abspath(props.lyrics_path))
                    built = lyrics_mod.build_lyrics(
                        context, lines, props.bake_fps, props.frame_start,
                        size=props.size * 0.15)
                    data["lyrics"] = len(built)
                except Exception as exc:
                    self.report({"WARNING"}, "Lyrics skipped: %s" % exc)

            if spec is not None and kind in RIPPLE_MODES:
                # Hand the analysis to the live-preview handler, so slider
                # tweaks can be seen without a re-bake.
                from . import preview
                cycles = waveform.ripple_component_cycles(
                    spec.shape[1], props.ripple_cycles, props.ripple_spread)
                preview.set_cache(spec, data, cycles, props.bake_fps)

            if props.sync_sound:
                try:
                    scene_build.add_sound_strip(scene, props.audio_path, props.frame_start)
                except Exception as exc:
                    self.report({"WARNING"}, "Sound strip skipped: %s" % exc)

            wm.progress_update(100)
        

        except Exception as exc:
            wm.progress_end()
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        wm.progress_end()

        msg = "%d verts, %d frames @ %d fps (%.1fs audio) in %.1fs" % (
            len(obj.data.vertices), n_frames, props.bake_fps, data["duration"], time.time() - t0,
        )
        if data.get("bpm", 0.0) > 0.0:
            msg += " | %.0f BPM, %d beats" % (data["bpm"], len(data["beats"]))
        if data.get("key"):
            msg += ", key %s" % data["key"]
        if len(data.get("sections", ())):
            msg += ", %d sections" % len(data["sections"])
        if data.get("modes"):
            msg += ", %d plate modes" % data["modes"]
        if data.get("midi_channels"):
            msg += ", %d MIDI channels" % data["midi_channels"]
        if data.get("lyrics"):
            msg += ", %d lyric lines" % data["lyrics"]
        props.last_report = msg
        self.report({"INFO"}, msg)
        return {"FINISHED"}


class AUDIOGEO_OT_export(Operator):
    bl_idname = "audiogeo.export"
    bl_label = "Export"
    bl_description = "Export the generated object with its baked animation"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(OBJ_NAME) is not None

    def _target_path(self, props):
        path = bpy.path.abspath(props.export_path)
        ext = {"BLEND": ".blend", "GLB": ".glb", "GLTF": ".gltf", "FBX": ".fbx"}[props.export_format]
        root, cur = os.path.splitext(path)
        if cur.lower() != ext:
            path = root + ext
        folder = os.path.dirname(path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)
        return path

    def execute(self, context):
        props = context.scene.audiogeo
        obj = bpy.data.objects.get(OBJ_NAME)
        if obj is None:
            self.report({"ERROR"}, "Nothing generated yet.")
            return {"CANCELLED"}

        for other in context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj

        path = self._target_path(props)
        fmt = props.export_format
        only = props.export_selected_only

        try:
            if fmt == "BLEND":
                bpy.ops.wm.save_as_mainfile(filepath=path, copy=True)
            elif fmt == "FBX":
                # Keyword sets shift between versions; fall back to the minimum
                # that is guaranteed to exist.
                try:
                    bpy.ops.export_scene.fbx(
                        filepath=path,
                        use_selection=only,
                        object_types={"MESH"},
                        bake_anim=True,
                        bake_anim_use_all_bones=False,
                        bake_anim_use_nla_strips=False,
                        bake_anim_use_all_actions=False,
                        bake_anim_step=1.0,
                        bake_anim_simplify_factor=0.0,
                        add_leaf_bones=False,
                        apply_unit_scale=True,
                    )
                except TypeError:
                    bpy.ops.export_scene.fbx(filepath=path, use_selection=only, bake_anim=True)
            else:
                gltf_format = "GLB" if fmt == "GLB" else "GLTF_SEPARATE"
                try:
                    bpy.ops.export_scene.gltf(
                        filepath=path,
                        export_format=gltf_format,
                        use_selection=only,
                        export_animations=True,
                        export_morph=True,
                        export_frame_range=True,
                        export_apply=False,
                    )
                except TypeError:
                    bpy.ops.export_scene.gltf(filepath=path, export_format=gltf_format)
        except Exception as exc:
            self.report({"ERROR"}, "Export failed: %s" % exc)
            return {"CANCELLED"}

        props.last_report = "Exported to %s" % path
        self.report({"INFO"}, props.last_report)
        return {"FINISHED"}


class AUDIOGEO_OT_clear(Operator):
    bl_idname = "audiogeo.clear"
    bl_label = "Clear Result"
    bl_description = "Delete the generated object"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return bpy.data.objects.get(OBJ_NAME) is not None

    def execute(self, context):
        scene_build.remove_previous(context)
        context.scene.audiogeo.last_report = ""
        return {"FINISHED"}


class AUDIOGEO_OT_export_web(Operator):
    bl_idname = "audiogeo.export_web"
    bl_label = "Export Web Page"
    bl_description = ("Write a self-contained WebGL page. Only the two "
                      "coefficient curves per wave travel, so the file stays "
                      "small where a mesh cache would not")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        obj = bpy.data.objects.get(OBJ_NAME)
        return obj is not None and obj.data.shape_keys is not None

    def execute(self, context):
        from . import webexport

        props = context.scene.audiogeo
        scene = context.scene
        obj = bpy.data.objects.get(OBJ_NAME)

        n_comp = sum(1 for k in obj.data.shape_keys.key_blocks
                     if k.name.endswith("_SIN"))
        if n_comp == 0:
            self.report({"ERROR"},
                        "Web export needs a ripple bake (no SIN keys found).")
            return {"CANCELLED"}

        n_frames = int(scene.frame_end - scene.frame_start) + 1
        sin_c, cos_c = baking.collect_ripple_curves(obj, n_comp, n_frames)
        if sin_c is None:
            self.report({"ERROR"}, "No baked shape-key animation to export.")
            return {"CANCELLED"}

        cycles = waveform.ripple_component_cycles(
            n_comp, props.ripple_cycles, props.ripple_spread)
        path = bpy.path.abspath(props.web_export_path)
        folder = os.path.dirname(path)
        if folder and not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)

        try:
            size = webexport.export_web(
                path, sin_c, cos_c, cycles,
                audio_name=os.path.basename(bpy.path.abspath(props.audio_path)),
                title="Spectroform")
        except Exception as exc:
            self.report({"ERROR"}, "Web export failed: %s" % exc)
            return {"CANCELLED"}

        props.last_report = "Web page: %s (%.0f KB, %d waves x %d frames)" % (
            path, size / 1024.0, n_comp, n_frames)
        self.report({"INFO"}, props.last_report)
        return {"FINISHED"}


classes = (
    AUDIOGEO_OT_generate,
    AUDIOGEO_OT_export,
    AUDIOGEO_OT_export_web,
    AUDIOGEO_OT_clear,
)