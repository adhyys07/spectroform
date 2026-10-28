import os
import time

import bpy
import numpy as np
from bpy.types import Operator

from . import analysis, audio_io, baking, geometry, scene_build, waveform
from .constants import OBJ_NAME, RIPPLE_MODES, SPECTRAL_BARS, SPECTRAL_STATIC



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
            samples, rate = audio_io.load_audio(props.audio_path)

            wm.progress_update(20)
            data = analysis.analyse_audio(
                samples, rate, props.bake_fps,
                props.bass_max, props.mid_max, props.dynamic_range,
                per_band=props.per_band_level,
            )
            n_frames = data["n_frames"]
            if n_frames > 30000:
                self.report({"WARNING"}, "Very long clip (%d frames) - bake may be slow." % n_frames)

            wm.progress_update(45)
            if props.replace_existing:
                scene_build.remove_previous(context)

            kind = props.geometry_type
            frames = np.arange(n_frames, dtype=np.float32) + float(props.frame_start)

            if kind in SPECTRAL_STATIC or kind in SPECTRAL_BARS:
                spec = analysis.spectrogram(
                    samples, rate, props.bake_fps,
                    bins=props.spectrum_bins,
                    dynamic_range=props.dynamic_range,
                )
            elif kind in RIPPLE_MODES and props.ripple_source == "SPECTRUM":
                spec = analysis.spectrogram(
                    samples, rate, props.bake_fps,
                    bins=props.ripple_components,
                    dynamic_range=props.dynamic_range,
                )

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
                    props.resolution, props.size, props.ripple_pattern)
            elif kind in SPECTRAL_BARS:
                geo = waveform.build_spectrum_bars(
                    props.spectrum_bins, props.size, radial=(kind == "BARS_RADIAL"))
            elif kind == "RIPPLE":
                geo = waveform.build_ripple(
                    props.resolution, props.size, props.ripple_pattern,
                    arms=props.ripple_arms, warp=props.ripple_turbulence,
                    seed=props.seed)
            else:
                geo = geometry.build_geometry(props)

            obj = scene_build.create_object(context, geo, props.shade_smooth)

            wm.progress_update(65)
            if kind in RIPPLE_MODES:
                if props.ripple_source == "SPECTRUM":
                    pairs = waveform.make_spectral_ripple_offsets(
                        geo, props, spec.shape[1])
                    scene_build.add_ripple_shape_keys(obj, pairs)
                    wm.progress_update(80)
                    baked = baking.bake_spectral_ripple(obj, spec, data, props, frames)
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


classes = (
    AUDIOGEO_OT_generate,
    AUDIOGEO_OT_export,
    AUDIOGEO_OT_clear,
)