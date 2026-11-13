import math

import bpy
import numpy as np
from bpy.types import Operator

PARENT_NAME = "Spectroform_Orchestra"


def layout_positions(count, size, arrangement="RING", curve_obj=None):
    count = max(1, int(count))
    locs, rots = [], []

    if arrangement == "RING":
        radius = size * 1.2
        for i in range(count):
            a = 2.0 * math.pi * i / count
            locs.append((math.cos(a) * radius, math.sin(a) * radius, 0.0))
            # Face outward, so an asymmetric shape reads the same all round.
            rots.append((0.0, 0.0, a))
    elif arrangement == "GRID":
        cols = max(1, int(round(math.sqrt(count))))
        step = size * 1.3
        for i in range(count):
            row, col = divmod(i, cols)
            locs.append(((col - (cols - 1) * 0.5) * step,
                         (row - ((count - 1) // cols) * 0.5) * step, 0.0))
            rots.append((0.0, 0.0, 0.0))
    elif arrangement == "LINE":
        step = size * 1.3
        for i in range(count):
            locs.append(((i - (count - 1) * 0.5) * step, 0.0, 0.0))
            rots.append((0.0, 0.0, 0.0))
    elif arrangement == "CURVE" and curve_obj is not None:
        locs, rots = _along_curve(curve_obj, count)
    else:
        locs = [(0.0, 0.0, i * size * 0.4) for i in range(count)]
        rots = [(0.0, 0.0, 0.0)] * count

    return locs, rots


def _along_curve(curve_obj, count):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = curve_obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        n = len(mesh.vertices)
        if n == 0:
            return [(0.0, 0.0, 0.0)] * count, [(0.0, 0.0, 0.0)] * count
        pts = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", pts)
        pts = pts.reshape(n, 3)
    finally:
        evaluated.to_mesh_clear()

    idx = np.linspace(0, n - 1, count).astype(int)
    picked = pts[idx]
    matrix = curve_obj.matrix_world
    locs = [tuple(matrix @ bpy.types.bpy_prop_array.__class__ and
                  matrix @ __import__("mathutils").Vector(p)) for p in picked]
    return locs, [(0.0, 0.0, 0.0)] * count


def build_orchestra(context, source_obj, count, size, arrangement="RING",
                    curve_obj=None, share_mesh=True):
    parent = bpy.data.objects.get(PARENT_NAME)
    if parent is None:
        parent = bpy.data.objects.new(PARENT_NAME, None)
        context.collection.objects.link(parent)

    locs, rots = layout_positions(count, size, arrangement, curve_obj)
    members = []
    for i in range(count):
        name = "Spectroform_Member_%02d" % i
        obj = bpy.data.objects.get(name)
        if obj is None:
            if share_mesh:
                obj = bpy.data.objects.new(name, source_obj.data)
            else:
                obj = bpy.data.objects.new(name, source_obj.data.copy())
            context.collection.objects.link(obj)
        obj.parent = parent
        obj.location = locs[i]
        obj.rotation_euler = rots[i]
        members.append(obj)
    return parent, members


def bake_members(members, spec, props, frames, stems=None, mode="BIN"):
    from . import analysis, baking

    fps = float(props.bake_fps)
    count = len(members)

    for i, obj in enumerate(members):
        if mode == "STEM" and stems:
            names = [n for n in stems if not n.startswith("_")]
            curve = stems[names[i % len(names)]]
        else:
            # One frequency bin each, spread across the spectrum.
            col = int(round(i * (spec.shape[1] - 1) / max(count - 1, 1)))
            curve = spec[:, col]

        peak = float(np.max(curve))
        if props.per_band_level and peak > 1e-6:
            curve = curve / peak
        env = analysis.envelope(
            np.clip(np.asarray(curve, np.float32) * props.master_gain, 0.0, 2.0),
            fps, props.attack_ms, props.release_ms)

        height = props.base_scale * (1.0 + env * props.spec_height)
        baking.bake_curve(obj, "scale", 2, frames, height.astype(np.float32))
        baking.bake_curve(obj, "location", 2, frames,
                          (env * props.size * 0.2).astype(np.float32))
    return count


class AUDIOGEO_OT_orchestra(Operator):
    bl_idname = "audiogeo.orchestra"
    bl_label = "Build Orchestra"
    bl_description = ("Instance the result many times, each driven by its own "
                      "frequency band or stem")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        from .constants import OBJ_NAME
        from . import dataexport
        return (bpy.data.objects.get(OBJ_NAME) is not None
                and dataexport.LAST_RUN.get("spec") is not None)

    def execute(self, context):
        from . import dataexport
        from .constants import OBJ_NAME

        props = context.scene.audiogeo
        source = bpy.data.objects.get(OBJ_NAME)
        spec = dataexport.LAST_RUN.get("spec")
        if spec is None:
            self.report({"ERROR"}, "Generate a spectral mode first.")
            return {"CANCELLED"}

        frames = (np.arange(spec.shape[0], dtype=np.float32)
                  + float(props.frame_start))
        _parent, members = build_orchestra(
            context, source, props.orchestra_count, props.size,
            props.orchestra_layout, props.orchestra_curve,
            share_mesh=(props.orchestra_drive == "TRANSFORM"))
        n = bake_members(members, spec, props, frames,
                         dataexport.LAST_RUN.get("stems"),
                         mode="STEM" if props.orchestra_drive == "STEM" else "BIN")

        source.hide_viewport = source.hide_render = props.orchestra_hide_source
        props.last_report = "Orchestra: %d members" % n
        self.report({"INFO"}, props.last_report)
        return {"FINISHED"}


classes = (AUDIOGEO_OT_orchestra,)
