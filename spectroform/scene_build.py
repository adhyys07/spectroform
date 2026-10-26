import bmesh
import bpy
import numpy as np

from . import baking
from .constants import MAT_NAME, OBJ_NAME, SK_BASIS, SK_BASS, SK_MID, SK_TREBLE, STRIP_NAME

def remove_previous(context):
    obj = bpy.data.objects.get(OBJ_NAME)
    if obj is None:
        return
    mesh = obj.data
    for coll in list(obj.users_collection):
        coll.objects.unlink(obj)
    bpy.data.objects.remove(obj, do_unlink=True)
    if mesh is not None and mesh.users == 0:
        bpy.data.meshes.remove(mesh)


def create_object(context, geo, smooth):
    mesh = bpy.data.meshes.new(OBJ_NAME)
    mesh.from_pydata(geo["verts"].tolist(), [], geo["faces"])
    mesh.validate(verbose=False)

    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
 
    if smooth and len(mesh.polygons):
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    mesh.update()

    obj = bpy.data.objects.new(OBJ_NAME, mesh)
    context.collection.objects.link(obj)
    obj.rotation_mode = "XYZ"
 
    for other in context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj


def add_shape_keys(obj, bass_co, mid_co, treble_co):
    obj.shape_key_add(name=SK_BASIS, from_mix=False)
    obj.data.shape_keys.use_relative = True
 
    for name, coords in ((SK_BASS, bass_co), (SK_MID, mid_co), (SK_TREBLE, treble_co)):
        sk = obj.shape_key_add(name=name, from_mix=False)
        sk.slider_min = 0.0
        sk.slider_max = 2.0  # allows transient overshoot above 100 %
        flat = np.ascontiguousarray(coords, dtype=np.float32).ravel()
        sk.data.foreach_set("co", flat)

    obj.active_shape_key_index = 0

def setup_material(obj, frames, energy):
    mat = bpy.data.materials.get(MAT_NAME)
    if mat is None:
        mat = bpy.data.materials.new(MAT_NAME)
    mat.use_nodes = True
 
    nodes = mat.node_tree.nodes
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        out = next((n for n in nodes if n.type == "OUTPUT_MATERIAL"), None)
        if out:
            mat.node_tree.links.new(bsdf.outputs[0], out.inputs[0])
 
    bsdf.inputs["Base Color"].default_value = (0.02, 0.05, 0.12, 1.0)
    emit = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
    if emit is not None:
        emit.default_value = (0.1, 0.5, 1.0, 1.0)
 
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
 
    strength = bsdf.inputs.get("Emission Strength")
    if strength is None:
        return
    idx = next((i for i, s in enumerate(bsdf.inputs) if s == strength), None)
    if idx is None:
        return
 
    path = 'nodes["%s"].inputs[%d].default_value' % (bsdf.name, idx)
    values = (0.4 + energy * 6.0).astype(np.float32)
    try:
        baking.bake_curve(mat.node_tree, path, -1, frames, values)
    except Exception:
        pass  
 
def add_sound_strip(scene, filepath, frame_start):
    if scene.sequence_editor is None:
        scene.sequence_editor_create()
    se = scene.sequence_editor
    strips = se.strips if hasattr(se, "strips") else se.sequences
 
    existing = strips.get(STRIP_NAME) if hasattr(strips, "get") else None
    if existing is not None:
        strips.remove(existing)
 
    return strips.new_sound(
        name=STRIP_NAME,
        filepath=bpy.path.abspath(filepath),
        channel=1,
        frame_start=int(frame_start),
    )

def add_bin_shape_keys(obj, offsets):
    from .constants import SK_BIN

    obj.shape_key_add(name=SK_BASIS, from_mix=False)
    obj.data.shape_keys.use_relative = True

    for i, coords in enumerate(offsets):
        sk = obj.shape_key_add(name=SK_BIN % i, from_mix=False)
        sk.slider_min = 0.0
        sk.slider_max = 2.0
        sk.data.foreach_set("co", np.ascontiguousarray(coords, dtype=np.float32).ravel())

    obj.active_shape_key_index = 0

def add_ripple_shape_keys(obj, pairs):
    from .constants import SK_RIPPLE_COS, SK_RIPPLE_SIN

    obj.shape_key_add(name=SK_BASIS, from_mix=False)
    obj.data.shape_keys.use_relative = True

    for i, (sin_co, cos_co) in enumerate(pairs):
        for template, coords in ((SK_RIPPLE_SIN, sin_co), (SK_RIPPLE_COS, cos_co)):
            sk = obj.shape_key_add(name=template % i, from_mix=False)
            sk.slider_min = -2.0   # cos(wt) goes negative; 0.0 would clip half the wave
            sk.slider_max = 2.0
            sk.data.foreach_set("co", np.ascontiguousarray(coords, dtype=np.float32).ravel())

    obj.active_shape_key_index = 0
