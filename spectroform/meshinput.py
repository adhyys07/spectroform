import bmesh
import bpy
import numpy as np

def geo_from_object(obj, depsgraph=None, apply_modifiers=True):
    if obj is None or obj.type != "MESH":
        raise RuntimeError("Pick a mesh object as the base.")

    if apply_modifiers:
        depsgraph = depsgraph or bpy.context.evaluated_depsgraph_get()
        source = obj.evaluated_get(depsgraph)
        mesh = source.to_mesh()
    else:
        source, mesh = obj, obj.data

    try:
        n = len(mesh.vertices)
        if n == 0:
            raise RuntimeError("That mesh has no vertices.")

        verts = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", verts)
        verts = verts.reshape(n, 3)

        normals = np.empty(n * 3, dtype=np.float32)
        mesh.vertices.foreach_get("normal", normals)
        normals = normals.reshape(n, 3)

        faces = [tuple(p.vertices) for p in mesh.polygons]
    finally:
        if apply_modifiers:
            source.to_mesh_clear()

    centre = verts.mean(axis=0)
    local = verts - centre
    radius = float(np.linalg.norm(local, axis=1).max())
    radius = radius if radius > 1e-9 else 1.0

    r = np.clip(np.linalg.norm(local, axis=1) / radius, 0.0, 1.0)
    f = 1.0 - r
    falloff = (f * f * (3.0 - 2.0 * f)).astype(np.float32)

    extent = local.max(axis=0) - local.min(axis=0)
    axis = int(np.argmax(extent))
    span = float(extent[axis]) or 1.0
    wave = ((local[:, axis] - local[:, axis].min()) / span).astype(np.float32)

    return {
        "verts": verts,
        "normals": normals,
        "faces": faces,
        "falloff": falloff,
        "wave": wave,
        "radius": radius,
        "centre": centre.astype(np.float32),
    }

def add_ripple_fields(geo, size, pattern, arms=3.0, warp=0.4, seed=0):
    from .waveform import _phase_field

    # The patterns are built around the origin, so they are measured from the
    # mesh's own centre -- an off-origin logo would otherwise get the edge of
    # the pattern instead of its middle.
    local = geo["verts"] - geo["centre"]
    geo["phase"] = _phase_field(local, size, pattern, arms, warp, seed)
    geo["direction"] = direction_field_mesh(
        geo["verts"], geo["faces"], geo["phase"], geo["normals"])
    return geo

def direction_field_mesh(verts, faces, phase, normals):
    verts = np.asarray(verts, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    acc = np.zeros_like(verts)
    weight = np.zeros(verts.shape[0])

    for face in faces:
        k = len(face)
        for i in range(k):
            a, b = face[i], face[(i + 1) % k]
            edge = verts[b] - verts[a]
            length2 = float(edge @ edge)
            if length2 < 1e-18:
                continue
            d = (phase[b] - phase[a]) / length2
            acc[a] += edge * d
            acc[b] += edge * d
            weight[a] += 1.0
            weight[b] += 1.0

    acc /= np.maximum(weight, 1.0)[:, None]
    nrm = np.asarray(normals, dtype=np.float64)
    acc -= nrm * (acc * nrm).sum(axis=1)[:, None]

    mag = np.linalg.norm(acc, axis=1, keepdims=True)
    out = np.where(mag > 1e-9, acc / np.maximum(mag, 1e-12), 0.0)
    return out.astype(np.float32)


def create_from_source(context, source, geo, smooth, keep_transform=True):
    from .constants import OBJ_NAME

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
    if keep_transform:
        obj.matrix_world = source.matrix_world.copy()

    if source.data.materials:
        for mat in source.data.materials:
            obj.data.materials.append(mat)


    for other in context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    return obj


def suggested_size(geo):
    return float(geo["radius"]) * 2.0
