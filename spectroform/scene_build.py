import math

import bmesh
import bpy
import numpy as np

from . import baking
from .constants import MAT_NAME, OBJ_NAME, SK_BASIS, SK_BASS, SK_MID, SK_TREBLE, STRIP_NAME

CAM_NAME = "Spectroform_Camera"
RIG_NAME = "Spectroform_Rig"

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

def add_camera_rig(context, obj, frames, props):
    from . import baking

    rig = bpy.data.objects.get(RIG_NAME)
    if rig is None:
        rig = bpy.data.objects.new(RIG_NAME, None)
        context.collection.objects.link(rig)
    rig.rotation_mode = "XYZ"

    cam = bpy.data.objects.get(CAM_NAME)
    if cam is None:
        cam = bpy.data.objects.new(CAM_NAME, bpy.data.cameras.new(CAM_NAME))
        context.collection.objects.link(cam)
    cam.parent = rig

    dist = props.size * 1.8
    cam.location = (0.0, -dist, props.size * 0.75)
    cam.rotation_euler = (math.radians(66.0), 0.0, 0.0)

    n = len(frames)
    turns = float(props.camera_orbit)
    baking.bake_curve(rig, "rotation_euler", 2, frames,
                      np.linspace(0.0, 2.0 * np.pi * turns, n).astype(np.float32))
    context.scene.camera = cam
    return cam

def bake_emission_hue(mat, frames, hue, sat=0.85):
    """Animate the emission colour. `hue` and `sat` are per-frame, 0..1.

    A colour socket's default_value is a 4-float array, so these take a real
    channel index -- unlike the scalar sockets, which need -1.
    """
    import colorsys
    from . import baking

    nodes = mat.node_tree.nodes
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return
    sock = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
    if sock is None:
        return
    idx = next((i for i, s in enumerate(bsdf.inputs) if s == sock), None)
    if idx is None:
        return

    hue = np.asarray(hue, dtype=np.float32).ravel()
    sat = np.broadcast_to(np.asarray(sat, dtype=np.float32).ravel(), hue.shape)
    path = 'nodes["%s"].inputs[%d].default_value' % (bsdf.name, idx)
    rgb = np.array([colorsys.hsv_to_rgb(float(h) % 1.0, float(sv), 1.0)
                    for h, sv in zip(hue, sat)], dtype=np.float32)
    for c in range(3):
        baking.bake_curve(mat.node_tree, path, c, frames, rgb[:, c])

def add_section_markers(scene, bounds, frame_start):
    for m in [m for m in scene.timeline_markers if m.name.startswith("SF_")]:
        scene.timeline_markers.remove(m)
    for i, b in enumerate(bounds):
        scene.timeline_markers.new("SF_Section_%d" % (i + 1),
                                   frame=int(b) + int(frame_start))


def add_colour_attribute(obj, rgba, name="Spectroform_Colour"):
    mesh = obj.data
    attr = mesh.color_attributes.get(name)
    if attr is None:
        attr = mesh.color_attributes.new(name=name, type="FLOAT_COLOR",
                                         domain="POINT")
    flat = np.ascontiguousarray(rgba, dtype=np.float32).ravel()
    if len(attr.data) * 4 != flat.size:
        return None
    attr.data.foreach_set("color", flat)
    mesh.update()
    return attr

def wire_colour_into_material(mat, attr_name="Spectroform_Colour"):
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return
    node = next((n for n in nodes if n.type == "VERTEX_COLOR"), None)
    if node is None:
        node = nodes.new("ShaderNodeVertexColor")
        node.location = (bsdf.location.x - 320, bsdf.location.y - 180)
    node.layer_name = attr_name
    sock = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
    if sock is not None:
        links.new(node.outputs["Color"], sock)

def build_bar_objects(context, bins, size, radial=True):
    import math as _m

    parent = bpy.data.objects.new("Spectroform_Bars", None)
    context.collection.objects.link(parent)

    mesh = bpy.data.meshes.new("Spectroform_Bar")
    bm = mesh.new()
    bmesh.ops.create_cube(bm, size = 1.0)
    for v in bm.verts:
        v.co.z += 0.5
    bm.to_mesh(mesh)
    bm.free

    objs = []
    for b in range(bins):
        o = bpy.data.objects.new("Spectroform_Bar_%02d" % b, mesh)
        context.collection.objects.link(o)
        o.parent = parent
        if radial:
            a = 2.0 * _m.pi * b/bins
            o.location = (_m.cos(a) * size * 0.4, _m.sin(a) * size * 0.4, 0.0)
            o.rotation_euler = (0.0, 0.0, a)
        else:
            o.location = (-size * 0.5 + size * (b + 0.5) / bins, 0.0, 0.0)
        o.scale = (size / bins * 0.6, size / bins * 0.6, 0.05)
        objs.append(o)
    return parent, objs


def bake_bar_objects(objs, spec, props, frames):
    from . import baking
    fps = float(props.bake_fps)
    for b, o in enumerate(objs):
        col = spec[:, b]
        peak = float(col.max())
        if  props.per_band_level and peak > 1e-6:
            col = col / peak
        h = analysis.envelope(np.clip(col * props.master_gain, 0.0, 2.0),
                              fps, props.attack_ms, props.release_ms)
        baking.bake_curve(o, "scale", 2, frames,
                          (0.02 + h * props.spec_height * props.size).astype(np.float32))

def _compositor(scene):
    """(tree, source socket, destination socket) for the scene compositor.

    Blender 5.0 replaced scene.node_tree with a compositing node *group*,
    fed by a Group Input instead of a Render Layers node, so the two have
    to be found differently.
    """
    if hasattr(scene, "compositing_node_group"):          # 5.0+
        tree = scene.compositing_node_group
        if tree is None:
            tree = bpy.data.node_groups.new("Compositing", "CompositorNodeTree")
            tree.interface.new_socket(
                "Image", in_out="INPUT", socket_type="NodeSocketColor")
            tree.interface.new_socket(
                "Image", in_out="OUTPUT", socket_type="NodeSocketColor")
            tree.nodes.new("NodeGroupInput")
            tree.nodes.new("NodeGroupOutput")
            scene.compositing_node_group = tree
        source = next((n for n in tree.nodes if n.type == "GROUP_INPUT"), None)
        dest = next((n for n in tree.nodes if n.type == "GROUP_OUTPUT"), None)
    else:                                                  # 3.6 - 4.x
        scene.use_nodes = True
        tree = scene.node_tree
        source = next((n for n in tree.nodes if n.type == "R_LAYERS"), None)
        dest = next((n for n in tree.nodes if n.type == "COMPOSITE"), None)
    return tree, source, dest


def _glare_settings(glare, threshold, size, mix):
    """Glare is configured by node properties up to 4.x and by input
    sockets from 5.0, where the menus take their display names."""
    if hasattr(glare, "glare_type"):
        glare.glare_type = "FOG_GLOW"
        if "quality" in glare.bl_rna.properties:
            glare.quality = "HIGH"
        glare.threshold = threshold
        glare.size = size
        glare.mix = mix
        return

    values = (("Type", "Fog Glow"), ("Quality", "High"),
              ("Threshold", threshold), ("Size", size))
    for name, value in values:
        socket = glare.inputs.get(name)
        if socket is not None:
            socket.default_value = value


def setup_glare(scene, threshold=1.0, size=8, mix=0.0):
    tree, source, dest = _compositor(scene)
    glare = next((n for n in tree.nodes if n.type == "GLARE"), None)
    if glare is None:
        glare = tree.nodes.new("CompositorNodeGlare")
    _glare_settings(glare, threshold, size, mix)

    if source is not None and dest is not None:
        tree.links.new(source.outputs["Image"], glare.inputs["Image"])
        tree.links.new(glare.outputs["Image"], dest.inputs["Image"])
    return glare


def add_camera_cuts(context, props, bounds, frame_start):
    import math as _m

    scene = context.scene
    angles = ((0.0, 1.9, 0.55), (_m.pi * 0.6, 1.2, 0.25),
              (_m.pi * 1.2, 2.6, 0.9), (_m.pi * 1.7, 0.9, 0.15))
    frames = [int(frame_start)] + [int(b) + int(frame_start) for b in bounds]

    cams = []
    for i, frame in enumerate(frames):
        name = "Spectroform_Cam_%d" % i
        cam = bpy.data.objects.get(name)
        if cam is None:
            cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
            context.collection.objects.link(cam)
        ang, dist, height = angles[i % len(angles)]
        cam.location = (_m.cos(ang) * props.size * dist,
                        _m.sin(ang) * props.size * dist,
                        props.size * height)
        track = cam.constraints.get("SF_Track") or cam.constraints.new("TRACK_TO")
        track.name = "SF_Track"
        track.target = bpy.data.objects.get(OBJ_NAME)
        track.track_axis = "TRACK_NEGATIVE_Z"
        track.up_axis = "UP_Y"

        marker = next((m for m in scene.timeline_markers if m.frame == frame), None)
        if marker is None:
            marker = scene.timeline_markers.new("SF_Cut_%d" % i, frame=frame)
        marker.camera = cam
        cams.append(cam)

    scene.camera = cams[0]
    return cams


def add_light_rig(context, props, frames, bands):
    from . import baking

    colours = ((1.0, 0.25, 0.2), (0.2, 1.0, 0.45), (0.35, 0.5, 1.0))
    places = ((-1.0, -1.0, 1.0), (1.0, -0.8, 1.2), (0.0, 1.2, 1.4))
    out = []
    for i, (name, band) in enumerate(bands):
        key = "Spectroform_Light_%s" % name
        obj = bpy.data.objects.get(key)
        if obj is None:
            data = bpy.data.lights.new(key, type="AREA")
            obj = bpy.data.objects.new(key, data)
            context.collection.objects.link(obj)
        obj.data.color = colours[i % 3]
        obj.data.size = props.size * 0.8
        obj.location = tuple(c * props.size for c in places[i % 3])

        watts = (props.light_power * (0.1 + np.asarray(band, dtype=np.float32))
                 * props.size ** 2)
        baking.bake_curve(obj.data, "energy", -1, frames, watts)
        out.append(obj)
    return out


def add_onset_sparks(context, obj, props, onset, frames):
    psys = obj.modifiers.get("SF_Sparks")
    if psys is None:
        psys = obj.modifiers.new("SF_Sparks", type="PARTICLE_SYSTEM")
    settings = psys.particle_system.settings
    settings.count = int(props.spark_count)
    settings.lifetime = max(4, int(props.bake_fps * 0.8))
    settings.frame_start = int(frames[0])
    settings.frame_end = int(frames[-1])
    settings.emit_from = "FACE"
    settings.use_emit_random = True
    settings.normal_factor = props.size * 0.4
    settings.factor_random = props.size * 0.2
    settings.physics_type = "NEWTON"
    settings.effector_weights.gravity = 0.25
    settings.render_type = "HALO"
    settings.particle_size = props.size * 0.004
    return psys
