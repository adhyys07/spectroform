import bpy
import numpy as np

TREE_NAME = "Spectroform_Waves"
MOD_NAME = "SF_Waves"

_IS_4X = bpy.app.version >= (4, 0, 0)

def _new_socket(tree, kind, name, in_out, default=0.0, lo=None, hi=None):
    if _IS_4X:
        sock = tree.interface.new_socket(name=name, in_out=in_out,
                                         socket_type=kind)
    else:
        sock = (tree.inputs if in_out == "INPUT" else tree.outputs).new(kind, name)
    if kind == "NodeSocketFloat" and hasattr(sock, "default_value"):
        sock.default_value = float(default)
        if lo is not None:
            sock.min_value = float(lo)
        if hi is not None:
            sock.max_value = float(hi)
    return sock

def build_wave_tree(n_components, cycles):
    tree = bpy.data.node_groups.get(TREE_NAME)
    if tree is not None:
        bpy.data.node_groups.remove(tree)
    tree = bpy.data.node_groups.new(TREE_NAME, "GeometryNodeTree")

    _new_socket(tree, "NodeSocketGeometry", "Geometry", "INPUT")
    _new_socket(tree, "NodeSocketGeometry", "Geometry", "OUTPUT")
    for c in range(n_components):
        _new_socket(tree, "NodeSocketFloat", "Amp %d" % c, "INPUT", 0.0, -4.0, 4.0)
        _new_socket(tree, "NodeSocketFloat", "Phase %d" % c, "INPUT", 0.0, -1e6, 1e6)

    nodes, links = tree.nodes, tree.links
    group_in = nodes.new("NodeGroupInput"); group_in.location = (-1200, 0)
    group_out = nodes.new("NodeGroupOutput"); group_out.location = (900, 0)

    phase_attr = nodes.new("GeometryNodeInputNamedAttribute")
    phase_attr.location = (-1000, -220)
    phase_attr.data_type = "FLOAT"
    phase_attr.inputs["Name"].default_value = "sf_phase"

    falloff_attr = nodes.new("GeometryNodeInputNamedAttribute")
    falloff_attr.location = (-1000, -360)
    falloff_attr.data_type = "FLOAT"
    falloff_attr.inputs["Name"].default_value = "sf_falloff"

    normal = nodes.new("GeometryNodeInputNormal"); normal.location = (-1000, -500)

    total = None
    for c in range(n_components):
        scale = nodes.new("ShaderNodeMath")
        scale.operation = "MULTIPLY"
        scale.location = (-800, -c * 180)
        scale.inputs[1].default_value = float(cycles[c]) * 2.0 * np.pi
        links.new(phase_attr.outputs["Attribute"], scale.inputs[0])

        add = nodes.new("ShaderNodeMath")
        add.operation = "ADD"
        add.location = (-620, -c * 180)
        links.new(scale.outputs["Value"], add.inputs[0])
        links.new(group_in.outputs["Phase %d" % c], add.inputs[1])

        sine = nodes.new("ShaderNodeMath")
        sine.operation = "SINE"
        sine.location = (-440, -c * 180)
        links.new(add.outputs["Value"], sine.inputs[0])

        amp = nodes.new("ShaderNodeMath")
        amp.operation = "MULTIPLY"
        amp.location = (-260, -c * 180)
        links.new(sine.outputs["Value"], amp.inputs[0])
        links.new(group_in.outputs["Amp %d" % c], amp.inputs[1])

        if total is None:
            total = amp
        else:
            acc = nodes.new("ShadeNodeMath")
            acc.operation = "ADD"
            acc.location = (-80, -c * 180)
            links.new(total.outputs["Value"], acc.inputs[0])
            links.new(amp.outputs["Value"], acc.inputs[1])
            total = acc

    masked = nodes.new("ShaderNodeMath")
    masked.operation = "MULTIPLY"
    masked.location = (120, 0)
    links.new(total.outputs["Value"], masked.inputs[0])
    links.new(falloff_attr.outputs["Attribute"], masked.inputs[1])

    offset = nodes.new("ShaderNodeVectorMath")
    offset.operation = "SCALE"
    offset.location = (320, -120)
    links.new(normal.outputs["Normal"], offset.inputs[0])
    links.new(masked.outputs["Value"], offset.inputs["Scale"])

    set_pos = nodes.new("GeometryNodeSetPosition")
    set_pos.location = (560, 0)
    links.new(group_in.outputs["Geometry"], set_pos.inputs["Geometry"])
    links.new(offset.outputs["Vector"], set_pos.inputs["Offset"])
    links.new(set_pos.outputs["Geometry"], group_out.inputs["Geometry"])
    return tree

def write_phase_attribute(obj, geo):
    mesh = obj.data
    for name, values in (("sf_phase", geo.get("phase")),
                         ("sf_falloff", geo.get("falloff"))):
        if values is None:
            continue
        attr = mesh.attributes.get(name)
        if attr is None:
            attr = mesh.attributes.new(name=name, type="FLOAT", domain="POINT")
        attr.data.foreach_set(
            "value", np.ascontiguousarray(values, dtype=np.float32).ravel())
    mesh.update()

def _socket_key(tree, name):
    if _IS_4X:
        for item in tree.interface.items_tree:
            if (getattr(item, "item_type", "SOCKET") == "SOCKET"
                    and item.name == name and item.in_out == "INPUT"):
                return item.identifier
        return None
    sock = tree.inputs.get(name)
    return sock.identifier if sock else None

def bake_tree_inputs(obj, tree, spec, cycles, props, frames, fps):
    from . import analysis, baking, physics
    mod = obj.modifiers.get(MOD_NAME) or obj.modifiers.new(MOD_NAME, type="MODES")
    mod.node_group = tree

    n_frames = len(frames)
    t = np.arange(n_frames, dtype=np.float64) / float(fps)
    speeds = (physics.dispersion_speeds(cycles, props.size,
                                        scale=props.ripple_speed)
              if props.use_dispersion
              else np.full(len(cycles), float(props.ripple_speed)))

    for c in range(spec.shape[1]):
        col = spec[:n_frames, c].astype(np.float32)
        peak = float(col.max())
        if props.per_band_level and peak > 1e-6:
            col = col / peak
        amp = analysis.envelope(np.clip(col * props.master_gain, 0.0, 2.0),
                                fps, props.attack_ms, props.release_ms)
        amp = amp * (props.terrain_strength * max(props.size, 1e-4)
                     / np.sqrt(float(spec.shape[1])))
        phase = (2.0 * np.pi * float(speeds[c]) * t).astype(np.float32)

        for name, values in (("Amp %d" % c, amp), ("Phase %d" % c, phase)):
            key = _socket_key(tree, name)
            if key is not None:
                baking.bake_curve(mod, '["%s"]' % key, -1, frames, values)
    return mod