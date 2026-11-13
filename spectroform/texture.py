import bpy
import numpy as np

IMG_NAME = "Spectroform_Spectrum"
NODE_LABEL = "SF_Spectrum"

def write_spectrum_image(spec, name=IMG_NAME):
    spec = np.asarray(spec, dtype=np.float32)
    h, w = spec.shape

    img = bpy.data.images.get(name)
    if img is not None and (img.size[0] != w or img.size[1] != h):
        bpy.data.images.remove(img)
        img = None
    if img is None:
        img = bpy.data.images.new(name, width=w, height=h,
                                  float_buffer=True, is_data=True)
    img.colorspace_settings.name = "Non-Color"

    px = np.ones((h, w, 4), dtype=np.float32)
    px[..., 0] = px[..., 1] = px[..., 2] = spec

    img.pixels.foreach_set(px.ravel())
    img.update()

    if not img.packed_file:
        try:
            img.pack()
        except RuntimeError:
            pass
    return img

def _drive_time(value_node, frame_start, n_frames):
    fcurve = value_node.outputs[0].driver_add("default_value")
    drv = fcurve.driver
    drv.type = "SCRIPTED"
    drv.expression = "((frame - %d) + 0.5) / %d" % (
        int(frame_start), max(int(n_frames), 1))
    for var in list(drv.variables):
        drv.variables.remove(var)
    return fcurve

def displace_with_texture(obj, img, frame_start, n_frames, height=0.3):
    tex = bpy.data.textures.get("Spectroform_Displace")
    if tex is None:
        tex = bpy.data.textures.new("Spectroform_Displace", type="IMAGE")
    tex.image = img
    tex.extension = "EXTEND"

    empty = bpy.data.objects.get("Spectroform_TexCoord")
    if empty is None:
        empty = bpy.data.objects.new("Spectroform_TexCoord", None)
        bpy.context.collection.objects.link(empty)
        empty.empty_display_size = 0.2

    mod = obj.modifiers.get("SF_Displace")
    if mod is None:
        mod = obj.modifiers.new("SF_Displace", type="DISPLACE")
    mod.texture = tex
    mod.direction = "Z"
    mod.mid_level = 0.0
    mod.strength = float(height)
    mod.texture_coords = "OBJECT"
    mod.texture_coords_object = empty

    empty.animation_data_clear()
    fcurve = empty.driver_add("location", 1)
    fcurve.driver.type = "SCRIPTED"
    fcurve.driver.expression = "-((frame - %d) / %d)" % (
        int(frame_start), max(int(n_frames), 1))
    for var in list(fcurve.driver.variables):
        fcurve.driver.variables.remove(var)
    return mod


def wire_spectrum_into_material(mat, img, frame_start, n_frames, strength=6.0):
    mat.use_nodes = True
    tree = mat.node_tree
    nodes, links = tree.nodes, tree.links

    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return None

    def node_of(kind, label, x, y):
        found = next((n for n in nodes if n.label == label), None)
        if found is None:
            found = nodes.new(kind)
            found.label = label
            found.location = (bsdf.location.x + x, bsdf.location.y + y)
        return found

    coord = node_of("ShaderNodeTexCoord", NODE_LABEL + "_Coord", -1100, 160)
    sep = node_of("ShaderNodeSeparateXYZ", NODE_LABEL + "_Sep", -920, 160)
    time = node_of("ShaderNodeValue", NODE_LABEL + "_Time", -920, -40)
    comb = node_of("ShaderNodeCombineXYZ", NODE_LABEL + "_Comb", -740, 80)
    tex = node_of("ShaderNodeTexImage", NODE_LABEL + "_Tex", -560, 80)
    ramp = node_of("ShaderNodeValToRGB", NODE_LABEL + "_Ramp", -340, -120)
    mul = node_of("ShaderNodeMath", NODE_LABEL + "_Gain", -340, 120)

    tex.image = img
    tex.interpolation = "Linear"

    tex.extension = "EXTEND"
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = float(strength)

    links.new(coord.outputs["Generated"], sep.inputs["Vector"])
    links.new(sep.outputs["X"], comb.inputs["X"])
    links.new(time.outputs["Value"], comb.inputs["Y"])
    links.new(comb.outputs["Vector"], tex.inputs["Vector"])
    links.new(tex.outputs["Color"], mul.inputs[0])
    links.new(tex.outputs["Color"], ramp.inputs["Fac"])

    strength_sock = bsdf.inputs.get("Emission Strength")
    if strength_sock is not None:
        links.new(mul.outputs["Value"], strength_sock)
    colour = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
    if colour is not None:
        links.new(ramp.outputs["Color"], colour)

    ramp.color_ramp.elements[0].color = (0.0, 0.05, 0.25, 1.0)
    ramp.color_ramp.elements[1].color = (1.0, 0.95, 0.85, 1.0)

    _drive_time(time, frame_start, n_frames)
    return tex