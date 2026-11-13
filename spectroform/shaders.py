import bpy

LOOK_ITEMS = [
    ("EMISSION", "Emission", "The current look: dark base, audio-driven glow"),
    ("IRIDESCENT", "Iridescent", "Thin-film colour shifting with view angle"),
    ("GLASS", "Glass", "Refractive, with audio driving the roughness"),
    ("HOLOGRAM", "Hologram", "Scanlines and fresnel edge glow"),
    ("CHROME", "Chrome", "Polished metal -- reads entirely by reflection"),
    ("WIREFRAME", "Wireframe Glow", "Emissive edges over a dark body"),
]


def _fresh_tree(mat):
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    out = tree.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    return tree, out


def _principled(tree, out, x=100):
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (x, 0)
    tree.links.new(bsdf.outputs[0], out.inputs[0])
    return bsdf


def apply_look(mat, look, size=4.0):
    builder = {
        "EMISSION": _build_emission,
        "IRIDESCENT": _build_iridescent,
        "GLASS": _build_glass,
        "HOLOGRAM": _build_hologram,
        "CHROME": _build_chrome,
        "WIREFRAME": _build_wireframe,
    }.get(look, _build_emission)
    return builder(mat, size)


def _build_emission(mat, size):
    tree, out = _fresh_tree(mat)
    bsdf = _principled(tree, out)
    bsdf.inputs["Base Color"].default_value = (0.02, 0.05, 0.12, 1.0)
    emit = bsdf.inputs.get("Emission Color") or bsdf.inputs.get("Emission")
    if emit is not None:
        emit.default_value = (0.1, 0.5, 1.0, 1.0)
    return bsdf, bsdf.inputs.get("Emission Strength")


def _build_iridescent(mat, size):
    tree, out = _fresh_tree(mat)
    nodes, links = tree.nodes, tree.links
    bsdf = _principled(tree, out)

    layer = nodes.new("ShaderNodeLayerWeight")
    layer.location = (-600, 0)
    layer.inputs["Blend"].default_value = 0.35

    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.location = (-400, 0)
    ramp.color_ramp.interpolation = "EASE"
    for pos, colour in ((0.0, (0.8, 0.1, 0.5, 1.0)),
                        (0.45, (0.1, 0.6, 0.9, 1.0)),
                        (1.0, (0.9, 0.8, 0.2, 1.0))):
        element = (ramp.color_ramp.elements[0] if pos == 0.0 else
                   ramp.color_ramp.elements[-1] if pos == 1.0 else
                   ramp.color_ramp.elements.new(pos))
        element.position = pos
        element.color = colour

    links.new(layer.outputs["Fresnel"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Metallic"].default_value = 0.9
    bsdf.inputs["Roughness"].default_value = 0.15
    return bsdf, bsdf.inputs["Roughness"]


def _build_glass(mat, size):
    tree, out = _fresh_tree(mat)
    bsdf = _principled(tree, out)
    bsdf.inputs["Base Color"].default_value = (0.9, 0.95, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.05
    for name, value in (("Transmission Weight", 1.0), ("Transmission", 1.0),
                        ("IOR", 1.45)):
        sock = bsdf.inputs.get(name)
        if sock is not None:
            sock.default_value = value
    # EEVEE needs these or transmission renders opaque.
    mat.use_backface_culling = False
    mat.blend_method = "BLEND" if hasattr(mat, "blend_method") else "OPAQUE"
    return bsdf, bsdf.inputs["Roughness"]


def _build_hologram(mat, size):
    tree, out = _fresh_tree(mat)
    nodes, links = tree.nodes, tree.links

    emit = nodes.new("ShaderNodeEmission")
    emit.location = (100, 0)
    links.new(emit.outputs[0], out.inputs[0])

    coord = nodes.new("ShaderNodeTexCoord"); coord.location = (-900, 0)
    sep = nodes.new("ShaderNodeSeparateXYZ"); sep.location = (-720, 0)
    mul = nodes.new("ShaderNodeMath"); mul.location = (-540, 0)
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 180.0 / max(size, 1e-4)
    sine = nodes.new("ShaderNodeMath"); sine.location = (-360, 0)
    sine.operation = "SINE"
    gain = nodes.new("ShaderNodeMath"); gain.location = (-180, 0)
    gain.operation = "MULTIPLY_ADD"
    gain.inputs[1].default_value = 0.25
    gain.inputs[2].default_value = 0.75

    fres = nodes.new("ShaderNodeLayerWeight"); fres.location = (-540, -220)
    fres.inputs["Blend"].default_value = 0.2
    mix = nodes.new("ShaderNodeMath"); mix.location = (-180, -220)
    mix.operation = "ADD"

    links.new(coord.outputs["Object"], sep.inputs["Vector"])
    links.new(sep.outputs["Z"], mul.inputs[0])
    links.new(mul.outputs["Value"], sine.inputs[0])
    links.new(sine.outputs["Value"], gain.inputs[0])
    links.new(gain.outputs["Value"], mix.inputs[0])
    links.new(fres.outputs["Fresnel"], mix.inputs[1])
    links.new(mix.outputs["Value"], emit.inputs["Strength"])
    emit.inputs["Color"].default_value = (0.2, 0.9, 1.0, 1.0)
    return emit, emit.inputs["Strength"]


def _build_chrome(mat, size):
    tree, out = _fresh_tree(mat)
    bsdf = _principled(tree, out)
    bsdf.inputs["Base Color"].default_value = (0.85, 0.87, 0.9, 1.0)
    bsdf.inputs["Metallic"].default_value = 1.0
    bsdf.inputs["Roughness"].default_value = 0.08
    return bsdf, bsdf.inputs["Roughness"]


def _build_wireframe(mat, size):
    tree, out = _fresh_tree(mat)
    nodes, links = tree.nodes, tree.links

    mix = nodes.new("ShaderNodeMixShader"); mix.location = (200, 0)
    links.new(mix.outputs[0], out.inputs[0])

    body = nodes.new("ShaderNodeBsdfPrincipled"); body.location = (-100, 150)
    body.inputs["Base Color"].default_value = (0.01, 0.01, 0.02, 1.0)
    body.inputs["Roughness"].default_value = 0.6

    edge = nodes.new("ShaderNodeEmission"); edge.location = (-100, -150)
    edge.inputs["Color"].default_value = (0.3, 0.9, 1.0, 1.0)
    edge.inputs["Strength"].default_value = 4.0

    wire = nodes.new("ShaderNodeWireframe"); wire.location = (-360, 0)
    wire.use_pixel_size = False
    wire.inputs["Size"].default_value = max(size * 0.004, 1e-4)

    links.new(wire.outputs["Fac"], mix.inputs["Fac"])
    links.new(body.outputs[0], mix.inputs[1])
    links.new(edge.outputs[0], mix.inputs[2])
    # Cycles only -- the Wireframe node has no EEVEE equivalent, so the
    # caller should warn rather than render a black object.
    return edge, edge.inputs["Strength"]


def needs_cycles(look):
    return look == "WIREFRAME"
