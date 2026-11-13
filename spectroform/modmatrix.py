import bpy
import numpy as np
from bpy.types import Operator, PropertyGroup, UIList
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       IntProperty, StringProperty)

SOURCES = [
    ("BASS", "Bass", "Low band envelope"),
    ("MID", "Mid", "Mid band envelope"),
    ("TREBLE", "Treble", "High band envelope"),
    ("RMS", "Loudness", "Overall level"),
    ("ONSET", "Onset", "Transient flux -- spiky"),
    ("CENTROID", "Brightness", "Spectral centroid: dark to bright"),
    ("KICK", "Kick", "Low-band onset share"),
    ("SNARE", "Snare", "Mid-band onset share"),
    ("HAT", "Hats", "High-band onset share"),
    ("BEAT", "Beat Pulse", "Decaying spike on every detected beat"),
    ("BAR", "Bar Pulse", "Decaying spike on every bar line"),
    ("PHRASE", "Phrase Pulse", "Decaying spike where a phrase turns over"),
    ("SECTION", "Section Level", "Steps between detected sections"),
    ("PITCH", "Melody Pitch", "Lead pitch, log-scaled"),
    ("CORRELATION", "Stereo Width", "Phase correlation: mono to wide"),
    ("CONSTANT", "Constant", "A fixed value -- for offsetting a target"),
]

TARGETS = {
    "SCALE": ("Object Scale", "object", "scale", -1),
    "SCALE_Z": ("Object Height", "object", "scale", 2),
    "LOC_Z": ("Object Height Offset", "object", "location", 2),
    "ROT_Z": ("Spin", "object", "rotation_euler", 2),
    "ROT_X": ("Tilt", "object", "rotation_euler", 0),
    "EMIT": ("Emission Strength", "emission", None, -1),
    "HUE": ("Emission Hue", "hue", None, -1),
    "ROUGH": ("Roughness", "bsdf:Roughness", None, -1),
    "METALLIC": ("Metallic", "bsdf:Metallic", None, -1),
    "LIGHT": ("Light Power", "light", "energy", -1),
    "CAM_SHAKE": ("Camera Shake", "camera", "location", 0),
    "WAVE_AMP": ("Wave Height", "wave_amp", None, -1),
    "WAVE_SPEED": ("Wave Speed", "wave_speed", None, -1),
}

class AUDIOGEO_ModRoute(PropertyGroup):
    enabled: BoolProperty(name="On", default=True)
    source: EnumProperty(name="From", items=SOURCES, default="BASS")
    target: EnumProperty(
        name="To",
        items=[(k, v[0], v[0]) for k, v in sorted(TARGETS.items())],
        default="SCALE")
    gain: FloatProperty(name="Amount", default=0.25, min=-8.0, max=8.0,
                        soft_min=-2.0, soft_max=2.0)
    offset: FloatProperty(name="Offset", default=0.0, min=-8.0, max=8.0,
                          soft_min=-1.0, soft_max=1.0)
    curve: EnumProperty(
        name="Response",
        items=[("LINEAR", "Linear", "Straight through"),
               ("SQUARE", "Soft Knee", "Squared -- quiet parts stay calm"),
               ("SQRT", "Hard Knee", "Square root -- quiet parts already move"),
               ("SMOOTH", "S-Curve", "Smoothstep -- eases both ends")],
        default="LINEAR")
    attack_ms: FloatProperty(name="Attack", default=25.0, min=0.0, max=2000.0)
    release_ms: FloatProperty(name="Release", default=120.0, min=0.0, max=4000.0)
    invert: BoolProperty(name="Invert", default=False)


def build_sources(data, spec, bin_hz, props, n_frames, fps):
    from . import analysis

    zero = np.zeros(n_frames, dtype=np.float32)

    def fit(values):
        arr = np.asarray(values, dtype=np.float32).ravel()
        if arr.size >= n_frames:
            return arr[:n_frames]
        return np.pad(arr, (0, n_frames - arr.size))
    
    out = {
        "BASS": fit(data.get("bass", zero)),
        "MID": fit(data.get("mid", zero)),
        "TREBLE": fit(data.get("treble", zero)),
        "RMS": fit(data.get("rms", zero)),
        "ONSET": fit(data.get("onset", zero)),
        "PITCH": fit(data.get("pitch_unit", data.get("pitch_conf", zero))),
        "CORRELATION": fit((np.asarray(data["correlation"]) * 0.5 + 0.5)
                           if data.get("correlation") is not None else zero),
        "SECTION": fit(data.get("section_gain", zero)),
        "CONSTANT": np.ones(n_frames, dtype=np.float32),
    }

    if spec is not None and bin_hz is not None:
        out["CENTROID"] = fit(analysis.spectral_centroid(spec, bin_hz))
        low, mid, high = analysis.onset_bands(spec, bin_hz)
        out["KICK"], out["SNARE"], out["HAT"] = fit(low), fit(mid), fit(high)
    else:
        out["CENTROID"] = out["KICK"] = out["SNARE"] = out["HAT"] = zero

    for name, key, decay in (("BEAT", "beats", 0.18),
                             ("BAR", "bars", 0.3),
                             ("PHRASE", "phrases", 0.6)):
        out[name] = _event_pulse(data.get(key), n_frames, fps, decay,
                                 props.frame_start)
    return out


def _event_pulse(frames, n_frames, fps, decay, start_frame):
    out = np.zeros(int(n_frames), dtype=np.float32)
    if frames is None or not len(np.asarray(frames).ravel()):
        return out
    tau = max(float(decay) * float(fps), 1.0)
    ramp = np.exp(-np.arange(0, int(tau * 4.0)) / tau).astype(np.float32)
    for f in np.asarray(frames, dtype=np.int64):
        i = int(f) - int(start_frame)
        if 0 <= i < out.size:
            end = min(out.size, i + ramp.size)
            out[i:end] = np.maximum(out[i:end], ramp[:end - i])
    return out

def shape_route(route, source, fps):
    from . import analysis
    x = np.clip(np.asarray(source, dtype=np.float32), 0.0, 1.0)

    if route.curve == "SQUARE":
        x = x * x
    elif route.curve == "SQRT":
        x = np.sqrt(x)
    elif route.curve == "SMOOTH":
        x = x * x * (3.0 - 2.0 * x)

    if route.attack_ms > 0.0 or route.release_ms > 0.0:
        x = analysis.envelope(x, fps, max(route.attack_ms, 1.0),
                              max(route.release_ms, 1.0))
    if route.invert:
        x = 1.0 - x
    return (x * float(route.gain) + float(route.offset)).astype(np.float32)

def apply_routes(context, obj, props, frames, sources, fps):
    from . import baking, scene_build
    from .constants import MAT_NAME

    n = len(frames)
    totals = {}
    for route in props.mod_routes:
        if not route.enabled:
            continue
        curve = shape_route(route, sources.get(route.source,
                                               np.zeros(n, np.float32)), fps)
        totals[route.target] = totals.get(
            route.target, np.zeros(n, dtype=np.float32)) + curve

    mat = bpy.data.materials.get(MAT_NAME)
    applied = 0

    for target, curve in total.items():
        label, kind, path, index = TARGETS[target]

        if kind == "object":
            if target == "SCALE":
                values = props.base_scale * (1.0 + curve)
                for axis in range(3):
                    baking.bake_curve(obj, path, axis, frames, values)
            elif path == "scale":
                baking.bake_curve(obj, path, index, frames,
                                  props.base_scale * (1.0 + curve))
            else:
                baking.bake_curve(obj, path, index, frames, curve)
            applied += 1

        elif kind == "emission" and mat is not None:
            scene_build.setup_material(obj, frames, np.clip(curve, 0.0, 20.0))
            applied += 1

        elif kind == "hue" and mat is not None:
            scene_build.bake_emission_hue(mat, frames, np.clip(curve, 0.0, 1.0))
            applied += 1

        elif kind.startswith("bsdf:") and mat is not None:
            _bake_bsdf_input(mat, kind.split(":", 1)[1], frames,
                             np.clip(curve, 0.0, 1.0))
            applied += 1

        elif kind == "light":
            for light in [o for o in bpy.data.objects
                          if o.name.startswith("Spectroform_Light")]:
                baking.bake_curve(light.data, path, index, frames,
                                  np.maximum(curve, 0.0) * props.light_power)
            applied += 1

        elif kind == "camera" and context.scene.camera is not None:
            cam = context.scene.camera
            base = float(cam.location[index])
            baking.bake_curve(cam, path, index, frames,
                              base + curve * props.size * 0.05)
            applied += 1

    return applied, totals

def _bake_bsdf_input(mat, socket_name, frames, values):
    from . import baking

    nodes = mat.node_tree.nodes
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return
    sock = bsdf.inputs.get(socket_name)
    if sock is None:
        return
    idx = next((i for i, s in enumerate(bsdf.inputs) if s == sock), None)
    if idx is None:
        return
    baking.bake_curve(mat.node_tree,
                      'nodes["%s"].inputs[%d].default_value' % (bsdf.name, idx),
                      -1, frames, values)


class AUDIOGEO_UL_routes(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_prop, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        sub = row.row(align=True)
        sub.enabled = item.enabled
        sub.prop(item, "source", text="")
        sub.label(icon="FORWARD")
        sub.prop(item, "target", text="")
        sub.prop(item, "gain", text="")


class AUDIOGEO_OT_route_add(Operator):
    bl_idname = "audiogeo.route_add"
    bl_label = "Add Route"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.audiogeo
        props.mod_routes.add()
        props.mod_index = len(props.mod_routes) - 1
        return {"FINISHED"}

class AUDIOGEO_OT_route_remove(Operator):
    bl_idname = "audiogeo.route_remove"
    bl_label = "Remove Route"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return len(context.scene.audiogeo.mod_routes) > 0

    def execute(self, context):
        props = context.scene.audiogeo
        props.mod_routes.remove(props.mod_index)
        props.mod_index = max(0, props.mod_index - 1)
        return {"FINISHED"}

class AUDIOGEO_OT_route_defaults(Operator):
    bl_idname = "audiogeo.route_defaults"
    bl_label = "Load Classic Mapping"
    bl_description = ("Fill the matrix with the fixed mapping the add-on used "
                      "before routes existed, as a starting point")
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.audiogeo
        props.mod_routes.clear()
        for source, target, gain in (("RMS", "SCALE", 0.25),
                                     ("ONSET", "SCALE", 0.15),
                                     ("BASS", "WAVE_AMP", 0.25),
                                     ("CENTROID", "HUE", 1.0),
                                     ("RMS", "EMIT", 6.0)):
            route = props.mod_routes.add()
            route.source, route.target, route.gain = source, target, gain
        props.mod_index = 0
        return {"FINISHED"}

classes = (AUDIOGEO_ModRoute, AUDIOGEO_UL_routes, AUDIOGEO_OT_route_add,
           AUDIOGEO_OT_route_remove, AUDIOGEO_OT_route_defaults)
