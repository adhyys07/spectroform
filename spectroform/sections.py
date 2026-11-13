import bpy
import numpy as np
from bpy.types import Operator, PropertyGroup
from bpy.props import EnumProperty, FloatProperty, StringProperty

BAKEABLE = ("terrain_strength", "wave_strength", "spike_strength",
            "ripple_speed", "ripple_beat_gain", "master_gain",
            "scale_gain", "pulse_gain", "spin_speed", "ripple_steepness",
            "attack_ms", "release_ms")


def preset_items(self, context):
    from .presets import PRESETS
    return [(k, k.replace("_", " ").title(), "") for k in PRESETS]


class AUDIOGEO_SectionSlot(PropertyGroup):
    preset: EnumProperty(name="Preset", items=preset_items)
    label: StringProperty(name="Label", default="")


def section_bounds(sections, n_frames, start_frame=0):
    picks = [int(s) for s in np.asarray(sections, dtype=np.int64).ravel()]
    bounds = [0] + sorted(p for p in picks if 0 < p < n_frames) + [int(n_frames)]
    return bounds


def preset_weights(bounds, n_frames, blend_frames=24):
    n_sections = max(1, len(bounds) - 1)
    w = np.zeros((n_sections, int(n_frames)), dtype=np.float64)
    half = max(1, int(blend_frames) // 2)

    for i in range(n_sections):
        a, b = bounds[i], bounds[i + 1]
        w[i, a:b] = 1.0
        if i > 0:
            lo, hi = max(0, a - half), min(int(n_frames), a + half)
            if hi > lo:
                t = np.linspace(0.0, 1.0, hi - lo)
                w[i, lo:hi] = t * t * (3.0 - 2.0 * t)
                w[i - 1, lo:hi] = 1.0 - w[i, lo:hi]

    total = np.maximum(w.sum(axis=0, keepdims=True), 1e-9)
    return (w / total).astype(np.float32)


def blended_values(props, slots, weights):
    from .presets import PRESETS

    n_sections, n_frames = weights.shape
    out = {}
    for name in BAKEABLE:
        current = float(getattr(props, name))
        per_section = np.empty(n_sections, dtype=np.float64)
        for i in range(n_sections):
            values = PRESETS.get(slots[i].preset if i < len(slots) else "", {})
            value = values.get(name, current)
            per_section[i] = float(value) if isinstance(value, (int, float)) else current
        out[name] = (weights.T @ per_section).astype(np.float32)
    return out


def amplitude_curve(props, slots, weights):
    """Per-frame amplitude multiplier from the blended presets.

    A ratio against the panel's own wave heights, not an absolute height: the
    bakers already apply the panel values, so handing them absolute numbers
    would multiply the amplitude twice.
    """
    values = blended_values(props, slots, weights)
    keys = ("terrain_strength", "wave_strength", "spike_strength")
    base = sum(max(float(getattr(props, k)), 0.0) for k in keys)
    if base <= 1e-6:
        # Every wave height is zero, so there is nothing to scale and a ratio
        # would divide by ~0.
        return np.ones(weights.shape[1], dtype=np.float32)
    blended = sum(values[k] for k in keys)
    return (blended / base).astype(np.float32)


def auto_assign(slots, sections, rms, n_frames, start_frame=0, order=None):
    from .presets import PRESETS

    order = order or ["CALM_OCEAN", "SPECTRUM_RING", "CLUB_VISUALIZER", "GLITCH"]
    order = [k for k in order if k in PRESETS] or list(PRESETS)
    bounds = section_bounds(sections, n_frames, start_frame)
    rms = np.asarray(rms, dtype=np.float64)

    energies = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], min(bounds[i + 1], rms.size)
        energies.append(float(rms[a:b].mean()) if b > a else 0.0)

    lo, hi = min(energies), max(energies)
    span = max(hi - lo, 1e-9)
    slots.clear()
    for i, energy in enumerate(energies):
        slot = slots.add()
        rank = int(np.clip((energy - lo) / span * (len(order) - 1) + 0.5,
                           0, len(order) - 1))
        slot.preset = order[rank]
        slot.label = "Section %d" % (i + 1)
    return len(energies)


class AUDIOGEO_OT_auto_sections(Operator):
    bl_idname = "audiogeo.auto_sections"
    bl_label = "Auto-Assign Presets"
    bl_description = ("Give each detected section a preset by how loud it is, "
                      "as a starting point")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        from . import dataexport
        return bool(dataexport.LAST_RUN.get("data"))

    def execute(self, context):
        from . import dataexport

        props = context.scene.audiogeo
        data = dataexport.LAST_RUN.get("data") or {}
        n = int(data.get("n_frames", 0))
        count = auto_assign(props.section_slots, data.get("sections", []),
                            data.get("rms", np.zeros(n)), n, props.frame_start)
        props.last_report = "Assigned presets to %d sections" % count
        return {"FINISHED"}


classes = (AUDIOGEO_SectionSlot, AUDIOGEO_OT_auto_sections)
