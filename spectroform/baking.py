import numpy as np

from . import analysis
from .constants import SK_BASS, SK_MID, SK_TREBLE

def find_fcurve(action, data_path, index):
    collections = []
    try:
        if action.fcurves is not None:
            collections.append(action.fcurves)
    except Exception:
        pass
    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for bag in getattr(strip, "channelbags", []):
                collections.append(bag.fcurves)
 
    for coll in collections:
        try:
            for fc in coll:
                if fc.data_path == data_path and fc.array_index == index:
                    return fc
        except Exception:
            continue
    return None

def bake_curve(id_data, data_path, index, frames, values):
    n = len(frames)
    if n == 0:
        return

    # Scalar properties (a shape key value, a node socket) reject an index,
    # while their F-Curve still reports array_index 0.
    if index < 0:
        id_data.keyframe_insert(data_path=data_path, frame=float(frames[0]))
    else:
        id_data.keyframe_insert(data_path=data_path, index=index, frame=float(frames[0]))

    action = id_data.animation_data.action
    fc = find_fcurve(action, data_path, max(index, 0))
    if fc is None:
        raise RuntimeError("Could not create an F-Curve for %s[%d]" % (data_path, index))
 
    kp = fc.keyframe_points
    if n > len(kp):
        kp.add(count=n - len(kp))
 
    f = np.asarray(frames, dtype=np.float32)
    v = np.asarray(values, dtype=np.float32)

    co = np.empty(n * 2, dtype=np.float32)
    co[0::2] = f
    co[1::2] = v
    kp.foreach_set("co", co)

    left = np.empty(n * 2, dtype=np.float32)
    left[0::2] = f - 0.33
    left[1::2] = v
    right = np.empty(n * 2, dtype=np.float32)
    right[0::2] = f + 0.33
    right[1::2] = v
    kp.foreach_set("handle_left", left)
    kp.foreach_set("handle_right", right)
 
    try:
        kp.foreach_set("interpolation", np.full(n, 1, dtype=np.int32))  # LINEAR
    except Exception:
        for k in kp:
            k.interpolation = "LINEAR"

    fc.update()

def bake_animation(obj, data, props, frames):
    gain = props.master_gain
    bass = np.clip(data["bass"] * props.bass_gain * gain, 0.0, 2.0)
    mid = np.clip(data["mid"] * props.mid_gain * gain, 0.0, 2.0)
    treble = np.clip(data["treble"] * props.treble_gain * gain, 0.0, 2.0)
    rms = data["rms"]
    onset = data["onset"]
 
    fps = float(props.bake_fps)
    bass = analysis.envelope(bass, fps, props.attack_ms, props.release_ms)
    mid = analysis.envelope(mid, fps, props.attack_ms, props.release_ms)
    # Treble is given a snappier envelope — spikes should flick, not swell.
    treble = analysis.envelope(treble, fps, props.attack_ms * 0.5, props.release_ms * 0.6)
    rms = analysis.envelope(rms, fps, props.attack_ms, props.release_ms)
    onset = analysis.envelope(onset, fps, 8.0, max(40.0, props.release_ms * 0.5))
 
    key = obj.data.shape_keys
    bake_curve(key, 'key_blocks["%s"].value' % SK_BASS, -1, frames, bass)
    bake_curve(key, 'key_blocks["%s"].value' % SK_MID, -1, frames, mid)
    bake_curve(key, 'key_blocks["%s"].value' % SK_TREBLE, -1, frames, treble)
 
    scale = props.base_scale * (1.0 + rms * props.scale_gain + onset * props.pulse_gain)
    for axis in range(3):
        bake_curve(obj, "scale", axis, frames, scale)
 
    if abs(props.spin_speed) > 1e-6 or props.spin_gain > 1e-6:
        step = (props.spin_speed + onset * props.spin_gain) / fps
        spin = np.cumsum(step).astype(np.float32)
        bake_curve(obj, "rotation_euler", 2, frames, spin)
 
    return {"bass": bass, "mid": mid, "treble": treble, "rms": rms, "onset": onset}

def bake_object_motion(obj, data, props, frames):
    fps = float(props.bake_fps)
    rms = analysis.envelope(data["rms"], fps, props.attack_ms, props.release_ms)
    onset = analysis.envelope(data["onset"], fps, 8.0, max(40.0, props.release_ms * 0.5))

    scale = props.base_scale * (1.0 + rms * props.scale_gain + onset * props.pulse_gain)
    for axis in range(3):
        bake_curve(obj, "scale", axis, frames, scale)

    if abs(props.spin_speed) > 1e-6 or props.spin_gain > 1e-6:
        step = (props.spin_speed + onset * props.spin_gain) / fps
        bake_curve(obj, "rotation_euler", 2, frames, np.cumsum(step).astype(np.float32))

    return {"rms": rms, "onset": onset,
            "energy": np.clip(rms * 0.6 + onset * 0.4, 0.0, 2.0)}

def bake_spectrum(obj, spec, props, frames):
    from .constants import SK_BIN

    fps = float(props.bake_fps)
    key = obj.data.shape_keys
    bins = spec.shape[1]

    for b in range(bins):
        v = np.clip(spec[:, b] * props.master_gain, 0.0, 2.0)
        lean = b / max(bins - 1, 1)
        v = analysis.envelope(v, fps,
                              props.attack_ms * (1.0 - 0.5 * lean),
                              props.release_ms * (1.0 - 0.4 * lean))
        bake_curve(key, 'key_blocks["%s"].value' % (SK_BIN % b), -1, frames, v)

