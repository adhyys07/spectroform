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

def audio_phase(base_speed, onset, beat_gain, fps):
    """Phase integrated from the music: steady drift plus a surge per onset.

    Returns radians per frame accumulated, so crests advance when the track
    does and stall when it is quiet.
    """
    rate = (float(base_speed) + onset * float(beat_gain)) / fps
    return (2.0 * np.pi * np.cumsum(rate)).astype(np.float32)


def bake_ripple(obj, data, props, frames):
    from .constants import SK_RIPPLE_COS, SK_RIPPLE_SIN
    from .waveform import BAND_SPEED_RATIO

    fps = float(props.bake_fps)
    key = obj.data.shape_keys
    t = (np.asarray(frames, dtype=np.float32) - float(frames[0])) / fps

    bands = (
        ("bass", props.bass_gain),
        ("mid", props.mid_gain),
        ("treble", props.treble_gain),
    )

    onset = analysis.envelope(data["onset"], fps, 8.0, max(40.0, props.release_ms * 0.5))

    baked = {}
    for i, ((name, gain), speed_ratio) in enumerate(zip(bands, BAND_SPEED_RATIO)):
        amp = np.clip(data[name] * gain * props.master_gain, 0.0, 2.0)
        amp = analysis.envelope(amp, fps, props.attack_ms, props.release_ms)
        baked[name] = amp

        ph = audio_phase(props.ripple_speed * speed_ratio, onset,
                         props.ripple_beat_gain, fps)
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_SIN % i), -1, frames,
                   (amp * np.cos(ph)).astype(np.float32))
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_COS % i), -1, frames,
                   (-amp * np.sin(ph)).astype(np.float32))

    baked.update(bake_object_motion(obj, data, props, frames))
    return baked

def bake_scroll(obj, props, frames):
    n = len(frames)
    x0 = props.size * 0.5
    bake_curve(obj, "location", 0, frames,
               np.linspace(x0, -x0, n).astype(np.float32))

def bake_spectral_ripple(obj, spec, data, props, frames):
    """One travelling wave per spectrum bin, phase locked to the audio."""
    from .constants import SK_RIPPLE_COS, SK_RIPPLE_SIN

    fps = float(props.bake_fps)
    key = obj.data.shape_keys
    n_comp = spec.shape[1]

    onset = analysis.envelope(data["onset"], fps, 8.0, max(40.0, props.release_ms * 0.5))
    ratio = (np.linspace(0.0, 1.0, n_comp, dtype=np.float32)
             if n_comp > 1 else np.zeros(1, dtype=np.float32))
    # Higher bins travel faster, matching their shorter wavelength.
    speeds = props.ripple_speed * (2.0 ** (ratio * props.ripple_spread * 0.5))

    for i in range(n_comp):
        col = spec[:, i]
        if props.per_band_level:
            # Real music has little energy up top, so without this the fine
            # ripples never show. Each bin then swings its own full range.
            peak = float(col.max())
            if peak > 1e-6:
                col = col / peak
        amp = np.clip(col * props.master_gain, 0.0, 2.0)
        amp = analysis.envelope(amp, fps, props.attack_ms, props.release_ms)
        ph = audio_phase(float(speeds[i]), onset, props.ripple_beat_gain, fps)
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_SIN % i), -1, frames,
                   (amp * np.cos(ph)).astype(np.float32))
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_COS % i), -1, frames,
                   (-amp * np.sin(ph)).astype(np.float32))

    return bake_object_motion(obj, data, props, frames)
