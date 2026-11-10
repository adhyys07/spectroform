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

    if index < 0:
        id_data.keyframe_insert(data_path=data_path, frame=float(frames[0]))
    else:
        id_data.keyframe_insert(data_path=data_path, index=index, frame=float(frames[0]))

    action = id_data.animation_data.action
    fc = find_fcurve(action, data_path, max(index, 0))
    if fc is None:
        raise RuntimeError("Could not create an F-Curve for %s[%d]" % (data_path, index))
 
    kp = fc.keyframe_points
    # Re-baking a shorter clip onto an ID that kept its action (the material
    # survives between bakes) leaves stale keys behind, and foreach_set then
    # fails with an array-length mismatch. Size the list exactly.
    if len(kp) > n:
        try:
            kp.clear()
        except AttributeError:
            for point in list(kp)[n:]:
                kp.remove(point, fast=True)
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
    rate = (float(base_speed) + onset * float(beat_gain)) / fps
    return (2.0 * np.pi * np.cumsum(rate)).astype(np.float32)


def ripple_phase(data, props, frames, base_speed, onset, fps):
    """Phase for one wave: locked to detected beats, or free-running.

    Beat Lock blends between the two, so 0 keeps the onset-surge behaviour
    and 1 puts a crest exactly on every beat.
    """
    beats = np.asarray(data.get("beats", ()), dtype=np.float64)
    if float(props.beat_lock) > 0.0 and beats.size >= 2:
        return beat_locked_phase(len(frames), beats - float(frames[0]), fps,
                                 base_speed,
                                 cycles_per_beat=props.cycles_per_beat,
                                 lock=props.beat_lock)
    return audio_phase(base_speed, onset, props.ripple_beat_gain, fps)


def loop_pair(amp, ph, props):
    """Apply the loop crossfade to a wave's amplitude and phase."""
    fade = int(getattr(props, "loop_fade", 0))
    if fade >= 2:
        amp = make_loopable(amp, fade)
        ph = make_loopable(ph, fade, wrap=True)
    return np.asarray(amp), np.asarray(ph)


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

        ph = ripple_phase(data, props, frames,
                          props.ripple_speed * speed_ratio, onset, fps)
        amp, ph = loop_pair(amp, ph, props)
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
    from .constants import SK_RIPPLE_COS, SK_RIPPLE_SIN

    fps = float(props.bake_fps)
    key = obj.data.shape_keys
    n_comp = spec.shape[1]

    onset = analysis.envelope(data["onset"], fps, 8.0, max(40.0, props.release_ms * 0.5))
    ratio = (np.linspace(0.0, 1.0, n_comp, dtype=np.float32)
             if n_comp > 1 else np.zeros(1, dtype=np.float32))
    speeds = props.ripple_speed * (2.0 ** (ratio * props.ripple_spread * 0.5))

    for i in range(n_comp):
        col = spec[:, i]
        if props.per_band_level:
            peak = float(col.max())
            if peak > 1e-6:
                col = col / peak
        amp = np.clip(col * props.master_gain, 0.0, 2.0)
        amp = analysis.envelope(amp, fps, props.attack_ms, props.release_ms)
        ph = ripple_phase(data, props, frames, float(speeds[i]), onset, fps)
        amp, ph = loop_pair(amp, ph, props)
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_SIN % i), -1, frames,
                   (amp * np.cos(ph)).astype(np.float32))
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_COS % i), -1, frames,
                   (-amp * np.sin(ph)).astype(np.float32))

    return bake_object_motion(obj, data, props, frames)

def beat_locked_phase(n_frames, beats, fps, free_speed,cycles_per_beat = 1.0, lock = 1.0):
    t = np.arange(n_frames, dtype=np.float64)
    free = 2.0 * np.pi * float(free_speed) * t / float(fps)
    b = np.asarray(beats, dtype=np.float64)
    if b.size < 2 or lock <= 0.0:
        return free.astype(np.float32)

    targets = 2.0 * np.pi * float(cycles_per_beat) * np.arange(b.size)
    locked = np.interp(t, b, targets)

    slope = (targets[1] - targets[0]) / max(b[1] - b[0], 1e-6)
    head = t < b[0]
    locked[head] = targets[0] - (b[0] - t[head]) * slope
    slope_end = (targets[-1] - targets[-2]) / max(b[-1] - b[-2], 1e-6)
    tail = t > b[-1]
    locked[tail] = targets[-1] + (t[tail] - b[-1]) * slope_end
    w = float(np.clip(lock, 0.0, 1.0))
    return ((1.0 - w) * free + w * locked).astype(np.float32)

def make_loopable(values, fade_frames, wrap=False):
    """Ramp the tail so the last frame lands on the first -- a clean loop.

    A plain crossfade to the head is wrong here: it ends on values[fade-1],
    not values[0], which leaves the seam open. Correcting by a ramped offset
    guarantees out[-1] == out[0] and keeps the shape of the curve.
    """
    v = np.asarray(values, dtype=np.float64).copy()
    n = v.size
    f = int(min(fade_frames, n))
    if f < 2 or n < 2:
        return v.astype(np.float32)

    target = v[0]
    if wrap:
        # Monotonic phase: aim at the nearest equivalent angle, so the
        # correction stays small and the wave does not run backwards.
        target = v[0] + np.round((v[-1] - v[0]) / (2.0 * np.pi)) * 2.0 * np.pi

    delta = target - v[-1]
    w = np.linspace(0.0, 1.0, f)
    w = w * w * (3.0 - 2.0 * w)          # smoothstep, so the join has no kink
    v[-f:] = v[-f:] + w * delta
    return v.astype(np.float32)


def bake_emitters(obj, data, props, frames, beats):
    from .constants import SK_RIPPLE_COS, SK_RIPPLE_SIN

    key = obj.data.shape_keys
    fps = float(props.bake_fps)
    n = len(frames)
    slots = max(1, int(props.emitter_slots))
    t = np.arange(n, dtype=np.float64) / fps
    decay = max(float(props.emitter_decay), 0.05)

    for s in range(slots):
        amp = np.zeros(n)
        ph = np.zeros(n)
        for j, b in enumerate(np.asarray(beats, dtype=int) - int(frames[0])):
            if j % slots != s or b < 0 or b >= n:
                continue
            age = t[b:] - t[b]
            amp[b:] = np.exp(-age / decay)
            ph[b:] = 2.0 * np.pi * props.ripple_speed * age
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_SIN % s), -1, frames,
                   (amp * np.cos(ph)).astype(np.float32))
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_COS % s), -1, frames,
                   (-amp * np.sin(ph)).astype(np.float32))

    return bake_object_motion(obj, data, props, frames)