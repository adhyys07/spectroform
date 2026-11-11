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
    beats = np.asarray(data.get("beats", ()), dtype=np.float64)
    if float(props.beat_lock) > 0.0 and beats.size >= 2:
        return beat_locked_phase(len(frames), beats - float(frames[0]), fps,
                                 base_speed,
                                 cycles_per_beat=props.cycles_per_beat,
                                 lock=props.beat_lock)
    return audio_phase(base_speed, onset, props.ripple_beat_gain, fps)


def loop_pair(amp, ph, props):
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

    if getattr(props, "use_dispersion", False):
        # omega = sqrt(g*k): long waves outrun short ones. The 2**ratio ramp
        # below does the opposite, which is why it reads as animated noise.
        from . import physics, waveform
        cycles = waveform.ripple_component_cycles(
            n_comp, props.ripple_cycles, props.ripple_spread)
        speeds = physics.dispersion_speeds(
            cycles, props.size, scale=props.ripple_speed)
    else:
        speeds = props.ripple_speed * (2.0 ** (ratio * props.ripple_spread * 0.5))

    for i in range(n_comp):
        col = spec[:, i]
        if props.per_band_level:
            peak = float(col.max())
            if peak > 1e-6:
                col = col / peak
        amp = np.clip(col * props.master_gain, 0.0, 2.0)
        if getattr(props, "use_resonance", False):
            # A resonator at the component's own frequency: the surface gains
            # mass, so it overshoots and rings instead of stopping dead.
            amp = resonant_amplitude(
                amp, props, fps, max(abs(float(speeds[i])), 0.05))
        else:
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

def bake_drum_emitter(obj,data,props,frames,spec,bin_hz):
    from .constants import SK_RIPPLE_COS, SK_RIPPLE_SIN

    key = obj.data.shape_keys
    fps = float(props.bake_fps)
    n = len(frames)
    bands = analysis.onset_bands(spec, bin_hz)
    slots = max(3, int(props.emitter_slots))
    decay = max(float(props.emitter_decay), 0.05)
    t = np.arange(n, dtype=np.float64) / fps

    for s in range(slots):
        band = bands[s % 3][:n]

        tau = decay * (1.0, 0.6, 0.3)[s % 3]
        amp = analysis.envelope(band.astype(np.float32), fps, 5.0, tau * 1000.0)
        speed = props.ripple_speed * (1.0, 1.7, 2.8)[s % 3]
        ph = (2.0 * np.pi * speed * t).astype(np.float32)
        amp, ph = loop_pair(amp, ph, props)
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_SIN % s), -1, frames,
                   (amp * np.cos(ph)).astype(np.float32))
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_COS % s), -1, frames,
                   (-amp * np.sin(ph)).astype(np.float32))

    return bake_object_motion(obj, data, props, frames)


def section_gain(n_frames, bounds, variation, seed=0):
    gain = np.ones(n_frames, dtype=np.float32)
    if variation <= 0.0 or len(bounds) == 0:
        return gain
    rng = np.random.RandomState(int(seed))
    edges = [0] + [int(b) for b in bounds] + [n_frames]
    for i in range(len(edges) - 1):
        level = 1.0 + float(variation) * rng.uniform(-1.0, 1.0)
        gain[edges[i]:edges[i + 1]] = level
    k = max(3, n_frames // 120) | 1
    pad = np.pad(gain, k // 2, mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid").astype(np.float32)

def harmony_hue(chroma_frames):
    dominant = np.argmax(chroma_frames, axis=1)
    confidence = chroma_frames.max(axis=1)
    hue = (dominant.astype(np.float32) / 12.0)
    # Desaturate where no pitch class really dominates.
    sat = np.clip((confidence - 0.12) * 4.0, 0.15, 0.95)
    return hue, sat


def bake_drum_emitters(obj, props, frames, spec, bin_hz):
    """Three emitter families: kick, snare and hats each drive their own slots."""
    from .constants import SK_RIPPLE_COS, SK_RIPPLE_SIN

    key = obj.data.shape_keys
    fps = float(props.bake_fps)
    n = len(frames)
    bands = analysis.onset_bands(spec, bin_hz)
    slots = max(3, int(props.emitter_slots))
    decay = max(float(props.emitter_decay), 0.05)
    t = np.arange(n, dtype=np.float64) / fps

    for s in range(slots):
        which = s % 3
        band = np.asarray(bands[which], dtype=np.float32)[:n]
        if band.size < n:
            band = np.pad(band, (0, n - band.size))
        # Hats flick, kicks swell: shorter release for the higher bands.
        tau = decay * (1.0, 0.6, 0.3)[which]
        amp = analysis.envelope(band, fps, 5.0, tau * 1000.0)
        speed = props.ripple_speed * (1.0, 1.7, 2.8)[which]
        ph = (2.0 * np.pi * speed * t).astype(np.float32)
        amp, ph = loop_pair(amp, ph, props)
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_SIN % s), -1, frames,
                   (amp * np.cos(ph)).astype(np.float32))
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_COS % s), -1, frames,
                   (-amp * np.sin(ph)).astype(np.float32))


def dispersion_phase(cycles, props, frames, fps, beats=None):
    from .physics import dispersion_speeds

    speeds = dispersion_speeds(cycles, props.size, scale=props.ripple_speed)
    t = (np.asarray(frames, dtype=np.float64) - float(frames[0])) / fps
    return [(2.0 * np.pi * float(s) * t).astype(np.float32) for s in speeds]


def resonant_amplitude(drive, props, fps, f0, q=None):
    from .physics import resonator

    return resonator(np.asarray(drive, dtype=np.float64), f0,
                     q if q is not None else props.resonance_q, fps)


def bake_faraday(obj, spec, data, props, frames):
    from .constants import SK_RIPPLE_COS, SK_RIPPLE_SIN

    fps = float(props.bake_fps)
    key = obj.data.shape_keys
    n_comp = spec.shape[1]
    t = (np.asarray(frames, dtype=np.float64) - float(frames[0])) / fps

    ratio = (np.linspace(0.0, 1.0, n_comp, dtype=np.float32)
             if n_comp > 1 else np.zeros(1, dtype=np.float32))

    drive_hz = props.ripple_speed * (2.0 ** (ratio * props.ripple_spread * 0.5))

    for i in range(n_comp):
        col = spec[:, i]
        peak = float(col.max())
        if props.per_band_level and peak > 1e-6:
            col = col / peak
        amp = resonant_amplitude(np.clip(col * props.master_gain, 0.0, 2.0),
                                 props, fps, max(drive_hz[i] * 0.5, 0.05))
        ph = (2.0 * np.pi * float(drive_hz[i]) * 0.5 * t).astype(np.float32)
        amp, ph = loop_pair(amp, ph, props)
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_SIN % i), -1, frames,
                   (amp * np.cos(ph)).astype(np.float32))
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_COS % i), -1, frames,
                   (-amp * np.sin(ph)).astype(np.float32))

    return bake_object_motion(obj, data, props, frames)


def bake_modal(obj, spec, data, props, frames, modes):
    """Standing plate modes, each ringing at its own true modal frequency.

    A drumhead's modal frequencies are proportional to the Bessel zeros, so
    mode (m, n) is driven at k_mn / k_01 times the fundamental -- the real
    ratios, not an even ramp.
    """
    from .constants import SK_RIPPLE_COS, SK_RIPPLE_SIN

    fps = float(props.bake_fps)
    key = obj.data.shape_keys
    n_comp = min(spec.shape[1], len(modes))
    t = (np.asarray(frames, dtype=np.float64) - float(frames[0])) / fps
    k0 = max(float(modes[0][2]), 1e-6)
    zeros = np.zeros(len(frames), dtype=np.float32)

    for i in range(n_comp):
        f_mode = abs(props.ripple_speed) * (float(modes[i][2]) / k0)
        f_mode = min(f_mode, fps * 0.45)          # never key above Nyquist
        col = spec[:, i]
        peak = float(col.max())
        if props.per_band_level and peak > 1e-6:
            col = col / peak
        drive = np.clip(col * props.master_gain, 0.0, 2.0)
        amp = resonant_amplitude(drive, props, fps, max(f_mode, 0.05))
        ph = (2.0 * np.pi * f_mode * t).astype(np.float32)
        amp, ph = loop_pair(amp, ph, props)
        # The mode shape lives in the COS key; the SIN key is a zero offset,
        # because a standing mode does not travel.
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_SIN % i), -1,
                   frames, zeros)
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_COS % i), -1,
                   frames, (-amp * np.sin(ph)).astype(np.float32))

    return bake_object_motion(obj, data, props, frames)


def bake_harmonic_rings(obj, spec, data, props, frames, count):
    """One ring per partial, each pulsing with its own band's energy."""
    from .constants import SK_RIPPLE_COS, SK_RIPPLE_SIN

    fps = float(props.bake_fps)
    key = obj.data.shape_keys
    n_comp = min(spec.shape[1], count)
    zeros = np.zeros(len(frames), dtype=np.float32)

    for i in range(n_comp):
        col = spec[:, i]
        peak = float(col.max())
        if props.per_band_level and peak > 1e-6:
            col = col / peak
        drive = np.clip(col * props.master_gain, 0.0, 2.0)
        if getattr(props, "use_resonance", False):
            amp = resonant_amplitude(drive, props, fps,
                                     max(abs(props.ripple_speed), 0.05))
        else:
            amp = analysis.envelope(drive, fps, props.attack_ms, props.release_ms)
        amp, _ph = loop_pair(amp, np.zeros_like(amp), props)
        # The ring band sits in the SIN key; COS is a zero offset here.
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_SIN % i), -1,
                   frames, amp.astype(np.float32))
        bake_curve(key, 'key_blocks["%s"].value' % (SK_RIPPLE_COS % i), -1,
                   frames, zeros)

    return bake_object_motion(obj, data, props, frames)


def collect_ripple_curves(obj, n_comp, n_frames):
    """Read the baked SIN/COS values back out, for the web exporter."""
    from .constants import SK_RIPPLE_COS, SK_RIPPLE_SIN

    keys = obj.data.shape_keys
    action = keys.animation_data.action if keys and keys.animation_data else None
    if action is None:
        return None, None

    sin_c = np.zeros((n_frames, n_comp), dtype=np.float32)
    cos_c = np.zeros((n_frames, n_comp), dtype=np.float32)
    for i in range(n_comp):
        for name, dest in ((SK_RIPPLE_SIN, sin_c), (SK_RIPPLE_COS, cos_c)):
            fc = find_fcurve(action, 'key_blocks["%s"].value' % (name % i), -1)
            if fc is None:
                continue
            pts = fc.keyframe_points
            for f in range(min(n_frames, len(pts))):
                dest[f, i] = pts[f].co[1]
    return sin_c, cos_c