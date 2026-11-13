import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

A4_HZ = 440.0
A4_MIDI = 69
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                          2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                          2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

STEM_NAMES = ("KICK", "BASS", "VOCAL", "OTHER")


def db_normalise(values, dynamic_range, top=None):
    db = 20.0 * np.log10(np.maximum(values, 1e-9))
    if top is None:
        top = float(db.max())
    if not np.isfinite(top):
        return np.zeros_like(values)
    out = (db - (top - dynamic_range)) / max(dynamic_range, 1e-3)
    return np.clip(out, 0.0 , 1.0).astype(np.float32)

def envelope(signal, fps, attack_ms, release_ms):
    tau_a = max(attack_ms, 1.0) / 1000.0
    tau_r = max(release_ms, 1.0) / 1000.0
    ca = math.exp(-1.0 / max(tau_a * fps, 1e-4))
    cr = math.exp(-1.0 / max(tau_r * fps, 1e-4))

    out = np.empty_like(signal)
    y = 0.0
    for i in range(signal.size):
        v = float(signal[i])
        c = ca if v > y else cr
        y = c * y + (1.0 - c) * v
        out[i] = y
    return out

def analyse_audio(samples, rate, fps, bass_max, mid_max, dynamic_range, per_band=True):
    hop = max(1, int(round(rate / float(fps))))
    win = 1024
    while win < hop * 2:
        win *= 2

    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / rate)
    nyquist = rate * 0.5
 
    edges = [
        (20.0, min(bass_max, nyquist)),
        (min(bass_max, nyquist), min(mid_max, nyquist)),
        (min(mid_max, nyquist), min(16000.0, nyquist)),
    ]
    masks = []
    for lo, hi in edges:
        m = (freqs >= lo) & (freqs < max(hi, lo + 1.0))
        if not m.any():  # degenerate crossover → fall back to a single bin
            m = np.zeros(freqs.shape, dtype=bool)
            m[min(len(freqs) - 1, 1)] = True
        masks.append(m)
 
    n_frames = max(2, int(math.ceil(samples.size / float(hop))))
    padded = np.zeros(n_frames * hop + win, dtype=np.float32)
    padded[: samples.size] = samples
 
    bands = np.zeros((3, n_frames), dtype=np.float32)
    rms = np.zeros(n_frames, dtype=np.float32)
    flux = np.zeros(n_frames, dtype=np.float32)


    prev_mag = None
    chunk = 256
    for start in range(0, n_frames, chunk):
        end = min(start + chunk, n_frames)
        offsets = np.arange(start, end) * hop
        block = np.stack([padded[o: o + win] for o in offsets]).astype(np.float32)
        rms[start:end] = np.sqrt((block ** 2).mean(axis=1))

        block *= window
        mag = np.abs(np.fft.rfft(block, axis=1)).astype(np.float32)

        for b, mask in enumerate(masks):
            bands[b, start:end] = np.sqrt((mag[:, mask] ** 2).mean(axis=1))

        if prev_mag is None:
            flux[start] = 0.0
            if end - start > 1:
                diff = np.diff(mag, axis=0)
                flux[start + 1: end] = np.maximum(diff, 0.0).sum(axis=1)
        else:
            diff = np.diff(np.vstack([prev_mag[None, :], mag]), axis=0)
            flux[start:end] = np.maximum(diff, 0.0).sum(axis=1)
        prev_mag = mag[-1]

    top = None
    if not per_band:
        top = max(20.0 * math.log10(max(float(bands[b].max()), 1e-9)) for b in range(3))

    result = {
        "bass": db_normalise(bands[0], dynamic_range, top),
        "mid": db_normalise(bands[1], dynamic_range, top),
        "treble": db_normalise(bands[2], dynamic_range, top),
        "rms": db_normalise(rms, dynamic_range),
        "n_frames": n_frames,
        "duration": samples.size / float(rate),
        "window": win,
        "hop": hop,
    }

    peak = float(flux.max())
    onset = (flux / peak) if peak > 0.0 else flux
    result["onset"] = np.clip(onset, 0.0, 1.0).astype(np.float32) ** 2  # sharpen
    return result

def log_band_masks(freqs, fmin, fmax, bins):
    edges = np.geomspace(max(fmin, 1.0), max(fmax, fmin * 2.0), bins + 1)
    masks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():  # band narrower than one FFT bin -> snap to nearest
            idx = int(np.argmin(np.abs(freqs - 0.5 * (lo + hi))))
            m = np.zeros(freqs.shape, dtype=bool)
            m[idx] = True
        masks.append(m)
    return masks

def spectrogram(samples, rate, fps, bins=64, fmin=30.0, fmax=16000.0, dynamic_range=60.0):
    hop = max(1, int(round(rate / float(fps))))
    win = 1024
    while win < hop * 2:
        win *= 2

    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / rate)
    masks = log_band_masks(freqs, fmin, min(fmax, rate * 0.5), int(bins))

    n_frames = max(2, int(math.ceil(samples.size / float(hop))))
    padded = np.zeros(n_frames * hop + win, dtype=np.float32)
    padded[: samples.size] = samples

    spec = np.zeros((n_frames, int(bins)), dtype=np.float32)
    chunk = 256
    for start in range(0, n_frames, chunk):
        end = min(start + chunk, n_frames)
        offsets = np.arange(start, end) * hop
        block = np.stack([padded[o: o + win] for o in offsets]).astype(np.float32)
        block *= window
        mag = np.abs(np.fft.rfft(block, axis=1)).astype(np.float32)
        for b, m in enumerate(masks):
            spec[start:end, b] = np.sqrt((mag[:, m] ** 2).mean(axis=1))

    db = 20.0 * np.log10(np.maximum(spec, 1e-9))
    top = float(db.max())
    if not np.isfinite(top):
        return np.zeros_like(spec)
    out = (db - (top - dynamic_range)) / max(dynamic_range, 1e-3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def detect_tempo(onset, fps, bpm_min=60.0, bpm_max=200.0,
                 bpm_pref=120.0, octave_width=0.9):
    x = np.asarray(onset, dtype=np.float64)
    x -= x.mean()
    if x.size < 8 or not np.any(x):
        return 0.0,0
    ac = np.correlate(x, x, mode="full")[x.size - 1:]
    lo = max(1, int(round(60.0 * fps / bpm_max)))
    hi = min(x.size - 1, int(round(60.0 * fps / bpm_min)))
    if hi <= lo:
        return 0.0, 0

    lags = np.arange(lo, hi + 1, dtype=np.float64)
    cand = ac[lo:hi + 1]
    bpms = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log2(bpms / bpm_pref) / octave_width) ** 2)
    lag = int(lags[int(np.argmax(cand * prior))])
    return 60.0 * fps / lag, lag


def beat_frames(onset, fps, bpm, start_frame=0):
    if bpm <= 0:
        return np.zeros(0, dtype=np.int32)
    period = 60.0 * fps / bpm
    n = onset.size
    best_off, best_score = 0.0, -1.0
    for off in np.arange(0.0, period, 0.25):
        idx = np.round(np.arange(off, n, period)).astype(int)
        idx = idx[idx < n]
        if idx.size == 0:
            continue
        score = float(onset[idx].sum())
        if score > best_score:
            best_score, best_off = score, off
    idx = np.round(np.arange(best_off, n, period)).astype(int)
    return (idx[idx < n] + int(start_frame)).astype(np.int32)

def bin_frequencies(bins, fmin = 30.0, fmax = 16000.0):
    edges = np.geomspace(max(fmin, 1.0), fmax, bins + 1)
    return np.sqrt(edges[:-1] * edges[1:]).astype(np.float32)

def spectral_centroid(spec, bin_hz):
    lg = np.log2(np.maximum(np.asarray(bin_hz, dtype=np.float64), 1.0))
    w = spec.sum(axis=1)
    c = (spec * lg[None, :]).sum(axis=1) / np.maximum(w, 1e-9)
    lo, hi = float(lg[0]), float(lg[-1])
    return np.clip((c - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)

def _median_filter(x, size, axis):
    size = max(1, int(size) | 1)
    pad = size // 2
    widths = [(pad, pad) if a == axis else (0, 0) for a in range(x.ndim)]
    win = sliding_window_view(np.pad(x, widths, mode="edge"), size, axis=axis)
    return np.median(win, axis=-1)


def hpss(spec, kernel_time = 17, kernel_freq=17, power=2.0):
    harm = _median_filter(spec, kernel_time, axis=0)
    perc = _median_filter(spec, kernel_freq, axis=1)
    hp, pp = harm ** power, perc ** power
    total = hp + pp + 1e-9
    return (spec * (hp / total)).astype(np.float32), \
           (spec * (pp / total)).astype(np.float32)

def onset_bands(spec, bin_hz, edges=(250.0, 4000.0)):
    hz = np.asarray(bin_hz)
    masks = (hz < edges[0],
             (hz >= edges[0]) & (hz < edges[1]),
             hz >= edges[1])
    diff = np.maximum(np.diff(spec, axis=0, prepend=spec[:1]), 0.0)
    out = []
    for m in masks:
        f = diff[:, m].sum(axis=1) if m.any() else np.zeros(spec.shape[0])
        peak = float(f.max())
        out.append((f / peak if peak > 0 else f).astype(np.float32))
    return tuple(out)

def classify_onsets(spec, bin_hz, onset, threshold=0.25):
    """Label onset peaks by which band carries most of their flux.

    Returns (frames, labels, shares). Prefer the shares: real drums overlap,
    so a hard label throws away the fact that a frame is both kick and hat.
    """
    low, mid, high = onset_bands(spec, bin_hz)
    n = min(len(onset), spec.shape[0])
    peaks = [i for i in range(1, n - 1)
             if onset[i] >= threshold
             and onset[i] >= onset[i - 1] and onset[i] > onset[i + 1]]

    labels, shares = [], []
    for i in peaks:
        v = np.array([low[i], mid[i], high[i]], dtype=np.float64)
        total = v.sum()
        sh = v / total if total > 1e-9 else np.zeros(3)
        shares.append(sh)
        labels.append(("KICK", "SNARE", "HAT")[int(np.argmax(sh))])
    return (np.array(peaks, dtype=np.int32), labels,
            np.array(shares, dtype=np.float32).reshape(-1, 3))


def chroma(samples, rate, fps, fmin=110.0, fmax = 2000.0, win = 8192):
    hop = max(1, int(round(rate / float(fps))))
    win = int(win)
    while win < hop * 2:
        win *= 2
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, 1.0 / rate)

    resolvable = (rate /win) / 0.0578
    keep = (freqs >= max(fmin, resolvable)) & (freqs <= fmax)
    if not keep.any():
        return np.zeros((2, 12), dtype=np.float32)
    midi = 12.0 * np.log2(np.maximum(freqs[keep], 1e-6) / A4_HZ) + A4_MIDI
    pc = np.rint(midi).astype(int) % 12

    n_frames = max(2, int(math.ceil(samples.size / float(hop))))
    padded = np.zeros(n_frames * hop + win, dtype=np.float32)
    padded[: samples.size] = samples

    out = np.zeros((n_frames, 12), dtype=np.float32)
    for start in range(0, n_frames, 256):
        end = min(start + 256, n_frames)
        block = np.stack([padded[o: o + win]
                          for o in np.arange(start, end) * hop]).astype(np.float32)
        block *= window
        mag = np.abs(np.fft.rfft(block, axis=1)).astype(np.float32)[:, keep]
        for c in range(12):
            sel = pc == c
            if sel.any():
                out[start:end, c] = mag[:, sel].sum(axis=1)
    return (out / np.maximum(out.sum(axis=1, keepdims=True), 1e-9)).astype(np.float32)

def detect_key(chroma_frames):
    profile = chroma_frames.mean(axis=0)
    profile -= profile.mean()
    best = (-2.0, 0, True)
    for tonic in range(12):
        for major in (True,False):
            ref = np.roll(MAJOR_PROFILE if major else MINOR_PROFILE, tonic)
            ref = ref - ref.mean()
            denom = np.linalg.norm(profile) * np.linalg.norm(ref)
            score = float(profile @ ref / denom) if denom > 1e-9 else 0.0
            if score > best[0]:
                best = (score, tonic, major)

    score, tonic, major = best
    return tonic, major, "%s%s" % (NOTE_NAMES[tonic], "" if major else "m"), score

def detect_sections(spec, fps, min_seconds=4.0, count=None):
    n = spec.shape[0]
    if n < 8:
        return np.zeros(0, dtype=np.int32)
    f = spec / np.maximum(np.linalg.norm(spec, axis=1, keepdims=True), 1e-9)
    sim = f @ f.T

    k = max(2, int(round(fps * min_seconds * 0.5)))
    novelty = np.zeros(n, dtype=np.float64)
    for i in range(k, n - k):
        a = sim[i - k:i, i - k:i].mean()
        b = sim[i:i + k, i:i + k].mean()
        cross = sim[i - k:i, i:i + k].mean()
        novelty[i] = (a + b) * 0.5 - cross
    if novelty.max() > 0:
        novelty /= novelty.max()
    
    gap = int(fps* min_seconds)
    picks = []
    for i in np.argsort(novelty)[::-1]:
        if novelty[i] <= 0.2:
            break
        if all(abs(int(i) - p) >= gap for p in picks):
            picks.append(int(i))
        if count is not None and len(picks) >= count:
            break
    return np.array(sorted(picks), dtype=np.int32)


def _local_contrast(onset, period):
    x = np.asarray(onset, dtype=np.float64)
    win = min(max(3, int(period * 2.0) | 1), (x.size - 1) | 1)
    if win < 3:
        return x - x.mean()
    return x - np.convolve(x, np.ones(win) / float(win), mode="same")

def beat_track(onset, fps, bpm, tightness=300.0, start_frame=0):
    x = np.asarray(onset, dtype=np.float64)
    n = x.size
    if n < 4 or bpm <= 0.0:
        return np.zeros(0, dtype=np.int32)
    period = 60.0 * fps / float(bpm)
    if period < 2.0 or period >= n:
        return np.zeros(0, dtype=np.int32)

    local = _local_contrast(x, period)

    back = np.arange(-int(round(2.0 * period)), -int(round(0.5 * period)) + 1)
    back = back[back < 0]
    if back.size == 0:
        return np.zeros(0, dtype=np.int32)
    penalty = -tightness * (np.log(-back / period) ** 2)

    score = np.full(n, -np.inf)
    prev = np.full(n, -1, dtype=np.int32)
    first = int(round(0.5 * period))
    score[:first] = local[:first]

    for i in range(first, n):
        cand = i + back
        ok = cand >= 0
        if not ok.any():
            score[i] = local[i]
            continue
        vals = score[cand[ok]] + penalty[ok]
        j = int(np.argmax(vals))
        if np.isfinite(vals[j]):
            score[i] = local[i] + vals[j]
            prev[i] = cand[ok][j]
        else:
            score[i] = local[i]

    tail = max(0, n - int(round(period)))
    i = int(tail + np.argmax(score[tail:]))
    beats = []
    while i >= 0:
        beats.append(i)
        i = int(prev[i])
    beats.reverse()
    return np.array(beats, dtype=np.int32) + int(start_frame)


def local_tempo(beats, fps, smooth=5):
    b = np.asarray(beats, dtype=np.float64)
    if b.size < 2:
        return np.zeros(0, dtype=np.float32)
    bpms = 60.0 * float(fps) / np.maximum(np.diff(b), 1e-6)
    k = max(1, int(smooth) | 1)
    if bpms.size >= k:
        pad = k // 2
        padded = np.pad(bpms, (pad, pad), mode="edge")
        bpms = np.array([np.median(padded[i:i + k]) for i in range(bpms.size)])
    return np.append(bpms, bpms[-1]).astype(np.float32)


def detect_downbeat(spec, bin_hz, beats, beats_per_bar=4, start_frame=0):
    spec = np.asarray(spec, dtype=np.float64)
    hz = np.asarray(bin_hz, dtype=np.float64)
    idx = np.asarray(beats, dtype=np.int64) - int(start_frame)
    idx = idx[(idx >= 0) & (idx < spec.shape[0])]
    npb = max(2, int(beats_per_bar))
    if idx.size < npb * 2:
        return 0


    low = hz < 150.0
    if not low.any():
        low = np.zeros(hz.shape, dtype=bool)
        low[0] = True
    bass = spec[:, low].sum(axis=1)
    bass = bass / max(float(bass.max()), 1e-9)

    change = np.abs(np.diff(spec, axis=0, prepend=spec[:1])).sum(axis=1)
    change = change / max(float(change.max()), 1e-9)

    strength = bass[idx] + change[idx]
    best_phase, best_score = 0, -np.inf
    for phase in range(npb):
        picked = strength[phase::npb]
        if picked.size == 0:
            continue
        others = np.delete(strength, np.arange(phase, strength.size, npb))
        score = float(picked.mean() - (others.mean() if others.size else 0.0))
        if score > best_score:
            best_score, best_phase = score, phase
    return best_phase

def bar_frames(beats, beats_per_bar=4, phase=0):
    b = np.asarray(beats, dtype=np.int32)
    npb = max(1, int(beats_per_bar))
    return b if b.size == 0 else b[int(phase) % npb::npb]


def phrase_frames(bars, bars_per_phrase=4):
    b = np.asarray(bars, dtype=np.int32)
    return b if b.size == 0 else b[::max(1, int(bars_per_phrase))]

def pitch_contout(samples, rate, fps, fmin=70.0, fmax=1200.0, win=2048, threshold=0.3):
    hop = max(1, int(round(rate / float(fps))))
    win = int(win)
    while win < hop * 2:
        win *= 2
    lag_min = max(2, int(math.floor(rate / float(fmax))))
    lag_max = min(win - 1, int(math.ceil(rate / float(fmin))))
    if lag_max <= lag_min:
        return np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)

    n_frames = max(2, int(math.ceil(samples.size / float(hop))))
    padded = np.zeros(n_frames * hop + win, dtype=np.float32)
    padded[: samples.size] = samples

    hz = np.zeros(n_frames, dtype=np.float32)
    conf = np.zeros(n_frames, dtype=np.float32)
    window = np.hanning(win).astype(np.float32)

    size = 1
    while size < win * 2:
        size *=2

    for start in range(0, n_frames, 128):
        end = min(start + 128, n_frames)
        block = np.stack([padded[o: o + win]
                          for o in np.arange(start, end) * hop])
        block = (block - block.mean(axis=1, keepdims=True)) * window
        energy = (block ** 2).sum(axis=1)

        spec = np.fft.rfft(block, n=size, axis=1)
        ac = np.fft.irfft(spec * np.conj(spec), n=size, axis=1)[:, :lag_max + 1]

        norm = ac[:, lag_min:lag_max + 1] / np.maximum(ac[:, 0], 1e-9)[:, None]
        peak = np.argmax(norm, axis=1)
        value = norm[np.arange(norm.shape[0]), peak]

        for r in range(norm.shape[0]):
            if energy[r] < 1e-6 or value[r] < threshold:
                continue
            lag = lag_min + int(peak[r])

            if lag_min < lag < lag_max:
                y0, y1, y2 = (float(ac[r, lag - 1]), float(ac[r, lag]),
                              float(ac[r, lag + 1]))
                denom = y0 - 2.0 * y1 + y2
                shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
                shift = float(np.clip(shift, -0.5, 0.5))
            else:
                shift = 0.0
            hz[start + r] = rate / (lag + shift)
            conf[start + r] = float(value[r])

    return hz, conf

def pitch_contour(samples, rate, fps, fmin=70.0, fmax=1200.0, win=2048,
                  threshold=0.3):
    """Per-frame fundamental and confidence.

    Normalised autocorrelation, which on a real mix tracks whatever is loudest
    and most periodic -- usually the lead vocal or lead synth. Monophonic by
    construction: a chord gives whichever partial dominates. hz is 0 where the
    frame is unvoiced, so callers must not interpolate across zeros blindly --
    use fill_unvoiced() for that.
    """
    hop = max(1, int(round(rate / float(fps))))
    win = int(win)
    while win < hop * 2:
        win *= 2

    n_frames = max(2, int(math.ceil(samples.size / float(hop))))

    lag_min = max(2, int(math.floor(rate / float(fmax))))
    lag_max = min(win - 1, int(math.ceil(rate / float(fmin))))
    if lag_max <= lag_min:
        # Callers index this by frame, so the degenerate case must still come
        # back frame-shaped.
        return (np.zeros(n_frames, dtype=np.float32),
                np.zeros(n_frames, dtype=np.float32))

    padded = np.zeros(n_frames * hop + win, dtype=np.float32)
    padded[: samples.size] = samples

    hz = np.zeros(n_frames, dtype=np.float32)
    conf = np.zeros(n_frames, dtype=np.float32)
    window = np.hanning(win).astype(np.float32)

    # FFT autocorrelation in blocks: np.correlate per frame is ~40x slower and
    # this runs over the whole track.
    size = 1
    while size < win * 2:
        size *= 2

    for start in range(0, n_frames, 128):
        end = min(start + 128, n_frames)
        block = np.stack([padded[o: o + win]
                          for o in np.arange(start, end) * hop])
        block = (block - block.mean(axis=1, keepdims=True)) * window
        energy = (block ** 2).sum(axis=1)

        spec = np.fft.rfft(block, n=size, axis=1)
        ac = np.fft.irfft(spec * np.conj(spec), n=size, axis=1)[:, :lag_max + 1]

        norm = ac[:, lag_min:lag_max + 1] / np.maximum(ac[:, 0], 1e-9)[:, None]
        peak = np.argmax(norm, axis=1)
        value = norm[np.arange(norm.shape[0]), peak]

        for r in range(norm.shape[0]):
            if energy[r] < 1e-6 or value[r] < threshold:
                continue
            lag = lag_min + int(peak[r])
            # Parabolic interpolation: without it the pitch quantises to
            # integer lags, which shows as stair-stepping in the high register.
            if lag_min < lag < lag_max:
                y0, y1, y2 = (float(ac[r, lag - 1]), float(ac[r, lag]),
                              float(ac[r, lag + 1]))
                denom = y0 - 2.0 * y1 + y2
                shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
                shift = float(np.clip(shift, -0.5, 0.5))
            else:
                shift = 0.0
            hz[start + r] = rate / (lag + shift)
            conf[start + r] = float(value[r])

    return hz, conf

def fill_unvoiced(hz, conf, max_gap_frames=8):
    hz = np.asarray(hz, dtype=np.float32).copy()
    voiced = hz > 0.0
    if not voiced.any():
        return hz
    idx = np.flatnonzero(voiced)
    hz[:idx[0]] = hz[idx[0]]
    hz[idx[-1] + 1:] = hz[idx[-1]]

    gap_start = None
    for i in range(idx[0], idx[-1] + 1):
        if hz[i] > 0.0:
            if gap_start is not None:
                length = i - gap_start
                if length <= max_gap_frames:
                    hz[gap_start:i] = np.linspace(
                        hz[gap_start - 1], hz[i], length + 2)[1:-1]
                gap_start = None
        elif gap_start is None:
            gap_start = i
    return hz


def pitch_to_unit(hz, fmin=70.0, fmax=1200.0):
    hz = np.asarray(hz, dtype=np.float64)
    lo, hi = math.log2(max(fmin, 1.0)), math.log2(max(fmax, fmin * 2.0))
    out = (np.log2(np.maximum(hz, 1e-6)) - lo) / max(hi - lo, 1e-9)
    return np.clip(np.where(hz > 0.0, out, 0.0), 0.0, 1.0).astype(np.float32)

def nmf(spec, components=6, iterations=120, seed=0):
    V = np.maximum(np.asarray(spec, dtype=np.float64).T, 1e-9)
    bins, frames = V.shape
    k = max(1, int(components))

    rng = np.random.RandomState(int(seed))
    W = rng.rand(bins, k) + 0.1
    H = rng.rand(k, frames) + 0.1
    ones = np.ones_like(V)

    for _ in range(int(iterations)):
        WH = W @ H + 1e-9
        H *= (W.T @ (V / WH)) / np.maximum(W.T @ ones, 1e-9)
        WH = W @ H + 1e-9
        W *= ((V / WH) @ H.T) / np.maximum(ones @ H.T, 1e-9)
        norm = np.maximum(W.sum(axis=0, keepdims=True), 1e-9)
        W /= norm
        H *= norm.T

    return W.T.astype(np.float32), H.T.astype(np.float32)

def _flatness(template):
    t = np.maximum(np.asarray(template, dtype=np.float64), 1e-9)
    return float(np.exp(np.log(t).mean()) / max(t.mean(), 1e-9))


def _percussiveness(activation):
    a = np.asarray(activation, dtype=np.float64)
    if a.size < 2 or a.max() <= 1e-9:
        return 0.0
    return float(np.maximum(np.diff(a, prepend=a[:1]), 0.0).sum()
                 / max(a.sum(), 1e-9))


def classify_components(W, H, bin_hz):
    hz = np.asarray(bin_hz, dtype=np.float64)
    labels = []
    for c in range(W.shape[0]):
        w = np.maximum(W[c].astype(np.float64), 1e-9)
        centroid = float((w * hz).sum() / max(w.sum(), 1e-9))
        flat, perc = _flatness(w), _percussiveness(H[:, c])

        if centroid < 160.0 and perc > 0.18:
            labels.append("KICK")
        elif centroid < 260.0:
            labels.append("BASS")
        elif 200.0 <= centroid <= 4000.0 and flat < 0.35 and perc < 0.25:
            labels.append("VOCAL")
        else:
            labels.append("OTHER")
    return labels


def nmf_stems(spec, bin_hz, components=6, iterations=120, seed=0, fps=None, attack_ms=10.0, release_ms=120.0):
    W, H = nmf(spec, components=components, iterations=iterations, seed=seed)
    labels = classify_components(W, H, bin_hz)

    out = {}
    for name in STEM_NAMES:
        cols = [i for i, lab in enumerate(labels) if lab == name]
        env = H[:, cols].sum(axis=1) if cols else np.zeros(H.shape[0], np.float32)
        peak = float(env.max())
        if peak > 1e-9:
            env = env / peak
        if fps is not None:
            env = envelope(env.astype(np.float32), float(fps),
                           attack_ms, release_ms)
        out[name] = np.asarray(env, dtype=np.float32)

    out["_labels"], out["_templates"] = labels, W
    return out

def stem_reconstruct(spec, W, H, columns):
    full = np.maximum(H @ W, 1e-9)
    part = H[:, columns] @ W[columns]
    return (np.asarray(spec, dtype=np.float32) * (part / full)).astype(np.float32)

def make_stem_ripple_offsets(geo, props, stem_names):
    unit = max(props.size, 1e-4)
    shape = {"KICK": (0.6, 1.0), "BASS": (1.0, 0.7),
             "VOCAL": (2.2, 0.5), "OTHER": (4.0, 0.35)}
    out = []
    for name in stem_names:
        mult, share = shape.get(name, (2.0, 0.5))
        k = 2.0 * np.pi * max(props.ripple_cycles * mult, 0.01)
        out.append(wave_pair(geo, props, k,
                             unit * float(props.terrain_strength) * share))
    return out
