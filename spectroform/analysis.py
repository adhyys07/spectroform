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