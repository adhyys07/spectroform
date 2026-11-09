import math

import numpy as np

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
