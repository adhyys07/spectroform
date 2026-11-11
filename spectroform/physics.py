import math
import numpy as np

GRAVITY = 9.81

def dispersion_speeds(cycles, size, gravity=GRAVITY, scale=1.0):
    k = 2.0 * np.pi * np.asarray(cycles, dtype=np.float64) / max(size, 1e-6)
    omega = np.sqrt(gravity * np.maximum(k, 1e-9))
    return (float(scale) * omega / (2.0 * np.pi)).astype(np.float32)

def resonator(drive, f0, q, fps):
    f0 = float(np.clip(f0, 1e-3, fps * 0.45))
    q = max(float(q), 0.5)
    w = 2.0 * np.pi * f0 / float(fps)
    r = math.exp(-w / (2.0 * q))          
    a1 = -2.0 * r * math.cos(w)
    a2 = r * r
    gain = 1.0 + a1 + a2  

    x = np.asarray(drive, dtype=np.float64)
    y = np.empty_like(x)
    y1 = y2 = 0.0
    for i in range(x.size):
        cur = gain * x[i] - a1 * y1 - a2 * y2
        y[i] = cur
        y2, y1 = y1, cur
    return y.astype(np.float32)

def bessel_j(n, x, steps=256):
    x = np.asarray(x, dtype=np.float64)
    tau = np.linspace(0.0, np.pi, steps)
    integrand = np.cos(n * tau[None, :] - x.ravel()[:, None] * np.sin(tau)[None, :])
    return (np.trapezoid(integrand, tau, axis=1) / np.pi).reshape(x.shape)

def bessel_zeros(n, count, search_to=80.0, steps=20000):
    xs = np.linspace(1e-6, search_to, steps)
    vals = bessel_j(n, xs)
    roots = []
    for i in range(len(xs) - 1):
        if vals[i] * vals[i + 1] < 0.0:
            lo, hi = xs[i], xs[i+1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if bessel_j(n, np.array([lo]))[0] * bessel_j(n, np.array([mid]))[0] <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(0.5 * (lo + hi))
        if len(roots) >= count:
            break
    return np.array(roots[:count], dtype=np.float64)

def mode_table(max_m=4, per_m=4):
    modes = []
    for m in range(max_m + 1):
        for n, k in enumerate(bessel_zeros(m, per_m)):
            modes.append((m, n, float(k)))
    modes.sort(key=lambda t: t[2])
    return modes

def thermal_erode(height, iterations=40, talus=0.012, rate=0.5):
    h = np.asarray(height, dtype=np.float64).copy()
    for _ in range(int(iterations)):
        delta = np.zeros_like(h)
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            diff = h - np.roll(h, shift, axis=axis)
            move = np.where(diff > talus, (diff - talus) * (rate * 0.25), 0.0)
            delta -= move
            delta += np.roll(move, -shift, axis=axis)
        h += delta
    return h.astype(np.float32)

def harmonic_radii(tonic_hz, count, size, octaves=3.0):
    partials = np.arange(1, count + 1, dtype=np.float64)
    cents = np.log2(partials)
    return (size * 0.5 * np.clip(cents / octaves, 0.0, 1.0)).astype(np.float32)