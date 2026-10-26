import math

import numpy as np

from .geometry import _quad_grid_faces

def _reduce_rows(mat, target):
    n = mat.shape[0]
    if target >= n:
        return mat.astype(np.float32)
    bounds = np.linspace(0, n, target + 1).astype(int)
    rows = [mat[a:max(b, a + 1)].max(axis=0) for a, b in zip(bounds[:-1], bounds[1:])]
    return np.asarray(rows, dtype=np.float32)

def _peak_envelope(samples, steps):
    n = samples.size
    bounds = np.linspace(0, n, steps + 1).astype(int)
    amp = np.array(
        [np.abs(samples[a:max(b, a + 1)]).max() for a, b in zip(bounds[:-1], bounds[1:])],
        dtype=np.float32,
    )
    return amp / max(float(amp.max()), 1e-6)

def build_spectrogram(spec, size, height, time_steps):
    rows = _reduce_rows(spec, int(time_steps))
    nt, nb = rows.shape

    x = np.linspace(-size * 0.5, size * 0.5, nt, dtype=np.float32)
    y = np.linspace(-size * 0.5, size * 0.5, nb, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    zz = rows * float(height) * size

    verts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1).astype(np.float32)
    count = verts.shape[0]

    return {
        "verts": verts,
        "normals": np.tile(np.array([0.0, 0.0, 1.0], np.float32), (count, 1)),
        "faces": _quad_grid_faces(nt, nb),
        "falloff": np.ones(count, dtype=np.float32),
        "wave": np.repeat(np.linspace(0.0, 1.0, nt, dtype=np.float32), nb),
        "time_steps": nt,
    }


def build_waveform_tube(samples, size, tube, segments, steps):
    amp = _peak_envelope(samples, int(steps))
    steps = amp.size
    seg = max(3, int(segments))

    x = np.linspace(-size * 0.5, size * 0.5, steps, dtype=np.float32)
    ang = (2.0 * np.pi * np.arange(seg) / seg).astype(np.float32)
    ca, sa = np.cos(ang), np.sin(ang)

    r = (float(tube) * size * (0.12 + 0.88 * amp))[:, None]
    vy = r * ca[None, :]
    vz = r * sa[None, :]
    vx = np.repeat(x[:, None], seg, axis=1)

    verts = np.stack([vx.ravel(), vy.ravel(), vz.ravel()], axis=1).astype(np.float32)
    normals = np.stack([
        np.zeros(steps * seg, dtype=np.float32),
        np.tile(ca, steps),
        np.tile(sa, steps),
    ], axis=1).astype(np.float32)

    return {
        "verts": verts,
        "normals": normals,
        "faces": _quad_grid_faces(steps, seg, wrap_cols=True),
        "falloff": np.ones(steps * seg, dtype=np.float32),
        "wave": np.repeat(np.linspace(0.0, 1.0, steps, dtype=np.float32), seg),
        "time_steps": steps,
    }


def build_waveform_ring(samples, size, tube, segments, steps):
    amp = _peak_envelope(samples, int(steps))
    steps = amp.size
    seg = max(3, int(segments))
    theta = (2.0 * np.pi * np.arange(steps) / steps).astype(np.float32)
    ct, st = np.cos(theta)[:, None], np.sin(theta)[:, None]

    phi = (2.0 * np.pi * np.arange(seg) / seg).astype(np.float32)
    cp, sp = np.cos(phi)[None, :], np.sin(phi)[None, :]

    major = size * 0.5
    r = (float(tube) * size * (0.12 + 0.88 * amp))[:, None]

    nx = cp * ct
    ny = cp * st
    nz = np.repeat(sp, steps, axis=0)

    vx = major * ct + r * nx
    vy = major * st + r * ny
    vz = r * nz

    verts = np.stack([vx.ravel(), vy.ravel(), vz.ravel()], axis=1).astype(np.float32)
    normals = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1).astype(np.float32)

    return {
        "verts": verts,
        "normals": normals,
        "faces": _quad_grid_faces(steps, seg, wrap_cols=True, wrap_rows=True),
        "falloff": np.ones(steps * seg, dtype=np.float32),
        "wave": np.repeat(np.linspace(0.0, 1.0, steps, dtype=np.float32), seg),
        "time_steps": steps,
    }

def build_spectrum_bars(bins, size, radial=True, bar_frac=0.65):
    bins = max(2, int(bins))
    base_h = 0.02 * size

    verts, normals, faces, bin_index, top_mask = [], [], [], [], []
    pad = (1.0 - float(bar_frac)) * 0.5

    for b in range(bins):
        if radial:
            a0 = 2.0 * math.pi * (b + pad) / bins
            a1 = 2.0 * math.pi * (b + 1.0 - pad) / bins
            r_in, r_out = size * 0.30, size * 0.50
            corners = [
                (r_in * math.cos(a0), r_in * math.sin(a0)),
                (r_out * math.cos(a0), r_out * math.sin(a0)),
                (r_out * math.cos(a1), r_out * math.sin(a1)),
                (r_in * math.cos(a1), r_in * math.sin(a1)),
            ]
        else:
            w = size / bins
            x0 = -size * 0.5 + b * w + w * pad
            x1 = x0 + w * float(bar_frac)
            d = size * 0.06
            corners = [(x0, -d), (x1, -d), (x1, d), (x0, d)]

        base = len(verts)
        for cx, cy in corners:
            verts.append((cx, cy, 0.0))
            normals.append((0.0, 0.0, 1.0))
            bin_index.append(b)
            top_mask.append(False)
        for cx, cy in corners:
            verts.append((cx, cy, base_h))
            normals.append((0.0, 0.0, 1.0))
            bin_index.append(b)
            top_mask.append(True)

        faces.append((base + 0, base + 3, base + 2, base + 1))          # bottom
        faces.append((base + 4, base + 5, base + 6, base + 7))          # top
        for i in range(4):
            j = (i + 1) % 4
            faces.append((base + i, base + j, base + 4 + j, base + 4 + i))

    count = len(verts)
    return {
        "verts": np.array(verts, dtype=np.float32),
        "normals": np.array(normals, dtype=np.float32),
        "faces": faces,
        "falloff": np.ones(count, dtype=np.float32),
        "wave": (np.array(bin_index, dtype=np.float32) / max(bins - 1, 1)),
        "bin_index": np.array(bin_index, dtype=np.int32),
        "top_mask": np.array(top_mask, dtype=bool),
        "bins": bins,
    }

def make_bin_offsets(geo, height, size):
    verts = geo["verts"]
    bin_index = geo["bin_index"]
    top = geo["top_mask"]
    peak = float(height) * size

    out = []

    for b in range(geo["bins"]):
        co = verts.copy()
        sel = top & (bin_index == b)
        co[sel, 2] = peak
        out.append(co)
    return out
