import math
import numpy as np

def _quad_grid_faces(rows, cols, wrap_cols=False, wrap_rows=False):
    idx = np.arange(rows * cols).reshape(rows, cols)
    if wrap_rows:
        r_curr, r_next = idx, np.roll(idx, -1, axis=0)
    else:
        r_curr, r_next = idx[:-1], idx[1:]
 
    def pair(arr):
        if wrap_cols:
            return arr, np.roll(arr, -1, axis=1)
        return arr[:, :-1], arr[:, 1:]
 
    a, b = pair(r_curr)
    d, c = pair(r_next)
    return np.stack([a.ravel(), b.ravel(), c.ravel(), d.ravel()], axis=1).tolist()


def build_grid(res, size):
    n = int(res) + 1
    lin = np.linspace(-size * 0.5, size * 0.5, n, dtype=np.float32)
    xx, yy = np.meshgrid(lin, lin, indexing="xy")
    x = xx.ravel()
    y = yy.ravel()
    verts = np.stack([x, y, np.zeros_like(x)], axis=1)
    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (verts.shape[0], 1))
 
    r = np.sqrt(x * x + y * y) / max(size * 0.5, 1e-6)
    r = np.clip(r, 0.0, 1.0)
    falloff = 1.0 - r
    falloff = (falloff * falloff * (3.0 - 2.0 * falloff)).astype(np.float32)  # smoothstep

    return {
        "verts": verts.astype(np.float32),
        "normals": normals.astype(np.float32),
        "faces": _quad_grid_faces(n, n),
        "falloff": falloff,
        "wave": r.astype(np.float32),
    }

def build_sphere(res, radius):
    seg = max(8, int(res))
    rings = max(4, int(res) // 2)

    verts, normals, wave = [], [], []
    for i in range(1, rings):
        theta = math.pi * i / rings
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(seg):
            phi = 2.0 * math.pi * j / seg
            nx, ny, nz = st * math.cos(phi), st * math.sin(phi), ct
            verts.append((nx * radius, ny * radius, nz * radius))
            normals.append((nx, ny, nz))
            wave.append(ct * 0.5 + 0.5)

    body = len(verts)
    verts.append((0.0, 0.0, radius))
    normals.append((0.0, 0.0, 1.0))
    wave.append(1.0)
    verts.append((0.0, 0.0, -radius))
    normals.append((0.0, 0.0, -1.0))
    wave.append(0.0)
    north, south = body, body + 1

    faces = _quad_grid_faces(rings - 1, seg, wrap_cols=True)
    for j in range(seg):
        faces.append((north, (j + 1) % seg, j))
    base = (rings - 2) * seg
    for j in range(seg):
        faces.append((south, base + j, base + (j + 1) % seg))

    count = len(verts)
    return {
        "verts": np.array(verts, dtype=np.float32),
        "normals": np.array(normals, dtype=np.float32),
        "faces": faces,
        "falloff": np.ones(count, dtype=np.float32),
        "wave": np.array(wave, dtype=np.float32),
    }

def build_cylinder(res, radius, height):
    seg = max(8, int(res))
    rings = max(2, int(res) // 2)

    verts, normals, wave = [], [], []
    for i in range(rings + 1):
        t = i / float(rings)
        z = -height * 0.5 + height * t
        for j in range(seg):
            phi = 2.0 * math.pi * j / seg
            nx, ny = math.cos(phi), math.sin(phi)
            verts.append((nx * radius, ny * radius, z))
            normals.append((nx, ny, 0.0))
            wave.append(t)

    w = np.array(wave, dtype=np.float32)
    edge = np.clip(np.minimum(w, 1.0 - w) * 4.0, 0.0, 1.0)
    falloff = (edge * edge * (3.0 - 2.0 * edge)).astype(np.float32)

    return {
        "verts": np.array(verts, dtype=np.float32),
        "normals": np.array(normals, dtype=np.float32),
        "faces": _quad_grid_faces(rings + 1, seg, wrap_cols=True),
        "falloff": falloff,
        "wave": w,
    }

def build_torus(res, radius, tube):
    major = max(8, int(res))
    minor = max(6, int(res) // 3)
 
    verts, normals, wave = [], [], []
    for i in range(major):
        a = 2.0 * math.pi * i / major
        ca, sa = math.cos(a), math.sin(a)
        for j in range(minor):
            b = 2.0 * math.pi * j / minor
            cb, sb = math.cos(b), math.sin(b)
            nx, ny, nz = ca * cb, sa * cb, sb
            verts.append((ca * radius + nx * tube, sa * radius + ny * tube, nz * tube))
            normals.append((nx, ny, nz))
            wave.append(i / float(major))
 
    count = len(verts)
    return {
        "verts": np.array(verts, dtype=np.float32),
        "normals": np.array(normals, dtype=np.float32),
        "faces": _quad_grid_faces(major, minor, wrap_cols=True, wrap_rows=True),
        "falloff": np.ones(count, dtype=np.float32),
        "wave": np.array(wave, dtype=np.float32),
    }


def build_geometry(props):
    size = props.size
    kind = props.geometry_type
    if kind == "GRID":
        return build_grid(props.resolution, size)
    if kind == "SPHERE":
        return build_sphere(props.resolution, size * 0.5)
    if kind == "CYLINDER":
        return build_cylinder(props.resolution, size * 0.35, size)
    return build_torus(props.resolution, size * 0.4, size * 0.15)

def sine_field(points, octaves, scale, seed):
    rng = np.random.RandomState(int(seed))
    out = np.zeros(points.shape[0], dtype=np.float32)
    amp = 1.0
    freq = float(scale)
    total = 0.0
    for _ in range(max(1, int(octaves))):
        for _ in range(3):
            d = rng.normal(size=3)
            norm = np.linalg.norm(d)
            d = d / (norm if norm > 1e-6 else 1.0)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            out += amp * np.sin(points @ d.astype(np.float32) * freq + phase)
            total += amp
        amp *= 0.5
        freq *= 2.0
    return (out / max(total, 1e-6)).astype(np.float32)

def make_shape_offsets(geo, props):
    verts = geo["verts"]
    normals = geo["normals"]
    falloff = geo["falloff"][:, None]
    wave = geo["wave"]
    unit = max(props.size, 1e-4)
 
    terrain = sine_field(verts, props.noise_octaves, props.noise_scale / unit, props.seed)
    bass_co = verts + normals * (terrain[:, None] * props.terrain_strength * unit) * falloff
 
    ripple = np.sin(wave * (2.0 * math.pi * max(1, props.wave_count))).astype(np.float32)
    mid_co = verts + normals * (ripple[:, None] * props.wave_strength * unit) * falloff
 
    spike_raw = sine_field(verts, 2, props.spike_density / unit, props.seed + 977)
    spike_mask = np.clip(spike_raw * 0.5 + 0.5, 0.0, 1.0) ** max(1.0, props.spike_sharpness)
    treble_co = verts + normals * (spike_mask[:, None] * props.spike_strength * unit)
 
    return bass_co, mid_co, treble_co