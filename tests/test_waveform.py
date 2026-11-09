import numpy as np
import pytest
from spectroform import waveform as W


@pytest.mark.parametrize("pattern", W.RIPPLE_PATTERNS)
def test_phase_fields_are_finite_and_distinct(pattern):
    geo = W.build_ripple(32, 4.0, pattern)
    assert np.isfinite(geo["phase"]).all()
    mag = np.linalg.norm(geo["direction"], axis=1)
    live = mag > 1e-6
    assert np.abs(mag[live] - 1.0).max() < 1e-5


def test_patterns_are_not_duplicates():
    sigs = {p: tuple(np.round([W.build_ripple(32, 4.0, p)["phase"].std()], 5))
            for p in W.RIPPLE_PATTERNS}
    assert len(set(sigs.values())) == len(sigs), sigs


def test_faces_index_valid():
    for p in W.RIPPLE_PATTERNS:
        geo = W.build_ripple(32, 4.0, p)
        f = np.asarray(geo["faces"])
        assert f.min() >= 0 and f.max() < geo["verts"].shape[0]


class FakeProps:                      # no bpy needed
    size = 4.0; ripple_profile = "GERSTNER"; ripple_steepness = 0.4
    ripple_cycles = 3.0; ripple_spread = 3.0
    terrain_strength = 0.25; wave_strength = 0.15; spike_strength = 0.3


def test_gerstner_matches_ground_truth():
    geo = W.build_ripple(32, 4.0, "RADIAL")
    props, k, amp, q = FakeProps(), 2 * np.pi * 3.0, 0.3, FakeProps.ripple_steepness
    sin_off, cos_off = W.wave_pair(geo, props, k, amp)
    A, B = sin_off - geo["verts"], cos_off - geo["verts"]
    qc = min(q, 2.0 / (amp * k))
    for phi in (0.0, 0.7, 1.9, 3.5):
        got = np.cos(phi) * A + (-np.sin(phi)) * B
        psi = k * geo["phase"] - phi
        want = (geo["normals"] * np.sin(psi)[:, None]
                - geo["direction"] * (qc * np.cos(psi)[:, None])) * amp * geo["falloff"][:, None]
        assert np.abs(got - want).max() < 1e-5


def test_gerstner_q_zero_is_sine():
    geo = W.build_ripple(32, 4.0, "RADIAL")
    p = FakeProps(); p.ripple_steepness = 0.0
    g = W.wave_pair(geo, p, 2 * np.pi * 3.0, 0.3)
    p.ripple_profile = "SINE"
    s = W.wave_pair(geo, p, 2 * np.pi * 3.0, 0.3)
    assert np.abs(g[0] - s[0]).max() < 1e-6
