"""Regression tests for bugs that shipped silently.

Every case here corresponds to something that was committed broken and gave
no error at bake time, only a wrong-looking result.
"""

import numpy as np

from spectroform import baking as B

FPS = 24.0


def _beats(n=16, period=12, offset=11):
    return np.arange(offset, offset + n * period, period, dtype=np.float64)


def test_beat_lock_puts_one_cycle_between_beats():
    beats = _beats()
    n = int(beats[-1]) + 20
    ph = B.beat_locked_phase(n, beats, FPS, free_speed=0.6,
                             cycles_per_beat=1.0, lock=1.0)
    at = np.interp(beats, np.arange(n), ph)
    cycles = np.diff(at) / (2.0 * np.pi)
    assert abs(cycles.mean() - 1.0) < 1e-3
    assert cycles.std() < 1e-3


def test_beat_lock_zero_is_free_running():
    beats = _beats()
    n = int(beats[-1]) + 20
    ph = B.beat_locked_phase(n, beats, FPS, free_speed=0.6, lock=0.0)
    want = 2.0 * np.pi * 0.6 * np.arange(n) / FPS
    assert np.abs(ph - want).max() < 1e-4


def test_beat_lock_phase_is_monotonic_including_the_head():
    # A misplaced paren in the head extrapolation made phase run backwards
    # before the first beat, which reads as the wave briefly reversing.
    beats = _beats(offset=30)
    n = int(beats[-1]) + 20
    for lock in (0.0, 0.5, 1.0):
        ph = B.beat_locked_phase(n, beats, FPS, 0.6, lock=lock)
        assert np.all(np.diff(ph) >= -1e-6), "phase reverses at lock=%s" % lock


def test_beat_lock_needs_at_least_two_beats():
    ph = B.beat_locked_phase(50, np.array([3.0]), FPS, 0.6, lock=1.0)
    assert np.all(np.diff(ph) >= -1e-9)
    assert np.isfinite(ph).all()


def test_make_loopable_closes_the_seam():
    # A plain crossfade to the head lands on values[fade-1], not values[0],
    # which left the seam open and actually made it worse.
    rng = np.random.default_rng(3)
    v = np.cumsum(rng.standard_normal(200)) * 0.1 + np.sin(np.linspace(0, 9, 200))
    before = abs(v[0] - v[-1])
    after = B.make_loopable(v, 24)
    assert abs(after[0] - after[-1]) < 1e-5
    assert abs(after[0] - after[-1]) < before


def test_make_loopable_keeps_phase_monotonic():
    ph = np.cumsum(np.full(200, 0.21))
    out = B.make_loopable(ph, 24, wrap=True)
    assert np.all(np.diff(out) >= -1e-9)
    gap = (out[-1] - out[0]) % (2.0 * np.pi)
    assert min(gap, 2.0 * np.pi - gap) < 1e-4


def test_make_loopable_leaves_the_head_alone():
    v = np.linspace(0.0, 1.0, 100)
    out = B.make_loopable(v, 20)
    assert np.abs(out[:79] - v[:79]).max() < 1e-6


def test_make_loopable_ignores_tiny_fades():
    v = np.linspace(0.0, 1.0, 10)
    assert np.abs(B.make_loopable(v, 1) - v).max() < 1e-6
    assert np.abs(B.make_loopable(v, 0) - v).max() < 1e-6
