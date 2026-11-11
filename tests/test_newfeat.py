"""Tests for the HPSS / onset-band / chroma / section analysers.

All pure numpy -- they run without Blender, in well under a second.
"""

import numpy as np

from spectroform import analysis as A

RATE, FPS = 44100, 24


def _tone_and_bursts(seconds=4.0):
    """A steady 220 Hz tone plus broadband bursts, and the burst frame mask."""
    t = np.arange(int(RATE * seconds)) / RATE
    tone = (0.6 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    rng = np.random.default_rng(0)
    clicks = np.zeros_like(t)
    frames = []
    for i in range(0, t.size, RATE // 2):
        n = min(600, t.size - i)
        clicks[i:i + n] += rng.standard_normal(n) * np.exp(-np.arange(n) / 120.0)
        frames.append(int(round(i / (RATE / FPS))))
    clicks = clicks.astype(np.float32)
    mix = ((tone + 0.8 * clicks) / 1.8).astype(np.float32)
    return mix, tone, clicks, frames


def test_hpss_conserves_energy():
    mix, _t, _c, _f = _tone_and_bursts()
    spec = A.spectrogram(mix, RATE, FPS, bins=48)
    h, p = A.hpss(spec)
    assert abs(h.sum() + p.sum() - spec.sum()) / spec.sum() < 0.02


def test_hpss_is_non_negative():
    mix, _t, _c, _f = _tone_and_bursts()
    h, p = A.hpss(A.spectrogram(mix, RATE, FPS, bins=48))
    assert h.min() >= 0.0 and p.min() >= 0.0


def test_hpss_concentrates_transients_in_the_percussive_part():
    # Kurtosis is the wrong measure here: a steady tone with a few leaked
    # spikes is more peaked, relative to its own variance, than a signal that
    # is mostly spikes. What matters is WHERE the transient energy lands.
    mix, _tone, _clicks, burst_frames = _tone_and_bursts()
    spec = A.spectrogram(mix, RATE, FPS, bins=48)
    h, p = A.hpss(spec)

    n = spec.shape[0]
    mask = np.zeros(n, dtype=bool)
    for f in burst_frames:
        if f < n:
            mask[max(0, f - 1):f + 2] = True

    pe, he = p.sum(axis=1), h.sum(axis=1)
    assert pe[mask].mean() > pe[~mask].mean() * 5.0      # measured ~20x
    assert 0.5 < he[mask].mean() / he[~mask].mean() < 2.0  # steady: ~0.89


def test_hpss_outputs_resemble_their_sources():
    mix, tone, clicks, _f = _tone_and_bursts()
    h, p = A.hpss(A.spectrogram(mix, RATE, FPS, bins=48))
    sp_t = A.spectrogram(tone, RATE, FPS, bins=48)
    sp_c = A.spectrogram(clicks, RATE, FPS, bins=48)

    def corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    assert corr(h, sp_t) > corr(h, sp_c)      # measured +0.993 vs -0.023
    assert corr(p, sp_c) > corr(p, sp_t)      # measured +0.827 vs -0.000


def test_chroma_finds_the_right_note():
    t = np.arange(int(RATE * 3.0)) / RATE
    a4 = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    ch = A.chroma(a4, RATE, FPS)
    assert A.NOTE_NAMES[int(np.argmax(ch.mean(axis=0)))] == "A"


def test_chroma_rows_sum_to_one():
    t = np.arange(int(RATE * 2.0)) / RATE
    x = np.sin(2 * np.pi * 330.0 * t).astype(np.float32)
    ch = A.chroma(x, RATE, FPS)
    assert np.abs(ch.sum(axis=1) - 1.0).max() < 1e-4


def test_detect_key_searches_every_tonic():
    # An early version returned inside the tonic loop, so only C was ever
    # compared and every track came back as C.
    t = np.arange(int(RATE * 3.0)) / RATE
    triad = sum(np.sin(2 * np.pi * f * t) for f in (440.0, 554.37, 659.25))
    triad = (triad / 3.0).astype(np.float32)
    _tonic, major, name, conf = A.detect_key(A.chroma(triad, RATE, FPS))
    assert name == "A" and major and conf > 0.5


def test_detect_key_hears_a_minor_triad_as_minor():
    t = np.arange(int(RATE * 3.0)) / RATE
    # A minor: A C E
    triad = sum(np.sin(2 * np.pi * f * t) for f in (440.0, 523.25, 659.25))
    _tonic, major, name, _conf = A.detect_key(A.chroma((triad / 3.0).astype(np.float32),
                                                       RATE, FPS))
    assert not major and name.endswith("m")


def test_detect_sections_finds_a_real_change():
    t = np.arange(int(RATE * 8.0)) / RATE
    half = t.size // 2
    x = np.zeros_like(t)
    x[:half] = np.sin(2 * np.pi * 80 * t[:half])
    x[half:] = 0.6 * np.sin(2 * np.pi * 3000 * t[half:])
    x = (x / np.max(np.abs(x))).astype(np.float32)
    spec = A.spectrogram(x, RATE, FPS, bins=48)
    bounds = A.detect_sections(spec, FPS, min_seconds=2.0, count=1)
    assert bounds.size == 1
    assert abs(int(bounds[0]) - spec.shape[0] // 2) <= FPS


def test_detect_sections_ranks_most_novel_first():
    # A reversed slice once picked the LEAST novel frames as boundaries.
    t = np.arange(int(RATE * 12.0)) / RATE
    third = t.size // 3
    x = np.zeros_like(t)
    x[:third] = np.sin(2 * np.pi * 90 * t[:third])
    x[third:2 * third] = np.sin(2 * np.pi * 1200 * t[third:2 * third])
    x[2 * third:] = np.sin(2 * np.pi * 90 * t[2 * third:])
    x = (x / np.max(np.abs(x))).astype(np.float32)
    spec = A.spectrogram(x, RATE, FPS, bins=48)
    bounds = A.detect_sections(spec, FPS, min_seconds=2.0, count=2)
    assert bounds.size == 2
    want = [spec.shape[0] // 3, 2 * spec.shape[0] // 3]
    for got, expect in zip(bounds, want):
        assert abs(int(got) - expect) <= FPS, (bounds.tolist(), want)


def test_detect_sections_handles_tiny_input():
    assert A.detect_sections(np.zeros((4, 8), dtype=np.float32), FPS).size == 0


def _kick_and_hat(seconds=4.0):
    t = np.arange(int(RATE * seconds)) / RATE
    x = np.zeros_like(t)
    rng = np.random.default_rng(0)
    for i in range(0, t.size, RATE // 2):            # kick on quarters
        n = min(2000, t.size - i)
        x[i:i + n] += 0.9 * np.sin(2 * np.pi * 60 * t[:n]) * np.exp(-np.arange(n) / 2000)
    for i in range(0, t.size, RATE // 4):            # hat on eighths
        n = min(500, t.size - i)
        x[i:i + n] += 0.3 * rng.standard_normal(n) * np.exp(-np.arange(n) / 300)
    return (x / np.max(np.abs(x))).astype(np.float32)


def test_onset_bands_are_not_the_same_signal():
    spec = A.spectrogram(_kick_and_hat(), RATE, FPS, bins=48)
    low, _mid, high = A.onset_bands(spec, A.bin_frequencies(48))
    assert np.corrcoef(low, high)[0, 1] < 0.9


def test_onset_bands_are_normalised():
    spec = A.spectrogram(_kick_and_hat(), RATE, FPS, bins=48)
    for band in A.onset_bands(spec, A.bin_frequencies(48)):
        assert band.min() >= 0.0 and band.max() <= 1.0 + 1e-6


def test_classify_onsets_shares_sum_to_one():
    x = _kick_and_hat()
    spec = A.spectrogram(x, RATE, FPS, bins=48)
    data = A.analyse_audio(x, RATE, FPS, 250.0, 4000.0, 48.0)
    _frames, labels, shares = A.classify_onsets(spec, A.bin_frequencies(48),
                                                data["onset"])
    if shares.size:
        assert np.abs(shares.sum(axis=1) - 1.0).max() < 1e-4
        assert len(labels) == shares.shape[0]
