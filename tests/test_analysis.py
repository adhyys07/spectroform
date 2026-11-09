import numpy as np
from spectroform import analysis as A

FPS, RATE = 24, 44100

def _click_track(bpm, seconds=8.0):
    n = int(RATE * seconds)
    x = np.zeros(n, dtype=np.float32)
    step = int(RATE * 60.0 / bpm)
    for i in range(0, n, step):
        x[i:i + 400] = np.hanning(400)[:len(x[i:i + 400])]
    return x

def test_detect_tempo_no_octave_error():
    for bpm in (90.0, 120.0, 140.0):
        onset = A.analyse_audio(_click_track(bpm), RATE, FPS, 250., 4000., 48.)["onset"]
        got, _ = A.detect_tempo(onset, FPS)
        assert abs(got - bpm) < 5.0, (bpm, got)

def test_centroid_follows_a_sweep():
    t = np.arange(int(RATE * 6.0)) / RATE
    inst = 80.0 * (10000.0 / 80.0) ** (t / 6.0)
    sweep = np.sin(2 * np.pi * np.cumsum(inst) / RATE).astype(np.float32)
    spec = A.spectrogram(sweep, RATE, FPS, bins=24)
    cen = A.spectral_centroid(spec, A.bin_frequencies(24))
    assert np.corrcoef(cen, np.arange(cen.size))[0, 1] > 0.9


def test_centroid_is_level_invariant():
    t = np.arange(int(RATE * 3.0)) / RATE
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    hz = A.bin_frequencies(24)
    a = A.spectral_centroid(A.spectrogram(x, RATE, FPS, bins=24), hz)
    b = A.spectral_centroid(A.spectrogram(x * 0.25, RATE, FPS, bins=24), hz)
    assert np.abs(a - b).mean() < 0.05
