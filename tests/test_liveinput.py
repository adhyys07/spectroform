import os
import sys
import types
import wave

import numpy as np
import pytest

RATE = 44100


def _stub_bpy():
    if "bpy" in sys.modules and hasattr(sys.modules["bpy"], "app"):
        return sys.modules["bpy"]

    bpy = types.ModuleType("bpy")
    bpy.__path__ = []
    bpy.types = types.ModuleType("bpy.types")
    bpy.types.Operator = object
    bpy.path = types.ModuleType("bpy.path")
    bpy.data = types.SimpleNamespace(is_saved=False, objects={})
    bpy.app = types.SimpleNamespace(tempdir="", timers=None)
    for name, mod in (("bpy", bpy), ("bpy.types", bpy.types),
                      ("bpy.path", bpy.path)):
        sys.modules[name] = mod
    return bpy


@pytest.fixture
def live(tmp_path):
    bpy = _stub_bpy()
    bpy.app.tempdir = str(tmp_path)
    bpy.data.is_saved = False
    bpy.path.abspath = lambda p: p.replace("//", str(tmp_path) + os.sep)
    from spectroform import liveinput
    return liveinput


class Props:

    bass_max, mid_max = 250.0, 4000.0
    dynamic_range = 60.0
    live_gain, live_gate_db, live_smoothing = 1.0, -45.0, 0.0
    live_rate, live_fps = RATE, 60
    live_device = ""
    live_record, live_record_max = True, 300
    live_record_path = ""
    live_auto_bake = False
    master_gain = 1.0
    bass_gain = mid_gain = treble_gain = 1.0
    base_scale, scale_gain, pulse_gain = 1.0, 0.5, 0.25
    spin_speed, spin_gain = 0.0, 0.0
    ripple_speed, ripple_spread, ripple_beat_gain = 1.0, 2.0, 0.0


def _tone(hz, seconds=0.1, amp=0.5):
    t = np.arange(int(RATE * seconds)) / RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _run(analyser, props, block, ticks=12):
    for _ in range(ticks):
        analyser.push(block)
        out = analyser.update(props)
    return out



def test_silence_leaves_the_surface_still(live):
    levels, bands, rms, onset = _run(
        live.Analyser(RATE, 16, Props()), Props(),
        np.zeros(live.FFT, dtype=np.float32))
    assert float(levels.max()) == 0.0
    assert float(bands.max()) == 0.0
    assert rms == pytest.approx(0.0)
    assert onset == pytest.approx(0.0)


def test_room_noise_stays_gated(live):
    rng = np.random.default_rng(0)
    hiss = (rng.standard_normal(live.FFT) * 10 ** (-65.0 / 20.0)).astype(np.float32)
    levels, _b, _r, _o = _run(live.Analyser(RATE, 16, Props()), Props(), hiss)
    assert float(levels.max()) < 0.02


def test_loud_tone_reaches_full_level(live):
    levels, _b, _r, _o = _run(
        live.Analyser(RATE, 16, Props()), Props(), _tone(1000.0, amp=0.9))
    assert float(levels.max()) > 0.9


def test_level_follows_the_tone_up_the_spectrum(live):
    peaks = []
    for hz in (120.0, 1000.0, 8000.0):
        levels, _b, _r, _o = _run(
            live.Analyser(RATE, 24, Props()), Props(), _tone(hz, amp=0.8))
        peaks.append(int(np.argmax(levels)))
    assert peaks[0] < peaks[1] < peaks[2]


def test_bands_split_bass_from_treble(live):
    props = Props()
    _l, bass, _r, _o = _run(live.Analyser(RATE, 16, props), props,
                            _tone(80.0, amp=0.8))
    _l, treble, _r, _o = _run(live.Analyser(RATE, 16, props), props,
                              _tone(9000.0, amp=0.8))
    assert bass[0] > bass[2]
    assert treble[2] > treble[0]


def test_gate_threshold_is_respected(live):
    quiet = _tone(1000.0, amp=0.004)          # about -48 dBFS
    props = Props()
    props.live_gate_db = -30.0
    shut, _b, _r, _o = _run(live.Analyser(RATE, 16, props), props, quiet)
    props_open = Props()
    props_open.live_gate_db = -70.0
    opened, _b, _r, _o = _run(live.Analyser(RATE, 16, props_open), props_open,
                              quiet)
    assert float(shut.max()) == 0.0
    assert float(opened.max()) > 0.1


def test_onset_fires_on_a_hit_and_decays(live):
    props = Props()
    analyser = live.Analyser(RATE, 16, props)
    _run(analyser, props, np.zeros(live.FFT, dtype=np.float32), ticks=4)

    rng = np.random.default_rng(1)
    hit = (rng.standard_normal(live.FFT) * 0.8).astype(np.float32)
    analyser.push(hit)
    peak = analyser.update(props)[3]
    assert peak > 0.05

    analyser.push(hit)
    assert analyser.update(props)[3] < peak


def test_smoothing_slows_the_rise(live):
    props, slow = Props(), Props()
    props.live_smoothing, slow.live_smoothing = 0.0, 0.9
    block = _tone(1000.0, amp=0.9)
    fast_level = _run(live.Analyser(RATE, 16, props), props, block, ticks=2)[0]
    slow_level = _run(live.Analyser(RATE, 16, slow), slow, block, ticks=2)[0]
    assert float(slow_level.max()) < float(fast_level.max())


def test_analyser_survives_short_blocks(live):
    props = Props()
    analyser = live.Analyser(RATE, 16, props)
    for size in (64, 300, 1024, 4096):
        analyser.push(_tone(500.0, seconds=size / RATE, amp=0.5))
        levels = analyser.update(props)[0]
        assert np.isfinite(levels).all()


class FakeKey:
    def __init__(self):
        self.value = 0.0


class FakeKeys(dict):
    def get(self, name, default=None):
        return dict.get(self, name, default)


class FakeEuler:

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

    def __iter__(self):
        return iter((self.x, self.y, self.z))


class FakeObject:
    def __init__(self, names):
        keys = FakeKeys({name: FakeKey() for name in names})
        self.data = types.SimpleNamespace(
            shape_keys=types.SimpleNamespace(key_blocks=keys))
        self.scale = (1.0, 1.0, 1.0)
        self.rotation_euler = FakeEuler()


def _ripple_object(pairs=4):
    names = []
    for i in range(pairs):
        names += ["RIPPLE_%d_SIN" % i, "RIPPLE_%d_COS" % i]
    return FakeObject(names)


def test_driver_finds_ripple_pairs_and_bins(live):
    assert len(live.Driver(_ripple_object(5), Props()).pairs) == 5
    bins = FakeObject(["BIN_%02d" % i for i in range(7)])
    driver = live.Driver(bins, Props())
    assert len(driver.bins) == 7 and driver.components == 7


def test_driver_writes_a_travelling_wave(live):
    props = Props()
    obj = _ripple_object(3)
    driver = live.Driver(obj, props)
    levels = np.full(3, 0.5, dtype=np.float32)

    driver.update(levels, np.zeros(3), 0.0, 0.0, 0.1, props)
    keys = obj.data.shape_keys.key_blocks
    for i in range(3):
        phase = driver.phase[i]
        amp = 0.5
        assert keys["RIPPLE_%d_SIN" % i].value == pytest.approx(
            amp * np.cos(phase), abs=1e-6)
        assert keys["RIPPLE_%d_COS" % i].value == pytest.approx(
            -amp * np.sin(phase), abs=1e-6)
        energy = (keys["RIPPLE_%d_SIN" % i].value ** 2
                  + keys["RIPPLE_%d_COS" % i].value ** 2)
        assert energy == pytest.approx(amp ** 2, abs=1e-6)


def test_phase_advances_with_elapsed_time(live):
    props = Props()
    driver = live.Driver(_ripple_object(2), props)
    levels = np.ones(2, dtype=np.float32)

    driver.update(levels, np.zeros(3), 0.0, 0.0, 0.25, props)
    first = driver.phase.copy()
    driver.update(levels, np.zeros(3), 0.0, 0.0, 0.25, props)
    assert np.all(driver.phase > first)
    assert driver.phase[1] > driver.phase[0]


def test_driver_scales_the_object_with_level(live):
    props = Props()
    obj = _ripple_object(1)
    driver = live.Driver(obj, props)
    driver.update(np.zeros(1), np.zeros(3), 1.0, 1.0, 0.1, props)
    assert obj.scale[0] == pytest.approx(
        props.base_scale * (1.0 + props.scale_gain + props.pulse_gain))


def test_driver_restores_what_it_found(live):
    props = Props()
    obj = FakeObject(["BASS_Terrain", "MID_Wave", "TREBLE_Spikes"])
    keys = obj.data.shape_keys.key_blocks
    keys["MID_Wave"].value = 0.42
    obj.scale = (3.0, 3.0, 3.0)

    driver = live.Driver(obj, props)
    driver.update(np.ones(1), np.ones(3), 1.0, 0.0, 0.1, props)
    assert keys["MID_Wave"].value != 0.42

    driver.restore()
    assert keys["MID_Wave"].value == pytest.approx(0.42)
    assert obj.scale == (3.0, 3.0, 3.0)


def test_levels_clip_to_the_slider_range(live):
    props = Props()
    props.master_gain = 50.0
    obj = _ripple_object(2)
    driver = live.Driver(obj, props)
    driver.update(np.ones(2), np.ones(3), 0.0, 0.0, 0.1, props)
    for key in obj.data.shape_keys.key_blocks.values():
        assert -2.0 <= key.value <= 2.0


def test_wav_round_trips(live, tmp_path):
    path = str(tmp_path / "take.wav")
    tone = _tone(440.0, seconds=0.5, amp=0.2)
    written, seconds, peak = live.write_wav(path, tone, RATE)

    assert written == path
    assert seconds == pytest.approx(0.5, abs=1e-3)
    assert peak == pytest.approx(0.2, abs=1e-3)

    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1
        assert f.getsampwidth() == 2
        assert f.getframerate() == RATE
        assert f.getnframes() == tone.size
        back = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")

    back = back.astype(np.float32) / 32767.0
    assert float(np.max(np.abs(back))) == pytest.approx(0.98, abs=0.01)
    scaled = tone * (0.98 / peak)
    assert float(np.max(np.abs(back - scaled))) < 1e-3


def test_wav_refuses_an_empty_take(live, tmp_path):
    with pytest.raises(RuntimeError):
        live.write_wav(str(tmp_path / "empty.wav"),
                       np.zeros(0, dtype=np.float32), RATE)


def test_silent_take_is_written_not_amplified(live, tmp_path):
    path = str(tmp_path / "silent.wav")
    _p, seconds, peak = live.write_wav(
        path, np.zeros(RATE, dtype=np.float32), RATE)
    assert peak == 0.0 and seconds == pytest.approx(1.0)
    with wave.open(path, "rb") as f:
        back = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    assert not back.any()


def test_wav_creates_missing_folders(live, tmp_path):
    path = str(tmp_path / "takes" / "nested" / "a.wav")
    live.write_wav(path, _tone(440.0, seconds=0.05), RATE)
    assert os.path.isfile(path)


def test_take_path_names_a_file_in_a_folder(live, tmp_path):
    props = Props()
    props.live_record_path = str(tmp_path)
    path = live.take_path(props)
    assert os.path.dirname(path) == str(tmp_path)
    assert path.lower().endswith(".wav")


def test_take_path_falls_back_to_temp_when_unsaved(live, tmp_path):
    props = Props()
    props.live_record_path = "//"
    assert live.take_path(props).startswith(str(tmp_path))


def test_take_path_keeps_an_explicit_name(live, tmp_path):
    props = Props()
    props.live_record_path = str(tmp_path / "gig")
    assert live.take_path(props) == str(tmp_path / "gig.wav")


def _capture(live, record=True, seconds=1.0):
    cap = live.Capture.__new__(live.Capture)
    cap.rate, cap.channels, cap.record = RATE, 1, record
    cap.max_samples = int(seconds * RATE)
    import queue
    import threading
    cap.monitor = queue.Queue(maxsize=4)
    cap._blocks, cap._lock = [], threading.Lock()
    cap.recorded = cap.captured = cap.dropped = 0
    cap.full, cap.started = False, 0.0
    return cap


def test_callback_downmixes_and_records(live):
    cap = _capture(live)
    stereo = np.stack([np.full(256, 1.0, dtype=np.float32),
                       np.full(256, -1.0, dtype=np.float32)], axis=1)
    cap._callback(stereo, 256, None, None)

    assert cap.recorded == 256
    take = cap.take()
    assert take.size == 256
    assert np.allclose(take, 0.0)          # +1 and -1 cancel
    assert cap.seconds == pytest.approx(256 / RATE)


def test_recording_stops_at_the_limit(live):
    cap = _capture(live, seconds=0.01)     # 441 samples
    block = np.full((256, 1), 0.5, dtype=np.float32)
    for _ in range(4):
        cap._callback(block, 256, None, None)

    assert cap.full is True
    assert cap.recorded <= 512             # stopped after the second block
    assert cap.captured == 1024            # but kept feeding the viewport


def test_monitor_only_keeps_recent_blocks(live):
    cap = _capture(live)
    block = np.full((16, 1), 0.25, dtype=np.float32)
    for _ in range(40):
        cap._callback(block, 16, None, None)

    blocks = cap.latest()
    assert len(blocks) <= cap.monitor.maxsize
    assert cap.latest() is None            
    assert cap.recorded == 40 * 16         


def test_monitoring_without_recording_keeps_nothing(live):
    cap = _capture(live, record=False)
    cap._callback(np.full((128, 1), 0.3, dtype=np.float32), 128, None, None)
    assert cap.recorded == 0
    assert cap.take().size == 0
    assert cap.captured == 128


def test_available_reports_the_missing_module(live):
    assert live.available() is (live.sd is not None)
    if live.sd is None:
        assert live.device_list() == []
