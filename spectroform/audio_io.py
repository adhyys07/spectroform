import os
import bpy
import numpy as np
from .constants import (AUDIO_EXTENSIONS, SUPPORTED_EXTENSIONS,
                        VIDEO_EXTENSIONS)

try:
    import aud
except ImportError:
    aud = None


def _decode(filepath):
    """Open and decode an audio file to a float32 array plus its sample rate.

    The array keeps its channels: (N,) or (N, channels).
    """
    if aud is None:
        raise RuntimeError("Blender's 'aud' audio module is unavailable in this build.")
    path = bpy.path.abspath(filepath)
    if not path or not os.path.isfile(path):
        raise RuntimeError("Audio file not found: %s" % path)

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise RuntimeError(
            "Unsupported file type '%s'. Audio: %s. Video: %s."
            % (ext, ", ".join(sorted(AUDIO_EXTENSIONS)),
               ", ".join(sorted(VIDEO_EXTENSIONS))))

    try:
        snd = aud.Sound.file(path)
    except AttributeError:
        snd = aud.Sound(path)

    rate = int(snd.specs[0])
    # Check this BEFORE data(): a file aud cannot decode reports rate 0, and
    # calling data() on one crashes Blender outright rather than raising.
    if rate <= 0:
        raise RuntimeError("Could not read the sample rate from the audio file.")

    try:
        data = snd.data()
    except Exception:
        data = snd.cache().data()

    data = np.asarray(data, dtype=np.float32)
    if data.size == 0:
        raise RuntimeError("The audio file decoded to zero samples.")
    return data, rate


def load_audio(filepath):
    """Mono samples normalised to peak 1.0, plus the sample rate."""
    data, rate = _decode(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono
    peak = float(np.max(np.abs(data)))
    if peak > 0.0:
        data = data / peak  # so the gain sliders behave the same for any file
    return data.astype(np.float32), rate


def load_audio_channels(filepath):
    """(mono, left, right, rate). Mono files give left is right.

    All three share one peak, or the stereo image collapses.
    """
    data, rate = _decode(filepath)
    if data.ndim > 1 and data.shape[1] >= 2:
        left, right = data[:, 0].copy(), data[:, 1].copy()
    else:
        left = right = data.ravel()
    mono = data.mean(axis=1) if data.ndim > 1 else data

    peak = max(float(np.max(np.abs(mono))), 1e-9)
    return ((mono / peak).astype(np.float32),
            (left / peak).astype(np.float32),
            (right / peak).astype(np.float32), rate)
