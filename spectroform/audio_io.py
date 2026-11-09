import os
import bpy
import numpy as np
from .constants import SUPPORTED_EXTENSIONS

try:
    import aud
except ImportError:
    aud = None


def load_audio(filepath):
    if aud is None:
        raise RuntimeError("Blender's 'aud' audio module is unavailable in this build.")
    path = bpy.path.abspath(filepath)
    if not path or not os.path.isfile(path):
        raise RuntimeError("Audio file not found: %s" % path)
 
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise RuntimeError("Unsupported file type '%s' (use WAV or MP3)." % ext)
 
    try:
        snd = aud.Sound.file(path)
    except AttributeError:
        snd = aud.Sound(path)
 
    rate = int(snd.specs[0])
    if rate <= 0:
        raise RuntimeError("Could not read the sample rate from the audio file.")

    try:
        data = snd.data()
    except Exception:
        data = snd.cache().data()
 
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono
    if data.size == 0:
        raise RuntimeError("The audio file decoded to zero samples.")
 
    peak = float(np.max(np.abs(data)))
    if peak > 0.0:
        data = data / peak  # so the gain sliders behave the same for any file
 
    return data, rate

def load_audio_channels(filepath):
    path = bpy.path.abspath(filepath)
    data, rate = _decode(path)
    if data.ndim > 1 and data.shape[1] >= 2:
        left, right = data[:, 0].copy(), data[:, 1].copy()
    else:
        flat = data.ravel()
        left = right = flat
    mono = data.mean(axis=1) if data.ndim > 1 else data

    peak = max(float(np.max(np.abs(mono))), 1e-9)
    return (mono / peak).astype(np.float32), \
           (left / peak).astype(np.float32), \
           (right / peak).astype(np.float32), rate
