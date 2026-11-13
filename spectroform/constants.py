
OBJ_NAME = "Spectroform_Mesh"
MAT_NAME = "Spectroform_Material"
STRIP_NAME = "Spectroform_Track"
 
SK_BASIS = "Basis"
SK_BASS = "BASS_Terrain"
SK_MID = "MID_Wave"
SK_TREBLE = "TREBLE_Spikes"
 
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".aif", ".aiff",
                    ".m4a"}
# Blender bundles ffmpeg, so aud decodes the audio track straight out of a
# video container -- no separate extraction step is needed.
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".webm", ".mkv", ".mov", ".avi"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

MIN_BLENDER = (3, 6, 0)

SK_BIN = "BIN_%02d"
SPECTRAL_STATIC = {"SPECTROGRAM", "WAVE_TUBE", "WAVE_RING", "MELODY",
                   "VECTORSCOPE"}
SPECTRAL_BARS = {"BARS_RADIAL", "BARS_LINE"}

SK_RIPPLE_SIN = "RIPPLE_%d_SIN"
SK_RIPPLE_COS = "RIPPLE_%d_COS"

RIPPLE_MODES = {"RIPPLE"}