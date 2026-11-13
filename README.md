## Spectroform
Spectroform is a blender extension which generates meshes from a sound wave from audio file. It takes the waves from the audio file and the python modules render them as a sine and cos wave in multiple directions that generates a continuous mesh.

# What it does?
Basically, it caputes audio from any audio files like MP3, WAV, AAC etc. then analyse its soundwaves then using python modules it converts them into 3D waves in blender, as the sound continues the waves changes according to amplitude and wavelength. It uses sine and cos waves to generate waves on 3D Models

# Framework used
1) Numpy: To handle all calculation part
2) Sound Device: To capture audio device
3) yt_dlp: Youtube module to capture audios using links
4) Aud: To decode audio files

# Features
1) 12 Shapes: Water ripple, spectrogram terrain, waveform tube and ring, spectrum bars, melody ribbon, stereo vectorscope, and noise grid/sphere/cylinder/torus
2) Custom Mesh: Deform your own model instead -- a logo, a character, a scan -- the original is left untouched
3) 9 Wave Sources: Spectrum, three bands, beat emitters, tonal/percussive, plate modes, Faraday, stereo interference, harmonics, MIDI
4) Beat Tracking: Finds the tempo, beats, bars and phrases, and accents the downbeats
5) Key and Sections: Detects the key and where each part of the song starts, with a different look per section
6) Stems: Splits the track into drums, bass and harmony, each driving its own waves
7) Melody: Follows the lead pitch, so the ribbon shows the whole tune at once
8) Live Mic: Drives the mesh in real time, records the take and bakes it (needs Sound Device)
9) Baked Keyframes: Everything becomes real F-curves, so you can scrub it, hand-edit it and render without the add-on
10) Material and Lights: Glow follows the energy, colour follows the key, plus bloom, a three-light rig and sparks on onsets
11) Camera Cuts: One camera per section, cutting between them on playback
12) Export: Animation render, mesh with morph targets, playable web page, analysis JSON, whole-folder batch, loop trim to bars
13) Presets and Modulation: Ready-made starting points, or route any analysed signal to any channel yourself
14) Caching: Nudge a slider and it re-bakes at once instead of re-analysing the song
15) Input: WAV, MP3, FLAC, OGG, AIFF and M4A, a direct link, or a page link with yt-dlp
