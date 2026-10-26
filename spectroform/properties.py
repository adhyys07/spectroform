from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

# sine_field() offsets the spike seed by 977, so keep headroom below 2**31.
_SEED_MAX = 2 ** 31 - 1000


class AUDIOGEO_Props(PropertyGroup):
    # --- source ---------------------------------------------------------
    audio_path: StringProperty(
        name="Audio File",
        description="WAV, MP3, FLAC, OGG, AIFF or M4A file to analyse",
        subtype="FILE_PATH",
        default="",
    )
    bake_fps: IntProperty(
        name="Bake FPS",
        description="Analysis and keyframe rate; also applied to the scene",
        default=24, min=1, max=240, soft_max=60,
    )
    frame_start: IntProperty(
        name="Start Frame",
        description="Frame the baked animation begins on",
        default=1, min=0, soft_max=1000,
    )
    replace_existing: BoolProperty(
        name="Replace Existing",
        description="Delete the previously generated object before building a new one",
        default=True,
    )
    last_report: StringProperty(
        name="Last Report",
        description="Result of the most recent generate or export",
        default="",
    )

    # --- frequency response ---------------------------------------------
    bass_gain: FloatProperty(
        name="Bass Gain",
        description="Multiplier on the bass band before enveloping",
        default=1.0, min=0.0, max=8.0, soft_max=4.0,
    )
    mid_gain: FloatProperty(
        name="Mid Gain",
        description="Multiplier on the mid band before enveloping",
        default=1.0, min=0.0, max=8.0, soft_max=4.0,
    )
    treble_gain: FloatProperty(
        name="Treble Gain",
        description="Multiplier on the treble band before enveloping",
        default=1.0, min=0.0, max=8.0, soft_max=4.0,
    )
    master_gain: FloatProperty(
        name="Master Gain",
        description="Applied to all three bands on top of their own gain",
        default=1.0, min=0.0, max=8.0, soft_max=4.0,
    )

    bass_max: FloatProperty(
        name="Bass / Mid Split",
        description="Upper edge of the bass band, in hertz",
        default=250.0, min=20.0, max=2000.0,
    )
    mid_max: FloatProperty(
        name="Mid / Treble Split",
        description="Upper edge of the mid band, in hertz",
        default=4000.0, min=100.0, max=16000.0,
    )
    dynamic_range: FloatProperty(
        name="Dynamic Range",
        description="Decibels below the peak that map to zero response",
        default=48.0, min=6.0, max=96.0,
    )
    per_band_level: BoolProperty(
        name="Normalise Per Band",
        description=(
            "Scale each band against its own peak. Disable to share one peak, "
            "which preserves the real balance between bass, mid and treble"
        ),
        default=True,
    )

    attack_ms: FloatProperty(
        name="Attack",
        description="Envelope rise time in milliseconds",
        default=25.0, min=1.0, max=1000.0, soft_max=200.0,
    )
    release_ms: FloatProperty(
        name="Release",
        description="Envelope fall time in milliseconds",
        default=120.0, min=1.0, max=4000.0, soft_max=1000.0,
    )

    # --- geometry -------------------------------------------------------
    geometry_type: EnumProperty(
        name="Shape",
        description="Base mesh the shape keys deform",
        items=[
            ("RIPPLE", "Water Ripple", "Travelling waves whose height follows each band"),
            ("SPECTROGRAM", "Spectrogram Terrain", "Time x frequency x magnitude -- the 3D waveform"),
            ("WAVE_TUBE", "Waveform Tube", "Tube along X whose radius follows the waveform"),
            ("WAVE_RING", "Waveform Ring", "The waveform tube bent into a closed ring"),
            ("BARS_RADIAL", "Spectrum Bars (Radial)", "Animated bar per frequency bin, in a circle"),
            ("BARS_LINE", "Spectrum Bars (Line)", "Animated bar per frequency bin, in a row"),
            ("GRID", "Grid (Noise)", "Flat plane displaced by a noise field"),
            ("SPHERE", "Sphere (Noise)", "UV sphere displaced by a noise field"),
            ("CYLINDER", "Cylinder (Noise)", "Open tube displaced by a noise field"),
            ("TORUS", "Torus (Noise)", "Ring displaced by a noise field"),
        ],
        default="RIPPLE",
    )
    
    resolution: IntProperty(
        name="Resolution",
        description="Subdivisions per side; vertex count grows with its square",
        default=64, min=8, max=512, soft_max=256,
    )
    size: FloatProperty(
        name="Size",
        description="Overall extent in metres; all deformations scale with it",
        default=4.0, min=0.01, soft_max=50.0, unit="LENGTH",
    )
    base_scale: FloatProperty(
        name="Base Scale",
        description="Object scale at silence, before pulse and loudness gain",
        default=1.0, min=0.01, soft_max=10.0,
    )
    shade_smooth: BoolProperty(
        name="Shade Smooth",
        description="Mark the generated faces smooth",
        default=True,
    )

    terrain_strength: FloatProperty(
        name="Strength",
        description="How far bass pushes vertices along their normals, relative to Size",
        default=0.25, min=0.0, max=4.0, soft_max=1.0,
    )
    noise_scale: FloatProperty(
        name="Noise Scale",
        description="Feature size of the terrain field; higher is busier",
        default=2.0, min=0.01, max=100.0, soft_max=20.0,
    )
    noise_octaves: IntProperty(
        name="Octaves",
        description="Layers of detail in the terrain field",
        default=3, min=1, max=8,
    )

    wave_strength: FloatProperty(
        name="Strength",
        description="How far mids push the ripple, relative to Size",
        default=0.15, min=0.0, max=4.0, soft_max=1.0,
    )
    wave_count: IntProperty(
        name="Wave Count",
        description="Number of ripples across the shape",
        default=4, min=1, max=64, soft_max=16,
    )

    spike_strength: FloatProperty(
        name="Strength",
        description="How far treble extrudes the spikes, relative to Size",
        default=0.3, min=0.0, max=4.0, soft_max=1.0,
    )
    spike_density: FloatProperty(
        name="Density",
        description="How many spikes appear across the shape",
        default=8.0, min=0.1, max=100.0, soft_max=32.0,
    )
    spike_sharpness: FloatProperty(
        name="Sharpness",
        description="Exponent on the spike mask; higher gives narrower points",
        default=4.0, min=1.0, max=32.0, soft_max=16.0,
    )

    seed: IntProperty(
        name="Seed",
        description="Random seed for the terrain and spike fields",
        default=0, min=0, max=_SEED_MAX, soft_max=9999,
    )

    # --- motion & look --------------------------------------------------
    scale_gain: FloatProperty(
        name="Loudness Scale",
        description="How much overall loudness inflates the object",
        default=0.25, min=0.0, max=4.0, soft_max=1.0,
    )
    pulse_gain: FloatProperty(
        name="Beat Pulse",
        description="How much detected onsets kick the object scale",
        default=0.15, min=0.0, max=4.0, soft_max=1.0,
    )
    spin_speed: FloatProperty(
        name="Spin Speed",
        description="Constant Z rotation per second",
        default=0.0, min=-100.0, max=100.0, soft_min=-12.0, soft_max=12.0,
        subtype="ANGLE", unit="ROTATION",
    )
    spin_gain: FloatProperty(
        name="Spin On Beat",
        description="Extra Z rotation per second driven by onsets",
        default=0.0, min=0.0, max=100.0, soft_max=12.0,
        subtype="ANGLE", unit="ROTATION",
    )

    create_material: BoolProperty(
        name="Create Material",
        description="Add an emissive material whose strength tracks the audio energy",
        default=True,
    )
    sync_sound: BoolProperty(
        name="Sync Sound",
        description="Add the audio file to the sequencer so playback stays in sync",
        default=True,
    )

    ripple_source: EnumProperty(
        name="Wave Source",
        description="What decides each wave's size and speed",
        items=[
            ("SPECTRUM", "Spectrum", "One wave per frequency bin -- follows the track closely"),
            ("BANDS", "Three Bands", "One wave each for bass, mid and treble -- simpler, calmer"),
        ],
        default="SPECTRUM",
    )
    ripple_components: IntProperty(
        name="Wave Components",
        description="Spectrum bins, each becoming its own travelling wave. "
                    "More means a closer match to the audio and more shape keys",
        default=12, min=1, max=32, soft_max=16,
    )
    ripple_spread: FloatProperty(
        name="Pitch Spread",
        description="Octaves of wavelength between the lowest and highest wave. "
                    "0 makes every wave the same size",
        default=3.0, min=0.0, max=6.0,
    )
    ripple_beat_gain: FloatProperty(
        name="Beat Surge",
        description="How hard each onset drives the waves forward. 0 gives a "
                    "constant drift that ignores the rhythm",
        default=2.0, min=0.0, max=20.0, soft_max=8.0,
    )
    ripple_pattern: EnumProperty(
        name="Ripple Pattern",
        description="Direction the waves travel",
        items=[
            ("RADIAL", "Radial", "Rings spreading from the centre, like a dropped stone"),
            ("LINEAR", "Linear", "Parallel rollers crossing along X"),
            ("DIAGONAL", "Diagonal", "Parallel rollers crossing corner to corner"),
        ],
        default="RADIAL",
    )
    ripple_cycles: FloatProperty(
        name="Wave Count",
        description="Crests across the surface for the bass wave. "
                    "Mid and treble use 2.2x and 4.5x this",
        default=3.0, min=0.25, max=40.0, soft_max=12.0,
    )
    ripple_speed: FloatProperty(
        name="Wave Speed",
        description="Bass wave cycles per second. Mid and treble travel "
                    "1.6x and 2.7x faster",
        default=0.6, min=-8.0, max=8.0, soft_min=-3.0, soft_max=3.0,
    )
    scroll_terrain: BoolProperty(
        name="Scroll With Audio",
        description="Slide a Spectrogram Terrain past the origin so the "
                    "playhead stays centred",
        default=True,
    )

    # --- export ---------------------------------------------------------
    export_format: EnumProperty(
        name="Format",
        description="File type written by the Export button",
        items=[
            ("GLB", "glTF Binary (.glb)", "Single-file glTF, shape keys included"),
            ("GLTF", "glTF Separate (.gltf)", "glTF with external buffers and textures"),
            ("FBX", "FBX (.fbx)", "FBX with baked animation"),
            ("BLEND", "Blend File (.blend)", "Copy of the current file"),
        ],
        default="GLB",
    )
    export_path: StringProperty(
        name="Path",
        description="Destination file; the extension is corrected to match Format",
        subtype="FILE_PATH",
        default="//audio_reactive_geometry.glb",
    )
    export_selected_only: BoolProperty(
        name="Selected Only",
        description="Export just the generated object rather than the whole scene",
        default=True,
    )

    spectrum_bins: IntProperty(
        name="Frequency Bins",
        description="Log-spaced bands across the spectrum. Keep at 32 or below "
                    "for glTF export -- each bin becomes a morph target",
        default=48, min=4, max=128, soft_max=64,
    )
    time_steps: IntProperty(
        name="Time Steps",
        description="Rows of the spectrogram mesh. Longer audio is reduced to "
                    "this many rows with a block max, so peaks survive",
        default=256, min=16, max=2048, soft_max=512,
    )
    spec_height: FloatProperty(
        name="Height",
        description="Peak displacement as a fraction of Size",
        default=0.35, min=0.0, max=4.0, soft_max=1.0,
    )
    tube_radius: FloatProperty(
        name="Tube Radius",
        description="Tube thickness as a fraction of Size",
        default=0.12, min=0.001, max=1.0, soft_max=0.3,
    )
    tube_segments: IntProperty(
        name="Tube Segments",
        description="Vertices around the tube cross-section",
        default=16, min=3, max=64,
    )


classes = (AUDIOGEO_Props,)
