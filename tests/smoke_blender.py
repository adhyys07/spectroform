"""Headless smoke test: register the add-on and bake for real.

Run it with:
    blender --background --factory-startup --python smoke_blender.py

--factory-startup keeps your own preferences and any installed copy of the
add-on out of it, so this tests the source tree and nothing else.
"""

import os
import sys
import traceback

# This file lives in <repo>/tests, and Blender runs it with __file__ set.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO = os.path.join(REPO, "demo_beat.wav")
STEREO = os.path.join(REPO, "demo_stereo.wav")

if REPO not in sys.path:
    sys.path.insert(0, REPO)

import bpy  # noqa: E402

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print("  %s %s%s" % ("ok  " if condition else "FAIL", name,
                         (" -- " + detail) if detail else ""))


def fresh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene.audiogeo


def base(props, audio=AUDIO):
    props.audio_path = audio
    props.bake_fps = 24
    props.frame_start = 1
    props.resolution = 24          # keep the vertex count small
    props.ripple_components = 4
    props.time_steps = 32
    props.spectrum_bins = 8
    props.create_material = True
    props.add_camera = False
    props.light_rig = False
    props.spark_count = 0
    props.sync_sound = True
    return props


def result():
    obj = bpy.data.objects.get("Spectroform_Mesh")
    if obj is None:
        return None, 0, 0, 0
    keys = obj.data.shape_keys
    n_keys = len(keys.key_blocks) if keys else 0
    # Blender 4.4+ moved f-curves into action layers/strips/channelbags,
    # so count them the way the add-on itself finds them.
    from spectroform import baking
    curves = 0
    for holder in (obj, keys):
        anim = getattr(holder, "animation_data", None)
        if anim and anim.action:
            action = anim.action
            seen = []
            if getattr(action, "fcurves", None) is not None:
                seen += list(action.fcurves)
            for layer in getattr(action, "layers", []):
                for strip in getattr(layer, "strips", []):
                    for bag in getattr(strip, "channelbags", []):
                        seen += list(bag.fcurves)
            curves += len(seen)
    assert baking.find_fcurve is not None
    return obj, len(obj.data.vertices), n_keys, curves


def case(name, setup, deforms=True):
    # deforms=False for the SPECTRAL_STATIC shapes: spectrogram, melody
    # and vectorscope are built *from* the audio, so only the object
    # moves and there are no shape keys to look for.
    print("\n[%s]" % name)
    try:
        props = base(fresh())
        setup(props)
        out = bpy.ops.audiogeo.generate()
        check("%s: operator finished" % name, out == {"FINISHED"}, str(out))
        obj, verts, keys, curves = result()
        check("%s: object built" % name, obj is not None)
        check("%s: has geometry" % name, verts > 0, "%d verts" % verts)
        if deforms:
            check("%s: has shape keys" % name, keys > 1, "%d keys" % keys)
        else:
            check("%s: static mesh, no shape keys" % name, keys == 0,
                  "%d keys" % keys)
        check("%s: has keyframes" % name, curves > 0, "%d f-curves" % curves)
        check("%s: frame range set" % name,
              bpy.context.scene.frame_end > bpy.context.scene.frame_start,
              "%d-%d" % (bpy.context.scene.frame_start,
                         bpy.context.scene.frame_end))
        print("  report: %s" % props.last_report)
    except Exception:
        check("%s: no exception" % name, False, traceback.format_exc(limit=3))


print("Blender %s" % bpy.app.version_string)
print("repo    %s" % REPO)

# --- registration -------------------------------------------------------
print("\n[register]")
try:
    import spectroform
    spectroform.register()
    check("add-on registers", True)
except Exception:
    check("add-on registers", False, traceback.format_exc(limit=4))
    raise SystemExit(1)

check("scene property group", hasattr(bpy.context.scene, "audiogeo"))
for op in ("generate", "live", "live_stop", "live_bake", "render", "export",
           "export_web", "export_data", "rebake", "preset", "batch"):
    check("operator audiogeo.%s" % op, hasattr(bpy.ops.audiogeo, op))

from spectroform import liveinput  # noqa: E402
print("  sounddevice present: %s" % liveinput.available())
check("live input degrades without sounddevice",
      liveinput.device_list() == [] or liveinput.available())

# --- the shapes ---------------------------------------------------------
case("ripple", lambda p: setattr(p, "geometry_type", "RIPPLE"))
case("spectrogram", lambda p: setattr(p, "geometry_type", "SPECTROGRAM"),
     deforms=False)
case("bars", lambda p: setattr(p, "geometry_type", "BARS_LINE"))
case("melody", lambda p: setattr(p, "geometry_type", "MELODY"),
     deforms=False)
case("sphere-noise", lambda p: setattr(p, "geometry_type", "SPHERE"))


def _stereo(props):
    props.geometry_type = "VECTORSCOPE"
    props.use_stereo = True
    props.audio_path = STEREO


case("vectorscope-stereo", _stereo, deforms=False)


def _emitter(props):
    props.geometry_type = "RIPPLE"
    props.ripple_source = "EMITTER"


case("beat-emitters", _emitter)


def _plates(props):
    props.geometry_type = "RIPPLE"
    props.ripple_source = "CHLADNI"
    props.chladni_modes = 4


case("plate-modes", _plates)

# --- custom mesh input --------------------------------------------------
print("\n[custom-mesh]")
try:
    props = base(fresh())
    bpy.ops.mesh.primitive_monkey_add()
    source = bpy.context.active_object
    source.name = "MyLogo"
    before = (len(source.data.vertices), tuple(source.scale))

    props.geometry_type = "RIPPLE"
    props.mesh_source = "CUSTOM"
    props.custom_mesh = source
    props.custom_auto_size = True
    out = bpy.ops.audiogeo.generate()

    check("custom: operator finished", out == {"FINISHED"}, str(out))
    obj, verts, keys, curves = result()
    check("custom: built from the source mesh", obj is not None and verts > 0,
          "%d verts" % verts)
    check("custom: source left untouched",
          (len(source.data.vertices), tuple(source.scale)) == before)
    check("custom: separate object", obj is not source)
    check("custom: has ripple keys", keys > 1, "%d keys" % keys)
    check("custom: size matched the object", props.size > 0,
          "size %.2f" % props.size)
except Exception:
    check("custom: no exception", False, traceback.format_exc(limit=3))

# --- re-bake and export -------------------------------------------------
print("\n[rebake + export]")
try:
    props = base(fresh())
    props.geometry_type = "RIPPLE"
    bpy.ops.audiogeo.generate()
    check("stamp readable", bpy.ops.audiogeo.rebake.poll())
    check("re-bake runs", bpy.ops.audiogeo.rebake() == {"FINISHED"})

    out_dir = bpy.app.tempdir
    props.export_path = os.path.join(out_dir, "smoke.glb")
    props.web_export_path = os.path.join(out_dir, "smoke_web")
    props.data_export_path = os.path.join(out_dir, "smoke_data")
    for name, folder in (("export_web", props.web_export_path),
                         ("export_data", props.data_export_path)):
        try:
            res = getattr(bpy.ops.audiogeo, name)()
            target = folder if os.path.isdir(folder) else os.path.dirname(folder)
            written = [f for f in os.listdir(target or ".")
                       if f.startswith("spectroform") or f.startswith("smoke")]
            check("%s wrote a file" % name,
                  res == {"FINISHED"} and bool(written), ", ".join(written))
        except Exception as exc:
            check("%s wrote a file" % name, False, str(exc))
except Exception:
    check("rebake/export: no exception", False, traceback.format_exc(limit=3))

print("\n%s\n%d passed, %d failed" % ("-" * 60, len(PASS), len(FAIL)))
for name in FAIL:
    print("  FAILED: %s" % name)
sys.exit(1 if FAIL else 0)
