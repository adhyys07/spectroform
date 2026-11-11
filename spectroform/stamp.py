import json
SETTINGS_KEY = "spectroform_settings"
VERSION_KEY = "spectroform_version"

_SKIP = {"last_report", "export_path", "render_path", "web_export_path"}
_LOCAL = {"audio_path", "midi_path", "lyrics_path"}

def snapshot(props, include_local=True):
    out = {}
    for name, prop in props.bl_rna.properties.items():
        if name in {"rna_type", "name"} or name in _SKIP or prop.is_readonly:
            continue
        if not include_local and getattr(prop, "subtype", "NONE") in {
                "FILE_PATH", "DIR_PATH", "FILE_NAME"}:
            continue
        if name in _LOCAL and not include_local:
            continue
        value = getattr(props, name)
        if isinstance(value, (bool, int, float, str)):
            out[name] = value
        else:
            try:
                out[name] = list(value)
            except TypeError:
                continue
    return out


def restore(props, values, report=None):
    applied, skipped = 0, []
    for name, value in values.items():
        if not hasattr(props, name):
            skipped.append(name)
            continue
        try:
            setattr(props, name, value)
            applied += 1
        except (TypeError, ValueError):
            skipped.append(name)
    if report is not None and skipped:
        report({"INFO"}, "Skipped %d unknown settings: %s"
               % (len(skipped), ", ".join(sorted(skipped)[:5])))
    return applied, skipped

def write(obj, props, version=(1, 0, 0)):
    obj[SETTINGS_KEY] = json.dumps(snapshot(props), sort_keys=True)
    obj[VERSION_KEY] = "%d.%d.%d" % version

def read(obj):
    raw = obj.get(SETTINGS_KEY) if obj is not None else None
    if not raw:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None


def as_preset_source(props, name="MY_LOOK"):
    values = snapshot(props, include_local=False)
    lines = ['    "%s": {' % name]
    for key in sorted(values):
        default = getattr(props.bl_rna.properties[key], "default", None)
        if default is not None and values[key] == default:
            continue
        lines.append("        %r: %r," % (key, values[key]))
    lines.append("    },")
    return "\n".join(lines)