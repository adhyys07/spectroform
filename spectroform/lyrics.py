import re
import bpy

LINE = re.compile(r"\[(\d+):(\d+)(?:[.:](\d+))?\]\s*(.*)")

def read_lrc(path):
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            for m in LINE.finditer(raw):
                mm, ss, frac, text = m.groups()
                t = int(mm) * 60 + int(ss)
                if frac:
                    t += int(frac) / (100.0 if len(frac) == 2 else 1000.0)
                out.append((t, text.strip()))
    return sorted(out)

def build_lyrics(context, lines, fps, frame_start, size=1.0):
    coll = context.collection
    objs = []
    for i, (t, text) in enumerate(lines):
        if not text:
            continue
        curve = bpy.data.curves.new("SF_Lyric_%03d" % i, type="FONT")
        curve.body = text
        curve.align_x = "CENTER"
        curve.align_y = "CENTER"
        curve.size = size
        obj = bpy.data.objects.new("SF_Lyric_%03d" % i, curve)
        coll.objects.link(obj)

        start = int(frame_start + t * fps)
        nxt = lines[i + 1][0] if i + 1 < len(lines) else t + 4.0
        end = int(frame_start + nxt * fps)

        for frame, shown in ((start - 1, True), (start, False),
                             (end - 1, False), (end, True)):
            obj.hide_viewport = shown
            obj.hide_render = shown
            obj.keyframe_insert("hide_viewport", frame=frame)
            obj.keyframe_insert("hide_render", frame=frame)
        objs.append(obj)
    return objs