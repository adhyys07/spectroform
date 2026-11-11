import json
import os

import numpy as np

_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>%(title)s</title>
<style>html,body{margin:0;height:100%%;background:#06080f;overflow:hidden}
canvas{display:block;width:100%%;height:100%%}</style></head>
<body><canvas id=c></canvas><audio id=a src="%(audio)s"></audio>
<script>
const DATA = %(data)s;
const cv = document.getElementById('c'), gl = cv.getContext('webgl');
const VS = `
precision highp float;
attribute vec2 uv;
uniform float amp[%(n)d];
uniform float cyc[%(n)d];
uniform mat4 mvp;
varying float h;
void main(){
  float r = length(uv);
  float z = 0.0;
  for (int i = 0; i < %(n)d; i++) {
    z += amp[i] * sin(r * cyc[i] * 6.2831853);
  }
  h = z;
  gl_Position = mvp * vec4(uv.x, uv.y, z, 1.0);
}`;
const FS = `
precision highp float; varying float h;
void main(){ gl_FragColor = vec4(0.1+h, 0.45+h*0.5, 1.0, 1.0); }`;
// ... compile, build a grid, and on each frame read DATA.frames[i] into amp[]
</script></body></html>
"""


def export_web(path, sin_curves, cos_curves, cycles, audio_name="", title="Spectroform"):
    sin_c = np.asarray(sin_curves, dtype=np.float32)
    cos_c = np.asarray(cos_curves, dtype=np.float32)
    amp = np.sqrt(sin_c ** 2 + cos_c ** 2)
    phase = np.arctan2(-cos_c, sin_c)

    payload = {
        "fps": 24,
        "cycles": [round(float(c), 4) for c in np.asarray(cycles).ravel()],
        "amp": [[round(float(v), 3) for v in row] for row in amp],
        "phase": [[round(float(v), 3) for v in row] for row in phase],
    }
    data = json.dumps(payload, separators=(",", ":"))

    html = _HTML % {"title": title, "audio": audio_name,
                    "data": data, "n": sin_c.shape[1]}
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.getsize(path)

    