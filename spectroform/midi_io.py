import struct

def _vlq(data, i):
    value = 0
    while True:
        byte = data[i]
        i += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, i


def read_midi(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:4] != b"MThd":
        raise ValueError("Not a MIDI file")
    _hlen, _fmt, ntracks, division = struct.unpack(">IHHH", data[4:14])
    if division & 0x8000:
        raise ValueError("SMPTE timing is not supported")

    notes, tempo = [], 500000              
    pos = 14
    for _ in range(ntracks):
        if data[pos:pos + 4] != b"MTrk":
            break
        length = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        end = pos + 8 + length
        i = pos + 8
        tick, running, open_notes = 0, None, {}
        while i < end:
            delta, i = _vlq(data, i)
            tick += delta
            status = data[i]
            if status & 0x80:
                i += 1
                running = status          
            else:
                status = running
            if status is None:
                break

            kind, chan = status & 0xF0, status & 0x0F
            if kind in (0x80, 0x90):
                pitch, vel = data[i], data[i + 1]
                i += 2
                # A note-on with velocity 0 is a note-off.
                if kind == 0x90 and vel > 0:
                    open_notes[(chan, pitch)] = (tick, vel)
                else:
                    started = open_notes.pop((chan, pitch), None)
                    if started:
                        notes.append((started[0], tick, pitch, started[1], chan))
            elif kind in (0xA0, 0xB0, 0xE0):
                i += 2
            elif kind in (0xC0, 0xD0):
                i += 1
            elif status == 0xFF:
                meta = data[i]
                i += 1
                mlen, i = _vlq(data, i)
                if meta == 0x51 and mlen == 3:
                    tempo = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
                i += mlen
            elif status in (0xF0, 0xF7):
                mlen, i = _vlq(data, i)
                i += mlen
            else:
                break
        pos = end
    notes.sort()
    return notes, division, tempo


def midi_to_frames(notes, ticks_per_beat, tempo_us, fps):
    sec_per_tick = (tempo_us / 1e6) / float(ticks_per_beat)
    return [(s * sec_per_tick * fps, e * sec_per_tick * fps, p, v, c)
            for s, e, p, v, c in notes]


def midi_envelopes(frames_notes, n_frames, fps, channels=None, release=0.25):
    import numpy as np

    chans = sorted({c for *_rest, c in frames_notes}) if channels is None else channels
    out = {}
    decay = max(float(release), 1e-3) * fps
    for c in chans:
        env = np.zeros(n_frames, dtype=np.float32)
        for s, e, _p, v, ch in frames_notes:
            if ch != c:
                continue
            a, b = int(max(0, s)), int(min(n_frames, max(s + 1, e)))
            env[a:b] = np.maximum(env[a:b], v / 127.0)
            tail = np.arange(b, min(n_frames, b + int(decay * 3)))
            if tail.size:
                env[tail] = np.maximum(env[tail],
                                       (v / 127.0) * np.exp(-(tail - b) / decay))
        out[c] = env
    return out
