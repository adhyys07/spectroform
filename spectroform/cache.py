import hashlib
import os

_STORE, _ORDER = {},[]
_MAX_ENTRIES = 3
_STATS = {"hits" : 0, "misses": 0}

def _file_identity(path):
    try:
        st = os.stat(path)
    except OSError:
        return "missing"
    return "%d:%d" % (st.st_size, int(st.st_mtime))


def key_for(path, kind, **params):
    parts = [os.path.normcase(os.path.abspath(path)), _file_identity(path), kind]
    parts += ["%s=%r" % kv for kv in sorted(params.items())]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

def memo(path, kind, builder, **params):
    k = key_for(path, kind, **params)
    if k in _STORE:
        _STATS["hits"] += 1
        hit = _STORE[k]
        return (dict(hit) if isinstance(hit, dict) else hit), True

    _STATS["misses"] += 1
    value = builder()
    stored = dict(value) if isinstance(value, dict) else value
    if k not in _STORE and len(_ORDER) >= _MAX_ENTRIES:
        _STORE.pop(_ORDER.pop(0), None)
    if k not in _STORE:
        _ORDER.append(k)
    _STORE[k] = stored
    return (dict(stored) if isinstance(stored, dict) else stored), False

def clear():
    _STORE.clear()
    del _ORDER[:]
    _STATS.update(hits=0, misses=0)

def info():
    return "%d entries, %d hits / %d misses" % (
        len(_STORE), _STATS["hits"], _STATS["misses"])