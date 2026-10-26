import os
import re 
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = "spectroform"
PKG_DIR = os.path.join(HERE, PKG)
DIST = os.path.join(HERE, "dist")

EXCLUDE_DIRS = {"__pycache__", "tests", ".git"}
EXCLUDE_SUFFIX = (".pyc", ".pyo", ".zip", ".DS_Store")

def read_version(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as fh:
        text = fh.read()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit("No version field in blender_manifest.toml")
    return match.group(1)

def collect():
    for root, dirs, files in os.walk(PKG_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in sorted(files):
            if name.endswith(EXCLUDE_SUFFIX):
                continue
            abs_path = os.path.join(root, name)
            yield abs_path, os.path.relpath(abs_path, PKG_DIR)

def write_zip(out_path, members, prefix=""):
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_path, rel in members:
            zf.write(abs_path, os.path.join(prefix, rel) if prefix else rel)
    return out_path

def main():
    manifest = os.path.join(PKG_DIR, "blender_manifest.toml")
    if not os.path.isfile(manifest):
        raise SystemExit("blender_manifest.toml not found next to the package.")

    version = read_version(manifest)
    members = list(collect())
    if not any(rel == "__init__.py" for _, rel in members):
        raise SystemExit("__init__.py missing from the package.")

    os.makedirs(DIST, exist_ok=True)

    ext_zip = write_zip(
        os.path.join(DIST, "%s-%s.zip" % (PKG, version)), members
    )
    legacy_zip = write_zip(
        os.path.join(DIST, "%s-%s-legacy.zip" % (PKG, version)), members, prefix=PKG
    )

    print("Packaged %d files (version %s)" % (len(members), version))
    print("  extension (4.2+)   : %s" % os.path.relpath(ext_zip, HERE))
    print("  legacy add-on (3.6): %s" % os.path.relpath(legacy_zip, HERE))
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())