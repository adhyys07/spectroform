"""Let the pure-numpy modules import without Blender.

`spectroform/__init__.py` imports bpy, so importing the package normally fails
outside Blender. Registering a stub package with the right __path__ lets
`from spectroform import analysis` load analysis.py directly, and lets the
relative imports inside it resolve, without __init__.py ever running.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "spectroform"

if "spectroform" not in sys.modules:
    stub = types.ModuleType("spectroform")
    stub.__path__ = [str(PKG)]
    stub.__doc__ = "Test stub: real __init__ skipped so bpy is not needed."
    sys.modules["spectroform"] = stub

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
