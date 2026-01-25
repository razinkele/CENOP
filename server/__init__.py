"""
Top-level `server` package shim.

This package makes imports like `import server.renderers` work by
adding the actual `cenop/server` directory to this package search path.
"""
import os

_this_dir = os.path.dirname(__file__)
_real_server_dir = os.path.normpath(os.path.join(_this_dir, '..', 'cenop', 'server'))
if os.path.isdir(_real_server_dir) and _real_server_dir not in __path__:
    __path__.insert(0, _real_server_dir)

# Re-export commonly used attributes for convenience
try:
    from .main import server  # type: ignore
    from .reactive_state import SimulationState  # type: ignore
except Exception:
    # Fail silently during import-time checks; real errors will surface later
    pass
