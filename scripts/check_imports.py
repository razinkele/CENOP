import importlib, pkgutil

def try_import(name):
    try:
        m = importlib.import_module(name)
        print(f"{name} OK -> {getattr(m, '__file__', None)}")
    except Exception as e:
        print(f"{name} ERROR: {type(e).__name__}: {e}")

try_import('cenop')
try_import('cenop.agents')
try_import('cenop.core.simulation')
try_import('cenop.landscape.loader')
try_import('cenop.ui.tabs.settings')
try_import('server.main')
