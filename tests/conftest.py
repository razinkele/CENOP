# Ensure the local `src` package dir is available during tests
# This helps pytest discover `cenop` when running tests without an editable install.
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)
