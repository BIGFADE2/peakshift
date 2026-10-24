"""Dependency fallback: import from a local vendored folder when pip is unavailable.

Normally this does nothing -- if `pip install -r requirements.txt` succeeded,
`import scipy` just works. Only when dependencies sit in a `vendor/` folder or a
`.pylibs/` folder somewhere above the repo (offline bundle, restricted sandbox)
does it add them to sys.path.
"""
import os
import sys

_MAX_UP = 5


def _candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    yield os.path.join(here, "vendor")
    folder = here
    for _ in range(_MAX_UP):
        parent = os.path.dirname(folder)
        if not parent or parent == folder:
            break
        folder = parent
        yield os.path.join(folder, ".pylibs")
        yield os.path.join(folder, "vendor")


def ensure():
    for path in _candidates():
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
