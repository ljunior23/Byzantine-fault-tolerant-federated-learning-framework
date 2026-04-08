import sys
import os

_ROOT = os.path.dirname(os.path.abspath(__file__))  # project root (where conftest.py lives)

for _candidate in [
    _ROOT,                          # flat layout: aggregators/ sits right here
    os.path.join(_ROOT, "src"),     # nested layout: aggregators/ is inside src/
]:
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
