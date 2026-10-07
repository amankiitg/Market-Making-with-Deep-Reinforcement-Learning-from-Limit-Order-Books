"""Make the package importable when pytest is run as `pytest` rather than `python -m pytest`.

`python -m pytest` puts the working directory on the path; a bare `pytest` does not, and
pytest's own rootdir insertion stops at the first directory without an __init__.py, which is
this one. Adding the repository root here means README commands work either way.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
