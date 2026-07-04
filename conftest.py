"""Ensure the repo root is importable so `import ghostprint` works under a bare
`pytest` invocation (which, unlike `python -m pytest`, does not add the cwd to
sys.path)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
