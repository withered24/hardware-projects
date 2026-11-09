import sys
from pathlib import Path

# tests import the software models (ewformat, golden) from sw/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sw"))
