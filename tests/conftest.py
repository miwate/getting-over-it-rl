# Lets `pytest tests/...` import goi_jax from the repo root without installing it.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
