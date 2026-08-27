"""Test configuration.

These are UNIT tests of code mechanics -- masking, aggregation arithmetic,
standardization, partitioning. The small arrays they use are hand-built fixtures
chosen to make an invariant checkable by hand; they are NOT data, and no test
result feeds any reported number. Everything reported in the paper comes from real
NHANES records only (see data/README.md).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RESULTS_JSON = ROOT / "results" / "results.json"
README = ROOT / "README.md"
