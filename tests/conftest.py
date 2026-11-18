import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLD = str(ROOT / "data/gold/v2/gold.jsonl")
BASELINE = str(ROOT / "data/predictions/baseline.jsonl")
REGRESSED = str(ROOT / "data/predictions/regressed.jsonl")
GATES = str(ROOT / "config/gates.yaml")
