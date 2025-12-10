from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA = BASE_DIR / "data"
HEA = DATA / "healthy"
IMP = DATA / "impaired"

REP = BASE_DIR / "reports"