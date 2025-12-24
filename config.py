from pathlib import Path

BASEDIR = Path(__file__).resolve().parent

DATA = BASEDIR / "data"
HEA = DATA / "healthy"
IMP = DATA / "impaired"

REP = BASEDIR / "reports"