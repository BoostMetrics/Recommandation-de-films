from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

DATA_RAW_IMDB      = DATA_RAW / "imdb"
DATA_RAW_MOVIELENS = DATA_RAW / "movielens"