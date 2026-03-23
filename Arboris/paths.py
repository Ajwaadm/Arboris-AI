"""
Arboris - Path Configuration (Version 2)

Purpose:
- Manage dataset + outputs + checkpoints
"""

from imports import *
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_ROOT = BASE_DIR / "datasets"
INAT21_PATH = DATA_ROOT / "iNat21"

TRAIN_DIR = INAT21_PATH / "train"
TRAIN_JSON = INAT21_PATH / "train.json"

OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_FILE = OUTPUT_DIR / "training_log.csv"

def create_dirs():
    for p in [DATA_ROOT, OUTPUT_DIR, CHECKPOINT_DIR]:
        os.makedirs(p, exist_ok=True)