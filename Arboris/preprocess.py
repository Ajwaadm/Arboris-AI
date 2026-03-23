"""
Arboris - Path Configuration

Purpose:
- Centralize all paths
- Make project portable
"""

from imports import *
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_ROOT = BASE_DIR / "datasets"
INAT21_PATH = DATA_ROOT / "iNat21"

TRAIN_DIR = INAT21_PATH / "train"
TRAIN_JSON = INAT21_PATH / "train.json"

OUTPUT_DIR = BASE_DIR / "outputs"

def create_dirs():
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)