"""
Arboris - Paths & Config (Final Version)

Purpose:
- Central configuration
- Dataset + outputs + training settings
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

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 3

DATASET_STAGES = [0.001, 0.01, 0.1, 1.0]

TAXONOMY_LEVELS = [
    "kingdom", "phylum", "class",
    "order", "family", "genus", "species"
]

def create_dirs():
    for p in [DATA_ROOT, OUTPUT_DIR, CHECKPOINT_DIR]:
        os.makedirs(p, exist_ok=True)