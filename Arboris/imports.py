"""
Arboris - Central Imports (Final Version)

Purpose:
- Unified imports
- Device + reproducibility setup
"""

import os
import json
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd

from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns

from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)