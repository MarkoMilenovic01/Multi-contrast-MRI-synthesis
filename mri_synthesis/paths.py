"""Stable project paths shared by commands and local MLflow storage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT/'configs'/'quick_experiments.json'
