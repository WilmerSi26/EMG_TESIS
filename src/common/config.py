"""Central configuration for final EMG characterization and training."""

from __future__ import annotations

from pathlib import Path

from src.common.labels import FINAL_MOVEMENT_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PROJECT_ROOT = Path(r"D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN")

RAW_CHANNELS = ["emg_ch1", "emg_ch2", "emg_ch3", "emg_ch4"]
NUM_CHANNELS = 4

SAMPLING_RATE_HZ = 200
WINDOW_SIZE = 40
STRIDE = 10

FINAL_CLASSES = FINAL_MOVEMENT_CLASSES

PILOT_SOURCE_GLOB = "20260609_*_S001_trial-1_rutina_4ch.csv"
PILOT_SOURCE_DIR = SOURCE_PROJECT_ROOT / "data" / "raw"
PILOT_REMAP_DIR = PROJECT_ROOT / "data" / "pilot_remap_14_to_10"
FINAL_DATA_DIR = PROJECT_ROOT / "data" / "final_10clases"

