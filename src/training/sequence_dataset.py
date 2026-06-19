"""Dataset utilities for sequence-based EMG models."""

from __future__ import annotations

import fnmatch
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.labels import LABEL_ALIASES


RAW_CHANNELS = ["emg_ch1", "emg_ch2", "emg_ch3", "emg_ch4"]


def normalize_label(label: object) -> str | None:
    if not isinstance(label, str):
        return None
    return LABEL_ALIASES.get(label.strip().lower())


def filter_csv_files(files: list[Path], include_glob: str | None, exclude_glob: list[str]) -> list[Path]:
    selected = files
    if include_glob:
        selected = [path for path in selected if fnmatch.fnmatch(path.name, include_glob)]
    for pattern in exclude_glob:
        selected = [path for path in selected if not fnmatch.fnmatch(path.name, pattern)]
    return selected


def load_dataset(input_dir: Path, include_glob: str | None = None, exclude_glob: list[str] | None = None) -> pd.DataFrame:
    files = sorted(input_dir.rglob("*.csv"))
    files = filter_csv_files(files, include_glob, exclude_glob or [])
    if not files:
        raise RuntimeError(f"No se encontraron CSV en {input_dir}")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        missing = [column for column in RAW_CHANNELS + ["movement_label"] if column not in frame.columns]
        if missing:
            print(f"Saltando {path.name}: faltan columnas {missing}")
            continue
        frame = frame.copy()
        frame["source_file"] = path.name
        frame["target_label"] = frame["movement_label"].map(normalize_label)
        frame = frame.dropna(subset=["target_label"])
        if not frame.empty:
            frames.append(frame)

    if not frames:
        raise RuntimeError("No hay muestras validas para entrenar.")

    dataset = pd.concat(frames, ignore_index=True)
    for channel in RAW_CHANNELS:
        dataset[channel] = pd.to_numeric(dataset[channel], errors="coerce")
    return dataset.dropna(subset=RAW_CHANNELS)


def segment_groups(dataset: pd.DataFrame):
    for source_file, file_frame in dataset.groupby("source_file", sort=False):
        file_frame = file_frame.reset_index(drop=True)
        change = file_frame["target_label"].ne(file_frame["target_label"].shift()).cumsum()
        for segment_id, segment in file_frame.groupby(change, sort=False):
            label = str(segment["target_label"].iloc[0])
            yield source_file, int(segment_id), label, segment


def build_sequence_windows(dataset: pd.DataFrame, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for _source_file, _segment_id, label, segment in segment_groups(dataset):
        values = segment[RAW_CHANNELS].to_numpy(dtype=np.float32)
        if len(values) < window_size:
            continue
        for start in range(0, len(values) - window_size + 1, stride):
            x_rows.append(values[start : start + window_size])
            y_rows.append(label)

    if not x_rows:
        raise RuntimeError("No se pudieron crear ventanas. Baja --window-size o adquiere sesiones mas largas.")
    return np.stack(x_rows).astype(np.float32), np.array(y_rows)


def standardize_sequence_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    x_train_scaled = (x_train - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    x_test_scaled = (x_test - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    stats = {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}
    return x_train_scaled.astype(np.float32), x_test_scaled.astype(np.float32), stats

