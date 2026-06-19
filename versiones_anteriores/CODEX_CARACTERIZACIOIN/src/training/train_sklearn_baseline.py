"""
Entrenamiento base sin TensorFlow para validar sesiones EMG.

Este script usa scikit-learn y puede ejecutarse en el entorno actual aunque
TensorFlow no este disponible. No es una RNN, pero sirve para comprobar si las
10 sesiones tienen etiquetas y senales separables antes de crear el entorno de
entrenamiento profundo.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
        valid_labels = ", ".join(sorted(set(LABEL_ALIASES.values())))
        raise RuntimeError(f"No hay muestras con etiquetas validas: {valid_labels}.")

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


def extract_features(window: np.ndarray) -> np.ndarray:
    features = []
    for channel in range(window.shape[1]):
        values = window[:, channel].astype(np.float64)
        diff = np.diff(values)
        features.extend(
            [
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
                np.mean(np.abs(values)),
                np.sqrt(np.mean(values**2)),
                np.sum(np.abs(diff)),
            ]
        )
    return np.array(features, dtype=np.float32)


def build_feature_windows(dataset: pd.DataFrame, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    x_rows = []
    y_rows = []
    for _source_file, _segment_id, label, segment in segment_groups(dataset):
        values = segment[RAW_CHANNELS].to_numpy(dtype=np.float32)
        if len(values) < window_size:
            continue
        for start in range(0, len(values) - window_size + 1, stride):
            window = values[start : start + window_size]
            x_rows.append(extract_features(window))
            y_rows.append(label)

    if not x_rows:
        raise RuntimeError("No se pudieron crear ventanas. Baja --window-size o adquiere sesiones mas largas.")
    return np.vstack(x_rows), np.array(y_rows)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    figure, axis = plt.subplots(figsize=(8, 7))
    display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar clasificador base EMG sin TensorFlow.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "results" / "figures")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "results" / "metrics")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--trees", type=int, default=250)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--include-glob", type=str, default=None)
    parser.add_argument("--exclude-glob", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.input_dir, args.include_glob, args.exclude_glob)
    source_files = sorted(str(name) for name in dataset["source_file"].unique())
    x, labels_text = build_feature_windows(dataset, args.window_size, args.stride)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_text)
    labels = [str(label) for label in encoder.classes_]

    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = RandomForestClassifier(
        n_estimators=args.trees,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    accuracy = float(model.score(x_test, y_test))
    y_pred = model.predict(x_test)

    joblib.dump(model, args.models_dir / "emg_sklearn_baseline.joblib")
    joblib.dump(scaler, args.models_dir / "sklearn_feature_scaler.joblib")
    joblib.dump(encoder, args.models_dir / "sklearn_label_encoder.joblib")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": "RandomForest baseline",
        "window_size": args.window_size,
        "stride": args.stride,
        "channels": RAW_CHANNELS,
        "labels": labels,
        "trees": args.trees,
        "test_size": args.test_size,
        "test_accuracy": accuracy,
        "input_dir": str(args.input_dir),
        "include_glob": args.include_glob,
        "exclude_glob": args.exclude_glob,
        "source_file_count": len(source_files),
        "source_files": source_files,
    }
    (args.models_dir / "sklearn_baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    report = classification_report(y_test, y_pred, target_names=labels)
    (args.metrics_dir / "sklearn_baseline_report.txt").write_text(report, encoding="utf-8")
    plot_confusion(y_test, y_pred, labels, args.figures_dir / "sklearn_baseline_confusion_matrix.png")

    print("Entrenamiento base terminado.")
    print(f"Ventanas: {len(x)} | Clases: {labels}")
    print(f"Exactitud test: {accuracy:.4f}")
    print(f"Modelo: {args.models_dir / 'emg_sklearn_baseline.joblib'}")


if __name__ == "__main__":
    main()
