"""
Entrena un modelo pequeno compatible con TFLite para XIAO ESP32S3.

Usa caracteristicas por ventana, no una RNN. Es una alternativa embebible para
TensorFlow Lite Micro cuando LSTM/RNN no convierte bien o consume demasiada RAM.
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
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight


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
            yield source_file, int(segment_id), str(segment["target_label"].iloc[0]), segment


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
            x_rows.append(extract_features(values[start : start + window_size]))
            y_rows.append(label)
    if not x_rows:
        raise RuntimeError("No se pudieron crear ventanas.")
    return np.vstack(x_rows), np.array(y_rows)


def make_model(input_dim: int, classes: int, learning_rate: float):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_tflite(model, output_path: Path) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    output_path.write_bytes(converter.convert())


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
    parser = argparse.ArgumentParser(description="Entrenar MLP pequeno TFLite para XIAO ESP32S3.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "results" / "figures")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "results" / "metrics")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
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
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight = {int(label): float(weight) for label, weight in zip(np.unique(y_train), class_weight_values)}

    model = make_model(x_train.shape[1], len(labels), args.learning_rate)
    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    model.save(args.models_dir / "emg_tiny_mlp.keras")
    save_tflite(model, args.models_dir / "emg_tiny_mlp.tflite")
    joblib.dump(scaler, args.models_dir / "tiny_feature_scaler.joblib")
    joblib.dump(encoder, args.models_dir / "tiny_label_encoder.joblib")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": "Tiny MLP feature classifier",
        "window_size": args.window_size,
        "stride": args.stride,
        "feature_count": int(x_train.shape[1]),
        "labels": labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "test_size": args.test_size,
        "class_weight": class_weight,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "input_dir": str(args.input_dir),
        "include_glob": args.include_glob,
        "exclude_glob": args.exclude_glob,
        "source_file_count": len(source_files),
        "source_files": source_files,
    }
    (args.models_dir / "tiny_mlp_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = classification_report(y_test, y_pred, target_names=labels)
    (args.metrics_dir / "tiny_mlp_report.txt").write_text(report, encoding="utf-8")
    plot_confusion(y_test, y_pred, labels, args.figures_dir / "tiny_mlp_confusion_matrix.png")

    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(history.history.get("loss", []), label="train")
    axis.plot(history.history.get("val_loss", []), label="val")
    axis.grid(True, alpha=0.3)
    axis.legend()
    axis.set_title("Tiny MLP loss")
    figure.tight_layout()
    figure.savefig(args.figures_dir / "tiny_mlp_history.png", dpi=160)
    plt.close(figure)

    print("Entrenamiento Tiny MLP terminado.")
    print(f"Ventanas: {len(x)} | Clases: {labels}")
    print(f"Exactitud test: {test_accuracy:.4f}")
    print(f"TFLite: {args.models_dir / 'emg_tiny_mlp.tflite'}")


if __name__ == "__main__":
    main()
