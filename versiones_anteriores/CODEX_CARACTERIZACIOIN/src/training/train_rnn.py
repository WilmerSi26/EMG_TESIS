"""
Entrenamiento de RNN/LSTM para clasificar movimientos EMG.

Lee CSV de data/raw, crea ventanas temporales de 4 canales y entrena un modelo
Keras. Las etiquetas usadas por defecto separan apertura y cierre de cada dedo,
ademas de cierre de mano, apertura de mano y pinza.
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


def validate_tensorflow_python_version() -> None:
    if sys.version_info >= (3, 13):
        version = ".".join(str(part) for part in sys.version_info[:3])
        raise RuntimeError(
            "TensorFlow no esta disponible para este entorno de Python. "
            f"Version actual: Python {version}. "
            "Crea un entorno de entrenamiento con Python 3.11 o 3.12 e instala "
            "requirements-training.txt."
        )


def normalize_label(label: object) -> str | None:
    if not isinstance(label, str):
        return None
    key = label.strip().lower()
    return LABEL_ALIASES.get(key)


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
    dataset = dataset.dropna(subset=RAW_CHANNELS)
    return dataset


def segment_groups(dataset: pd.DataFrame):
    for source_file, file_frame in dataset.groupby("source_file", sort=False):
        file_frame = file_frame.reset_index(drop=True)
        change = file_frame["target_label"].ne(file_frame["target_label"].shift()).cumsum()
        for segment_id, segment in file_frame.groupby(change, sort=False):
            label = str(segment["target_label"].iloc[0])
            yield source_file, int(segment_id), label, segment


def build_windows(dataset: pd.DataFrame, window_size: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    labels: list[str] = []

    for _source_file, _segment_id, label, segment in segment_groups(dataset):
        values = segment[RAW_CHANNELS].to_numpy(dtype=np.float32)
        if len(values) < window_size:
            continue
        for start in range(0, len(values) - window_size + 1, stride):
            windows.append(values[start : start + window_size])
            labels.append(label)

    if not windows:
        raise RuntimeError(
            "No se pudieron crear ventanas. Baja --window-size o adquiere sesiones mas largas."
        )

    return np.stack(windows).astype(np.float32), np.array(labels)


def scale_windows(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_shape = x_train.shape
    test_shape = x_test.shape
    x_train_2d = x_train.reshape(-1, train_shape[-1])
    x_test_2d = x_test.reshape(-1, test_shape[-1])
    scaler.fit(x_train_2d)
    x_train_scaled = scaler.transform(x_train_2d).reshape(train_shape).astype(np.float32)
    x_test_scaled = scaler.transform(x_test_2d).reshape(test_shape).astype(np.float32)
    return x_train_scaled, x_test_scaled, scaler


def make_model(window_size: int, channels: int, classes: int, units: int, learning_rate: float):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, channels)),
            tf.keras.layers.LSTM(units, return_sequences=False),
            tf.keras.layers.Dropout(0.25),
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
    try:
        tflite_model = converter.convert()
    except Exception as exc:
        warning_path = output_path.with_suffix(".conversion_warning.txt")
        warning_path.write_text(
            "No se pudo convertir automaticamente la RNN/LSTM a TFLite.\n\n"
            "El modelo Keras si fue guardado y puede usarse en PC.\n"
            "Para XIAO ESP32S3/TensorFlow Lite Micro conviene entrenar un modelo "
            "mas compacto, por ejemplo MLP o CNN 1D sobre ventanas/caracteristicas.\n\n"
            f"Detalle tecnico:\n{exc}\n",
            encoding="utf-8",
        )
        print(f"Advertencia: no se genero TFLite. Detalle en {warning_path}")
        return
    output_path.write_bytes(tflite_model)


def plot_history(history, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history.get("loss", []), label="train")
    axes[0].plot(history.history.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(history.history.get("accuracy", []), label="train")
    axes[1].plot(history.history.get("val_accuracy", []), label="val")
    axes[1].set_title("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    figure, axis = plt.subplots(figsize=(6, 5))
    display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar RNN para movimientos EMG.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "results" / "figures")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "results" / "metrics")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--units", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--include-glob", type=str, default=None)
    parser.add_argument("--exclude-glob", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_tensorflow_python_version()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.input_dir, args.include_glob, args.exclude_glob)
    source_files = sorted(str(name) for name in dataset["source_file"].unique())
    x, labels_text = build_windows(dataset, args.window_size, args.stride)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels_text)
    labels = list(label_encoder.classes_)
    class_counts = pd.Series(labels_text).value_counts().to_dict()

    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify,
    )
    x_train, x_test, scaler = scale_windows(x_train, x_test)
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight = {int(label): float(weight) for label, weight in zip(np.unique(y_train), class_weight_values)}

    model = make_model(
        window_size=args.window_size,
        channels=len(RAW_CHANNELS),
        classes=len(labels),
        units=args.units,
        learning_rate=args.learning_rate,
    )

    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
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

    model_path = args.models_dir / "emg_rnn.keras"
    tflite_path = args.models_dir / "emg_rnn.tflite"
    scaler_path = args.models_dir / "scaler.joblib"
    encoder_path = args.models_dir / "label_encoder.joblib"
    metadata_path = args.models_dir / "metadata.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": "LSTM",
        "window_size": args.window_size,
        "stride": args.stride,
        "channels": RAW_CHANNELS,
        "labels": labels,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "units": args.units,
        "learning_rate": args.learning_rate,
        "test_size": args.test_size,
        "class_counts_windows": class_counts,
        "class_weight": class_weight,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "input_dir": str(args.input_dir),
        "include_glob": args.include_glob,
        "exclude_glob": args.exclude_glob,
        "source_file_count": len(source_files),
        "source_files": source_files,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = classification_report(y_test, y_pred, target_names=labels)
    (args.metrics_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_history(history, args.figures_dir / "training_history.png")
    plot_confusion(y_test, y_pred, labels, args.figures_dir / "confusion_matrix.png")
    save_tflite(model, tflite_path)

    print("Entrenamiento terminado.")
    print(f"Ventanas: {len(x)} | Clases: {labels}")
    print(f"Exactitud test: {test_accuracy:.4f}")
    print(f"Modelo Keras: {model_path}")
    print(f"Modelo TFLite: {tflite_path}")


if __name__ == "__main__":
    main()
