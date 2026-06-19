"""Train reference deep-learning models adapted from selected EMG GitHub repositories."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.sequence_dataset import build_sequence_windows, load_dataset, standardize_sequence_train_test


MODEL_DESCRIPTIONS = {
    "ocjorge_cnn_lstm": "TensorFlow CNN-LSTM adapted from ocjorge/CNN-LSTM",
    "iyeleswarapu_cnn1d": "TensorFlow CNN1D equivalent adapted from iyeleswarapu/emg-gesture-recognition",
    "laboratorio_inception_lstm": "Light Inception-CNN-LSTM inspired by laboratorioAI/2023-HGR5-CNN_LSTM",
}


def build_ocjorge_cnn_lstm(input_shape: tuple[int, int], classes: int, learning_rate: float):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, kernel_size=9, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=9, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LSTM(96, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_iyeleswarapu_cnn1d(input_shape: tuple[int, int], classes: int, learning_rate: float):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def inception_block_1d(inputs, filters: int):
    import tensorflow as tf

    b1 = tf.keras.layers.Conv1D(filters, kernel_size=1, padding="same", activation="relu")(inputs)
    b3 = tf.keras.layers.Conv1D(filters, kernel_size=3, padding="same", activation="relu")(inputs)
    b5 = tf.keras.layers.Conv1D(filters, kernel_size=5, padding="same", activation="relu")(inputs)
    bp = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(inputs)
    bp = tf.keras.layers.Conv1D(filters, kernel_size=1, padding="same", activation="relu")(bp)
    x = tf.keras.layers.Concatenate()([b1, b3, b5, bp])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def build_laboratorio_inception_lstm(input_shape: tuple[int, int], classes: int, learning_rate: float):
    import tensorflow as tf

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inception_block_1d(inputs, filters=12)
    x = tf.keras.layers.Dropout(0.20)(x)
    x = inception_block_1d(x, filters=16)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(model_name: str, input_shape: tuple[int, int], classes: int, learning_rate: float):
    if model_name == "ocjorge_cnn_lstm":
        return build_ocjorge_cnn_lstm(input_shape, classes, learning_rate)
    if model_name == "iyeleswarapu_cnn1d":
        return build_iyeleswarapu_cnn1d(input_shape, classes, learning_rate)
    if model_name == "laboratorio_inception_lstm":
        return build_laboratorio_inception_lstm(input_shape, classes, learning_rate)
    raise RuntimeError(f"Modelo no soportado: {model_name}")


def save_tflite(model, output_path: Path) -> str | None:
    import tensorflow as tf

    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        output_path.write_bytes(converter.convert())
        return None
    except Exception as exc:  # noqa: BLE001
        output_path.with_suffix(".conversion_warning.txt").write_text(str(exc), encoding="utf-8")
        return str(exc)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    figure, axis = plt.subplots(figsize=(9, 8))
    display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_history(history, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history.get("accuracy", []), label="train")
    axes[0].plot(history.history.get("val_accuracy", []), label="val")
    axes[0].set_title("Accuracy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(history.history.get("loss", []), label="train")
    axes[1].plot(history.history.get("val_loss", []), label="val")
    axes[1].set_title("Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar modelos profundos de referencia para EMG.")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_DESCRIPTIONS),
        required=True,
    )
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "pilot_remap_14_to_10_200hz")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "results" / "figures")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "results" / "metrics")
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--include-glob", type=str, default=None)
    parser.add_argument("--exclude-glob", action="append", default=[])
    parser.add_argument("--no-tflite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = args.models_dir / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.input_dir, args.include_glob, args.exclude_glob)
    source_files = sorted(str(name) for name in dataset["source_file"].unique())
    x, labels_text = build_sequence_windows(dataset, args.window_size, args.stride)

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
    x_train, x_test, normalization = standardize_sequence_train_test(x_train, x_test)

    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight = {int(label): float(weight) for label, weight in zip(np.unique(y_train), class_weight_values)}

    model = build_model(args.model, (args.window_size, x_train.shape[-1]), len(labels), args.learning_rate)

    import tensorflow as tf

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
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

    model.save(model_dir / f"{args.model}.keras")
    joblib.dump(encoder, model_dir / f"{args.model}_label_encoder.joblib")
    (model_dir / f"{args.model}_normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")

    tflite_warning = None
    if not args.no_tflite:
        tflite_warning = save_tflite(model, model_dir / f"{args.model}.tflite")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_name": args.model,
        "model_description": MODEL_DESCRIPTIONS[args.model],
        "window_size": args.window_size,
        "stride": args.stride,
        "channels": int(x_train.shape[-1]),
        "labels": labels,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history.history.get("loss", [])),
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
        "tflite_conversion_warning": tflite_warning,
    }
    (model_dir / f"{args.model}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = classification_report(y_test, y_pred, target_names=labels)
    (args.metrics_dir / f"{args.model}_report.txt").write_text(report, encoding="utf-8")
    plot_confusion(y_test, y_pred, labels, args.figures_dir / f"{args.model}_confusion_matrix.png")
    plot_history(history, args.figures_dir / f"{args.model}_history.png")

    print("Entrenamiento terminado.")
    print(f"Modelo: {args.model}")
    print(f"Ventanas: {len(x)} | Clases: {labels}")
    print(f"Exactitud test: {test_accuracy:.4f}")
    print(f"Directorio modelo: {model_dir}")


if __name__ == "__main__":
    main()

