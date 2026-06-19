"""
Simulacion grafica de inferencia EMG.

Puede reproducir un CSV guardado o leer senales en tiempo real desde el puerto
serial. Usa el modelo entrenado por src/training/train_rnn.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from serial.tools import list_ports


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CHANNELS = ["emg_ch1", "emg_ch2", "emg_ch3", "emg_ch4"]


def parse_device_line(line: str) -> dict[str, str] | None:
    line = line.strip()
    if not line or line.startswith("timestamp_ms"):
        return None
    parts = line.split(",")
    if len(parts) != 8:
        return None
    timestamp_ms, sample_index, ch1, ch2, ch3, ch4, board_id, sampling_rate_hz = parts
    try:
        int(timestamp_ms)
        int(sample_index)
        int(ch1)
        int(ch2)
        int(ch3)
        int(ch4)
        int(sampling_rate_hz)
    except ValueError:
        return None
    return {
        "emg_ch1": ch1,
        "emg_ch2": ch2,
        "emg_ch3": ch3,
        "emg_ch4": ch4,
        "sampling_rate_hz": sampling_rate_hz,
        "board_id": board_id,
    }


def list_serial_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def load_artifacts(models_dir: Path):
    import tensorflow as tf

    model_path = models_dir / "emg_rnn.keras"
    scaler_path = models_dir / "scaler.joblib"
    encoder_path = models_dir / "label_encoder.joblib"
    metadata_path = models_dir / "metadata.json"

    missing = [path for path in [model_path, scaler_path, encoder_path, metadata_path] if not path.exists()]
    if missing:
        raise RuntimeError("Faltan artefactos de entrenamiento: " + ", ".join(str(path) for path in missing))

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, scaler, label_encoder, metadata


def create_plot(labels: list[str], plot_window: int):
    figure, (axis_signal, axis_prob) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [2, 1]})
    colors = ["#2563eb", "#f97316", "#16a34a", "#dc2626"]
    offset = 4500
    signal_buffers = [deque([0] * plot_window, maxlen=plot_window) for _ in RAW_CHANNELS]
    x_values = list(range(plot_window))
    signal_lines = []
    for channel, color in enumerate(colors):
        (line,) = axis_signal.plot(x_values, list(signal_buffers[channel]), color=color, linewidth=0.8, label=f"CH{channel + 1}")
        signal_lines.append(line)

    axis_signal.set_title("Senales EMG recientes")
    axis_signal.set_xlim(0, plot_window)
    axis_signal.set_ylim(0, offset * len(RAW_CHANNELS))
    axis_signal.grid(True, alpha=0.3)
    axis_signal.legend(loc="upper right")

    bars = axis_prob.bar(labels, [0.0] * len(labels), color="#2563eb")
    axis_prob.set_ylim(0, 1)
    axis_prob.set_ylabel("Probabilidad")
    axis_prob.grid(True, axis="y", alpha=0.3)
    title = axis_prob.set_title("Esperando ventana completa...")
    figure.tight_layout()
    plt.ion()
    return figure, signal_lines, signal_buffers, bars, title, offset


def predict_window(model, scaler, label_encoder, window: np.ndarray) -> tuple[str, np.ndarray]:
    window_scaled = scaler.transform(window.reshape(-1, window.shape[-1])).reshape(1, *window.shape)
    probabilities = model.predict(window_scaled, verbose=0)[0]
    label = str(label_encoder.inverse_transform([int(np.argmax(probabilities))])[0])
    return label, probabilities


def update_plot(signal_lines, signal_buffers, bars, title, values: list[int], offset: int, prediction: str | None, probabilities: np.ndarray | None) -> None:
    for channel, value in enumerate(values):
        signal_buffers[channel].append(value + (offset * channel))
        signal_lines[channel].set_ydata(list(signal_buffers[channel]))

    if prediction is not None and probabilities is not None:
        for bar, probability in zip(bars, probabilities):
            bar.set_height(float(probability))
        confidence = float(np.max(probabilities))
        title.set_text(f"Prediccion: {prediction} | confianza: {confidence:.2f}")
    plt.pause(0.001)


def iter_csv_samples(csv_file: Path, sample_delay: float | None):
    frame = pd.read_csv(csv_file)
    missing = [column for column in RAW_CHANNELS if column not in frame.columns]
    if missing:
        raise RuntimeError(f"El CSV no tiene columnas {missing}")
    for _index, row in frame.iterrows():
        values = [int(row[column]) for column in RAW_CHANNELS]
        if sample_delay:
            time.sleep(sample_delay)
        yield values


def iter_serial_samples(port: str, baudrate: int):
    try:
        serial_port = serial.Serial(port=port, baudrate=baudrate, timeout=0.02)
        time.sleep(1.5)
        serial_port.reset_input_buffer()
    except serial.SerialException as exc:
        ports = ", ".join(list_serial_ports()) or "ninguno"
        raise RuntimeError(f"No se pudo abrir {port}. Puertos disponibles: {ports}") from exc

    with serial_port:
        while True:
            line = serial_port.readline().decode("utf-8", errors="replace")
            parsed = parse_device_line(line)
            if parsed is None:
                continue
            yield [int(parsed[column]) for column in RAW_CHANNELS]


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia grafica EMG con RNN entrenada.")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--csv-file", type=Path, help="CSV para replay/simulacion.")
    parser.add_argument("--csv-real-time", action="store_true", help="Reproduce CSV a velocidad aproximada de muestreo.")
    parser.add_argument("--port", help="Puerto serial para lectura en vivo, por ejemplo COM5.")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--list-ports", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_ports:
        print("Puertos disponibles:")
        for port in list_serial_ports():
            print(f"- {port}")
        return

    if not args.csv_file and not args.port:
        print("Indica --csv-file para replay o --port para tiempo real.", file=sys.stderr)
        sys.exit(2)

    model, scaler, label_encoder, metadata = load_artifacts(args.models_dir)
    labels = list(label_encoder.classes_)
    window_size = int(metadata["window_size"])
    sample_window: deque[list[int]] = deque(maxlen=window_size)

    figure, signal_lines, signal_buffers, bars, title, offset = create_plot(labels, plot_window=window_size)

    if args.csv_file:
        sample_rate = 1000.0
        if args.csv_real_time:
            try:
                sample_rate = float(pd.read_csv(args.csv_file, nrows=1)["sampling_rate_hz"].iloc[0])
            except Exception:
                sample_rate = 1000.0
        delay = (1.0 / sample_rate) if args.csv_real_time else None
        samples = iter_csv_samples(args.csv_file, delay)
    else:
        samples = iter_serial_samples(args.port, args.baudrate)

    try:
        for values in samples:
            sample_window.append(values)
            prediction = None
            probabilities = None
            if len(sample_window) == window_size:
                window = np.array(sample_window, dtype=np.float32)
                prediction, probabilities = predict_window(model, scaler, label_encoder, window)

            update_plot(signal_lines, signal_buffers, bars, title, values, offset, prediction, probabilities)
            if not plt.fignum_exists(figure.number):
                break
    except KeyboardInterrupt:
        print("Simulacion detenida.")
    finally:
        plt.ioff()


if __name__ == "__main__":
    main()
