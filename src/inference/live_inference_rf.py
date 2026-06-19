"""Live EMG inference on the computer using the trained RandomForest model."""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import serial
from serial.tools import list_ports


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CHANNELS = ["emg_ch1", "emg_ch2", "emg_ch3", "emg_ch4"]


def list_serial_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def choose_serial_port() -> str:
    ports = list_serial_ports()
    if not ports:
        raise RuntimeError("No se detectaron puertos COM.")
    if len(ports) == 1:
        return ports[0]
    print("Puertos disponibles:")
    for index, port in enumerate(ports, start=1):
        print(f"{index}. {port}")
    selected = int(input("Selecciona puerto: ").strip())
    return ports[selected - 1]


def parse_device_line(line: str) -> list[int] | None:
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("timestamp_ms"):
        return None
    parts = line.split(",")
    if len(parts) != 8:
        return None
    try:
        return [int(parts[index]) for index in range(2, 6)]
    except ValueError:
        return None


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
    return np.array(features, dtype=np.float32).reshape(1, -1)


def load_artifacts(model_dir: Path):
    baseline_model = model_dir / "emg_sklearn_baseline.joblib"
    baseline_scaler = model_dir / "sklearn_feature_scaler.joblib"
    baseline_encoder = model_dir / "sklearn_label_encoder.joblib"
    rf_model = model_dir / "emg_rf_final.joblib"
    rf_scaler = model_dir / "rf_feature_scaler.joblib"
    rf_encoder = model_dir / "rf_label_encoder.joblib"

    if baseline_model.exists():
        return joblib.load(baseline_model), joblib.load(baseline_scaler), joblib.load(baseline_encoder)
    if rf_model.exists():
        return joblib.load(rf_model), joblib.load(rf_scaler), joblib.load(rf_encoder)
    raise RuntimeError(f"No se encontraron artefactos RandomForest en {model_dir}")


def create_plot(plot_window: int):
    figure, axis = plt.subplots(figsize=(11, 5))
    buffers = [deque([0] * plot_window, maxlen=plot_window) for _ in range(4)]
    colors = ["#2563eb", "#f97316", "#16a34a", "#dc2626"]
    offset = 4500
    x_values = list(range(plot_window))
    lines = []
    for channel in range(4):
        (line,) = axis.plot(x_values, list(buffers[channel]), color=colors[channel], linewidth=0.9, label=f"CH{channel + 1}")
        lines.append(line)
    axis.set_xlim(0, plot_window)
    axis.set_ylim(0, offset * 4)
    axis.set_xlabel("Muestras recientes")
    axis.set_ylabel("ADC raw + offset")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")
    plt.ion()
    plt.tight_layout()
    return figure, axis, lines, buffers, offset


def update_plot(axis, lines, buffers, values: list[int], offset: int, label: str, confidence: float) -> None:
    for channel, value in enumerate(values):
        buffers[channel].append(value + offset * channel)
        lines[channel].set_ydata(list(buffers[channel]))
    axis.set_title(f"Inferencia PC: {label} | confianza {confidence:.2f}")
    plt.pause(0.001)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia EMG en vivo en computador.")
    parser.add_argument("--port", help="Puerto serial, por ejemplo COM5.")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "baseline_final")
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--smooth", type=int, default=5, help="Numero de predicciones para voto mayoritario.")
    parser.add_argument("--plot-window", type=int, default=400)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_ports:
        for port in list_serial_ports():
            print(port)
        return
    port = args.port or choose_serial_port()
    model, scaler, encoder = load_artifacts(args.model_dir)

    serial_port = serial.Serial(port=port, baudrate=args.baudrate, timeout=0.02)
    time.sleep(1.5)
    serial_port.reset_input_buffer()

    sample_buffer: deque[list[int]] = deque(maxlen=args.window_size)
    prediction_buffer: deque[str] = deque(maxlen=args.smooth)
    step_counter = 0
    latest_label = "esperando"
    latest_confidence = 0.0
    plot_objects = None if args.no_plot else create_plot(args.plot_window)

    print(f"Inferencia en vivo usando {args.model_dir}")
    print(f"Puerto: {port} | ventana: {args.window_size} muestras | paso: {args.step_size} muestras")
    print("Ctrl+C para detener.")

    try:
        with serial_port:
            while True:
                line = serial_port.readline().decode("utf-8", errors="replace")
                values = parse_device_line(line)
                if values is None:
                    continue
                sample_buffer.append(values)
                step_counter += 1

                if len(sample_buffer) == args.window_size and step_counter >= args.step_size:
                    step_counter = 0
                    window = np.array(sample_buffer, dtype=np.float32)
                    features = extract_features(window)
                    features_scaled = scaler.transform(features)
                    probabilities = model.predict_proba(features_scaled)[0]
                    predicted_index = int(np.argmax(probabilities))
                    raw_label = str(encoder.inverse_transform([predicted_index])[0])
                    prediction_buffer.append(raw_label)
                    latest_label = Counter(prediction_buffer).most_common(1)[0][0]
                    latest_confidence = float(probabilities[predicted_index])
                    print(f"{latest_label:26s} confianza={latest_confidence:.2f}", end="\r")

                if plot_objects is not None:
                    _figure, axis, lines, buffers, offset = plot_objects
                    update_plot(axis, lines, buffers, values, offset, latest_label, latest_confidence)
    except KeyboardInterrupt:
        print("\nInferencia detenida.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
