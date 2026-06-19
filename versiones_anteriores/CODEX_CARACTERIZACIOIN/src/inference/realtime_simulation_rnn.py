"""Simulacion grafica con la RNN/LSTM entrenada en TensorFlow."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

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
    if not line or line.startswith("timestamp_ms") or line.startswith("#"):
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
        "board_id": board_id,
        "sampling_rate_hz": sampling_rate_hz,
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
        raise RuntimeError("Faltan artefactos RNN: " + ", ".join(str(path) for path in missing))
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, scaler, encoder, metadata


def create_plot(labels: list[str], plot_window: int):
    figure, (axis_signal, axis_prob) = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    colors = ["#2563eb", "#f97316", "#16a34a", "#dc2626"]
    offset = 4500
    signal_buffers = [deque([0] * plot_window, maxlen=plot_window) for _ in RAW_CHANNELS]
    x_values = list(range(plot_window))
    signal_lines = []
    for channel, color in enumerate(colors):
        (line,) = axis_signal.plot(
            x_values,
            list(signal_buffers[channel]),
            color=color,
            linewidth=0.8,
            label=f"CH{channel + 1}",
        )
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
    axis_prob.tick_params(axis="x", labelrotation=35)
    title = axis_prob.set_title("Esperando ventana completa...")
    figure.tight_layout()
    plt.ion()
    return figure, signal_lines, signal_buffers, bars, title, offset


def predict_window(model, scaler, encoder, window: np.ndarray) -> tuple[str, np.ndarray]:
    scaled = scaler.transform(window.reshape(-1, window.shape[-1])).reshape(1, *window.shape)
    probabilities = model.predict(scaled, verbose=0)[0]
    index = int(np.argmax(probabilities))
    label = str(encoder.inverse_transform([index])[0])
    return label, probabilities


def update_plot(signal_lines, signal_buffers, bars, title, values, offset, prediction, probabilities) -> None:
    for channel, value in enumerate(values):
        signal_buffers[channel].append(value + (offset * channel))
        signal_lines[channel].set_ydata(list(signal_buffers[channel]))
    if prediction is not None and probabilities is not None:
        for bar, probability in zip(bars, probabilities):
            bar.set_height(float(probability))
        title.set_text(f"Prediccion RNN: {prediction} | confianza: {float(np.max(probabilities)):.2f}")
    plt.pause(0.001)


def iter_csv_samples(csv_file: Path, sample_delay: float | None):
    frame = pd.read_csv(csv_file)
    for _index, row in frame.iterrows():
        values = [int(row[channel]) for channel in RAW_CHANNELS]
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
            yield [int(parsed[channel]) for channel in RAW_CHANNELS]


def run_simulation(models_dir: Path, csv_file: Path | None, csv_real_time: bool, port: str | None, baudrate: int) -> None:
    model, scaler, encoder, metadata = load_artifacts(models_dir)
    labels = [str(label) for label in encoder.classes_]
    window_size = int(metadata["window_size"])
    sample_window: deque[list[int]] = deque(maxlen=window_size)
    figure, signal_lines, signal_buffers, bars, title, offset = create_plot(labels, window_size)

    if csv_file:
        delay = None
        if csv_real_time:
            try:
                sample_rate = float(pd.read_csv(csv_file, nrows=1)["sampling_rate_hz"].iloc[0])
            except Exception:
                sample_rate = 1000.0
            delay = 1.0 / sample_rate
        samples = iter_csv_samples(csv_file, delay)
    elif port:
        samples = iter_serial_samples(port, baudrate)
    else:
        raise RuntimeError("Indica CSV o puerto.")

    for values in samples:
        sample_window.append(values)
        prediction = None
        probabilities = None
        if len(sample_window) == window_size:
            window = np.array(sample_window, dtype=np.float32)
            prediction, probabilities = predict_window(model, scaler, encoder, window)
        update_plot(signal_lines, signal_buffers, bars, title, values, offset, prediction, probabilities)
        if not plt.fignum_exists(figure.number):
            break


class Launcher:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Simulacion EMG - RNN TensorFlow")
        self.root.geometry("640x260")
        self.csv_file: Path | None = None
        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="921600")
        self.csv_var = tk.StringVar(value="Ningun CSV seleccionado")
        self.status_var = tk.StringVar(value="Selecciona CSV o puerto COM.")
        self.realtime_var = tk.BooleanVar(value=False)
        self._build_ui()
        self.refresh_ports()

    def _build_ui(self) -> None:
        panel = ttk.Frame(self.root, padding=12)
        panel.pack(fill=tk.BOTH, expand=True)
        ttk.Label(panel, text="Replay desde CSV").grid(row=0, column=0, sticky="w")
        ttk.Button(panel, text="Escoger CSV", command=self.choose_csv).grid(row=1, column=0, sticky="w")
        ttk.Label(panel, textvariable=self.csv_var, width=68).grid(row=1, column=1, columnspan=3, sticky="w", padx=8)
        ttk.Checkbutton(panel, text="Velocidad aproximada real", variable=self.realtime_var).grid(row=2, column=1, sticky="w", padx=8)
        ttk.Button(panel, text="Simular CSV", command=self.start_csv).grid(row=2, column=0, sticky="w")
        ttk.Separator(panel, orient=tk.HORIZONTAL).grid(row=3, column=0, columnspan=4, sticky="ew", pady=10)
        ttk.Label(panel, text="Tiempo real desde XIAO").grid(row=4, column=0, sticky="w")
        ttk.Label(panel, text="Puerto").grid(row=5, column=0, sticky="w")
        self.port_combo = ttk.Combobox(panel, textvariable=self.port_var, width=12)
        self.port_combo.grid(row=5, column=1, sticky="w")
        ttk.Button(panel, text="Actualizar", command=self.refresh_ports).grid(row=5, column=2, sticky="w", padx=6)
        ttk.Label(panel, text="Baudios").grid(row=6, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(panel, textvariable=self.baud_var, width=14).grid(row=6, column=1, sticky="w", pady=(6, 0))
        ttk.Button(panel, text="Simular puerto", command=self.start_port).grid(row=6, column=2, sticky="w", padx=6, pady=(6, 0))
        status = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def refresh_ports(self) -> None:
        ports = list_serial_ports()
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
        self.status_var.set(f"Puertos detectados: {', '.join(ports) if ports else 'ninguno'}")

    def choose_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleccionar CSV EMG",
            initialdir=PROJECT_ROOT / "data" / "raw",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
        )
        if path:
            self.csv_file = Path(path)
            self.csv_var.set(self.csv_file.name)

    def start_csv(self) -> None:
        if self.csv_file is None:
            messagebox.showerror("CSV requerido", "Selecciona un CSV primero.")
            return
        self.root.destroy()
        run_simulation(PROJECT_ROOT / "models", self.csv_file, self.realtime_var.get(), None, 921600)

    def start_port(self) -> None:
        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("Puerto requerido", "Selecciona un puerto COM.")
            return
        try:
            baudrate = int(self.baud_var.get())
        except ValueError:
            messagebox.showerror("Baudios invalidos", "El valor de baudios debe ser numerico.")
            return
        self.root.destroy()
        run_simulation(PROJECT_ROOT / "models", None, False, port, baudrate)


def parse_args():
    parser = argparse.ArgumentParser(description="Inferencia grafica EMG con RNN TensorFlow.")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--csv-file", type=Path)
    parser.add_argument("--csv-real-time", action="store_true")
    parser.add_argument("--port")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--list-ports", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_ports:
        for port in list_serial_ports():
            print(f"- {port}")
        return
    if not args.csv_file and not args.port:
        root = tk.Tk()
        Launcher(root)
        root.mainloop()
        return
    run_simulation(args.models_dir, args.csv_file, args.csv_real_time, args.port, args.baudrate)


if __name__ == "__main__":
    main()
