"""Fast live EMG inference dashboard.

Reads the XIAO serial stream in a background thread, updates plots in batches,
and predicts with the lightweight RandomForest model trained for real time.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import tkinter as tk
from collections import Counter, deque
from pathlib import Path
from tkinter import messagebox, ttk

import joblib
import numpy as np
import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from serial.tools import list_ports


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CHANNELS = ["emg_ch1", "emg_ch2", "emg_ch3", "emg_ch4"]


def list_serial_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


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
    model_path = model_dir / "emg_sklearn_baseline.joblib"
    scaler_path = model_dir / "sklearn_feature_scaler.joblib"
    encoder_path = model_dir / "sklearn_label_encoder.joblib"
    metadata_path = model_dir / "sklearn_baseline_metadata.json"
    missing = [path for path in [model_path, scaler_path, encoder_path, metadata_path] if not path.exists()]
    if missing:
        raise RuntimeError("Faltan artefactos: " + ", ".join(str(path) for path in missing))
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return model, scaler, encoder, metadata


class FastInferenceDashboard:
    def __init__(self, root: tk.Tk, model_dir: Path) -> None:
        self.root = root
        self.root.title("Inferencia EMG en vivo - RF rapido")
        self.root.geometry("1180x760")
        self.root.minsize(1000, 650)

        self.model_dir = model_dir
        self.model, self.scaler, self.encoder, self.metadata = load_artifacts(model_dir)
        self.labels = [str(label) for label in self.encoder.classes_]
        self.window_size = int(self.metadata.get("window_size", 40))
        self.predict_stride = int(self.metadata.get("stride", 10))
        self.plot_window = 400
        self.draw_interval_s = 0.05

        self.serial_port: serial.Serial | None = None
        self.reader_thread: threading.Thread | None = None
        self.stop_reader = threading.Event()
        self.sample_queue: queue.SimpleQueue[list[int]] = queue.SimpleQueue()
        self.reader_error: str | None = None
        self.sample_buffer: deque[list[int]] = deque(maxlen=self.window_size)
        self.plot_buffers = [deque([0] * self.plot_window, maxlen=self.plot_window) for _ in RAW_CHANNELS]
        self.prediction_history: deque[str] = deque(maxlen=5)
        self.current_probabilities = np.zeros(len(self.labels), dtype=np.float32)
        self.sample_count = 0
        self.last_prediction_sample = 0
        self.last_draw_time = 0.0
        self.last_rx_time = 0.0

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="921600")
        self.status_var = tk.StringVar(value="Desconectado")
        self.prediction_var = tk.StringVar(value="esperando datos")
        self.confidence_var = tk.StringVar(value="0.00")
        self.samples_var = tk.StringVar(value="000000")
        self.ch_vars = [tk.StringVar(value="----") for _ in RAW_CHANNELS]
        self.top_vars = [tk.StringVar(value="--") for _ in range(3)]

        self.build_ui()
        self.refresh_ports()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(20, self.update_loop)

    def build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="Puerto").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(toolbar, textvariable=self.port_var, width=14, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(toolbar, text="Actualizar", command=self.refresh_ports).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Baudios").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Entry(toolbar, textvariable=self.baud_var, width=10).pack(side=tk.LEFT)
        self.connect_button = ttk.Button(toolbar, text="Conectar", command=self.toggle_connection)
        self.connect_button.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(toolbar, text="Limpiar", command=self.reset_stream_state).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT)

        metrics = ttk.Frame(main)
        metrics.pack(fill=tk.X, pady=(10, 8))
        metrics.columnconfigure(0, weight=1, minsize=620)
        metrics.columnconfigure(1, weight=0, minsize=150)
        metrics.columnconfigure(2, weight=0, minsize=160)

        prediction_box = ttk.Frame(metrics)
        prediction_box.grid(row=0, column=0, sticky="ew")
        ttk.Label(prediction_box, text="Prediccion", font=("Segoe UI", 10)).pack(anchor=tk.W)
        self.prediction_label = tk.Label(
            prediction_box,
            textvariable=self.prediction_var,
            font=("Segoe UI", 26, "bold"),
            fg="#1d4ed8",
            anchor="w",
            width=30,
        )
        self.prediction_label.pack(anchor=tk.W)

        confidence_box = ttk.Frame(metrics)
        confidence_box.grid(row=0, column=1, sticky="nw", padx=(18, 0))
        ttk.Label(confidence_box, text="Confianza", font=("Segoe UI", 10)).pack(anchor=tk.W)
        tk.Label(
            confidence_box,
            textvariable=self.confidence_var,
            font=("Consolas", 20, "bold"),
            width=6,
            anchor="e",
        ).pack(anchor=tk.W)

        samples_box = ttk.Frame(metrics)
        samples_box.grid(row=0, column=2, sticky="nw", padx=(18, 0))
        ttk.Label(samples_box, text="Muestras", font=("Segoe UI", 10)).pack(anchor=tk.W)
        tk.Label(
            samples_box,
            textvariable=self.samples_var,
            font=("Consolas", 20, "bold"),
            width=7,
            anchor="e",
        ).pack(anchor=tk.W)

        ttk.Label(main, text=f"Modelo: {self.model_dir}", font=("Segoe UI", 10)).pack(anchor=tk.W)

        details = ttk.Frame(main)
        details.pack(fill=tk.X, pady=(6, 8))
        for column in range(8):
            details.columnconfigure(column, weight=0, minsize=72)
        for channel, variable in enumerate(self.ch_vars):
            ttk.Label(details, text=f"CH{channel + 1}:", font=("Segoe UI", 10)).grid(
                row=0,
                column=channel * 2,
                sticky="w",
            )
            tk.Label(
                details,
                textvariable=variable,
                font=("Consolas", 11),
                width=5,
                anchor="e",
            ).grid(row=0, column=channel * 2 + 1, sticky="w", padx=(0, 14))

        top_frame = ttk.Frame(main)
        top_frame.pack(fill=tk.X, pady=(0, 8))
        for column in range(3):
            top_frame.columnconfigure(column, weight=1, uniform="top3")
            ttk.Label(
                top_frame,
                textvariable=self.top_vars[column],
                font=("Consolas", 10),
                width=38,
                anchor="w",
            ).grid(row=0, column=column, sticky="w")

        self.figure = Figure(figsize=(11, 6.2), dpi=100)
        self.signal_axis = self.figure.add_subplot(211)
        self.prob_axis = self.figure.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        colors = ["#2563eb", "#f97316", "#16a34a", "#dc2626"]
        x_values = list(range(self.plot_window))
        self.signal_lines = []
        for index, color in enumerate(colors):
            (line,) = self.signal_axis.plot(
                x_values,
                list(self.plot_buffers[index]),
                color=color,
                linewidth=0.85,
                label=f"CH{index + 1}",
            )
            self.signal_lines.append(line)
        self.signal_axis.set_title("Senales EMG en vivo")
        self.signal_axis.set_xlim(0, self.plot_window)
        self.signal_axis.set_ylim(0, 18000)
        self.signal_axis.grid(alpha=0.25)
        self.signal_axis.legend(loc="upper right", ncol=4)

        self.prob_bars = self.prob_axis.bar(self.labels, [0.0] * len(self.labels), color="#94a3b8")
        self.prob_axis.set_ylim(0, 1)
        self.prob_axis.set_title("Probabilidad por clase")
        self.prob_axis.tick_params(axis="x", labelrotation=28, labelsize=9)
        self.prob_axis.grid(axis="y", alpha=0.25)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def refresh_ports(self) -> None:
        ports = list_serial_ports()
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
        self.status_var.set(f"Puertos: {', '.join(ports) if ports else 'ninguno'}")

    def toggle_connection(self) -> None:
        if self.reader_thread is not None:
            self.disconnect()
        else:
            self.connect()

    def connect(self) -> None:
        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("Puerto requerido", "Selecciona el puerto COM de la XIAO.")
            return
        try:
            baudrate = int(self.baud_var.get())
            self.serial_port = serial.Serial(port=port, baudrate=baudrate, timeout=0.01)
            time.sleep(0.5)
            self.serial_port.reset_input_buffer()
        except (ValueError, serial.SerialException) as exc:
            self.serial_port = None
            messagebox.showerror("No se pudo conectar", str(exc))
            return

        self.reset_stream_state()
        self.stop_reader.clear()
        self.reader_thread = threading.Thread(target=self.serial_reader_loop, daemon=True)
        self.reader_thread.start()
        self.connect_button.configure(text="Desconectar")
        self.status_var.set(f"Conectado a {port}")

    def disconnect(self) -> None:
        self.stop_reader.set()
        if self.serial_port is not None:
            self.serial_port.close()
            self.serial_port = None
        self.reader_thread = None
        self.connect_button.configure(text="Conectar")
        self.status_var.set("Desconectado")

    def reset_stream_state(self) -> None:
        while True:
            try:
                self.sample_queue.get_nowait()
            except queue.Empty:
                break
        self.reader_error = None
        self.sample_buffer.clear()
        self.prediction_history.clear()
        self.current_probabilities = np.zeros(len(self.labels), dtype=np.float32)
        self.sample_count = 0
        self.last_prediction_sample = 0
        self.last_rx_time = 0.0
        for buffer in self.plot_buffers:
            buffer.clear()
            buffer.extend([0] * self.plot_window)
        self.prediction_var.set("esperando datos")
        self.confidence_var.set("0.00")
        self.samples_var.set("000000")
        for variable in self.ch_vars:
            variable.set("----")
        for variable in self.top_vars:
            variable.set("--")
        self.update_plot(force=True)

    def serial_reader_loop(self) -> None:
        while not self.stop_reader.is_set():
            port = self.serial_port
            if port is None:
                break
            try:
                line = port.readline().decode("utf-8", errors="replace")
            except (serial.SerialException, OSError) as exc:
                self.reader_error = str(exc)
                break
            values = parse_device_line(line)
            if values is not None:
                self.sample_queue.put(values)

    def update_loop(self) -> None:
        self.read_queue_batch()
        if self.reader_error:
            self.status_var.set(f"Error serial: {self.reader_error}")
        elif self.reader_thread is not None and self.last_rx_time and time.time() - self.last_rx_time > 2:
            self.status_var.set("Conectado, sin datos recientes")
        self.root.after(20, self.update_loop)

    def read_queue_batch(self) -> None:
        changed = False
        for _ in range(1000):
            try:
                values = self.sample_queue.get_nowait()
            except queue.Empty:
                break
            self.handle_sample(values)
            changed = True

        now = time.time()
        if changed and now - self.last_draw_time >= self.draw_interval_s:
            self.update_plot()
            self.last_draw_time = now

    def handle_sample(self, values: list[int]) -> None:
        self.sample_count += 1
        self.last_rx_time = time.time()
        self.sample_buffer.append(values)

        offset = 4500
        for channel, value in enumerate(values):
            self.plot_buffers[channel].append(value + channel * offset)

        self.samples_var.set(f"{self.sample_count:06d}")
        for variable, value in zip(self.ch_vars, values):
            variable.set(f"{value:4d}")

        ready = len(self.sample_buffer) == self.window_size
        due = self.sample_count - self.last_prediction_sample >= self.predict_stride
        if ready and due:
            self.predict_current_window()
            self.last_prediction_sample = self.sample_count

    def predict_current_window(self) -> None:
        window = np.array(self.sample_buffer, dtype=np.float32)
        features = extract_features(window)
        features_scaled = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features_scaled)[0]
        class_index = int(np.argmax(probabilities))
        raw_label = str(self.encoder.inverse_transform([class_index])[0])
        confidence = float(probabilities[class_index])

        self.prediction_history.append(raw_label)
        voted_label = Counter(self.prediction_history).most_common(1)[0][0]
        voted_index = int(np.where(self.encoder.classes_ == voted_label)[0][0])

        self.current_probabilities = np.array(probabilities, dtype=np.float32)
        self.prediction_var.set(voted_label)
        self.confidence_var.set(f"{float(probabilities[voted_index]):.2f}")
        self.prediction_label.configure(fg="#15803d" if confidence >= 0.55 else "#ca8a04")
        self.update_top3(probabilities)

    def update_top3(self, probabilities: np.ndarray) -> None:
        order = np.argsort(probabilities)[::-1][:3]
        for rank, index in enumerate(order):
            label = str(self.encoder.inverse_transform([int(index)])[0])
            self.top_vars[rank].set(f"Top {rank + 1}: {label:<24} {float(probabilities[index]):.2f}")

    def update_plot(self, force: bool = False) -> None:
        for channel, line in enumerate(self.signal_lines):
            line.set_ydata(list(self.plot_buffers[channel]))

        best_index = int(np.argmax(self.current_probabilities)) if len(self.current_probabilities) else 0
        top3 = set(np.argsort(self.current_probabilities)[::-1][:3])
        for index, (bar, probability) in enumerate(zip(self.prob_bars, self.current_probabilities)):
            bar.set_height(float(probability))
            if index == best_index:
                bar.set_color("#15803d")
            elif index in top3:
                bar.set_color("#f59e0b")
            else:
                bar.set_color("#94a3b8")
        if force:
            self.canvas.draw()
        else:
            self.canvas.draw_idle()

    def on_close(self) -> None:
        self.disconnect()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dashboard rapido de inferencia EMG.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "baseline_realtime")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    try:
        FastInferenceDashboard(root, args.model_dir)
    except RuntimeError as exc:
        messagebox.showerror("Modelo no disponible", str(exc))
        root.destroy()
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    root.mainloop()


if __name__ == "__main__":
    main()
