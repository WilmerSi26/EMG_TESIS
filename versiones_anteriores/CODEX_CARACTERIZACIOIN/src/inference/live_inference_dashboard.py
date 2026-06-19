"""Live EMG inference dashboard with current sensor signals.

The dashboard reads the XIAO serial stream, plots the current four EMG channels
and shows the movement predicted by the trained scikit-learn baseline model.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import joblib
import numpy as np
import pandas as pd
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
    if not line or line.startswith("timestamp_ms") or line.startswith("#"):
        return None

    parts = line.split(",")
    if len(parts) != 8:
        return None

    try:
        _timestamp_ms = int(parts[0])
        _sample_index = int(parts[1])
        values = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
        _sampling_rate_hz = int(parts[7])
    except ValueError:
        return None
    return values


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


class LiveInferenceDashboard:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Inferencia EMG en vivo - XIAO ESP32S3")
        self.root.geometry("1120x760")
        self.root.minsize(960, 640)

        self.models_dir = PROJECT_ROOT / "models"
        self.model, self.scaler, self.encoder, self.metadata = self.load_artifacts()
        self.labels = [str(label) for label in self.encoder.classes_]
        self.window_size = int(self.metadata.get("window_size", 200))
        self.plot_window = max(300, self.window_size * 2)
        self.predict_stride = int(self.metadata.get("stride", 50))

        self.serial_port: serial.Serial | None = None
        self.reader_thread: threading.Thread | None = None
        self.stop_reader = threading.Event()
        self.sample_queue: queue.SimpleQueue[list[int]] = queue.SimpleQueue()
        self.reader_error: str | None = None
        self.reader_done = False
        self.active_source = "ninguna"
        self.csv_file: Path | None = None
        self.sample_buffer: deque[list[int]] = deque(maxlen=self.window_size)
        self.plot_buffers = [deque([0] * self.plot_window, maxlen=self.plot_window) for _ in RAW_CHANNELS]
        self.probability_history: deque[np.ndarray] = deque(maxlen=7)
        self.sample_count = 0
        self.last_prediction_sample = 0
        self.last_rx_time = 0.0
        self.last_draw_time = 0.0
        self.draw_interval_s = 0.10
        self.current_probabilities = np.zeros(len(self.labels), dtype=np.float32)

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value="921600")
        self.status_var = tk.StringVar(value="Desconectado")
        self.csv_var = tk.StringVar(value="Ningun CSV seleccionado")
        self.csv_realtime_var = tk.BooleanVar(value=True)
        self.prediction_var = tk.StringVar(value="Esperando datos")
        self.confidence_var = tk.StringVar(value="Confianza: 0.00")
        self.top_predictions_var = tk.StringVar(value="Top 3: --")
        self.samples_var = tk.StringVar(value="Muestras: 0")
        self.channels_var = tk.StringVar(value="CH1: -- | CH2: -- | CH3: -- | CH4: --")

        self.build_ui()
        self.refresh_ports()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(30, self.update_loop)

    def load_artifacts(self):
        model_path = self.models_dir / "emg_sklearn_baseline.joblib"
        scaler_path = self.models_dir / "sklearn_feature_scaler.joblib"
        encoder_path = self.models_dir / "sklearn_label_encoder.joblib"
        metadata_path = self.models_dir / "sklearn_baseline_metadata.json"
        missing = [path for path in [model_path, scaler_path, encoder_path, metadata_path] if not path.exists()]
        if missing:
            raise RuntimeError("Faltan artefactos del modelo base: " + ", ".join(str(path) for path in missing))
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        encoder = joblib.load(encoder_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return model, scaler, encoder, metadata

    def build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Prediction.TLabel", font=("Segoe UI", 30, "bold"))
        style.configure("Metric.TLabel", font=("Segoe UI", 12))

        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="CSV").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Escoger", command=self.choose_csv).pack(side=tk.LEFT, padx=(6, 4))
        ttk.Button(toolbar, text="Replay", command=self.start_csv_replay).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Checkbutton(toolbar, text="vel. real", variable=self.csv_realtime_var).pack(side=tk.LEFT, padx=(0, 14))
        ttk.Label(toolbar, text="Puerto COM").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(toolbar, textvariable=self.port_var, width=14, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=(6, 10))
        ttk.Button(toolbar, text="Actualizar", command=self.refresh_ports).pack(side=tk.LEFT)
        ttk.Label(toolbar, text="Baudios").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Entry(toolbar, textvariable=self.baud_var, width=10).pack(side=tk.LEFT)
        self.connect_button = ttk.Button(toolbar, text="Conectar", command=self.toggle_connection)
        self.connect_button.pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(toolbar, text="Detener", command=self.disconnect).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT)
        ttk.Label(main, textvariable=self.csv_var).pack(anchor=tk.W, pady=(8, 0))

        metrics = ttk.Frame(main)
        metrics.pack(fill=tk.X, pady=(8, 8))
        self.prediction_label = tk.Label(
            metrics,
            textvariable=self.prediction_var,
            font=("Segoe UI", 30, "bold"),
            fg="#1d4ed8",
        )
        self.prediction_label.pack(side=tk.LEFT)
        ttk.Label(metrics, textvariable=self.confidence_var, style="Metric.TLabel").pack(side=tk.LEFT, padx=(28, 0))
        ttk.Label(metrics, textvariable=self.samples_var, style="Metric.TLabel").pack(side=tk.LEFT, padx=(28, 0))

        ttk.Label(main, textvariable=self.channels_var, style="Metric.TLabel").pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(main, textvariable=self.top_predictions_var, style="Metric.TLabel").pack(anchor=tk.W, pady=(0, 8))

        self.figure = Figure(figsize=(11, 6.4), dpi=100)
        self.signal_axis = self.figure.add_subplot(211)
        self.prob_axis = self.figure.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.figure, master=main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.signal_lines = []
        colors = ["#2563eb", "#f97316", "#16a34a", "#dc2626"]
        x_values = list(range(self.plot_window))
        for index, color in enumerate(colors):
            (line,) = self.signal_axis.plot(
                x_values,
                list(self.plot_buffers[index]),
                color=color,
                linewidth=0.85,
                label=f"CH{index + 1}",
            )
            self.signal_lines.append(line)
        self.signal_axis.set_title("Senal actual de los 4 sensores EMG")
        self.signal_axis.set_xlim(0, self.plot_window)
        self.signal_axis.set_ylim(0, 18000)
        self.signal_axis.grid(alpha=0.25)
        self.signal_axis.legend(loc="upper right", ncol=4)

        self.prob_bars = self.prob_axis.bar(self.labels, [0.0] * len(self.labels), color="#2563eb")
        self.prob_axis.set_title("Probabilidad por movimiento")
        self.prob_axis.set_ylim(0, 1.0)
        self.prob_axis.grid(axis="y", alpha=0.25)
        self.prob_axis.tick_params(axis="x", labelrotation=28)
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
            return
        self.connect()

    def choose_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Seleccionar CSV EMG",
            initialdir=PROJECT_ROOT / "data" / "raw",
            filetypes=[("CSV", "*.csv"), ("Todos", "*.*")],
        )
        if not path:
            return
        self.csv_file = Path(path)
        self.csv_var.set(f"CSV: {self.csv_file.name}")

    def reset_stream_state(self) -> None:
        self.stop_reader.set()
        while True:
            try:
                self.sample_queue.get_nowait()
            except queue.Empty:
                break
        self.stop_reader.clear()
        self.reader_error = None
        self.reader_done = False
        self.sample_buffer.clear()
        self.probability_history.clear()
        self.current_probabilities = np.zeros(len(self.labels), dtype=np.float32)
        self.sample_count = 0
        self.last_prediction_sample = 0
        self.last_draw_time = 0.0
        self.last_rx_time = time.time()
        for channel in range(len(self.plot_buffers)):
            self.plot_buffers[channel].clear()
            self.plot_buffers[channel].extend([0] * self.plot_window)
        self.prediction_var.set("Esperando datos")
        self.confidence_var.set("Confianza: 0.00")
        self.top_predictions_var.set("Top 3: --")
        self.samples_var.set("Muestras: 0")
        self.channels_var.set("CH1: -- | CH2: -- | CH3: -- | CH4: --")
        self.update_plot()

    def connect(self) -> None:
        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("Puerto requerido", "Selecciona el puerto COM de la XIAO.")
            return
        try:
            baudrate = int(self.baud_var.get())
        except ValueError:
            messagebox.showerror("Baudios invalidos", "El valor de baudios debe ser numerico.")
            return

        self.disconnect()
        try:
            self.serial_port = serial.Serial(port=port, baudrate=baudrate, timeout=0.02)
            time.sleep(0.4)
            self.serial_port.reset_input_buffer()
        except serial.SerialException as exc:
            self.serial_port = None
            messagebox.showerror("No se pudo conectar", str(exc))
            self.refresh_ports()
            return

        self.reset_stream_state()
        self.active_source = f"puerto {port}"
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
        self.active_source = "ninguna"
        self.connect_button.configure(text="Conectar")
        self.status_var.set("Desconectado")

    def start_csv_replay(self) -> None:
        if self.csv_file is None:
            messagebox.showerror("CSV requerido", "Selecciona primero un CSV.")
            return
        self.disconnect()
        self.reset_stream_state()
        self.active_source = f"CSV {self.csv_file.name}"
        self.reader_thread = threading.Thread(target=self.csv_reader_loop, daemon=True)
        self.reader_thread.start()
        self.status_var.set(f"Reproduciendo {self.csv_file.name}")

    def csv_reader_loop(self) -> None:
        assert self.csv_file is not None
        try:
            frame = pd.read_csv(self.csv_file)
            missing = [column for column in RAW_CHANNELS if column not in frame.columns]
            if missing:
                self.reader_error = f"CSV sin columnas {missing}"
                self.reader_done = True
                return
            sample_rate = 1000.0
            if "sampling_rate_hz" in frame.columns:
                sample_rate = float(pd.to_numeric(frame["sampling_rate_hz"], errors="coerce").dropna().iloc[0])
            delay = 1.0 / sample_rate if self.csv_realtime_var.get() and sample_rate > 0 else 0.0
            for _index, row in frame.iterrows():
                if self.stop_reader.is_set():
                    break
                values = [int(row[column]) for column in RAW_CHANNELS]
                self.sample_queue.put(values)
                if delay:
                    time.sleep(delay)
        except Exception as exc:
            self.reader_error = str(exc)
        finally:
            self.reader_done = True

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
        if self.reader_thread is not None:
            self.read_queue_batch()
            if self.reader_error:
                self.status_var.set(f"Error serial: {self.reader_error}")
            elif self.reader_done:
                self.status_var.set(f"Finalizado: {self.active_source}")
            if self.last_rx_time and time.time() - self.last_rx_time > 2.0:
                self.status_var.set(f"{self.active_source}: sin datos recientes")
        self.root.after(25, self.update_loop)

    def read_queue_batch(self) -> None:
        changed = False
        for _ in range(800):
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
        self.samples_var.set(f"Muestras: {self.sample_count}")
        self.channels_var.set(
            f"CH1: {values[0]} | CH2: {values[1]} | CH3: {values[2]} | CH4: {values[3]}"
        )

        ready = len(self.sample_buffer) == self.window_size
        due = self.sample_count - self.last_prediction_sample >= self.predict_stride
        if ready and due:
            self.predict_current_window()

    def predict_current_window(self) -> None:
        window = np.array(self.sample_buffer, dtype=np.float32)
        features = extract_features(window).reshape(1, -1)
        features = self.scaler.transform(features)
        probabilities = self.model.predict_proba(features)[0]
        self.probability_history.append(np.array(probabilities, dtype=np.float32))
        smoothed = np.mean(np.vstack(self.probability_history), axis=0)
        class_index = int(np.argmax(smoothed))
        prediction = str(self.encoder.inverse_transform([class_index])[0])
        confidence = float(smoothed[class_index])

        self.current_probabilities = np.array(smoothed, dtype=np.float32)
        if confidence >= 0.60:
            self.prediction_var.set(f"Prediccion: {prediction}")
            self.prediction_label.configure(fg="#15803d")
        elif confidence >= 0.42:
            self.prediction_var.set(f"Mas cercano: {prediction}")
            self.prediction_label.configure(fg="#ca8a04")
        else:
            self.prediction_var.set(f"Duda: {prediction}")
            self.prediction_label.configure(fg="#dc2626")
        self.confidence_var.set(f"Confianza: {confidence:.2f}")
        self.top_predictions_var.set(self.format_top_predictions(smoothed))
        self.last_prediction_sample = self.sample_count

    def format_top_predictions(self, probabilities: np.ndarray) -> str:
        order = np.argsort(probabilities)[::-1][:3]
        items = []
        for index in order:
            label = str(self.encoder.inverse_transform([int(index)])[0])
            items.append(f"{label} {float(probabilities[index]):.2f}")
        return "Top 3: " + " | ".join(items)

    def update_plot(self) -> None:
        for channel, line in enumerate(self.signal_lines):
            line.set_ydata(list(self.plot_buffers[channel]))
        top_order = set(np.argsort(self.current_probabilities)[::-1][:3])
        for index, (bar, probability) in enumerate(zip(self.prob_bars, self.current_probabilities)):
            bar.set_height(float(probability))
            if index == int(np.argmax(self.current_probabilities)):
                bar.set_color("#15803d")
            elif index in top_order:
                bar.set_color("#f59e0b")
            else:
                bar.set_color("#94a3b8")
        self.canvas.draw_idle()

    def on_close(self) -> None:
        self.disconnect()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        LiveInferenceDashboard(root)
    except RuntimeError as exc:
        messagebox.showerror("Modelo no disponible", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
