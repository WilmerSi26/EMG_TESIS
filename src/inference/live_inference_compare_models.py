"""Compare several EMG models in live PC inference.

The dashboard reads one XIAO serial stream and evaluates several trained
models over the same EMG window. It is intended for practical comparison on
the workstation before choosing which TFLite candidate should be moved to the
XIAO ESP32S3.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from collections import Counter, deque
from dataclasses import dataclass
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


MODEL_PRESETS = {
    "rf": {
        "title": "RandomForest rapido",
        "kind": "rf",
        "path": PROJECT_ROOT / "models" / "baseline_realtime",
        "embedded": "PC",
    },
    "tiny_mlp": {
        "title": "Tiny MLP TFLite",
        "kind": "tiny_tflite",
        "path": PROJECT_ROOT / "models" / "tiny_mlp_final",
        "embedded": "XIAO candidato",
    },
    "cnn1d": {
        "title": "CNN1D iyeleswarapu",
        "kind": "keras_sequence",
        "path": PROJECT_ROOT / "models" / "iyeleswarapu_cnn1d",
        "model_name": "iyeleswarapu_cnn1d",
        "embedded": "XIAO candidato",
    },
    "cnn_lstm": {
        "title": "CNN-LSTM ocjorge",
        "kind": "keras_sequence",
        "path": PROJECT_ROOT / "models" / "ocjorge_cnn_lstm",
        "model_name": "ocjorge_cnn_lstm",
        "embedded": "PC",
    },
    "inception_lstm": {
        "title": "Inception-LSTM labAI",
        "kind": "keras_sequence",
        "path": PROJECT_ROOT / "models" / "laboratorio_inception_lstm",
        "model_name": "laboratorio_inception_lstm",
        "embedded": "PC",
    },
}


@dataclass
class LivePrediction:
    label: str
    confidence: float
    top3: str
    latency_ms: float


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


class BaseAdapter:
    def __init__(self, key: str, title: str, model_dir: Path, smooth: int, embedded: str) -> None:
        self.key = key
        self.title = title
        self.model_dir = model_dir
        self.embedded = embedded
        self.history: deque[str] = deque(maxlen=smooth)
        self.window_size = 40
        self.labels: list[str] = []

    def predict_probabilities(self, window: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def label_for_index(self, index: int) -> str:
        return self.labels[index]

    def predict(self, window: np.ndarray) -> LivePrediction:
        started = time.perf_counter()
        probabilities = np.asarray(self.predict_probabilities(window), dtype=np.float32).reshape(-1)
        raw_index = int(np.argmax(probabilities))
        raw_label = self.label_for_index(raw_index)
        self.history.append(raw_label)
        label = Counter(self.history).most_common(1)[0][0]
        label_index = self.labels.index(label) if label in self.labels else raw_index
        confidence = float(probabilities[label_index])
        top3 = self.format_top3(probabilities)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return LivePrediction(label=label, confidence=confidence, top3=top3, latency_ms=latency_ms)

    def format_top3(self, probabilities: np.ndarray) -> str:
        order = np.argsort(probabilities)[::-1][:3]
        parts = [f"{self.label_for_index(int(index))} {float(probabilities[index]):.2f}" for index in order]
        return " | ".join(parts)


class RandomForestAdapter(BaseAdapter):
    def __init__(self, key: str, title: str, model_dir: Path, smooth: int, embedded: str) -> None:
        super().__init__(key, title, model_dir, smooth, embedded)
        model_path = model_dir / "emg_sklearn_baseline.joblib"
        scaler_path = model_dir / "sklearn_feature_scaler.joblib"
        encoder_path = model_dir / "sklearn_label_encoder.joblib"
        metadata_path = model_dir / "sklearn_baseline_metadata.json"
        missing = [path for path in [model_path, scaler_path, encoder_path, metadata_path] if not path.exists()]
        if missing:
            raise RuntimeError("Faltan artefactos RF: " + ", ".join(str(path) for path in missing))
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.window_size = int(metadata.get("window_size", 40))
        self.labels = [str(label) for label in self.encoder.classes_]

    def predict_probabilities(self, window: np.ndarray) -> np.ndarray:
        features = extract_features(window[-self.window_size :])
        return self.model.predict_proba(self.scaler.transform(features))[0]


class TinyTFLiteAdapter(BaseAdapter):
    def __init__(self, key: str, title: str, model_dir: Path, smooth: int, embedded: str) -> None:
        super().__init__(key, title, model_dir, smooth, embedded)
        import tensorflow as tf

        model_path = model_dir / "emg_tiny_mlp.tflite"
        scaler_path = model_dir / "tiny_feature_scaler.joblib"
        encoder_path = model_dir / "tiny_label_encoder.joblib"
        metadata_path = model_dir / "tiny_mlp_metadata.json"
        missing = [path for path in [model_path, scaler_path, encoder_path, metadata_path] if not path.exists()]
        if missing:
            raise RuntimeError("Faltan artefactos Tiny MLP: " + ", ".join(str(path) for path in missing))
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.window_size = int(metadata.get("window_size", 40))
        self.labels = [str(label) for label in self.encoder.classes_]

    def predict_probabilities(self, window: np.ndarray) -> np.ndarray:
        features = extract_features(window[-self.window_size :])
        x = self.scaler.transform(features).astype(self.input_details["dtype"])
        self.interpreter.set_tensor(self.input_details["index"], x)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details["index"])
        return np.asarray(output[0], dtype=np.float32)


class KerasSequenceAdapter(BaseAdapter):
    def __init__(self, key: str, title: str, model_dir: Path, model_name: str, smooth: int, embedded: str) -> None:
        super().__init__(key, title, model_dir, smooth, embedded)
        import tensorflow as tf

        self.tf = tf
        model_path = model_dir / f"{model_name}.keras"
        encoder_path = model_dir / f"{model_name}_label_encoder.joblib"
        normalization_path = model_dir / f"{model_name}_normalization.json"
        metadata_path = model_dir / f"{model_name}_metadata.json"
        missing = [path for path in [model_path, encoder_path, normalization_path, metadata_path] if not path.exists()]
        if missing:
            raise RuntimeError("Faltan artefactos Keras: " + ", ".join(str(path) for path in missing))
        self.model = tf.keras.models.load_model(model_path)
        self.encoder = joblib.load(encoder_path)
        normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.mean = np.asarray(normalization["mean"], dtype=np.float32).reshape(1, 1, -1)
        self.std = np.asarray(normalization["std"], dtype=np.float32).reshape(1, 1, -1)
        self.window_size = int(metadata.get("window_size", 40))
        self.labels = [str(label) for label in self.encoder.classes_]

    def predict_probabilities(self, window: np.ndarray) -> np.ndarray:
        x = window[-self.window_size :].astype(np.float32).reshape(1, self.window_size, len(RAW_CHANNELS))
        x = (x - self.mean) / self.std
        tensor = self.tf.convert_to_tensor(x, dtype=self.tf.float32)
        return self.model(tensor, training=False).numpy()[0]


def load_adapter(key: str, smooth: int) -> BaseAdapter:
    preset = MODEL_PRESETS[key]
    kind = preset["kind"]
    title = preset["title"]
    model_dir = Path(preset["path"])
    embedded = preset["embedded"]
    if kind == "rf":
        return RandomForestAdapter(key, title, model_dir, smooth, embedded)
    if kind == "tiny_tflite":
        return TinyTFLiteAdapter(key, title, model_dir, smooth, embedded)
    if kind == "keras_sequence":
        return KerasSequenceAdapter(key, title, model_dir, str(preset["model_name"]), smooth, embedded)
    raise RuntimeError(f"Tipo de modelo no soportado: {kind}")


class MultiModelDashboard:
    def __init__(
        self,
        root: tk.Tk,
        model_keys: list[str],
        baudrate: int,
        step_size: int,
        plot_window: int,
        smooth: int,
    ) -> None:
        self.root = root
        self.root.title("Inferencia EMG en vivo - modelo seleccionable")
        self.root.geometry("1280x780")
        self.root.minsize(1120, 680)

        self.model_keys = model_keys
        self.baudrate = baudrate
        self.step_size = step_size
        self.plot_window = plot_window
        self.smooth = smooth
        self.draw_interval_s = 0.05

        self.active_adapter: BaseAdapter | None = None
        self.active_model_key: str | None = None
        self.inference_enabled = False
        self.loading_model = False
        self.max_window = 40

        self.serial_port: serial.Serial | None = None
        self.reader_thread: threading.Thread | None = None
        self.stop_reader = threading.Event()
        self.sample_queue: queue.SimpleQueue[list[int]] = queue.SimpleQueue()
        self.reader_error: str | None = None
        self.sample_buffer: deque[list[int]] = deque(maxlen=self.max_window)
        self.plot_buffers = [deque([0] * self.plot_window, maxlen=self.plot_window) for _ in RAW_CHANNELS]
        self.sample_count = 0
        self.last_prediction_sample = 0
        self.last_draw_time = 0.0
        self.last_rx_time = 0.0

        self.model_var = tk.StringVar(value=self.model_keys[0] if self.model_keys else "")
        self.model_status_var = tk.StringVar(value="Selecciona un modelo y pulsa Cargar modelo")
        self.port_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Desconectado")
        self.samples_var = tk.StringVar(value="000000")
        self.channel_vars = [tk.StringVar(value="----") for _ in RAW_CHANNELS]
        self.active_model_var = tk.StringVar(value="sin modelo cargado")
        self.prediction_var = tk.StringVar(value="esperando datos")
        self.confidence_var = tk.StringVar(value="0.00")
        self.latency_var = tk.StringVar(value="-- ms")
        self.top3_var = tk.StringVar(value="--")
        self.embedded_var = tk.StringVar(value="--")
        self.window_var = tk.StringVar(value="--")

        self.build_ui()
        self.refresh_ports()
        self.update_model_description()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(20, self.update_loop)

    def build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        model_bar = ttk.Frame(main)
        model_bar.pack(fill=tk.X)
        ttk.Label(model_bar, text="Modelo").pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(model_bar, textvariable=self.model_var, values=self.model_keys, width=20, state="readonly")
        self.model_combo.pack(side=tk.LEFT, padx=(6, 8))
        self.model_combo.bind("<<ComboboxSelected>>", lambda _event: self.update_model_description())
        self.load_model_button = ttk.Button(model_bar, text="Cargar modelo", command=self.load_selected_model)
        self.load_model_button.pack(side=tk.LEFT)
        ttk.Label(model_bar, textvariable=self.model_status_var).pack(side=tk.LEFT, padx=(14, 0))

        toolbar = ttk.Frame(main)
        toolbar.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(toolbar, text="Puerto").pack(side=tk.LEFT)
        self.port_combo = ttk.Combobox(toolbar, textvariable=self.port_var, width=14, state="readonly")
        self.port_combo.pack(side=tk.LEFT, padx=(6, 8))
        ttk.Button(toolbar, text="Actualizar", command=self.refresh_ports).pack(side=tk.LEFT)
        self.connect_button = ttk.Button(toolbar, text="Conectar", command=self.toggle_connection, state=tk.DISABLED)
        self.connect_button.pack(side=tk.LEFT, padx=(10, 0))
        self.start_button = ttk.Button(toolbar, text="Iniciar inferencia", command=self.toggle_inference, state=tk.DISABLED)
        self.start_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(toolbar, text="Limpiar", command=self.reset_stream_state).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(toolbar, text=f"Baudios {self.baudrate} | paso {self.step_size} muestras").pack(side=tk.LEFT, padx=(18, 0))
        ttk.Label(toolbar, textvariable=self.status_var).pack(side=tk.RIGHT)

        signal_info = ttk.Frame(main)
        signal_info.pack(fill=tk.X, pady=(10, 8))
        ttk.Label(signal_info, text="Muestras", font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w")
        tk.Label(signal_info, textvariable=self.samples_var, font=("Consolas", 16, "bold"), width=7, anchor="e").grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 24),
        )
        for channel, variable in enumerate(self.channel_vars, start=1):
            ttk.Label(signal_info, text=f"CH{channel}", font=("Segoe UI", 10)).grid(row=0, column=channel, sticky="w")
            tk.Label(signal_info, textvariable=variable, font=("Consolas", 13), width=5, anchor="e").grid(
                row=1,
                column=channel,
                sticky="w",
                padx=(0, 18),
            )

        table = ttk.Frame(main)
        table.pack(fill=tk.X, pady=(0, 10))
        headers = ["Modelo activo", "Prediccion", "Conf.", "Latencia", "Top 3", "Uso", "Ventana"]
        widths = [28, 27, 8, 10, 72, 16, 10]
        for column, (header, width) in enumerate(zip(headers, widths)):
            ttk.Label(table, text=header, font=("Segoe UI", 9, "bold"), width=width).grid(
                row=0,
                column=column,
                sticky="w",
                padx=(0, 8),
            )
        ttk.Label(table, textvariable=self.active_model_var, width=widths[0]).grid(row=1, column=0, sticky="w", padx=(0, 8))
        tk.Label(
            table,
            textvariable=self.prediction_var,
            font=("Segoe UI", 11, "bold"),
            width=widths[1],
            anchor="w",
            fg="#1d4ed8",
        ).grid(row=1, column=1, sticky="w", padx=(0, 8))
        tk.Label(table, textvariable=self.confidence_var, font=("Consolas", 11), width=widths[2], anchor="e").grid(
            row=1,
            column=2,
            sticky="w",
            padx=(0, 8),
        )
        tk.Label(table, textvariable=self.latency_var, font=("Consolas", 11), width=widths[3], anchor="e").grid(
            row=1,
            column=3,
            sticky="w",
            padx=(0, 8),
        )
        ttk.Label(table, textvariable=self.top3_var, font=("Consolas", 9), width=widths[4]).grid(
            row=1,
            column=4,
            sticky="w",
            padx=(0, 8),
        )
        ttk.Label(table, textvariable=self.embedded_var, width=widths[5]).grid(row=1, column=5, sticky="w")
        ttk.Label(table, textvariable=self.window_var, width=widths[6]).grid(row=1, column=6, sticky="w")

        self.figure = Figure(figsize=(11, 4.8), dpi=100)
        self.signal_axis = self.figure.add_subplot(111)
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
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def update_model_description(self) -> None:
        key = self.model_var.get().strip()
        preset = MODEL_PRESETS.get(key)
        if preset is None:
            self.model_status_var.set("Modelo no reconocido")
            return
        status = f"{preset['title']} | {preset['embedded']} | no cargado"
        if key == self.active_model_key and self.active_adapter is not None:
            status = f"{preset['title']} | {preset['embedded']} | cargado"
        self.model_status_var.set(status)

    def load_selected_model(self) -> None:
        if self.reader_thread is not None:
            messagebox.showwarning("Modelo bloqueado", "Desconecta la XIAO antes de cambiar o cargar otro modelo.")
            return
        if self.loading_model:
            return
        key = self.model_var.get().strip()
        if key not in MODEL_PRESETS:
            messagebox.showerror("Modelo no valido", "Selecciona un modelo disponible.")
            return

        old_adapter = self.active_adapter
        self.active_adapter = None
        self.active_model_key = None
        if isinstance(old_adapter, KerasSequenceAdapter):
            old_adapter.tf.keras.backend.clear_session()
        gc.collect()

        self.loading_model = True
        self.load_model_button.configure(state=tk.DISABLED)
        self.connect_button.configure(state=tk.DISABLED)
        self.status_var.set("Cargando modelo...")
        self.model_status_var.set(f"Cargando {MODEL_PRESETS[key]['title']}...")
        self.root.update_idletasks()
        try:
            adapter = load_adapter(key, self.smooth)
        except Exception as exc:  # noqa: BLE001
            self.active_adapter = None
            self.active_model_key = None
            self.reset_prediction_fields()
            self.status_var.set("Error al cargar modelo")
            messagebox.showerror("No se pudo cargar el modelo", str(exc))
        else:
            self.active_adapter = adapter
            self.active_model_key = key
            self.max_window = adapter.window_size
            self.sample_buffer = deque(maxlen=self.max_window)
            self.active_model_var.set(adapter.title)
            self.embedded_var.set(adapter.embedded)
            self.window_var.set(f"{adapter.window_size} muestras")
            self.prediction_var.set("modelo cargado")
            self.confidence_var.set("0.00")
            self.latency_var.set("-- ms")
            self.top3_var.set("--")
            self.connect_button.configure(state=tk.NORMAL)
            self.status_var.set("Modelo cargado; conecta la XIAO")
        finally:
            self.loading_model = False
            self.load_model_button.configure(state=tk.NORMAL)
            self.update_model_description()

    def refresh_ports(self) -> None:
        ports = list_serial_ports()
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])
        if self.reader_thread is None:
            self.status_var.set(f"Puertos: {', '.join(ports) if ports else 'ninguno'}")

    def toggle_connection(self) -> None:
        if self.reader_thread is not None:
            self.disconnect()
        else:
            self.connect()

    def connect(self) -> None:
        if self.active_adapter is None:
            messagebox.showwarning("Modelo requerido", "Primero selecciona y carga un modelo.")
            return
        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("Puerto requerido", "Selecciona el puerto COM de la XIAO.")
            return
        try:
            self.serial_port = serial.Serial(port=port, baudrate=self.baudrate, timeout=0.01)
            time.sleep(0.5)
            self.serial_port.reset_input_buffer()
        except serial.SerialException as exc:
            self.serial_port = None
            messagebox.showerror("No se pudo conectar", str(exc))
            return

        self.reset_stream_state()
        self.stop_reader.clear()
        self.reader_thread = threading.Thread(target=self.serial_reader_loop, daemon=True)
        self.reader_thread.start()
        self.connect_button.configure(text="Desconectar")
        self.start_button.configure(state=tk.NORMAL, text="Iniciar inferencia")
        self.model_combo.configure(state=tk.DISABLED)
        self.load_model_button.configure(state=tk.DISABLED)
        self.status_var.set(f"Conectado a {port}; pulsa Iniciar inferencia")

    def disconnect(self) -> None:
        self.inference_enabled = False
        self.stop_reader.set()
        if self.serial_port is not None:
            self.serial_port.close()
            self.serial_port = None
        self.reader_thread = None
        self.connect_button.configure(text="Conectar", state=tk.NORMAL if self.active_adapter is not None else tk.DISABLED)
        self.start_button.configure(text="Iniciar inferencia", state=tk.DISABLED)
        self.model_combo.configure(state="readonly")
        self.load_model_button.configure(state=tk.NORMAL)
        self.status_var.set("Desconectado; puedes escoger otro modelo")

    def toggle_inference(self) -> None:
        if self.reader_thread is None:
            messagebox.showwarning("XIAO no conectada", "Conecta la XIAO antes de iniciar inferencia.")
            return
        if self.active_adapter is None:
            messagebox.showwarning("Modelo requerido", "Carga un modelo antes de iniciar inferencia.")
            return
        self.inference_enabled = not self.inference_enabled
        if self.inference_enabled:
            self.active_adapter.history.clear()
            self.last_prediction_sample = self.sample_count
            self.start_button.configure(text="Pausar inferencia")
            self.status_var.set("Inferencia activa")
        else:
            self.start_button.configure(text="Iniciar inferencia")
            self.status_var.set("Inferencia pausada")

    def reset_prediction_fields(self) -> None:
        self.active_model_var.set("sin modelo cargado")
        self.prediction_var.set("esperando datos")
        self.confidence_var.set("0.00")
        self.latency_var.set("-- ms")
        self.top3_var.set("--")
        self.embedded_var.set("--")
        self.window_var.set("--")

    def reset_stream_state(self) -> None:
        while True:
            try:
                self.sample_queue.get_nowait()
            except queue.Empty:
                break
        self.reader_error = None
        self.sample_buffer.clear()
        self.sample_count = 0
        self.last_prediction_sample = 0
        self.last_rx_time = 0.0
        self.inference_enabled = False
        if self.active_adapter is not None:
            self.active_adapter.history.clear()
            self.prediction_var.set("modelo cargado")
            self.active_model_var.set(self.active_adapter.title)
            self.embedded_var.set(self.active_adapter.embedded)
            self.window_var.set(f"{self.active_adapter.window_size} muestras")
        else:
            self.reset_prediction_fields()
        self.start_button.configure(text="Iniciar inferencia")
        for buffer in self.plot_buffers:
            buffer.clear()
            buffer.extend([0] * self.plot_window)
        self.samples_var.set("000000")
        for variable in self.channel_vars:
            variable.set("----")
        self.confidence_var.set("0.00")
        self.latency_var.set("-- ms")
        self.top3_var.set("--")
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
        self.samples_var.set(f"{self.sample_count:06d}")
        for variable, value in zip(self.channel_vars, values):
            variable.set(f"{value:4d}")

        offset = 4500
        for channel, value in enumerate(values):
            self.plot_buffers[channel].append(value + channel * offset)

        ready = self.active_adapter is not None and len(self.sample_buffer) >= self.active_adapter.window_size
        due = self.sample_count - self.last_prediction_sample >= self.step_size
        if self.inference_enabled and ready and due:
            self.predict_active_model()
            self.last_prediction_sample = self.sample_count

    def predict_active_model(self) -> None:
        if self.active_adapter is None:
            return
        window = np.array(self.sample_buffer, dtype=np.float32)
        try:
            prediction = self.active_adapter.predict(window)
        except Exception as exc:  # noqa: BLE001
            self.prediction_var.set("error")
            self.confidence_var.set("0.00")
            self.latency_var.set("-- ms")
            self.top3_var.set(str(exc)[:90])
            self.inference_enabled = False
            self.start_button.configure(text="Iniciar inferencia")
            return
        self.prediction_var.set(prediction.label)
        self.confidence_var.set(f"{prediction.confidence:.2f}")
        self.latency_var.set(f"{prediction.latency_ms:5.1f} ms")
        self.top3_var.set(prediction.top3)

    def update_plot(self, force: bool = False) -> None:
        for channel, line in enumerate(self.signal_lines):
            line.set_ydata(list(self.plot_buffers[channel]))
        if force:
            self.canvas.draw()
        else:
            self.canvas.draw_idle()

    def on_close(self) -> None:
        self.disconnect()
        self.root.destroy()
def parse_model_keys(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(MODEL_PRESETS)
    keys = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [key for key in keys if key not in MODEL_PRESETS]
    if unknown:
        raise RuntimeError(f"Modelos no soportados: {', '.join(unknown)}")
    return keys


def requires_tensorflow(model_keys: list[str]) -> bool:
    return any(MODEL_PRESETS[key]["kind"] in {"tiny_tflite", "keras_sequence"} for key in model_keys)


def relaunch_with_tensorflow_env_if_needed(model_keys: list[str]) -> None:
    if not requires_tensorflow(model_keys):
        return
    if importlib.util.find_spec("tensorflow") is not None:
        return

    tf_python = PROJECT_ROOT / ".venv-tf" / "Scripts" / "python.exe"
    if not tf_python.exists():
        raise RuntimeError(
            "Los modelos seleccionados requieren TensorFlow. Usa el interprete "
            f"{tf_python} o crea el entorno .venv-tf."
        )

    current = Path(sys.executable).resolve()
    target = tf_python.resolve()
    if current == target:
        raise RuntimeError("Este entorno no tiene TensorFlow instalado. Reinstala dependencias en .venv-tf.")

    print(f"TensorFlow no esta en {current}. Abriendo dashboard con {target}...")
    command = [str(target), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    raise SystemExit(completed.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparar inferencia EMG en vivo con varios modelos.")
    parser.add_argument("--models", default="all")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--smooth", type=int, default=5)
    parser.add_argument("--plot-window", type=int, default=400)
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_models:
        for key, preset in MODEL_PRESETS.items():
            print(f"{key:15s} {preset['title']} | {preset['embedded']} | {preset['path']}")
        return

    try:
        model_keys = parse_model_keys(args.models)
        relaunch_with_tensorflow_env_if_needed(model_keys)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    root = tk.Tk()
    MultiModelDashboard(root, model_keys, args.baudrate, args.step_size, args.plot_window, args.smooth)
    root.mainloop()


if __name__ == "__main__":
    main()