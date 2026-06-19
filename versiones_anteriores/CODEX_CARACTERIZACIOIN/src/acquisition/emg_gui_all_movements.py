"""
Interfaz grafica para adquisicion sEMG multicanal.

Lee 4 canales desde ESP32-S3 o XIAO ESP32S3, muestra senales crudas y
filtradas en vivo, y guarda una sesion completa siguiendo una rutina.

Formato esperado desde Arduino:
timestamp_ms,sample_index,emg_ch1,emg_ch2,emg_ch3,emg_ch4,board_id,sampling_rate_hz
"""

from __future__ import annotations

import csv   
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import serial
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt, iirnotch
from serial.tools import list_ports


BAUD_RATE_DEFAULT = 921600
MUESTRAS_GRAFICA = 500
NUM_CHANNELS = 4
OFFSET_VISUAL = 4500
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_LOW_CUT_HZ = 10.0
DEFAULT_HIGH_CUT_HZ = 500.0
DEFAULT_NOTCH_HZ = 60.0

RUTINA = [
    ("preparacion", 5),
    ("reposo", 5),
    ("reposo", 2),
    ("dedo_pulgar_cierre", 6),
    ("reposo", 2),
    ("dedo_pulgar_apertura", 6),
    ("reposo", 2),
    ("dedo_indice_cierre", 6),
    ("reposo", 2),
    ("dedo_indice_apertura", 6),
    ("reposo", 2),
    ("dedo_medio_cierre", 6),
    ("reposo", 2),
    ("dedo_medio_apertura", 6),
    ("reposo", 2),
    ("dedo_anular_cierre", 6),
    ("reposo", 2),
    ("dedo_anular_apertura", 6),
    ("reposo", 2),
    ("dedo_menique_cierre", 6),
    ("reposo", 2),
    ("dedo_menique_apertura", 6),
    ("reposo", 2),
    ("cierre_mano", 6),
    ("reposo", 2),
    ("apertura_mano", 6),
    ("reposo", 2),
    ("pinza_fina", 8),
    ("reposo", 2),
    ("reposo", 5),
]

PHASE_DISPLAY = {
    "preparacion": "PREPARACION",
    "reposo_inicial": "REPOSO INICIAL",
    "reposo_final": "REPOSO FINAL",
    "pausa": "REPOSO / RELAJAR",
    "reposo": "REPOSO / RELAJACION",
    "dedo_pulgar_cierre": "CIERRE DEDO PULGAR",
    "dedo_pulgar_apertura": "APERTURA DEDO PULGAR",
    "dedo_indice_cierre": "CIERRE DEDO INDICE",
    "dedo_indice_apertura": "APERTURA DEDO INDICE",
    "dedo_medio_cierre": "CIERRE DEDO MEDIO",
    "dedo_medio_apertura": "APERTURA DEDO MEDIO",
    "dedo_anular_cierre": "CIERRE DEDO ANULAR",
    "dedo_anular_apertura": "APERTURA DEDO ANULAR",
    "dedo_menique_cierre": "CIERRE DEDO MENIQUE",
    "dedo_menique_apertura": "APERTURA DEDO MENIQUE",
    "cierre_mano": "CIERRE MANO",
    "apertura_mano": "APERTURA MANO",
    "pinza_fina": "PINZA FINA",
}

CSV_COLUMNS = [
    "pc_timestamp",
    "elapsed_seconds",
    "device_timestamp_ms",
    "sample_index",
    "emg_ch1",
    "emg_ch2",
    "emg_ch3",
    "emg_ch4",
    "emg_filt_ch1",
    "emg_filt_ch2",
    "emg_filt_ch3",
    "emg_filt_ch4",
    "movement_label",
    "routine_step",
    "trial_number",
    "subject_id",
    "board_id",
    "sampling_rate_hz",
]


def listar_puertos() -> list[str]:
    return [port.device for port in list_ports.comports()]


def parsear_linea_dispositivo(linea: str) -> dict[str, str] | None:
    linea = linea.strip()
    if not linea or linea.startswith("timestamp_ms"):
        return None

    partes = linea.split(",")
    if len(partes) != 8:
        return None

    timestamp_ms, sample_index, ch1, ch2, ch3, ch4, board_id, sampling_rate_hz = partes
    valores_numericos = [timestamp_ms, sample_index, ch1, ch2, ch3, ch4, sampling_rate_hz]

    try:
        for valor in valores_numericos:
            int(valor)
    except ValueError:
        return None

    return {
        "device_timestamp_ms": timestamp_ms,
        "sample_index": sample_index,
        "emg_ch1": ch1,
        "emg_ch2": ch2,
        "emg_ch3": ch3,
        "emg_ch4": ch4,
        "board_id": board_id,
        "sampling_rate_hz": sampling_rate_hz,
    }


def aplicar_filtros(
    muestras: list[float],
    fs_hz: float,
    low_cut_hz: float,
    high_cut_hz: float,
    notch_hz: float,
    notch_enabled: bool,
) -> list[float]:
    """Aplica notch y pasa banda a una ventana de muestras para visualizacion."""
    if len(muestras) < 30:
        return muestras

    nyquist = fs_hz / 2.0
    low = max(0.1, low_cut_hz)
    high = min(high_cut_hz, nyquist * 0.98)

    if low >= high:
        return muestras

    filtrada = muestras

    try:
        if notch_enabled and 0 < notch_hz < nyquist:
            b_notch, a_notch = iirnotch(w0=notch_hz, Q=30.0, fs=fs_hz)
            filtrada = filtfilt(b_notch, a_notch, filtrada)

        b_band, a_band = butter(N=4, Wn=[low, high], btype="bandpass", fs=fs_hz)
        filtrada = filtfilt(b_band, a_band, filtrada)
        return [float(valor) for valor in filtrada]
    except ValueError:
        return muestras


class EMGRecorderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Captura sEMG multicanal - Tesis IA")
        self.root.geometry("1400x820")
        self.root.protocol("WM_DELETE_WINDOW", self.cerrar)

        self.serial_port: serial.Serial | None = None
        self.connected = False
        self.recording = False
        self.start_time = 0.0
        self.fase_start_time = 0.0
        self.rutina_idx = 0
        self.sample_counter = 0
        self.duration_total = sum(duracion for _, duracion in RUTINA)

        self.csv_file = None
        self.csv_writer: csv.DictWriter | None = None
        self.output_path: Path | None = None

        self.raw_plot_buffers = [
            deque([0] * MUESTRAS_GRAFICA, maxlen=MUESTRAS_GRAFICA)
            for _ in range(NUM_CHANNELS)
        ]
        self.raw_filter_buffers = [
            deque([0] * MUESTRAS_GRAFICA, maxlen=MUESTRAS_GRAFICA)
            for _ in range(NUM_CHANNELS)
        ]
        self.filtered_buffers = [
            deque([0.0] * MUESTRAS_GRAFICA, maxlen=MUESTRAS_GRAFICA)
            for _ in range(NUM_CHANNELS)
        ]

        self.port_var = tk.StringVar()
        self.baud_var = tk.StringVar(value=str(BAUD_RATE_DEFAULT))
        self.subject_var = tk.StringVar(value="S001")
        self.trial_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="Listo. Selecciona el puerto y conecta la placa.")
        self.samples_var = tk.StringVar(value="Muestras guardadas: 0")
        self.board_var = tk.StringVar(value="Placa: -")
        self.total_time_var = tk.StringVar(value=f"Duracion rutina: {self.duration_total}s")
        self.low_cut_var = tk.StringVar(value=str(DEFAULT_LOW_CUT_HZ))
        self.high_cut_var = tk.StringVar(value=str(DEFAULT_HIGH_CUT_HZ))
        self.notch_freq_var = tk.StringVar(value=str(DEFAULT_NOTCH_HZ))
        self.notch_enabled_var = tk.BooleanVar(value=True)
        self.filter_status_var = tk.StringVar(value="Filtro: pasa banda 10-500 Hz + notch 60 Hz")
        self.latest_filtered_values = [0.0] * NUM_CHANNELS

        self._crear_interfaz()
        self.refrescar_puertos()

    def _crear_interfaz(self) -> None:
        self.lbl_instruccion = tk.Label(
            self.root,
            text="DESCONECTADO",
            font=("Arial", 28, "bold"),
            bg="#222222",
            fg="gray",
            pady=15,
        )
        self.lbl_instruccion.pack(fill=tk.X)

        panel = ttk.Frame(self.root, padding=10)
        panel.pack(fill=tk.X)

        ttk.Label(panel, text="Puerto").grid(row=0, column=0, sticky="w", padx=4)
        self.port_combo = ttk.Combobox(panel, textvariable=self.port_var, width=12)
        self.port_combo.grid(row=1, column=0, padx=4)
        ttk.Button(panel, text="Actualizar", command=self.refrescar_puertos).grid(row=1, column=1, padx=4)

        ttk.Label(panel, text="Baudios").grid(row=0, column=2, sticky="w", padx=4)
        ttk.Entry(panel, textvariable=self.baud_var, width=10).grid(row=1, column=2, padx=4)

        ttk.Label(panel, text="Sujeto anonimo").grid(row=0, column=3, sticky="w", padx=4)
        ttk.Entry(panel, textvariable=self.subject_var, width=12).grid(row=1, column=3, padx=4)

        ttk.Label(panel, text="Toma").grid(row=0, column=4, sticky="w", padx=4)
        ttk.Entry(panel, textvariable=self.trial_var, width=8).grid(row=1, column=4, padx=4)

        ttk.Label(panel, textvariable=self.total_time_var).grid(row=1, column=5, padx=12)

        self.btn_connect = tk.Button(
            panel,
            text="1. CONECTAR",
            font=("Arial", 11, "bold"),
            bg="#2563eb",
            fg="white",
            command=self.conectar_serial,
        )
        self.btn_connect.grid(row=1, column=6, padx=8)

        self.btn_record = tk.Button(
            panel,
            text="2. GRABAR RUTINA COMPLETA",
            font=("Arial", 11, "bold"),
            bg="gray",
            fg="white",
            state=tk.DISABLED,
            command=self.iniciar_grabacion,
        )
        self.btn_record.grid(row=1, column=7, padx=8)

        self.btn_stop = tk.Button(
            panel,
            text="DETENER",
            font=("Arial", 11, "bold"),
            bg="#991b1b",
            fg="white",
            state=tk.DISABLED,
            command=self.finalizar_grabacion,
        )
        self.btn_stop.grid(row=1, column=8, padx=4)

        ttk.Label(panel, text="Pasa banda min Hz").grid(row=2, column=0, sticky="w", padx=4, pady=(8, 0))
        ttk.Entry(panel, textvariable=self.low_cut_var, width=10).grid(row=3, column=0, padx=4)

        ttk.Label(panel, text="Pasa banda max Hz").grid(row=2, column=1, sticky="w", padx=4, pady=(8, 0))
        ttk.Entry(panel, textvariable=self.high_cut_var, width=10).grid(row=3, column=1, padx=4)

        ttk.Checkbutton(panel, text="Notch", variable=self.notch_enabled_var).grid(row=3, column=2, padx=4)

        ttk.Label(panel, text="Notch Hz").grid(row=2, column=3, sticky="w", padx=4, pady=(8, 0))
        ttk.Combobox(panel, textvariable=self.notch_freq_var, width=8, values=["50", "60"]).grid(row=3, column=3, padx=4)

        ttk.Label(panel, textvariable=self.filter_status_var).grid(row=3, column=4, columnspan=2, sticky="w", padx=4)

        self.fig = Figure(figsize=(12, 5), dpi=100)
        self.ax_raw = self.fig.add_subplot(121)
        self.ax_filt = self.fig.add_subplot(122)
        colores = ["#2563eb", "#f97316", "#16a34a", "#dc2626"]
        self.raw_lines = []
        self.filtered_lines = []
        for canal in range(NUM_CHANNELS):
            raw_line, = self.ax_raw.plot([], [], color=colores[canal], linewidth=0.8, label=f"CH{canal + 1}")
            filt_line, = self.ax_filt.plot([], [], color=colores[canal], linewidth=1.0, label=f"CH{canal + 1}")
            self.raw_lines.append(raw_line)
            self.filtered_lines.append(filt_line)

        self.ax_raw.set_title("Senales crudas")
        self.ax_raw.set_xlabel("Muestras recientes")
        self.ax_raw.set_ylabel("ADC raw + offset")
        self.ax_raw.set_xlim(0, MUESTRAS_GRAFICA)
        self.ax_raw.set_ylim(0, OFFSET_VISUAL * NUM_CHANNELS)
        self.ax_raw.grid(True, alpha=0.3)
        self.ax_raw.legend(loc="upper right")

        self.ax_filt.set_title("Senales filtradas")
        self.ax_filt.set_xlabel("Muestras recientes")
        self.ax_filt.set_ylabel("ADC filtrado + offset")
        self.ax_filt.set_xlim(0, MUESTRAS_GRAFICA)
        self.ax_filt.set_ylim(-2500, OFFSET_VISUAL * NUM_CHANNELS)
        self.ax_filt.grid(True, alpha=0.3)
        self.ax_filt.legend(loc="upper right")
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(fill=tk.X, padx=20, pady=8)

        info_panel = ttk.Frame(self.root, padding=(10, 2))
        info_panel.pack(fill=tk.X)
        ttk.Label(info_panel, textvariable=self.samples_var).pack(side=tk.LEFT, padx=8)
        ttk.Label(info_panel, textvariable=self.board_var).pack(side=tk.LEFT, padx=8)

        status = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status.pack(fill=tk.X, side=tk.BOTTOM)

    def refrescar_puertos(self) -> None:
        puertos = listar_puertos()
        self.port_combo["values"] = puertos
        if puertos and not self.port_var.get():
            self.port_var.set(puertos[0])
        self.status_var.set(f"Puertos detectados: {', '.join(puertos) if puertos else 'ninguno'}")

    def obtener_parametros_filtro(self, fs_hz: float) -> tuple[float, float, float, bool]:
        try:
            low = float(self.low_cut_var.get())
            high = float(self.high_cut_var.get())
            notch = float(self.notch_freq_var.get())
        except ValueError:
            low = DEFAULT_LOW_CUT_HZ
            high = DEFAULT_HIGH_CUT_HZ
            notch = DEFAULT_NOTCH_HZ

        nyquist = fs_hz / 2.0
        effective_high = min(high, nyquist * 0.98)
        self.filter_status_var.set(f"Filtro efectivo: {low:g}-{effective_high:g} Hz | notch {notch:g} Hz")
        return low, high, notch, self.notch_enabled_var.get()

    def actualizar_senales_filtradas(self, fs_hz: float) -> None:
        low, high, notch, notch_enabled = self.obtener_parametros_filtro(fs_hz)
        for canal in range(NUM_CHANNELS):
            raw_values = list(self.raw_filter_buffers[canal])
            filtered = aplicar_filtros(raw_values, fs_hz, low, high, notch, notch_enabled)
            self.filtered_buffers[canal].clear()
            for value in filtered[-MUESTRAS_GRAFICA:]:
                self.filtered_buffers[canal].append(value + (OFFSET_VISUAL * canal))
            self.latest_filtered_values[canal] = filtered[-1] if filtered else 0.0

    def conectar_serial(self) -> None:
        puerto = self.port_var.get().strip()
        if not puerto:
            messagebox.showerror("Puerto requerido", "Selecciona un puerto COM.")
            return

        try:
            baudrate = int(self.baud_var.get())
            self.serial_port = serial.Serial(puerto, baudrate, timeout=0.02)
            time.sleep(1.5)
            self.serial_port.reset_input_buffer()
        except (ValueError, serial.SerialException) as exc:
            messagebox.showerror(
                "Error de conexion",
                f"No se pudo abrir {puerto}.\n\n"
                f"Cierra el Monitor Serial de Arduino IDE y verifica el puerto.\n\nDetalle: {exc}",
            )
            return

        self.connected = True
        self.btn_connect.config(state=tk.DISABLED, text="CONECTADO", bg="#16a34a")
        self.btn_record.config(state=tk.NORMAL, bg="#dc2626")
        self.lbl_instruccion.config(text="MONITOREANDO", bg="#2563eb", fg="white")
        self.status_var.set(f"Conectado a {puerto}. Monitoreando 4 canales sEMG.")
        self.actualizar()

    def iniciar_grabacion(self) -> None:
        if not self.connected or self.serial_port is None:
            return

        try:
            trial_number = int(self.trial_var.get())
        except ValueError:
            messagebox.showerror("Toma invalida", "El numero de toma debe ser numerico.")
            return

        subject_id = self.subject_var.get().strip()
        if not subject_id:
            messagebox.showerror("Sujeto requerido", "Completa el codigo anonimo del sujeto.")
            return

        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{subject_id}_trial-{trial_number}_rutina_4ch.csv"
        self.output_path = DATA_RAW_DIR / filename

        self.csv_file = self.output_path.open("w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_COLUMNS)
        self.csv_writer.writeheader()

        self.recording = True
        self.start_time = time.time()
        self.fase_start_time = self.start_time
        self.rutina_idx = 0
        self.sample_counter = 0
        self.progress["value"] = 0
        self.serial_port.reset_input_buffer()

        self.btn_record.config(state=tk.DISABLED, text="GRABANDO...", bg="gray")
        self.btn_stop.config(state=tk.NORMAL)
        self.status_var.set(f"Grabando rutina completa. Archivo: {self.output_path}")

    def finalizar_grabacion(self) -> None:
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
        self.csv_writer = None
        self.recording = False
        self.progress["value"] = 0
        self.btn_record.config(
            state=tk.NORMAL if self.connected else tk.DISABLED,
            text="2. GRABAR RUTINA COMPLETA",
            bg="#dc2626",
        )
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_instruccion.config(text="FIN SESION", bg="#16a34a", fg="white")
        if self.output_path:
            self.status_var.set(f"Archivo guardado: {self.output_path}")

    def obtener_fase_actual(self) -> tuple[str, int, float]:
        nombre, duracion = RUTINA[self.rutina_idx]
        elapsed_fase = time.time() - self.fase_start_time
        return nombre, duracion, elapsed_fase

    def actualizar_rutina(self) -> str:
        fase_nombre, fase_duracion, elapsed_fase = self.obtener_fase_actual()

        if elapsed_fase >= fase_duracion:
            self.rutina_idx += 1
            if self.rutina_idx >= len(RUTINA):
                self.finalizar_grabacion()
                return fase_nombre
            self.fase_start_time = time.time()
            fase_nombre, fase_duracion, elapsed_fase = self.obtener_fase_actual()

        restante = max(0, int(fase_duracion - elapsed_fase))
        self.progress["value"] = min(100, (elapsed_fase / fase_duracion) * 100)

        if "pausa" in fase_nombre or "reposo" in fase_nombre or "preparacion" in fase_nombre:
            bg = "#444444"
        elif "apertura" in fase_nombre or "extension" in fase_nombre:
            bg = "#2563eb"
        else:
            bg = "#dc2626"

        fase_visible = PHASE_DISPLAY.get(fase_nombre, fase_nombre.replace("_", " ").upper())
        self.lbl_instruccion.config(text=f"{fase_visible}\n{restante}s", bg=bg, fg="white")
        return fase_nombre

    def actualizar(self) -> None:
        if not self.connected or self.serial_port is None:
            return

        movement_label = None
        if self.recording:
            movement_label = self.actualizar_rutina()

        fs_hz = 1000.0
        received_sample = False

        try:
            while self.serial_port.in_waiting:
                linea = self.serial_port.readline().decode("utf-8", errors="replace")
                datos = parsear_linea_dispositivo(linea)
                if datos is None:
                    continue

                valores = [int(datos[f"emg_ch{i}"]) for i in range(1, NUM_CHANNELS + 1)]
                for index, valor in enumerate(valores):
                    self.raw_plot_buffers[index].append(valor + (OFFSET_VISUAL * index))
                    self.raw_filter_buffers[index].append(valor)

                self.board_var.set(f"Placa: {datos['board_id']} | Fs: {datos['sampling_rate_hz']} Hz")
                fs_hz = float(datos["sampling_rate_hz"])
                received_sample = True

                if self.recording and self.csv_writer is not None and movement_label is not None:
                    elapsed = time.time() - self.start_time
                    self.csv_writer.writerow(
                        {
                            "pc_timestamp": datetime.now().isoformat(timespec="milliseconds"),
                            "elapsed_seconds": f"{elapsed:.4f}",
                            "device_timestamp_ms": datos["device_timestamp_ms"],
                            "sample_index": datos["sample_index"],
                            "emg_ch1": datos["emg_ch1"],
                            "emg_ch2": datos["emg_ch2"],
                            "emg_ch3": datos["emg_ch3"],
                            "emg_ch4": datos["emg_ch4"],
                            "emg_filt_ch1": f"{self.latest_filtered_values[0]:.6f}",
                            "emg_filt_ch2": f"{self.latest_filtered_values[1]:.6f}",
                            "emg_filt_ch3": f"{self.latest_filtered_values[2]:.6f}",
                            "emg_filt_ch4": f"{self.latest_filtered_values[3]:.6f}",
                            "movement_label": movement_label,
                            "routine_step": self.rutina_idx + 1,
                            "trial_number": self.trial_var.get().strip(),
                            "subject_id": self.subject_var.get().strip(),
                            "board_id": datos["board_id"],
                            "sampling_rate_hz": datos["sampling_rate_hz"],
                        }
                    )
                    self.sample_counter += 1
                    self.samples_var.set(f"Muestras guardadas: {self.sample_counter}")
        except serial.SerialException as exc:
            self.status_var.set(f"Error serial: {exc}")
            self.desconectar()
            return

        if received_sample:
            self.actualizar_senales_filtradas(fs_hz)

        x_values = list(range(MUESTRAS_GRAFICA))
        for index, line in enumerate(self.raw_lines):
            line.set_data(x_values, list(self.raw_plot_buffers[index]))
        for index, line in enumerate(self.filtered_lines):
            line.set_data(x_values, list(self.filtered_buffers[index]))
        self.canvas.draw_idle()

        self.root.after(20, self.actualizar)

    def desconectar(self) -> None:
        if self.recording:
            self.finalizar_grabacion()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.connected = False

    def cerrar(self) -> None:
        self.desconectar()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    EMGRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
