"""
Adquisicion de sesiones EMG para caracterizacion de movimientos.

Lee 4 canales desde XIAO ESP32S3 por USB Serial y guarda un CSV con fases de
reposo y tres movimientos: cierre, apertura y pinza.

Formato esperado desde la placa:
timestamp_ms,sample_index,emg_ch1,emg_ch2,emg_ch3,emg_ch4,board_id,sampling_rate_hz
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import serial
from serial.tools import list_ports


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_CHANNELS = 4
DEFAULT_MOVEMENTS = (
    "dedo_pulgar_cierre",
    "dedo_pulgar_apertura",
    "dedo_indice_cierre",
    "dedo_indice_apertura",
    "dedo_medio_cierre",
    "dedo_medio_apertura",
    "dedo_anular_cierre",
    "dedo_anular_apertura",
    "dedo_menique_cierre",
    "dedo_menique_apertura",
    "cierre",
    "apertura",
    "pinza",
)
CSV_COLUMNS = [
    "pc_timestamp",
    "elapsed_seconds",
    "device_timestamp_ms",
    "sample_index",
    "emg_ch1",
    "emg_ch2",
    "emg_ch3",
    "emg_ch4",
    "movement_label",
    "phase_kind",
    "repetition",
    "session_id",
    "subject_id",
    "board_id",
    "sampling_rate_hz",
]


@dataclass
class Phase:
    label: str
    phase_kind: str
    repetition: int
    duration_seconds: float


@dataclass
class Config:
    port: str
    baudrate: int
    subject_id: str
    session_id: str
    repetitions: int
    movement_seconds: float
    rest_seconds: float
    output_dir: Path
    plot_window: int
    no_plot: bool


def list_serial_ports() -> list[str]:
    return [port.device for port in list_ports.comports()]


def choose_serial_port() -> str:
    while True:
        ports = list_serial_ports()
        if not ports:
            print("No se detectaron puertos seriales.")
            print("1. Conecta la XIAO ESP32S3 con un cable USB de datos.")
            print("2. Cierra el Monitor Serial de Arduino IDE si esta abierto.")
            print("3. Presiona RESET/BOOT si Windows no crea el puerto COM.")
            answer = input("Presiona ENTER para volver a buscar o escribe q para salir: ").strip().lower()
            if answer in {"q", "quit", "salir"}:
                raise RuntimeError("No se detecto ningun puerto COM disponible.")
            continue

        if len(ports) == 1:
            print(f"Puerto detectado automaticamente: {ports[0]}")
            return ports[0]

        print("Puertos disponibles:")
        for index, port in enumerate(ports, start=1):
            print(f"{index}. {port}")

        selected = input("Escribe el numero del puerto a usar: ").strip()
        try:
            selected_index = int(selected)
        except ValueError:
            print("Entrada invalida. Usa un numero de la lista.")
            continue
        if 1 <= selected_index <= len(ports):
            return ports[selected_index - 1]
        print("Numero fuera de rango.")


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
        "device_timestamp_ms": timestamp_ms,
        "sample_index": sample_index,
        "emg_ch1": ch1,
        "emg_ch2": ch2,
        "emg_ch3": ch3,
        "emg_ch4": ch4,
        "board_id": board_id,
        "sampling_rate_hz": sampling_rate_hz,
    }


def build_phases(config: Config) -> list[Phase]:
    phases: list[Phase] = [Phase("preparacion", "preparacion", 0, 3.0)]
    for repetition in range(1, config.repetitions + 1):
        for movement in DEFAULT_MOVEMENTS:
            phases.append(Phase("reposo", "reposo", repetition, config.rest_seconds))
            phases.append(Phase(movement, "movimiento", repetition, config.movement_seconds))
    phases.append(Phase("reposo", "reposo_final", config.repetitions, 3.0))
    return phases


def build_output_path(config: Config) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = (
        f"{timestamp}_{config.subject_id}_{config.session_id}_"
        f"{config.repetitions}rep_dedos_apertura_cierre_4ch.csv"
    )
    return config.output_dir / filename


def open_serial_port(port: str, baudrate: int) -> serial.Serial:
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=0.02)
        time.sleep(1.5)
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as exc:
        ports = ", ".join(list_serial_ports()) or "ninguno"
        raise RuntimeError(
            f"No se pudo abrir {port}. Cierra Arduino Serial Monitor. "
            f"Puertos disponibles: {ports}"
        ) from exc


def create_live_plot(plot_window: int):
    buffers = [deque([0] * plot_window, maxlen=plot_window) for _ in range(NUM_CHANNELS)]
    figure, axis = plt.subplots(figsize=(11, 5))
    colors = ["#2563eb", "#f97316", "#16a34a", "#dc2626"]
    lines = []
    offset = 4500
    x_values = list(range(plot_window))
    for channel in range(NUM_CHANNELS):
        (line,) = axis.plot(x_values, list(buffers[channel]), color=colors[channel], linewidth=0.8, label=f"CH{channel + 1}")
        lines.append(line)
    axis.set_title("sEMG 4 canales en vivo")
    axis.set_xlabel("Muestras recientes")
    axis.set_ylabel("ADC raw + offset")
    axis.set_ylim(0, offset * NUM_CHANNELS)
    axis.set_xlim(0, plot_window)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")
    plt.tight_layout()
    plt.ion()
    return figure, axis, lines, buffers, offset


def update_live_plot(lines, buffers, values: list[int], offset: int) -> None:
    for channel, value in enumerate(values):
        buffers[channel].append(value + (offset * channel))
        lines[channel].set_ydata(list(buffers[channel]))
    plt.pause(0.001)


def run_acquisition(config: Config) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    phases = build_phases(config)
    output_path = build_output_path(config)

    serial_port = open_serial_port(config.port, config.baudrate)
    plot_objects = None if config.no_plot else create_live_plot(config.plot_window)
    rows_written = 0
    valid_samples = 0

    print(f"Puerto abierto: {config.port} a {config.baudrate} baudios")
    print(f"Archivo de salida: {output_path}")
    print("La rutina empieza en 2 segundos...")
    time.sleep(2.0)

    routine_start = time.time()
    try:
        with serial_port, output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
            writer.writeheader()

            for phase in phases:
                print(
                    f"[rep {phase.repetition}] {phase.phase_kind.upper()}: "
                    f"{phase.label} por {phase.duration_seconds:.1f}s"
                )
                phase_end = time.time() + phase.duration_seconds
                last_second_printed = -1

                while time.time() < phase_end:
                    raw_line = serial_port.readline().decode("utf-8", errors="replace")
                    parsed = parse_device_line(raw_line)
                    if parsed is None:
                        continue

                    values = [int(parsed[f"emg_ch{i}"]) for i in range(1, NUM_CHANNELS + 1)]
                    valid_samples += 1

                    writer.writerow(
                        {
                            "pc_timestamp": datetime.now().isoformat(timespec="milliseconds"),
                            "elapsed_seconds": f"{time.time() - routine_start:.4f}",
                            "device_timestamp_ms": parsed["device_timestamp_ms"],
                            "sample_index": parsed["sample_index"],
                            "emg_ch1": parsed["emg_ch1"],
                            "emg_ch2": parsed["emg_ch2"],
                            "emg_ch3": parsed["emg_ch3"],
                            "emg_ch4": parsed["emg_ch4"],
                            "movement_label": phase.label,
                            "phase_kind": phase.phase_kind,
                            "repetition": phase.repetition,
                            "session_id": config.session_id,
                            "subject_id": config.subject_id,
                            "board_id": parsed["board_id"],
                            "sampling_rate_hz": parsed["sampling_rate_hz"],
                        }
                    )
                    rows_written += 1

                    if plot_objects is not None:
                        figure, _axis, lines, buffers, offset = plot_objects
                        update_live_plot(lines, buffers, values, offset)
                        if not plt.fignum_exists(figure.number):
                            print("Ventana de grafica cerrada. La adquisicion continua sin grafica.")
                            plot_objects = None

                    remaining = int(max(0, phase_end - time.time()))
                    if remaining != last_second_printed:
                        print(f"  restante: {remaining}s", end="\r")
                        last_second_printed = remaining
                print()

    except KeyboardInterrupt:
        print("\nAdquisicion detenida por el usuario.")
    finally:
        plt.ioff()

    if valid_samples == 0:
        raise RuntimeError("No se recibieron muestras validas desde la placa.")

    print(f"Muestras guardadas: {rows_written}")
    print(f"CSV guardado: {output_path}")
    return output_path


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Adquirir CSV EMG para cierre, apertura y pinza.")
    parser.add_argument("--port", help="Puerto serial, por ejemplo COM5.")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--subject-id", default="S001")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--movement-seconds", type=float, default=4.0)
    parser.add_argument("--rest-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--plot-window", type=int, default=500)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--list-ports", action="store_true")
    args = parser.parse_args()

    if args.list_ports:
        ports = list_serial_ports()
        print("Puertos disponibles:")
        for port in ports:
            print(f"- {port}")
        if not ports:
            print("- No se detectaron puertos seriales.")
        sys.exit(0)

    if not args.port:
        args.port = choose_serial_port()
    if args.repetitions < 1:
        parser.error("--repetitions debe ser 1 o mayor.")

    session_id = args.session_id or datetime.now().strftime("SES%Y%m%d%H%M%S")
    return Config(
        port=args.port,
        baudrate=args.baudrate,
        subject_id=args.subject_id,
        session_id=session_id,
        repetitions=args.repetitions,
        movement_seconds=args.movement_seconds,
        rest_seconds=args.rest_seconds,
        output_dir=args.output_dir,
        plot_window=args.plot_window,
        no_plot=args.no_plot,
    )


def main() -> None:
    try:
        config = parse_args()
        run_acquisition(config)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
