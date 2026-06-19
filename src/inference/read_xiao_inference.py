"""Read TinyML inference results emitted by the XIAO ESP32S3 firmware."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial
from serial.tools import list_ports


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def parse_inference_line(line: str):
    line = line.strip()
    if not line:
        return None
    if line.startswith("#"):
        print(line)
        return None

    parts = line.split(",")
    if parts[0] == "INFER" and len(parts) == 4:
        try:
            return {"kind": "best", "best": (int(parts[1]), parts[2], float(parts[3]))}
        except ValueError:
            return None

    if parts[0] == "INFER_TOP3" and len(parts) >= 10:
        try:
            top3 = []
            cursor = 1
            for _rank in range(3):
                top3.append((int(parts[cursor]), parts[cursor + 1], float(parts[cursor + 2])))
                cursor += 3
            raw_mean = None
            if cursor < len(parts) and parts[cursor] == "RAW_MEAN" and cursor + 4 < len(parts):
                raw_mean = tuple(float(value) for value in parts[cursor + 1: cursor + 5])
            return {"kind": "top3", "top3": top3, "raw_mean": raw_mean}
        except ValueError:
            return None

    return None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leer inferencias TinyML emitidas por la XIAO.")
    parser.add_argument("--port", help="Puerto serial, por ejemplo COM5.")
    parser.add_argument("--baudrate", type=int, default=921600)
    parser.add_argument("--list-ports", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_ports:
        for port in list_serial_ports():
            print(port)
        return

    port = args.port or choose_serial_port()
    serial_port = serial.Serial(port=port, baudrate=args.baudrate, timeout=0.2)
    time.sleep(1.5)
    serial_port.reset_input_buffer()

    print(f"Leyendo inferencia de XIAO en {port}. Ctrl+C para detener.")
    try:
        with serial_port:
            while True:
                raw_line = serial_port.readline().decode("utf-8", errors="replace")
                parsed = parse_inference_line(raw_line)
                if parsed is None:
                    continue
                if parsed["kind"] == "best":
                    index, label, confidence = parsed["best"]
                    print(f"clase={index:02d} | {label:26s} | confianza={confidence:.4f}", end="\r")
                elif parsed["kind"] == "top3":
                    top3_text = " | ".join(
                        f"{index:02d}:{label}={score:.3f}" for index, label, score in parsed["top3"]
                    )
                    raw = parsed.get("raw_mean")
                    raw_text = "" if raw is None else " | raw=" + ",".join(f"{value:.0f}" for value in raw)
                    print(f"{top3_text}{raw_text}".ljust(150), end="\r")
    except KeyboardInterrupt:
        print("\nLectura detenida.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
