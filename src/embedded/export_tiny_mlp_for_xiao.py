"""Export the trained Tiny MLP TFLite model and preprocessing data as Arduino headers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def format_float_array(name: str, values: np.ndarray) -> str:
    formatted = ", ".join(f"{float(value):.9g}f" for value in values)
    return f"const float {name}[{len(values)}] = {{{formatted}}};"


def write_model_header(tflite_path: Path, output_path: Path) -> None:
    data = tflite_path.read_bytes()
    chunks = []
    for index in range(0, len(data), 12):
        chunk = ", ".join(f"0x{byte:02x}" for byte in data[index : index + 12])
        chunks.append(f"  {chunk},")
    content = [
        "#pragma once",
        "#include <cstdint>",
        "",
        f"const unsigned int g_emg_model_len = {len(data)};",
        "alignas(16) const unsigned char g_emg_model[] = {",
        *chunks,
        "};",
        "",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def write_metadata_header(metadata: dict, scaler, encoder, output_path: Path) -> None:
    labels = [str(label) for label in encoder.classes_]
    mean = np.asarray(scaler.mean_, dtype=np.float32)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    label_lines = [f'  "{label}",' for label in labels]
    content = [
        "#pragma once",
        "#include <cstddef>",
        "",
        f"const int EMG_NUM_CHANNELS = 4;",
        f"const int EMG_WINDOW_SIZE = {int(metadata.get('window_size', 40))};",
        f"const int EMG_FEATURE_COUNT = {len(mean)};",
        f"const int EMG_CLASS_COUNT = {len(labels)};",
        "",
        format_float_array("EMG_FEATURE_MEAN", mean),
        format_float_array("EMG_FEATURE_SCALE", scale),
        "",
        "const char* const EMG_LABELS[EMG_CLASS_COUNT] = {",
        *label_lines,
        "};",
        "",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exportar modelo Tiny MLP para firmware Arduino/XIAO.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "tiny_mlp_final")
    parser.add_argument("--tflite-file", type=Path, default=None, help="Archivo .tflite especifico para exportar; por defecto usa emg_tiny_mlp.tflite")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "firmware" / "xiao_esp32s3_tinyml_inference" / "generated",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tflite_path = args.tflite_file if args.tflite_file is not None else args.model_dir / "emg_tiny_mlp.tflite"
    scaler_path = args.model_dir / "tiny_feature_scaler.joblib"
    encoder_path = args.model_dir / "tiny_label_encoder.joblib"
    metadata_path = args.model_dir / "tiny_mlp_metadata.json"

    for path in [tflite_path, scaler_path, encoder_path, metadata_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)

    write_model_header(tflite_path, args.output_dir / "model_data.h")
    write_metadata_header(metadata, scaler, encoder, args.output_dir / "emg_metadata.h")

    print("Exportacion Tiny MLP lista.")
    print(f"Modelo: {tflite_path}")
    print(f"Headers: {args.output_dir}")


if __name__ == "__main__":
    main()

