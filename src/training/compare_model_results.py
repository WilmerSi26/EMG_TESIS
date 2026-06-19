"""Build a compact comparison report from trained EMG model metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_METADATA = [
    {
        "name": "RandomForest baseline",
        "origin": "Modelo base local de validacion rapida",
        "metadata": PROJECT_ROOT / "models" / "baseline_realtime" / "sklearn_baseline_metadata.json",
        "tflite": None,
        "embedded_note": "No es candidato directo para XIAO; sirve como referencia de escritorio.",
    },
    {
        "name": "Tiny MLP TFLite",
        "origin": "Modelo compacto local para despliegue embebido",
        "metadata": PROJECT_ROOT / "models" / "tiny_mlp_final" / "tiny_mlp_metadata.json",
        "tflite": PROJECT_ROOT / "models" / "tiny_mlp_final" / "emg_tiny_mlp.tflite",
        "embedded_note": "Candidato embebido minimo por tamano y simplicidad.",
    },
    {
        "name": "CNN1D iyeleswarapu adaptado",
        "origin": "Adaptado desde iyeleswarapu/emg-gesture-recognition",
        "metadata": PROJECT_ROOT / "models" / "iyeleswarapu_cnn1d" / "iyeleswarapu_cnn1d_metadata.json",
        "tflite": PROJECT_ROOT / "models" / "iyeleswarapu_cnn1d" / "iyeleswarapu_cnn1d.tflite",
        "embedded_note": "Mejor candidato profundo para probar en TFLite/TFLite Micro por no usar LSTM.",
    },
    {
        "name": "CNN-LSTM ocjorge adaptado",
        "origin": "Adaptado desde ocjorge/CNN-LSTM",
        "metadata": PROJECT_ROOT / "models" / "ocjorge_cnn_lstm" / "ocjorge_cnn_lstm_metadata.json",
        "tflite": PROJECT_ROOT / "models" / "ocjorge_cnn_lstm" / "ocjorge_cnn_lstm.tflite",
        "embedded_note": "Bueno para escritorio; la LSTM fallo al convertir a TFLite builtin.",
    },
    {
        "name": "Inception-LSTM laboratorioAI adaptado",
        "origin": "Inspirado en laboratorioAI/2023-HGR5-CNN_LSTM",
        "metadata": PROJECT_ROOT / "models" / "laboratorio_inception_lstm" / "laboratorio_inception_lstm_metadata.json",
        "tflite": PROJECT_ROOT / "models" / "laboratorio_inception_lstm" / "laboratorio_inception_lstm.tflite",
        "embedded_note": "Referencia temporal; no conviene como primer despliegue en XIAO por LSTM.",
    },
]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: object) -> str:
    if value is None or value == "":
        return "pendiente"
    return f"{float(value) * 100:.2f}%"


def kb(path: Path | None) -> str:
    if not path or not path.exists():
        return "-"
    return f"{path.stat().st_size / 1024:.1f} KB"


def warning_state(metadata: dict, tflite_path: Path | None) -> str:
    warning = metadata.get("tflite_conversion_warning")
    if tflite_path and tflite_path.exists() and not warning:
        return "OK"
    if warning:
        return "Fallo conversion builtin"
    if tflite_path is None:
        return "No aplica"
    return "No generado"


def collect_rows() -> list[dict[str, str]]:
    rows = []
    for item in MODEL_METADATA:
        metadata = load_json(item["metadata"])
        accuracy = metadata.get("test_accuracy")
        rows.append(
            {
                "modelo": item["name"],
                "origen": item["origin"],
                "exactitud_test": pct(accuracy),
                "exactitud_decimal": "" if accuracy is None else f"{float(accuracy):.6f}",
                "epocas": str(metadata.get("epochs_ran", metadata.get("epochs", "-"))),
                "ventana": str(metadata.get("window_size", "-")),
                "stride": str(metadata.get("stride", "-")),
                "clases": str(len(metadata.get("labels", [])) or "-"),
                "tflite": warning_state(metadata, item["tflite"]),
                "tamano_tflite": kb(item["tflite"]),
                "nota_embebida": item["embedded_note"],
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Comparacion final de modelos EMG",
        "",
        "Este reporte compara los modelos entrenados bajo las mismas condiciones experimentales: dataset funcional de 10 clases, cuatro sujetos, 34 CSV validos, muestreo de 200 Hz, ventana de 40 muestras y desplazamiento de 10 muestras.",
        "",
        "| Modelo | Origen | Exactitud test | Epocas | TFLite | Tamano TFLite | Lectura practica |",
        "|---|---|---:|---:|---|---:|---|",
    ]
    for row in rows:
        lines.append("| {modelo} | {origen} | {exactitud_test} | {epocas} | {tflite} | {tamano_tflite} | {nota_embebida} |".format(**row))
    lines.extend(
        [
            "",
            "## Lectura tecnica",
            "",
            "- RandomForest queda como referencia de escritorio para comprobar separabilidad sin exigir despliegue embebido.",
            "- CNN-LSTM de ocjorge conserva una lectura temporal util en computador, pero su conversion TFLite builtin falla por operaciones TensorList asociadas a LSTM.",
            "- CNN1D inspirado en iyeleswarapu es el candidato profundo mas razonable para despliegue, porque logro exportarse a TFLite y no usa capas recurrentes.",
            "- Tiny MLP es el mas compacto y se mantiene como alternativa embebida minima por tamano y simplicidad.",
            "- Inception-LSTM inspirado en laboratorioAI queda como referencia metodologica temporal; tambien falla la conversion TFLite builtin por LSTM.",
            "",
            "## Decision para la siguiente fase",
            "",
            "Para el despliegue en XIAO ESP32S3 conviene priorizar los modelos que generan TFLite sin operaciones recurrentes no soportadas. Los modelos con LSTM pueden mantenerse como comparadores de computador, pero no deberian bloquear el avance del firmware embebido.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    write_csv(rows, PROJECT_ROOT / "results" / "metrics" / "model_comparison.csv")
    write_markdown(rows, PROJECT_ROOT / "results" / "reports" / "comparacion_modelos_referencias.md")
    print("Comparacion generada.")
    for row in rows:
        print(f"- {row['modelo']}: {row['exactitud_test']} | {row['tflite']}")


if __name__ == "__main__":
    main()

