"""Audit and prepare the final functional 10-class EMG dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.labels import FINAL_MOVEMENT_CLASSES, LABEL_ALIASES


RAW_CHANNELS = ["emg_ch1", "emg_ch2", "emg_ch3", "emg_ch4"]
REQUIRED_COLUMNS = [
    "pc_timestamp",
    "elapsed_seconds",
    "sample_index",
    *RAW_CHANNELS,
    "movement_label",
    "subject_id",
    "sampling_rate_hz",
]
EXPECTED_LABELS = set(FINAL_MOVEMENT_CLASSES)


def normalize_label(label: object) -> str | None:
    if not isinstance(label, str):
        return None
    return LABEL_ALIASES.get(label.strip().lower())


def inspect_csv(path: Path, min_rows: int, expected_fs: int) -> dict[str, object]:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return {
            "file": path.name,
            "rows": 0,
            "subject_id": "",
            "sampling_rates": "",
            "label_count": 0,
            "labels": "",
            "missing_labels": ",".join(sorted(EXPECTED_LABELS)),
            "status": "error",
            "reason": f"read_error:{exc}",
        }

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    normalized_labels = frame["movement_label"].map(normalize_label) if "movement_label" in frame.columns else pd.Series([])
    labels = sorted(set(normalized_labels.dropna().astype(str)))
    missing_labels = sorted(EXPECTED_LABELS - set(labels))
    subject_values = (
        sorted(frame["subject_id"].dropna().astype(str).unique().tolist())
        if "subject_id" in frame.columns
        else []
    )
    sampling_rates = (
        sorted(frame["sampling_rate_hz"].dropna().astype(str).unique().tolist())
        if "sampling_rate_hz" in frame.columns
        else []
    )

    reasons = []
    if missing_columns:
        reasons.append(f"missing_columns:{'|'.join(missing_columns)}")
    if len(frame) < min_rows:
        reasons.append(f"short_file:{len(frame)}<{min_rows}")
    if missing_labels:
        reasons.append(f"missing_labels:{'|'.join(missing_labels)}")
    if not subject_values:
        reasons.append("missing_subject_id")
    if sampling_rates and str(expected_fs) not in sampling_rates:
        reasons.append(f"unexpected_fs:{'|'.join(sampling_rates)}")

    return {
        "file": path.name,
        "rows": int(len(frame)),
        "subject_id": ",".join(subject_values),
        "sampling_rates": ",".join(sampling_rates),
        "label_count": len(labels),
        "labels": ",".join(labels),
        "missing_labels": ",".join(missing_labels),
        "status": "valid" if not reasons else "exclude",
        "reason": ";".join(reasons),
    }


def write_markdown(summary: dict[str, object], rows: list[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# Auditoria del dataset funcional",
        "",
        f"Fecha de generacion: {summary['created_at']}",
        "",
        f"- Archivos revisados: {summary['file_count']}",
        f"- Archivos validos: {summary['valid_file_count']}",
        f"- Archivos excluidos: {summary['excluded_file_count']}",
        f"- Sujetos validos: {', '.join(summary['valid_subjects'])}",
        f"- Filas validas aproximadas: {summary['valid_rows']}",
        "",
        "## Criterio",
        "",
        "Un CSV se marca como valido si supera el minimo de filas, contiene los cuatro canales EMG, conserva identificador de sujeto, reporta la frecuencia esperada y contiene las 10 clases funcionales normalizadas.",
        "",
        "## Archivos excluidos",
        "",
    ]
    excluded = [row for row in rows if row["status"] != "valid"]
    if not excluded:
        lines.append("No se excluyeron archivos.")
    else:
        lines.append("| Archivo | Filas | Sujeto | Motivo |")
        lines.append("|---|---:|---|---|")
        for row in excluded:
            lines.append(f"| {row['file']} | {row['rows']} | {row['subject_id']} | {row['reason']} |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditar y preparar CSV funcionales de 10 clases.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "final_10clases")
    parser.add_argument("--valid-dir", type=Path, default=PROJECT_ROOT / "data" / "final_10clases_valid")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "results" / "reports")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "results" / "metrics")
    parser.add_argument("--min-rows", type=int, default=12000)
    parser.add_argument("--expected-fs", type=int, default=200)
    parser.add_argument("--copy-valid", action="store_true")
    parser.add_argument("--clear-valid-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.input_dir.glob("*.csv"))
    rows = [inspect_csv(path, args.min_rows, args.expected_fs) for path in csv_files]
    valid_rows = [row for row in rows if row["status"] == "valid"]
    valid_subjects = sorted({str(row["subject_id"]) for row in valid_rows if row["subject_id"]})

    audit_frame = pd.DataFrame(rows)
    audit_csv = args.metrics_dir / "final_dataset_audit.csv"
    audit_frame.to_csv(audit_csv, index=False, encoding="utf-8")

    if args.copy_valid:
        if args.clear_valid_dir and args.valid_dir.exists():
            shutil.rmtree(args.valid_dir)
        args.valid_dir.mkdir(parents=True, exist_ok=True)
        valid_names = {str(row["file"]) for row in valid_rows}
        for source in csv_files:
            if source.name in valid_names:
                shutil.copy2(source, args.valid_dir / source.name)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "valid_dir": str(args.valid_dir),
        "file_count": len(rows),
        "valid_file_count": len(valid_rows),
        "excluded_file_count": len(rows) - len(valid_rows),
        "valid_subjects": valid_subjects,
        "valid_rows": int(sum(int(row["rows"]) for row in valid_rows)),
        "expected_labels": FINAL_MOVEMENT_CLASSES,
        "audit_csv": str(audit_csv),
    }
    (args.metrics_dir / "final_dataset_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, rows, args.reports_dir / "final_dataset_audit.md")

    print("Auditoria terminada.")
    print(f"Archivos revisados: {len(rows)}")
    print(f"Archivos validos: {len(valid_rows)}")
    print(f"Archivos excluidos: {len(rows) - len(valid_rows)}")
    print(f"Sujetos validos: {', '.join(valid_subjects)}")
    if args.copy_valid:
        print(f"CSV validos copiados a: {args.valid_dir}")


if __name__ == "__main__":
    main()
