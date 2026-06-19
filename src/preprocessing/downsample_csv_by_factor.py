"""Downsample pilot CSV files by keeping one sample every N rows inside each label segment."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submuestrear CSV EMG por factor entero.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "pilot_remap_14_to_10")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "pilot_remap_14_to_10_200hz")
    parser.add_argument("--factor", type=int, default=5)
    parser.add_argument("--target-rate", type=int, default=200)
    parser.add_argument("--include-glob", type=str, default="*.csv")
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def downsample_frame(frame: pd.DataFrame, factor: int, target_rate: int) -> pd.DataFrame:
    if "movement_label" not in frame.columns:
        raise RuntimeError("El CSV no contiene la columna movement_label.")

    change = frame["movement_label"].ne(frame["movement_label"].shift()).cumsum()
    pieces = []
    for _segment_id, segment in frame.groupby(change, sort=False):
        pieces.append(segment.iloc[::factor].copy())

    output = pd.concat(pieces, ignore_index=True)
    if "sampling_rate_hz" in output.columns:
        output["sampling_rate_hz"] = target_rate
    return output


def main() -> None:
    args = parse_args()
    if args.factor < 2:
        raise RuntimeError("--factor debe ser 2 o mayor.")

    if args.clear_output and args.output_dir.exists():
        # The output directory is fully controlled by this script.
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.input_dir.glob(args.include_glob))
    if not files:
        raise RuntimeError(f"No se encontraron CSV en {args.input_dir}")

    summary_rows = []
    global_counts: Counter[str] = Counter()
    for source_path in files:
        frame = pd.read_csv(source_path)
        output = downsample_frame(frame, args.factor, args.target_rate)
        destination = args.output_dir / source_path.name.replace("_piloto_10clases", "_piloto_10clases_200hz")
        output.to_csv(destination, index=False)

        counts = Counter(output["movement_label"])
        global_counts.update(counts)
        summary_rows.append(
            {
                "source_file": source_path.name,
                "output_file": destination.name,
                "input_rows": int(len(frame)),
                "output_rows": int(len(output)),
                "labels": dict(sorted(counts.items())),
            }
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "factor": args.factor,
        "target_rate": args.target_rate,
        "input_file_count": len(files),
        "output_file_count": len(summary_rows),
        "global_counts": dict(sorted(global_counts.items())),
        "files": summary_rows,
    }
    (args.output_dir / "downsample_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Submuestreo terminado.")
    print(f"Archivos generados: {len(summary_rows)}")
    print(f"Directorio: {args.output_dir}")
    print("Conteo global por clase:")
    for label, count in sorted(global_counts.items()):
        print(f"- {label}: {count}")


if __name__ == "__main__":
    main()

