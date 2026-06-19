"""Create a pilot dataset by remapping the previous 14-class CSV files to 10 functional classes."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import PILOT_REMAP_DIR, PILOT_SOURCE_DIR, PILOT_SOURCE_GLOB
from src.common.labels import FINAL_MOVEMENT_CLASSES, normalize_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remapear CSV de 14 clases a 10 clases funcionales.")
    parser.add_argument("--source-dir", type=Path, default=PILOT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=PILOT_REMAP_DIR)
    parser.add_argument("--include-glob", type=str, default=PILOT_SOURCE_GLOB)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clear_output and args.output_dir.exists():
        # The output directory is fully controlled by this script.
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.source_dir.glob(args.include_glob))
    if not files:
        raise RuntimeError(f"No se encontraron CSV con patron {args.include_glob} en {args.source_dir}")

    summary_rows = []
    global_counts: Counter[str] = Counter()
    skipped_files = []

    for source_path in files:
        frame = pd.read_csv(source_path)
        if "movement_label" not in frame.columns:
            skipped_files.append({"file": source_path.name, "reason": "sin columna movement_label"})
            continue

        remapped = frame["movement_label"].map(normalize_label)
        valid_mask = remapped.notna()
        output = frame.loc[valid_mask].copy()
        output["original_movement_label"] = output["movement_label"]
        output["movement_label"] = remapped.loc[valid_mask]

        if output.empty:
            skipped_files.append({"file": source_path.name, "reason": "sin etiquetas remapeables"})
            continue

        destination = args.output_dir / source_path.name.replace("_rutina_4ch", "_piloto_10clases")
        output.to_csv(destination, index=False)

        counts = Counter(output["movement_label"])
        global_counts.update(counts)
        summary_rows.append(
            {
                "source_file": source_path.name,
                "output_file": destination.name,
                "input_rows": int(len(frame)),
                "output_rows": int(len(output)),
                "dropped_rows": int(len(frame) - len(output)),
                "labels": dict(sorted(counts.items())),
            }
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "include_glob": args.include_glob,
        "source_file_count": len(files),
        "output_file_count": len(summary_rows),
        "classes": FINAL_MOVEMENT_CLASSES,
        "global_counts": dict(sorted(global_counts.items())),
        "files": summary_rows,
        "skipped_files": skipped_files,
    }
    (args.output_dir / "remap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Remapeo terminado.")
    print(f"Archivos fuente: {len(files)}")
    print(f"Archivos generados: {len(summary_rows)}")
    print(f"Directorio: {args.output_dir}")
    print("Conteo global por clase:")
    for label in FINAL_MOVEMENT_CLASSES:
        print(f"- {label}: {global_counts.get(label, 0)}")


if __name__ == "__main__":
    main()

