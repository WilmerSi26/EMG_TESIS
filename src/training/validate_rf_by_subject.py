"""Validate EMG feature classification by leaving one subject out."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.labels import LABEL_ALIASES


RAW_CHANNELS = ["emg_ch1", "emg_ch2", "emg_ch3", "emg_ch4"]


def normalize_label(label: object) -> str | None:
    if not isinstance(label, str):
        return None
    return LABEL_ALIASES.get(label.strip().lower())


def load_dataset(input_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(input_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        missing = [column for column in RAW_CHANNELS + ["movement_label", "subject_id"] if column not in frame.columns]
        if missing:
            print(f"Saltando {path.name}: faltan columnas {missing}")
            continue
        frame = frame.copy()
        frame["source_file"] = path.name
        frame["target_label"] = frame["movement_label"].map(normalize_label)
        frame = frame.dropna(subset=["target_label", "subject_id"])
        for channel in RAW_CHANNELS:
            frame[channel] = pd.to_numeric(frame[channel], errors="coerce")
        frame = frame.dropna(subset=RAW_CHANNELS)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError(f"No hay CSV validos en {input_dir}")
    return pd.concat(frames, ignore_index=True)


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


def iter_segments(dataset: pd.DataFrame):
    for source_file, file_frame in dataset.groupby("source_file", sort=False):
        file_frame = file_frame.reset_index(drop=True)
        change = file_frame["target_label"].ne(file_frame["target_label"].shift()).cumsum()
        for segment_id, segment in file_frame.groupby(change, sort=False):
            yield source_file, int(segment_id), segment


def build_feature_windows(dataset: pd.DataFrame, window_size: int, stride: int):
    x_rows = []
    label_rows = []
    subject_rows = []
    source_rows = []
    for source_file, _segment_id, segment in iter_segments(dataset):
        label = str(segment["target_label"].iloc[0])
        subject_id = str(segment["subject_id"].iloc[0])
        values = segment[RAW_CHANNELS].to_numpy(dtype=np.float32)
        if len(values) < window_size:
            continue
        for start in range(0, len(values) - window_size + 1, stride):
            x_rows.append(extract_features(values[start : start + window_size]))
            label_rows.append(label)
            subject_rows.append(subject_id)
            source_rows.append(source_file)
    if not x_rows:
        raise RuntimeError("No se pudieron crear ventanas.")
    return np.vstack(x_rows), np.array(label_rows), np.array(subject_rows), np.array(source_rows)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    figure, axis = plt.subplots(figsize=(9, 8))
    display.plot(ax=axis, cmap="Blues", values_format="d", colorbar=False)
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def train_model(x_train: np.ndarray, y_train: np.ndarray, trees: int):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    model = RandomForestClassifier(
        n_estimators=trees,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(x_train_scaled, y_train)
    return model, scaler


def write_markdown(
    output_path: Path,
    metadata: dict[str, object],
    fold_rows: list[dict[str, object]],
    labels: list[str],
) -> None:
    lines = [
        "# Validacion RandomForest por sujeto",
        "",
        "Esta validacion usa Leave-One-Subject-Out: en cada pliegue se entrena con tres sujetos y se evalua con el sujeto restante. Sirve para medir generalizacion inter-sujeto, que es mas exigente que una particion aleatoria de ventanas.",
        "",
        f"- Dataset: `{metadata['input_dir']}`",
        f"- Sujetos: {', '.join(metadata['subjects'])}",
        f"- Ventanas: {metadata['window_count']}",
        f"- Ventana temporal: {metadata['window_size']} muestras",
        f"- Salto entre ventanas: {metadata['stride']} muestras",
        f"- Clases: {', '.join(labels)}",
        f"- Exactitud media por sujeto: {metadata['mean_accuracy']:.4f}",
        "",
        "| Sujeto evaluado | Ventanas test | Exactitud |",
        "|---|---:|---:|",
    ]
    for row in fold_rows:
        lines.append(f"| {row['held_out_subject']} | {row['test_windows']} | {row['accuracy']:.4f} |")
    lines.extend(
        [
            "",
            "## Interpretacion",
            "",
            "Si esta metrica es menor que la validacion aleatoria, no significa que el sistema falle; significa que las senales sEMG cambian entre personas. Para control protesico real se puede justificar calibracion por usuario y reportar tambien validacion intra-sujeto por trials.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validar clasificador EMG dejando un sujeto fuera.")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data" / "final_10clases_valid")
    parser.add_argument("--models-dir", type=Path, default=PROJECT_ROOT / "models" / "rf_final")
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "results" / "figures")
    parser.add_argument("--metrics-dir", type=Path, default=PROJECT_ROOT / "results" / "metrics")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "results" / "reports")
    parser.add_argument("--window-size", type=int, default=40)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--save-final-model", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.input_dir)
    x, labels_text, groups, source_files = build_feature_windows(dataset, args.window_size, args.stride)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_text)
    labels = [str(label) for label in encoder.classes_]

    logo = LeaveOneGroupOut()
    fold_rows = []
    all_true = []
    all_pred = []
    for fold_index, (train_index, test_index) in enumerate(logo.split(x, y, groups), start=1):
        held_out = sorted(set(groups[test_index]))
        model, scaler = train_model(x[train_index], y[train_index], args.trees)
        y_pred = model.predict(scaler.transform(x[test_index]))
        accuracy = float(accuracy_score(y[test_index], y_pred))
        fold_rows.append(
            {
                "fold": fold_index,
                "held_out_subject": ",".join(held_out),
                "train_windows": int(len(train_index)),
                "test_windows": int(len(test_index)),
                "accuracy": accuracy,
            }
        )
        all_true.extend(y[test_index].tolist())
        all_pred.extend(y_pred.tolist())

        report = classification_report(y[test_index], y_pred, target_names=labels, zero_division=0)
        (args.metrics_dir / f"rf_loso_{held_out[0]}_report.txt").write_text(report, encoding="utf-8")
        plot_confusion(
            y[test_index],
            y_pred,
            labels,
            args.figures_dir / f"rf_loso_{held_out[0]}_confusion_matrix.png",
        )

    all_true_np = np.array(all_true)
    all_pred_np = np.array(all_pred)
    aggregate_accuracy = float(accuracy_score(all_true_np, all_pred_np))
    mean_accuracy = float(np.mean([row["accuracy"] for row in fold_rows]))

    pd.DataFrame(fold_rows).to_csv(args.metrics_dir / "rf_loso_subject_metrics.csv", index=False)
    aggregate_report = classification_report(all_true_np, all_pred_np, target_names=labels, zero_division=0)
    (args.metrics_dir / "rf_loso_subject_report.txt").write_text(aggregate_report, encoding="utf-8")
    plot_confusion(all_true_np, all_pred_np, labels, args.figures_dir / "rf_loso_subject_confusion_matrix.png")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "window_size": args.window_size,
        "stride": args.stride,
        "trees": args.trees,
        "labels": labels,
        "subjects": sorted(set(groups.tolist())),
        "source_file_count": int(len(set(source_files.tolist()))),
        "window_count": int(len(x)),
        "folds": fold_rows,
        "mean_accuracy": mean_accuracy,
        "aggregate_accuracy": aggregate_accuracy,
    }
    (args.metrics_dir / "rf_loso_subject_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.reports_dir / "rf_loso_subject_validation.md", metadata, fold_rows, labels)

    if args.save_final_model:
        model, scaler = train_model(x, y, args.trees)
        joblib.dump(model, args.models_dir / "emg_rf_final.joblib")
        joblib.dump(scaler, args.models_dir / "rf_feature_scaler.joblib")
        joblib.dump(encoder, args.models_dir / "rf_label_encoder.joblib")
        (args.models_dir / "rf_final_metadata.json").write_text(
            json.dumps(metadata | {"trained_on_all_valid_windows": True}, indent=2),
            encoding="utf-8",
        )

    print("Validacion por sujeto terminada.")
    print(f"Ventanas: {len(x)} | Sujetos: {', '.join(sorted(set(groups.tolist())))}")
    print(f"Exactitud media LOSO: {mean_accuracy:.4f}")
    print(f"Exactitud agregada LOSO: {aggregate_accuracy:.4f}")
    if args.save_final_model:
        print(f"Modelo final RF: {args.models_dir / 'emg_rf_final.joblib'}")


if __name__ == "__main__":
    main()
