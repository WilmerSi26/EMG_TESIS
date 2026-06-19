"""Paso 4: entrenar la RNN/LSTM con TensorFlow."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_training_python() -> None:
    project_root = Path(__file__).resolve().parent
    training_python = project_root / ".venv-training" / "Scripts" / "python.exe"
    current_python = Path(sys.executable).resolve()
    if training_python.exists() and current_python != training_python.resolve():
        print(f"Reiniciando con entorno TensorFlow: {training_python}")
        os.execv(str(training_python), [str(training_python), str(Path(__file__).resolve()), *sys.argv[1:]])


ensure_training_python()

from src.training.train_rnn import main


if __name__ == "__main__":
    main()
