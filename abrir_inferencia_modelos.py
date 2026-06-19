from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PYTHON_TF = PROJECT_ROOT / ".venv-tf" / "Scripts" / "python.exe"
DASHBOARD = PROJECT_ROOT / "src" / "inference" / "live_inference_compare_models.py"


def main() -> int:
    if not PYTHON_TF.exists():
        print(f"No existe el entorno TensorFlow: {PYTHON_TF}")
        print("Ejecuta primero setup_workstation.ps1 o crea .venv-tf con Python 3.12.")
        return 1
    command = [str(PYTHON_TF), str(DASHBOARD), "--models", "all"]
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
