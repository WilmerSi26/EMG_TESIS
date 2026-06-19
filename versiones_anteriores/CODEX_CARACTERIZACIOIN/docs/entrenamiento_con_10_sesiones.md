# Entrenamiento con las sesiones adquiridas

Ya existen CSV en:

```text
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\data\raw
```

## Problema actual

Tu entorno actual es:

```text
Python 3.14.4
```

TensorFlow no instala ahi. El error:

```text
ERROR: No matching distribution found for tensorflow
```

significa que pip no encontro una rueda compatible con esa version de Python.

## Solucion para entrenar la RNN

Instala Python 3.11 o 3.12 de 64 bits. Luego abre una terminal en:

```text
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN
```

y ejecuta:

```powershell
.\scripts\setup_training_env_windows.ps1
```

Si el script no encuentra Python, indica la ruta manualmente:

```powershell
.\scripts\setup_training_env_windows.ps1 -PythonExe C:\Users\Wilmer\AppData\Local\Programs\Python\Python312\python.exe
```

Cuando termine, entrena con:

```powershell
.\.venv-training\Scripts\python.exe src\training\train_rnn.py --input-dir data\raw --epochs 60
```

## Prueba inmediata sin TensorFlow

Para validar que tus CSV sirven antes de instalar TensorFlow, ejecuta:

```powershell
.\.venv\Scripts\python.exe src\training\train_sklearn_baseline.py --input-dir data\raw
```

Este modelo no es RNN, pero da una primera exactitud y matriz de confusion.

Salidas:

```text
models/emg_sklearn_baseline.joblib
results/metrics/sklearn_baseline_report.txt
results/figures/sklearn_baseline_confusion_matrix.png
```

## En PyCharm

Para la RNN, cambia el interprete del proyecto o crea una configuracion nueva
usando:

```text
D:\ESPOCH\TESIS\CODEX\CARACTERIZACIOIN\.venv-training\Scripts\python.exe
```

No instales TensorFlow dentro de `.venv` si ese entorno sigue usando Python
3.14.
