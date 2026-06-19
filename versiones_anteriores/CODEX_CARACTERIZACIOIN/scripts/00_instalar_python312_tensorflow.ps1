$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ToolsDir = Join-Path $ProjectRoot "tools"
$DownloadDir = Join-Path $ToolsDir "downloads"
$PythonDir = Join-Path $ToolsDir "Python312"
$InstallerPath = Join-Path $DownloadDir "python-3.12.10-amd64.exe"
$PythonUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
$TrainingVenv = Join-Path $ProjectRoot ".venv-training"

New-Item -ItemType Directory -Path $DownloadDir -Force | Out-Null

if (-not (Test-Path $InstallerPath)) {
    Write-Host "Descargando Python 3.12.10 desde python.org..."
    curl.exe -f -L $PythonUrl -o $InstallerPath
}
else {
    Write-Host "Instalador encontrado: $InstallerPath"
}

$InstallerSize = (Get-Item $InstallerPath).Length
if ($InstallerSize -lt 20000000) {
    throw "El instalador descargado es demasiado pequeno ($InstallerSize bytes). Borra $InstallerPath y vuelve a ejecutar."
}

if (-not (Test-Path (Join-Path $PythonDir "python.exe"))) {
    Write-Host "Instalando Python 3.12.10 en: $PythonDir"
    Start-Process -FilePath $InstallerPath -ArgumentList @(
        "/quiet",
        "InstallAllUsers=0",
        "PrependPath=0",
        "Include_launcher=0",
        "Include_test=0",
        "TargetDir=$PythonDir"
    ) -Wait -NoNewWindow
}
else {
    Write-Host "Python 3.12 ya existe en: $PythonDir"
}

$PythonExe = Join-Path $PythonDir "python.exe"
& $PythonExe --version

if (-not (Test-Path (Join-Path $TrainingVenv "Scripts\python.exe"))) {
    Write-Host "Creando entorno de entrenamiento: $TrainingVenv"
    & $PythonExe -m venv $TrainingVenv
}
else {
    Write-Host "Entorno de entrenamiento existente: $TrainingVenv"
}

$TrainingPython = Join-Path $TrainingVenv "Scripts\python.exe"
& $TrainingPython -m pip install --upgrade pip
& $TrainingPython -m pip install -r (Join-Path $ProjectRoot "requirements-training.txt")
& $TrainingPython -c "import tensorflow as tf; print('TensorFlow instalado:', tf.__version__)"

Write-Host ""
Write-Host "Listo. Para entrenar la RNN:"
Write-Host ".\.venv-training\Scripts\python.exe 04_ENTRENAR_RNN_TENSORFLOW.py --input-dir data\raw --epochs 60"
