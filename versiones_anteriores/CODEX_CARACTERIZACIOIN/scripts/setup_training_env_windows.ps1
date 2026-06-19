param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VenvPath = Join-Path $ProjectRoot ".venv-training"

function Test-PythonVersion {
    param([string]$Exe)
    try {
        $versionText = & $Exe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
        $parts = $versionText.Trim().Split(".")
        $major = [int]$parts[0]
        $minor = [int]$parts[1]
        return ($major -eq 3 -and ($minor -eq 11 -or $minor -eq 12))
    }
    catch {
        return $false
    }
}

if (-not $PythonExe) {
    $candidates = @(
        "python",
        "python3",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "C:\Python312\python.exe",
        "C:\Python311\python.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-PythonVersion -Exe $candidate) {
            $PythonExe = $candidate
            break
        }
    }
}

if (-not $PythonExe -or -not (Test-PythonVersion -Exe $PythonExe)) {
    Write-Host "No se encontro Python 3.11 o 3.12."
    Write-Host "Instala Python 3.11 o 3.12 de 64 bits desde https://www.python.org/downloads/"
    Write-Host "Luego ejecuta, por ejemplo:"
    Write-Host ".\scripts\setup_training_env_windows.ps1 -PythonExe C:\Users\Wilmer\AppData\Local\Programs\Python\Python312\python.exe"
    exit 1
}

Write-Host "Usando Python: $PythonExe"
& $PythonExe -m venv $VenvPath
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
& "$VenvPath\Scripts\python.exe" -m pip install -r (Join-Path $ProjectRoot "requirements-training.txt")

Write-Host ""
Write-Host "Entorno listo: $VenvPath"
Write-Host "Para entrenar:"
Write-Host "$VenvPath\Scripts\python.exe src\training\train_rnn.py --input-dir data\raw --epochs 60"
