# Installe l'environnement Titanic (Windows) :
#   uv, Python 3.13, venv .venv avec les dépendances de uv.lock, kernel Jupyter « Python (titanic) ».
# Usage (PowerShell) : powershell -ExecutionPolicy Bypass -File .\install.ps1   (relançable sans risque)
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue  # ignorer un autre venv activé

# 1. uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host ">> Installation de uv"
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
Write-Host ">> $(uv --version)"

# 2. venv + dépendances (versions exactes de uv.lock)
Write-Host ">> Création de .venv et installation des dépendances"
uv sync --python 3.13
if ($LASTEXITCODE -ne 0) { throw "uv sync a échoué" }

# 3. kernel Jupyter
Write-Host ">> Enregistrement du kernel Jupyter « Python (titanic) »"
uv run python -m ipykernel install --user --name titanic --display-name "Python (titanic)"
if ($LASTEXITCODE -ne 0) { throw "l'enregistrement du kernel a échoué" }

Write-Host ""
Write-Host "Installation terminée."
Write-Host "  JupyterLab : uv run jupyter lab"
Write-Host "  VS Code    : ouvrir le dossier, puis kernel « Python (titanic) » ou .venv"
