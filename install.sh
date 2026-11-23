#!/usr/bin/env bash
# Installe l'environnement Titanic (Linux / macOS) :
#   uv, Python 3.13, venv .venv avec les dépendances de uv.lock, kernel Jupyter « Python (titanic) ».
# Usage : ./install.sh   (relançable sans risque)
set -euo pipefail

cd "$(dirname "$0")"
unset VIRTUAL_ENV  # ignorer un autre venv activé dans le shell

# 1. uv
if ! command -v uv >/dev/null 2>&1; then
    echo ">> Installation de uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo ">> $(uv --version)"

# 2. venv + dépendances (versions exactes de uv.lock)
echo ">> Création de .venv et installation des dépendances"
uv sync --python 3.13

# 3. kernel Jupyter
echo ">> Enregistrement du kernel Jupyter « Python (titanic) »"
uv run python -m ipykernel install --user --name titanic --display-name "Python (titanic)"

cat <<'EOF'

Installation terminée.
  JupyterLab : uv run jupyter lab
  VS Code    : ouvrir le dossier, puis kernel « Python (titanic) » ou .venv
EOF
