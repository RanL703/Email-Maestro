#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv-app
source .venv-app/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.app.txt
echo "App environment ready at .venv-app"
