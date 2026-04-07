#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv-training
source .venv-training/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.training.txt
python -m ipykernel install --user --name scalerhack2-training --display-name "Python (scalerhack2-training)"
echo "Training environment ready at .venv-training with Jupyter kernel scalerhack2-training"
