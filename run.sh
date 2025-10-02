#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data/raw/rn outputs

# Choose Python
PY=python3
command -v python3 >/dev/null 2>&1 || PY=python

# venv (local to repo)
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# --- pick the notebook to run ---
# 1) first CLI arg wins, 2) NOTEBOOK env var, 3) fallback = find any *.ipynb in repo root
NOTEBOOK="${1:-${NOTEBOOK:-}}"
if [ -z "${NOTEBOOK}" ]; then
  NOTEBOOK="$(ls -1 *.ipynb 2>/dev/null | head -n 1 || true)"
fi
if [ -z "${NOTEBOOK}" ]; then
  echo "❌ No notebook specified. Run: ./run.sh 2025H5_groupE_EDA.ipynb"
  exit 1
fi

echo "▶️ Executing notebook: ${NOTEBOOK}"
papermill "${NOTEBOOK}" "outputs/executed_$(basename "${NOTEBOOK}")" -k python3

echo "✅ Done. Check outputs/ for executed notebook and any CSV/figures your notebook saves."
