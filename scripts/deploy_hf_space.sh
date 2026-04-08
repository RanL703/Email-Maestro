#!/usr/bin/env bash
set -euo pipefail

if [[ -f ".env.hf.space" ]]; then
  while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="${raw_line#"${raw_line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    if [[ -z "${line}" || "${line}" == \#* || "${line}" != *=* ]]; then
      continue
    fi
    key="${line%%=*}"
    value="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    export "${key}=${value}"
  done < .env.hf.space
fi

PYTHON_BIN="${PYTHON_BIN:-.venv-training/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

exec "${PYTHON_BIN}" scripts/deploy_hf_space.py "$@"
