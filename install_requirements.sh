#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INCLUDE_3RD="false"
PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
TORCH_CPU="${TORCH_CPU:-true}"

usage() {
  cat <<'EOF'
Usage: ./install_requirements.sh [--include-3rd]

Installs Python requirements for:
  - 1st Place (pip -r requirements.txt)
  - 2nd Place (pip install ./2nd Place)

Optional:
  --include-3rd   Attempt to install the 3rd Place conda environment

Notes:
  - Set PYTHON_BIN to control which Python to use (default: python3).
  - 3rd Place expects conda + CUDA; this script will skip it unless requested.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-3rd)
      INCLUDE_3RD="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

echo "[info] Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -m pip install --upgrade pip

echo "[info] Installing 1st Place requirements"
if [[ "${PY_VERSION}" == "3.12" ]]; then
  TMP_REQ="$(mktemp)"
  "${PYTHON_BIN}" - <<PY
from pathlib import Path

req_path = Path(r"${ROOT_DIR}/1st Place/requirements.txt")
raw = req_path.read_bytes()
for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
    try:
        text = raw.decode(enc)
        break
    except UnicodeDecodeError:
        continue
else:
    raise SystemExit("Unable to decode requirements.txt")

lines = []
for line in text.splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        lines.append(line)
        continue
    name = stripped.split("==")[0].strip().lower()
    if name in {"numpy", "pandas", "scipy"}:
        continue
    lines.append(stripped)

Path(r"${TMP_REQ}").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
  "${PYTHON_BIN}" -m pip install -r "${TMP_REQ}"
  "${PYTHON_BIN}" -m pip install "numpy==1.26.4" "pandas==2.2.2" "scipy==1.11.4"
  rm -f "${TMP_REQ}"
else
  "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/1st Place/requirements.txt"
fi

echo "[info] Installing 2nd Place requirements"
if [[ "${TORCH_CPU}" == "true" ]]; then
  PIP_INDEX_URL="https://download.pytorch.org/whl/cpu" \
  PIP_EXTRA_INDEX_URL="https://pypi.org/simple" \
  "${PYTHON_BIN}" -m pip install "${ROOT_DIR}/2nd Place"
else
  "${PYTHON_BIN}" -m pip install "${ROOT_DIR}/2nd Place"
fi

if [[ "${INCLUDE_3RD}" == "true" ]]; then
  if command -v conda >/dev/null 2>&1; then
    echo "[info] Installing 3rd Place conda environment"
    conda env create -f "${ROOT_DIR}/3rd Place/requirements_snomed.yml"
  else
    echo "[warn] conda not found; skipping 3rd Place environment"
  fi
else
  echo "[info] Skipping 3rd Place (use --include-3rd to attempt conda install)"
fi
