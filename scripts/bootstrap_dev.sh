#!/usr/bin/env bash
set -euo pipefail

# One-click venv setup + dependency install (macOS/Linux).
#
# Usage:
#   ./scripts/bootstrap_dev.sh
#   REQ_FILE=requirements-yolov8.txt ./scripts/bootstrap_dev.sh
#
# Notes:
# - This script is intentionally non-destructive: it creates/uses .venv in repo root.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

REQ_FILE="${REQ_FILE:-requirements.txt}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR 找不到 python3，請先安裝 Python 3.10/3.11" >&2
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "ERROR 找不到 requirements 檔案：${REQ_FILE}" >&2
  echo "   你可以改用：REQ_FILE=requirements-yolov8.txt ./scripts/bootstrap_dev.sh" >&2
  exit 1
fi

echo "== Speedgun bootstrap =="
echo "Repo: ${ROOT_DIR}"
echo "Requirements: ${REQ_FILE}"

if [[ ! -d ".venv" ]]; then
  echo "建立虛擬環境：.venv"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "更新 pip..."
python -m pip install -U pip

echo "安裝依賴..."
python -m pip install -r "${REQ_FILE}"

echo
echo "OK 完成。你現在可以："
echo " - 執行環境健檢：python scripts/doctor.py"
echo " - 開啟 GUI：python gui_app.py"
echo " - 跑 CLI：python pitching_overlay.py -v path/to/video.mp4 --conf 0.03"

