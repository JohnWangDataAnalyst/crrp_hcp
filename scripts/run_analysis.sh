#!/usr/bin/env bash
# Run the CRRP HCP manuscript CSV analysis pipeline.
#
# Expected input files in DATA_DIR:
#   crrp_subject_parcel_biomarkers.csv
#   crrp_subject_network_biomarkers.csv
#   crrp_subject_global_biomarkers.csv
#   crrp_shuffle_parcel_biomarkers.csv
#   crrp_shuffle_network_biomarkers.csv
#   crrp_shuffle_global_biomarkers.csv
#   crrp_failed_subjects.csv
#
# Usage:
#   bash scripts/run_analysis.sh
#   bash scripts/run_analysis.sh data/processed outputs/crrp_hcp_analysis
#
# Environment overrides:
#   PYTHON=python3 bash scripts/run_analysis.sh data/processed outputs/crrp_hcp_analysis

set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
DATA_DIR="${1:-data/processed}"
OUT_DIR="${2:-outputs/crrp_hcp_analysis}"

# Resolve repository root as the parent of this scripts/ directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PIPELINE="${REPO_ROOT}/src/crrp_hcp_csv_analysis_pipeline.py"
if [[ ! -f "${PIPELINE}" ]]; then
  # Fallback for a flat folder layout.
  PIPELINE="${REPO_ROOT}/crrp_hcp_csv_analysis_pipeline.py"
fi

if [[ ! -f "${PIPELINE}" ]]; then
  echo "ERROR: Could not find crrp_hcp_csv_analysis_pipeline.py." >&2
  echo "Expected it at src/crrp_hcp_csv_analysis_pipeline.py or repo root." >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}/${DATA_DIR}" && ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: DATA_DIR not found: ${DATA_DIR}" >&2
  echo "Pass the folder containing the CRRP CSV files, e.g.:" >&2
  echo "  bash scripts/run_analysis.sh data/processed outputs/crrp_hcp_analysis" >&2
  exit 1
fi

# Prefer paths relative to the repo root when available.
if [[ -d "${REPO_ROOT}/${DATA_DIR}" ]]; then
  DATA_PATH="${REPO_ROOT}/${DATA_DIR}"
else
  DATA_PATH="${DATA_DIR}"
fi

if [[ "${OUT_DIR}" = /* ]]; then
  OUT_PATH="${OUT_DIR}"
else
  OUT_PATH="${REPO_ROOT}/${OUT_DIR}"
fi

mkdir -p "${OUT_PATH}"

echo "Running CRRP HCP CSV analysis"
echo "  Data:   ${DATA_PATH}"
echo "  Output: ${OUT_PATH}"
echo "  Script: ${PIPELINE}"

"${PYTHON_BIN}" "${PIPELINE}" \
  --input-dir "${DATA_PATH}" \
  --output-dir "${OUT_PATH}"

echo "Done. Tables and figures are in: ${OUT_PATH}"
