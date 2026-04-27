#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DATASET_SLUG="${KAGGLE_DATASET_SLUG:-picekl/accident}"
DATASET_PARENT="${1:-$REPO_ROOT/dataset}"

DOWNLOADS_DIR="$DATASET_PARENT/downloads"
EXTRACT_DIR="$DATASET_PARENT/raw/kaggle"

REAL_TARGET_DIR="$DATASET_PARENT/real_videos"
SYNTHETIC_TARGET_DIR="$DATASET_PARENT/synthetic_videos"

ZIP_NAME="${DATASET_SLUG##*/}.zip"
ZIP_PATH="$DOWNLOADS_DIR/$ZIP_NAME"

print_usage() {
  cat <<EOF
Download the ACCIDENT dataset from Kaggle and normalize it into:
  $REPO_ROOT/dataset/real_videos
  $REPO_ROOT/dataset/synthetic_videos

Usage:
  bash dataset/download_dataset.sh
  bash dataset/download_dataset.sh /custom/dataset/root

Requirements:
  - kaggle CLI installed and authenticated
  - unzip
  - rsync
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    if [[ "$cmd" == "kaggle" ]]; then
      echo "Install the Kaggle CLI first, then authenticate before rerunning:" >&2
      echo "  uv venv .venv" >&2
      echo "  source .venv/bin/activate" >&2
      echo "  uv pip install -r dataset/requirements.txt" >&2
      echo "See dataset/README.md for the expected setup." >&2
    fi
    exit 1
  fi
}

require_command kaggle
require_command unzip
require_command rsync

mkdir -p "$DOWNLOADS_DIR" "$EXTRACT_DIR"

echo "Downloading Kaggle dataset: $DATASET_SLUG"
kaggle datasets download -d "$DATASET_SLUG" -p "$DOWNLOADS_DIR" -o

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Expected archive not found at $ZIP_PATH" >&2
  exit 1
fi

echo "Extracting $ZIP_PATH"
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
unzip -oq "$ZIP_PATH" -d "$EXTRACT_DIR"

# --- Expected RAW STRUCTURE ---
REAL_SRC="$EXTRACT_DIR/real_videos"
SYNTH_SRC="$EXTRACT_DIR/synthetic_videos"

REAL_META="$EXTRACT_DIR/metadata-real.csv"
SYNTH_META="$EXTRACT_DIR/metadata-synthetic.csv"
CLASSES_FILE="$EXTRACT_DIR/annotation_classes.yaml"

# --- VALIDATION ---
if [[ ! -d "$REAL_SRC" ]]; then
  echo "Missing directory: real_videos" >&2
  exit 1
fi

if [[ ! -f "$REAL_META" ]]; then
  echo "Missing file: metadata-real.csv" >&2
  exit 1
fi

# --- PREPARE REAL DATASET ---
echo "Preparing real dataset → $REAL_TARGET_DIR"
rm -rf "$REAL_TARGET_DIR"
mkdir -p "$REAL_TARGET_DIR"

rsync -a "$REAL_SRC"/ "$REAL_TARGET_DIR"/
cp "$REAL_META" "$DATASET_PARENT/metadata-real.csv"

# --- PREPARE SYNTHETIC DATASET ---
if [[ -d "$SYNTH_SRC" && -f "$SYNTH_META" ]]; then
  echo "Preparing synthetic dataset → $SYNTHETIC_TARGET_DIR"
  rm -rf "$SYNTHETIC_TARGET_DIR"
  mkdir -p "$SYNTHETIC_TARGET_DIR"

  rsync -a "$SYNTH_SRC"/ "$SYNTHETIC_TARGET_DIR"/
  cp "$SYNTH_META" "$DATASET_PARENT/metadata-synthetic.csv"

  if [[ -f "$CLASSES_FILE" ]]; then
    cp "$CLASSES_FILE" "$DATASET_PARENT/"
  fi
else
  echo "Synthetic dataset not found or incomplete — skipping."
fi

echo "Done."
echo "Real dataset:      $REAL_TARGET_DIR"
echo "Synthetic dataset: $SYNTHETIC_TARGET_DIR"
