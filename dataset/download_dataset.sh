#!/usr/bin/env bash

set -euo pipefail
trap 'echo "Interrupted." >&2; exit 130' INT TERM TSTP

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

Recommended install:
  uv venv .venv
  source .venv/bin/activate
  uv pip install -r dataset/requirements.txt

Optional environment variable:
  KAGGLE_DATASET_SLUG   defaults to: $DATASET_SLUG
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

find_source_dir() {
  local name="$1"

  if [[ -d "$EXTRACT_DIR/$name" ]]; then
    printf '%s\n' "$EXTRACT_DIR/$name"
    return 0
  fi

  local discovered
  discovered="$(find "$EXTRACT_DIR" -type d -name "$name" -print -quit 2>/dev/null || true)"
  if [[ -n "$discovered" ]]; then
    printf '%s\n' "$discovered"
    return 0
  fi

  return 1
}

find_real_source_dir() {
  if [[ -d "$EXTRACT_DIR/real_videos" ]]; then
    printf '%s\n' "$EXTRACT_DIR/real_videos"
    return 0
  fi

  if [[ -f "$EXTRACT_DIR/labels.csv" && -f "$EXTRACT_DIR/test_metadata.csv" && -d "$EXTRACT_DIR/videos" ]]; then
    printf '%s\n' "$EXTRACT_DIR"
    return 0
  fi

  find_source_dir "real_videos"
}

require_command kaggle
require_command unzip
require_command rsync

mkdir -p "$DOWNLOADS_DIR" "$EXTRACT_DIR" "$REAL_TARGET_DIR"

if [[ -f "$ZIP_PATH" ]]; then
  echo "Reusing existing archive at $ZIP_PATH (delete it to force re-download)"
else
  echo "Downloading Kaggle dataset $DATASET_SLUG"
  kaggle datasets download -d "$DATASET_SLUG" -p "$DOWNLOADS_DIR"
fi

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "Expected archive not found at $ZIP_PATH" >&2
  exit 1
fi

echo "Extracting $ZIP_PATH"
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"
unzip -oq "$ZIP_PATH" -d "$EXTRACT_DIR"

REAL_SOURCE_DIR="$(find_real_source_dir)" || {
  echo "Could not locate a real_videos dataset layout after extraction." >&2
  exit 1
}

echo "Syncing real dataset into $REAL_TARGET_DIR"
rsync -a --delete "$REAL_SOURCE_DIR"/ "$REAL_TARGET_DIR"/
[[ -f "$EXTRACT_DIR/metadata-real.csv" ]] && cp -f "$EXTRACT_DIR/metadata-real.csv" "$DATASET_PARENT/"

SYNTHETIC_SOURCE_DIR="$(find_source_dir synthetic_videos || true)"
if [[ -n "$SYNTHETIC_SOURCE_DIR" ]]; then
  mkdir -p "$SYNTHETIC_TARGET_DIR"
  echo "Syncing synthetic dataset into $SYNTHETIC_TARGET_DIR"
  rsync -a --delete "$SYNTHETIC_SOURCE_DIR"/ "$SYNTHETIC_TARGET_DIR"/
  [[ -f "$EXTRACT_DIR/metadata-synthetic.csv" ]] && cp -f "$EXTRACT_DIR/metadata-synthetic.csv" "$DATASET_PARENT/"
else
  echo "Synthetic dataset not found in the downloaded archive; skipping synthetic sync."
fi

echo "Dataset ready at $REAL_TARGET_DIR"
