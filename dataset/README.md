# Dataset setup

This repository expects the baseline-ready dataset at:

```text
dataset/real_videos/
  labels.csv
  test_metadata.csv
  videos/
    ...
```

The Kaggle archive may also include:

```text
dataset/synthetic_videos/
  ...
```

## Download from Kaggle

The repository includes a helper that downloads the Kaggle dataset and normalizes the extracted files into the layout above:

Install the CLI with `uv`:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r dataset/requirements.txt
```

Then run:

```bash
bash dataset/download_dataset.sh
```

What that command does:

- calls [download_dataset.sh](/Users/lukaspicek/Documents/Projects/ACCIDENT/dataset/download_dataset.sh)
- downloads the Kaggle archive for `picekl/accident`
- extracts it into `dataset/raw/kaggle/`
- syncs the real split into `dataset/real_videos/`
- syncs the synthetic split into `dataset/synthetic_videos/` when present

You can also see its built-in usage text with:

```bash
bash dataset/download_dataset.sh --help
```

## Kaggle CLI requirements

You need the `kaggle` CLI installed and authenticated before running the script.

Typical setup:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r dataset/requirements.txt
```

Then authenticate with either:

- `kaggle auth login`
- a `~/.kaggle/kaggle.json` API token file
- `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables

## Notes

- The downloaded archive is stored under `dataset/downloads/`.
- The raw extracted files are stored under `dataset/raw/kaggle/`.
- The normalized baseline-ready files live under `dataset/real_videos/`.
- Synthetic files, when included in the Kaggle archive, are normalized under `dataset/synthetic_videos/`.
- The baselines accept either `--dataset-path dataset` or `--dataset-path dataset/real_videos`.
