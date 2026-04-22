# LLM Baselines Experiments

This folder contains the experiment pipeline used for the accident paper baselines.

The experiments are organized into three stages:

- `baselines/temporal`: predict accident timing from video
- `baselines/spatial`: predict accident location + VLM classification
- `baselines/classification`: feature-based classification experiments

---

## 1) Environment setup

From `baselines/llm`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Expected data layout

The tracked temporal script accepts an explicit dataset root:

```bash
python baselines/temporal/main.py --model qwen --dataset-path ../../dataset --range 0:10
```

Expected layout:

```text
dataset/
  real_videos/
    labels.csv
    test_metadata.csv
    videos/
      ...
```

If your local copy lives elsewhere, pass `--dataset-path /absolute/path/to/real_videos`.

Important: some of the notebooks may still contain project-specific paths saved in notebook cells. Before running them, search for path literals and update them to your local dataset location.

## 3) Recommended execution order

### Step A: Temporal reasoning

Use the script for full or chunked runs:

```bash
python baselines/temporal/main.py --model qwen --dataset-path ../../dataset --range 0:676
python baselines/temporal/main.py --model qwen --dataset-path ../../dataset --range 676:1352
python baselines/temporal/main.py --model qwen --dataset-path ../../dataset --range 1352:2027

python baselines/temporal/main.py --model molmo --dataset-path ../../dataset --range 0:676
python baselines/temporal/main.py --model molmo --dataset-path ../../dataset --range 676:1352
python baselines/temporal/main.py --model molmo --dataset-path ../../dataset --range 1352:2027
```

Then open `baselines/temporal/analysis.ipynb` to:

- merge part files into one table
- export `results/temporal_pred.csv`
- compute temporal metrics

`baselines/temporal/dev.ipynb` is local scratch space and is intentionally not part of the tracked experiment workflow.

### Step B: Spatial + VLM classification

Run:

- `baselines/spatial/qwen.ipynb`
- `baselines/spatial/molmo.ipynb`

Both notebooks require `../temporal/results/temporal_pred.csv`.

Then run `baselines/spatial/analysis.ipynb` to reproduce aggregate spatial/classification tables.

### Step C: Feature extraction baseline

Run:

- `baselines/classification/extract_features.ipynb`
- `baselines/classification/analysis.ipynb`

These notebooks consume temporal (`temporal_pred.csv`) and spatial outputs (`spatial/results/*.pkl`).

---

## Practical notes

- The temporal script writes partial outputs to `baselines/temporal/results/<model>_temporal_parts/`.
- Model weights are loaded from Hugging Face at runtime, so GPU availability and local model access need to be set up before long runs.
- Start with a very small range such as `--range 0:10` to confirm that environment, paths, and model loading all work.

## 4) Outputs

Main artifacts are stored under each baseline subfolder `results/`, for example:

- `baselines/temporal/results/temporal_pred.csv`
- `baselines/spatial/results/qwen_oracle_analysis.pkl`
- `baselines/spatial/results/molmo_pred_analysis.pkl`

Keep all generated files in local `results/` directories to preserve reproducibility.
These `results/` directories are treated as generated outputs and are not tracked in git.

---
