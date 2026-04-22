# LLM Baselines Experiments

This folder contains the experiment pipeline used for the accident paper baselines.

The experiments are organized into three stages:

- `baselines/temporal`: predict accident timing from video
- `baselines/spatial`: predict accident location + VLM classification
- `baselines/classification`: feature-based classification experiments

---

## 1) Environment setup

From `llm_baselines`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2) Expected data layout

Current code assumes project-relative paths (from baseline notebooks/scripts):

- `../../data/labels.csv`
- `../../data/videos/...`

Change the data path based on your project structure.

## 3) Recommended execution order

### Step A: Temporal reasoning

Use script for full runs:

```bash
cd baselines/temporal
python main.py --model qwen --range 0:676
python main.py --model qwen --range 676:1352
python main.py --model qwen --range 1352:2027

python main.py --model molmo --range 0:676
python main.py --model molmo --range 676:1352
python main.py --model molmo --range 1352:2027
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

## 4) Outputs

Main artifacts are stored under each baseline subfolder `results/`, for example:

- `baselines/temporal/results/temporal_pred.csv`
- `baselines/spatial/results/qwen_oracle_analysis.pkl`
- `baselines/spatial/results/molmo_pred_analysis.pkl`

Keep all generated files in local `results/` directories to preserve reproducibility.
These `results/` directories are treated as generated outputs and are not tracked in git.

---

