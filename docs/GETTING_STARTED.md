# Getting started

This is the shortest path from a fresh clone to a working local setup.

## 1. Download the dataset

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r dataset/requirements.txt
bash dataset/download_dataset.sh
```

This prepares:

```text
dataset/real_videos/
  labels.csv
  test_metadata.csv
  videos/
```

If you have the dataset somewhere else already, you can skip this and pass `--dataset-path` to the scripts.

## 2. Run the easiest baseline first

```bash
cd baselines/heuristic
uv sync
python naive.py
```

That verifies the dataset structure and metadata with the lowest possible cost.

## 3. Run a small test

```bash
python optical_flow.py --take 5
python bbox_dynamics.py --take 5
```

## 4. Move to the LLM/VLM baselines

```bash
cd ../llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python baselines/temporal/main.py --model qwen --dataset-path ../../dataset --range 0:10
```

## 5. Use notebooks only after the scripts work

The tracked scripts are the best first run because they make path issues easier to debug. Once they work, use the notebooks for analysis and paper tables.

## 6. Generate synthetic data with CARLA

```bash
cd generation/carla-simulation
docker compose up --build
```

Use this path when you want to create synthetic accident data rather than reproduce the baseline experiments.
