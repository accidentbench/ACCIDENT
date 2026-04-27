# ACCIDENT

Code and assets for the ACCIDENT dataset paper, including baseline methods and the CARLA-based synthetic data generation pipeline.

Start here if you are new to the repo:

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- [docs/REPOSITORY_LAYOUT.md](docs/REPOSITORY_LAYOUT.md)

## Repository map

- [docs/](docs/) - onboarding and repository navigation docs
- [baselines/](baselines/) - all baseline implementations, split into heuristic and LLM/VLM families
- [generation/](generation/) - dataset generation pipelines, currently including CARLA-based synthesis
- [dataset/](dataset/) - local dataset cache plus the normalized `real_videos/` layout used by the baselines

## Quick start

If you want to reproduce paper baselines, start with one of these:

### Download the dataset

Install the Kaggle CLI first:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r dataset/requirements.txt
```

Then download the dataset:

```bash
bash dataset/download_dataset.sh
```

This runs [dataset/download_dataset.sh](dataset/download_dataset.sh), downloads the Kaggle dataset `picekl/accident`, and prepares `dataset/real_videos/` for the baseline code.

You still need to authenticate the Kaggle CLI first. See [dataset/README.md](dataset/README.md) for setup.

### Heuristic baselines

These are the easiest entry point because they already expose command-line interfaces.

```bash
cd baselines/heuristic
uv sync
python naive.py
python optical_flow.py --take 5
python bbox_dynamics.py --take 5
```

See [baselines/heuristic/README.md](baselines/heuristic/README.md) for details.

### LLM / VLM baselines

These experiments mix one tracked script and several notebooks:

```bash
cd baselines/llm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python baselines/temporal/main.py --model qwen --dataset-path ../../dataset --range 0:10
```

See [baselines/llm/README.md](baselines/llm/README.md) for the full execution order.

### Synthetic data generation

Use the CARLA project when you want to generate or extend synthetic accident data rather than run baselines:

```bash
cd generation/carla-simulation
docker compose up --build
```

See [generation/carla-simulation/README.md](generation/carla-simulation/README.md) for requirements and workflow details.

## Dataset layout used by the baselines

The heuristic baseline scripts and the LLM temporal script now both accept an explicit dataset root. The expected layout is:

```text
dataset/
  real_videos/
    labels.csv
    test_metadata.csv
    videos/
      ...
  synthetic_videos/
    ...
```

If your dataset lives elsewhere, pass `--dataset-path /path/to/real_videos` to the supported scripts.
Passing `--dataset-path /path/to/dataset` also works as long as that directory contains `real_videos/`.

## Recommendations for new users

- Start with `baselines/heuristic/naive.py` to verify that your labels and metadata are readable.
- Use `--take 5` on heavier baselines before launching full runs.
- Keep generated outputs inside each subproject's local output folders so reruns stay reproducible.
- Treat notebooks as analysis companions; use the tracked scripts when you want a repeatable paper run.
