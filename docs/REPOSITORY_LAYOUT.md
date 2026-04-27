# Repository layout

This repository is organized around three main workstreams.

## Top-level folders

- `baselines/` - all baseline implementations, grouped by approach family
- `generation/` - dataset generation pipelines
- `dataset/` - local dataset cache and dataset setup notes
- `docs/` - onboarding and repository navigation docs

## What to treat as source of truth

- Use the `.py` scripts in `baselines/heuristic/` and `baselines/llm/baselines/temporal/` as the most reproducible entry points.
- Use notebooks for analysis, visualization, and paper-table assembly.
- Use `dataset/real_videos/` as the normalized local data layout for all baseline code.

## Generated outputs

These are local artifacts and should not be committed:

- downloaded dataset archives and extracted files under `dataset/`
- YOLO inference outputs in `baselines/heuristic/inference-yolo11x/`
- optical flow caches and output CSVs in `baselines/heuristic/`
- `results/` folders and progress logs inside `baselines/llm/baselines/`

## Why the repo is structured this way

The large research components already have working internal paths and notebooks. This cleanup keeps those stable while making the top-level repo easier to enter:

- project-wide docs live in `docs/`
- data setup lives in `dataset/`
- generation pipelines live in `generation/`

That gives new users a cleaner mental model without forcing a risky refactor of the experiment folders.
