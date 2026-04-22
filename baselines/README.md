# Baselines

This directory groups the baseline implementations used in the ACCIDENT dataset paper.

## Contents

- [heuristic/](heuristic/) - classical baselines with runnable Python scripts and matching notebooks
- [llm/](llm/) - VLM/LLM baselines for temporal, spatial, and classification experiments

## Recommended order

1. Start with [heuristic/README.md](heuristic/README.md) for the quickest smoke test.
2. Move to [llm/README.md](llm/README.md) once dataset paths and local environment are confirmed.

## Shared convention

Both baseline families accept the normalized dataset layout under `dataset/real_videos/`. The runnable scripts also accept `dataset/` as long as it contains `real_videos/`.
