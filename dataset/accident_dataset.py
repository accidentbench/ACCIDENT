from pathlib import Path


DATASET_DIRNAME = "real_videos"
REQUIRED_FILES = ("labels.csv", "test_metadata.csv")
REQUIRED_DIRS = ("videos",)


def default_dataset_path(repo_root: Path) -> Path:
    return repo_root / "dataset" / DATASET_DIRNAME


def is_dataset_root(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_FILES) and all(
        (path / name).is_dir() for name in REQUIRED_DIRS
    )


def resolve_dataset_path(candidate: Path) -> Path:
    candidate = candidate.expanduser().resolve()

    if is_dataset_root(candidate):
        return candidate

    nested = candidate / DATASET_DIRNAME
    if is_dataset_root(nested):
        return nested

    raise FileNotFoundError(
        "Could not find the ACCIDENT dataset layout. Expected either "
        f"{candidate} or {nested} to contain {', '.join(REQUIRED_FILES)} "
        f"and a {REQUIRED_DIRS[0]}/ directory."
    )
