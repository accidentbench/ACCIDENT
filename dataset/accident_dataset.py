from pathlib import Path


REQUIRED_FILES = ("metadata-real.csv",)
REQUIRED_DIRS = ("real_videos",)


def default_dataset_path(repo_root: Path) -> Path:
    return repo_root / "dataset"


def is_dataset_root(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED_FILES) and all(
        (path / name).is_dir() for name in REQUIRED_DIRS
    )


def resolve_dataset_path(candidate: Path) -> Path:
    candidate = candidate.expanduser().resolve()

    if is_dataset_root(candidate):
        return candidate

    raise FileNotFoundError(
        f"Could not find the ACCIDENT dataset layout at {candidate}. "
        f"Expected {', '.join(REQUIRED_FILES)} and a {REQUIRED_DIRS[0]}/ directory."
    )
