from pathlib import Path


REAL_DATASET_DIRNAME = "real_videos"
SYNTHETIC_DATASET_DIRNAME = "synthetic_videos"

REAL_META = "metadata-real.csv"
SYNTH_META = "metadata-synthetic.csv"
CLASSES_FILE = "annotation_classes.yaml"


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def default_dataset_root(repo_root: Path) -> Path:
    return repo_root / "dataset"


def has_videos(path: Path) -> bool:
    return any(
        p.is_file() and p.suffix.lower() in VIDEO_EXTS
        for p in path.iterdir()
    )


def is_dataset_root(path: Path) -> bool:
    if not path.is_dir():
        return False

    # must contain metadata at root
    if not (path / REAL_META).is_file():
        return False

    # must contain real videos
    real_dir = path / REAL_DATASET_DIRNAME
    if not (real_dir.is_dir() and has_videos(real_dir)):
        return False

    return True


def resolve_dataset_root(candidate: Path) -> Path:
    candidate = candidate.expanduser().resolve()

    if is_dataset_root(candidate):
        return candidate

    nested = candidate / "dataset"
    if is_dataset_root(nested):
        return nested

    raise FileNotFoundError(
        f"Could not find dataset root at:\n"
        f"  {candidate}\n"
        f"  {nested}\n"
        f"Expected structure:\n"
        f"  dataset/\n"
        f"    real_videos/\n"
        f"    synthetic_videos/\n"
        f"    {REAL_META}\n"
    )


def get_dataset_paths(root: Path, kind: str = "real") -> tuple[Path, Path]:
    """
    Returns (videos_dir, metadata_path)
    """
    root = resolve_dataset_root(root)

    if kind == "real":
        videos = root / REAL_DATASET_DIRNAME
        meta = root / REAL_META
    elif kind == "synthetic":
        videos = root / SYNTHETIC_DATASET_DIRNAME
        meta = root / SYNTH_META
    else:
        raise ValueError(f"Unknown dataset kind: {kind}")

    if not videos.is_dir():
        raise FileNotFoundError(f"Missing videos dir: {videos}")

    if not meta.is_file():
        raise FileNotFoundError(f"Missing metadata file: {meta}")

    return videos, meta