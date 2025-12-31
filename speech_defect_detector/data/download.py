import logging
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import dvc.api
from dvc.repo import Repo


logger = logging.getLogger(__name__)

def copy_files(files: list, target: Path, split_name: str):
    """Copy files to target directory."""
    for file in files:
        shutil.copy2(file, target / file.name)
    logger.info(f"Copied {len(files)} files to {split_name}")


def prepare_data(source_dir: Path, target_dir: Path, val_split: float = 0.2, random_state: int = 42,) -> None:
    """
    Splits data from 'good' and 'bad' folders into train/val folders.

    Args:
        source_dir: Directory containing raw data
        target_dir: Target directory for prepared data
        val_split: Fraction of data to use for validation
        random_state: Random state
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    bad_source = source_dir / "bad"
    good_source = source_dir / "good"

    if not bad_source.exists() or not good_source.exists():
        raise ValueError(f"Source directory for good or bad samples not found.")
    
    bad_files = list(bad_source.glob("*.wav"))
    good_files = list(good_source.glob("*.wav"))

    logger.info(f"Found {len(bad_files)} bad samples and {len(good_files)} good samples")

    bad_train, bad_val = train_test_split(
        bad_files, test_size=val_split, random_state=random_state
    )
    good_train, good_val = train_test_split(
        good_files, test_size=val_split, random_state=random_state
    )

    logger.info(
        f"Split: {len(bad_train)} bad train, {len(bad_val)} bad val, "
        f"{len(good_train)} good train, {len(good_val)} good val"
    )

    train_bad_dir = target_dir / "train" / "bad"
    train_good_dir = target_dir / "train" / "good"
    val_bad_dir = target_dir / "val" / "bad"
    val_good_dir = target_dir / "val" / "good"

    for dir_path in [train_bad_dir, train_good_dir, val_bad_dir, val_good_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    copy_files(bad_train, train_bad_dir, "train/bad")
    copy_files(bad_val, val_bad_dir, "val/bad")
    copy_files(good_train, train_good_dir, "train/good")
    copy_files(good_val, val_good_dir, "val/good")

    logger.info(f"Data preparation completed. Data saved to {target_dir}")


def download_data(data_dir: Path, dvc_remote: str, val_split: float = 0.2, random_state: int = 42) -> None:
    """
    Download data using DVC and split into train/val structure.

    Args:
        data_dir: Directory to download data
        dvc_remote: DVC remote name
        val_split: Fraction of data to use for validation
        random_state: Random state
    """
    logger.info(f"Downloading data to {data_dir}")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        repo = Repo(".")
        repo.pull(remote=dvc_remote)
        logger.info("Data downloaded successfully from DVC")

        train_dir = data_dir / "train"
        val_dir = data_dir / "val"

        if not train_dir.exists() or not val_dir.exists():
            logger.info("Splitting data into train/val structure...")
            if (data_dir / "good").exists() and (data_dir / "bad").exists():
                prepare_data(data_dir, data_dir, val_split, random_state)
                logger.info("Data split into train/val structure completed")
            else:
                logger.warning(f"Could not find good/bad folders in {data_dir}.")
        else:
            logger.info("Data already split into train/val structure")
    except Exception as e:
        logger.error(str(e))


def get_data_path(config_path: str = "data/data.dvc") -> Path:
    """
    Get data path from DVC config.

    Args:
        config_path: Path to DVC config file

    Returns:
        Path to data directory
    """
    try:
        path = dvc.api.get_url(config_path, repo=".")
        return Path(path)
    except Exception:
        return Path("data/raw")
