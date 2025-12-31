import logging
from pathlib import Path

import fire
from hydra import compose, initialize_config_dir

from speech_defect_detector.training.trainer import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_command() -> None:
    """
    Train the model.

    Usage:
        python -m speech_defect_detector.commands train
    """
    config_dir = Path(__file__).parent.parent / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config")
        train(cfg)


def main():
    """Main entry point."""
    fire.Fire({"train": train_command})


if __name__ == "__main__":
    main()
