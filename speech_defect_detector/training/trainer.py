import logging
from pathlib import Path

import git
import mlflow
import mlflow.pytorch
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from speech_defect_detector.data.dataset import SpeechDefectDataset
from speech_defect_detector.data.download import download_data
from speech_defect_detector.training.lightning_module import SpeechDefectLightningModule

logger = logging.getLogger(__name__)


def get_git_commit_id() -> str:
    """
    Get current git commit ID.

    Returns:
        Git commit hash or 'unknown'
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha[:8]
    except Exception:
        return "unknown"


def train(config) -> None:
    """
    Train the model.

    Args:
        config: Hydra config
    """
    seed_everything(config.seed, workers=True)

    data_dir = Path(config.data.data_dir)
    download_data(data_dir, config.data.dvc_remote)

    train_dataset = SpeechDefectDataset(
        data_dir=data_dir / "train",
        sample_rate=config.data.sample_rate,
        max_duration_seconds=config.data.max_duration_seconds,
    )

    val_dataset = SpeechDefectDataset(
        data_dir=data_dir / "val",
        sample_rate=config.data.sample_rate,
         max_duration_seconds=config.data.max_duration_seconds,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True,
    )

    input_length = int(config.data.max_duration_seconds * config.data.sample_rate)

    model = SpeechDefectLightningModule(
        input_length=input_length,
        num_classes=config.model.num_classes,
        hidden_dim=config.model.hidden_dim,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.training.checkpoint_dir),
        filename="best-{epoch:02d}-{val_loss:.2f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )

    early_stopping = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=config.training.patience,
        verbose=True,
    )

    commit_id = get_git_commit_id()

    mlflow_logger = MLFlowLogger(
        experiment_name=config.logging.experiment_name,
        tracking_uri=config.logging.mlflow_uri,
    )

    mlflow_logger.log_hyperparams(
        {
            "git_commit_id": commit_id,
            "model.input_length": config.model.input_length,
            "model.num_classes": config.model.num_classes,
            "model.hidden_dim": config.model.hidden_dim,
            "training.learning_rate": config.training.learning_rate,
            "training.weight_decay": config.training.weight_decay,
            "training.batch_size": config.training.batch_size,
            "training.max_epochs": config.training.max_epochs,
            "data.sample_rate": config.data.sample_rate,
            "data.max_duration_seconds": config.data.max_duration_seconds,
        }
    )

    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback, early_stopping],
        logger=mlflow_logger,
        log_every_n_steps=config.training.log_every_n_steps,
    )

    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    with mlflow.start_run(run_id=mlflow_logger.run_id):
        mlflow.pytorch.log_model(
            model.model,
            "model",
            registered_model_name=config.logging.model_name,
            input_example=torch.randn(1, input_length).numpy(),
        )

    logger.info("Training completed!")
