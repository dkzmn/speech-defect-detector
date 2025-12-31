"""PyTorch Lightning module for training."""

import logging

import torch
import torch.nn.functional as f
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torchmetrics import AUROC, Accuracy, F1Score

from speech_defect_detector.models.speech_classifier import SpeechClassifier

logger = logging.getLogger(__name__)


class SpeechDefectLightningModule(LightningModule):
    """PyTorch Lightning module for speech defect detection."""

    def __init__(
        self,
        input_length: int = 16000,
        num_classes: int = 2,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        """
        Initialize Lightning module.

        Args:
            input_length: Length of input audio in samples
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = SpeechClassifier(
            input_length=input_length,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.test_accuracy = Accuracy(task="binary")

        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")

        self.train_roc_auc = AUROC(task="binary")
        self.val_roc_auc = AUROC(task="binary")
        self.test_roc_auc = AUROC(task="binary")

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch):
        """Training step."""
        audio, labels = batch
        logits = self(audio)
        loss = f.cross_entropy(logits, labels)

        probs = f.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        self.train_accuracy(preds, labels)
        self.train_f1(preds, labels)
        self.train_roc_auc(probs[:, 1], labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_accuracy, on_step=True, on_epoch=True)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train/roc_auc", self.train_roc_auc, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch):
        """Validation step."""
        audio, labels = batch
        logits = self(audio)
        loss = f.cross_entropy(logits, labels)

        probs = f.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        self.val_accuracy(preds, labels)
        self.val_f1(preds, labels)
        self.val_roc_auc(probs[:, 1], labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val/roc_auc", self.val_roc_auc, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch):
        """Test step."""
        audio, labels = batch
        logits = self(audio)
        loss = f.cross_entropy(logits, labels)

        probs = f.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.test_accuracy(preds, labels)
        self.test_f1(preds, labels)
        self.test_roc_auc(probs[:, 1], labels)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        self.log("test/roc_auc", self.test_roc_auc, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
