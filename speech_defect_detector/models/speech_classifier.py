import torch.nn as nn
import torch.nn.functional as F


class SpeechClassifier(nn.Module):
    """Simple MLP classifier."""

    def __init__(
        self,
        input_length: int = 16000,
        num_classes: int = 2,
        hidden_dim: int = 128,
    ):
        """
        Initialize model.

        Args:
            input_length: Length of input audio in samples
            num_classes: Number of output classes
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.input_length = input_length
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, length)

        Returns:
            Logits of shape (batch, num_classes)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
