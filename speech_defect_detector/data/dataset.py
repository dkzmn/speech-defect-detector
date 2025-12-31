"""Dataset classes for speech defect detection."""

from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

SAMPLE_ELEMENTS = 4


class SpeechDefectDataset(Dataset):
    """Dataset class."""

    def __init__(
        self,
        data_dir: Path,
        sample_rate: int = 16000,
        max_duration_seconds: float = 5.0,
    ):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing audio files organized by class
            sample_rate: Sample rate for audio files
            max_duration_seconds: Maximum duration in seconds. Files longer than this
                will be split at points of silence
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration_seconds)
        self.max_duration_seconds = max_duration_seconds
        self.samples = []
        self._load_samples()

    def _find_split_points(self, audio: np.ndarray, window_size: int = 1024) -> list[int]:
        """
        Find points in audio with silence for splitting.

        Args:
            audio: Audio signal
            window_size: Size of window for energy calculation

        Returns:
            List of sample indices where audio can be split
        """
        num_windows = len(audio) // window_size
        energies = []
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = audio[start:end]
            energy = np.mean(np.abs(window) ** 2)
            energies.append(energy)

        if len(energies) == 0:
            return []

        energies = np.array(energies)
        threshold = np.percentile(energies, 20)
        split_points = []

        for i in range(1, len(energies) - 1):
            if (
                energies[i] < threshold
                and energies[i] < energies[i - 1]
                and energies[i] < energies[i + 1]
            ):
                split_points.append(i * window_size)

        return split_points

    def _split_audio(self, audio: np.ndarray) -> list[tuple[int, int]]:
        """
        Split audio into segments not longer than max_length.

        Args:
            audio: Audio signal to split

        Returns:
            List of (start, end) tuples for each segment
        """
        if len(audio) <= self.max_length:
            return [(0, len(audio))]

        segments = []
        current_start = 0

        while current_start < len(audio):
            current_end = min(current_start + self.max_length, len(audio))
            if current_end < len(audio):
                search_start = int(current_start + self.max_length * 0.8)
                search_end = current_end
                search_region = audio[search_start:search_end]

                if len(search_region) > 0:
                    energies = np.abs(search_region) ** 2
                    window_size = min(512, len(energies) // 4)
                    if window_size > 0:
                        smoothed_energies = np.convolve(
                            energies, np.ones(window_size) / window_size, mode="valid"
                        )
                        if len(smoothed_energies) > 0:
                            min_idx = np.argmin(smoothed_energies)
                            split_point = search_start + min_idx + window_size // 2
                            current_end = min(split_point, current_end)

            segments.append((current_start, current_end))
            current_start = current_end

        return segments

    def _load_samples(self):
        """Load all audio file paths and their labels, splitting long files."""
        good_dir = self.data_dir / "good"
        bad_dir = self.data_dir / "bad"
        if good_dir.exists():
            for audio_file in good_dir.glob("*.wav"):
                try:
                    duration = librosa.get_duration(path=str(audio_file), sr=self.sample_rate)
                    if duration > self.max_duration_seconds:
                        audio, _ = librosa.load(str(audio_file), sr=self.sample_rate, mono=True)
                        segments = self._split_audio(audio)
                        for start, end in segments:
                            self.samples.append((audio_file, 0, start, end))
                    else:
                        self.samples.append((audio_file, 0, None, None))
                except Exception:
                    continue

        if bad_dir.exists():
            for audio_file in bad_dir.glob("*.wav"):
                try:
                    duration = librosa.get_duration(path=str(audio_file), sr=self.sample_rate)
                    if duration > self.max_duration_seconds:
                        audio, _ = librosa.load(str(audio_file), sr=self.sample_rate, mono=True)
                        segments = self._split_audio(audio)
                        for start, end in segments:
                            self.samples.append((audio_file, 1, start, end))
                    else:
                        self.samples.append((audio_file, 1, None, None))
                except Exception:
                    continue

    def __len__(self):
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get item from dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (audio_tensor, label)
        """
        sample_info = self.samples[idx]
        if len(sample_info) == SAMPLE_ELEMENTS:
            audio_path, label, start, end = sample_info
        else:
            audio_path, label = sample_info
            start, end = None, None

        audio, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)

        if start is not None and end is not None:
            audio = audio[start:end]

        if self.max_length is not None:
            if len(audio) > self.max_length:
                audio = audio[: self.max_length]
            else:
                audio = np.pad(audio, (0, self.max_length - len(audio)), mode="constant")

        audio_tensor = torch.from_numpy(audio).float()
        return audio_tensor, torch.tensor(label, dtype=torch.long)
