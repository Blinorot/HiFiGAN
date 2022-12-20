import logging
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torchaudio
from src.utils.parse_config import ConfigParser
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            limit=None,
            max_audio_length=8192,
    ):
        self.config_parser = config_parser
        self._assert_index_is_valid(index)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)

        if limit is not None:
            index = index[:limit]

        self._index: List[dict] = index
        self.max_audio_length = max_audio_length

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path = data_dict["path"]
        audio_wave = self.load_audio(audio_path)

        if audio_wave.shape[-1] > self.max_audio_length:
            random_start = np.random.randint(0, audio_wave.shape[-1] - self.max_audio_length + 1)
            audio_wave = audio_wave[..., random_start:random_start+self.max_audio_length]

        return {
            "audio": audio_wave,
            "audio_length": audio_wave.shape[-1],
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)

        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]

        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        
        return audio_tensor

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - duration of audio (in seconds)."
            )
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
