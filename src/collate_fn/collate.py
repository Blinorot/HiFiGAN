import logging
from typing import List

import torch
from numpy import dtype
from src.preprocessing.melspec import MelSpectrogram
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    
    audio = [elem['audio'].view(-1, 1) for elem in dataset_items]
    batch_audio = pad_sequence(audio, batch_first=True, padding_value=0).squeeze(-1)
    
    melspec_transform = MelSpectrogram()

    batch_spectrogram = melspec_transform(batch_audio) # B x F x T

    result_batch = {}
    result_batch['spectrogram'] = batch_spectrogram
    
    return result_batch
