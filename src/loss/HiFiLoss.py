from nis import cat

import torch
from src.preprocessing.melspec import MelSpectrogram
from torch import nn


class HiFiLoss(nn.Module):
    def __init__(self, fm_coef=2, mel_coef=45):
        super().__init__()

        self.fm_coef = fm_coef
        self.mel_coef = mel_coef

        self.l1_loss = nn.L1Loss()
        self.mel_transform = MelSpectrogram()

    def forward(self, spectrogram, generated_audio, **kwargs):
        generated_spectrogram = self.mel_transform(generated_audio.squeeze(1)) # remove channel
        
        if spectrogram.shape[-1] < generated_spectrogram.shape[-1]:
            diff = generated_spectrogram.shape[-1] - spectrogram.shape[-1]
            pad_value = self.mel_transform.config.pad_value
            pad = torch.zeros((spectrogram.shape[0], spectrogram.shape[1], diff))
            pad = pad.fill_(pad_value).to(spectrogram.device)
            spectrogram = torch.cat([spectrogram, pad], dim=-1)

        mel_loss = self.mel_coef * self.l1_loss(generated_spectrogram, spectrogram)

        G_loss = mel_loss
        D_loss = torch.zeros(1)

        return G_loss, D_loss, mel_loss
