import torch
from src.base import BaseModel
from src.model.blocks import Descriminator, Generator
from torch import nn


class HiFiGANModel(BaseModel):
    def __init__(self, input_channels, hidden_channels, upsample_kernels,
                 upsample_stride, resblock_kernels,
                 resblock_dilations):
        super().__init__()
        
        self.generator = Generator(input_channels, hidden_channels, upsample_kernels,
                                   upsample_stride, resblock_kernels,
                                   resblock_dilations)
        self.descriminator = Descriminator()

    def forward(self, spectrogram, **batch):
        return self.generator(spectrogram)

    def generate(self, **batch):
        return self.forward(**batch)

    def descriminate(self, generated_audio, real_audio, **batch):
        return self.descriminator(generated_audio, real_audio)
