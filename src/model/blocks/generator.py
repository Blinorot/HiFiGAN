from src.model.utils import get_conv_padding_size
from torch import nn


class MRFBlock(nn.Module):
    def __init__(self, channels, kernel,  dilations):
        super().__init__()

        self.kernel = kernel
        self.dilations = dilations

        layers = []

        for m in range(len(dilations)):
            layer = nn.Sequential(
                nn.LeakyReLU(),
                nn.Conv1d(channels, channels, kernel_size=kernel, 
                          dilation=dilations[m], 
                          padding=get_conv_padding_size(kernel, 1, dilations[m])),
                nn.LeakyReLU(),
                nn.Conv1d(channels, channels, kernel_size=kernel, 
                          dilation=1, 
                          padding=get_conv_padding_size(kernel, 1, 1)),
            )
            layers.append(layer)
        
        self.block = nn.ModuleList(layers)

    def forward(self, x):
        result = 0
        for layer in self.block:
            result = result + layer(x)
        return result


class MRF(nn.Module):
    def __init__(self, channels, resblock_kernels, resblock_dilations):
        super().__init__()

        resblocks = []
        for i in range(len(resblock_kernels)):
            resblocks.append(MRFBlock(channels, resblock_kernels[i], resblock_dilations[i]))

        self.resblocks = nn.ModuleList(resblocks)

    def forward(self, x):
        result = 0
        for block in self.resblocks:
            result = result + block(x)
        return result


class Generator(nn.Module):
    def __init__(self, input_channels, hidden_channels, upsample_kernels,
                 upsample_stride, resblock_kernels,
                 resblock_dilations):
        super().__init__()

        self.in_conv = nn.Conv1d(input_channels, hidden_channels, 7, 
                                  padding=get_conv_padding_size(7, 1, 1))

        blocks = []
        current_channels = hidden_channels
        for i in range(len(upsample_kernels)):
            upsample = nn.ConvTranspose1d(current_channels, current_channels // 2,
                                          upsample_kernels[i], upsample_stride[i],
                                          padding=(upsample_kernels[i] - upsample_stride[i]) // 2)
            mrf = MRF(current_channels // 2, resblock_kernels, resblock_dilations)
            block = nn.Sequential(upsample, mrf)
            blocks.append(block)

            current_channels = current_channels // 2

        self.blocks = nn.Sequential(*blocks)

        self.out_conv = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(current_channels, 1, 7, 
                      padding=get_conv_padding_size(7, 1, 1)),
            nn.Tanh()
        )

    def forward(self, spectrogram):
        spectrogram = self.in_conv(spectrogram)
        spectrogram = self.blocks(spectrogram)
        generated_audio = self.out_conv(spectrogram)
        return {"generated_audio": generated_audio}
