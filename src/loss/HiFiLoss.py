from nis import cat

import torch
from src.preprocessing.melspec import MelSpectrogram
from torch import nn


class GeneratorLoss(nn.Module):
    def __init__(self, fm_coef=2, mel_coef=45):
        super().__init__()

        self.fm_coef = fm_coef
        self.mel_coef = mel_coef

        self.l1_loss = nn.L1Loss()
        self.mel_transform = MelSpectrogram()

    def forward(self, 
        spectrogram, 
        generated_audio,
        p_gen_outs,
        p_real_feat,
        p_gen_feat,
        s_gen_outs,
        s_real_feat,
        s_gen_feat,
        **kwargs):
        generated_audio = generated_audio.squeeze(1) # remove channel
        generated_spectrogram = self.mel_transform(generated_audio) 
        
        if spectrogram.shape[-1] < generated_spectrogram.shape[-1]:
            diff = generated_spectrogram.shape[-1] - spectrogram.shape[-1]
            pad_value = self.mel_transform.config.pad_value
            pad = torch.zeros((spectrogram.shape[0], spectrogram.shape[1], diff))
            pad = pad.fill_(pad_value).to(spectrogram.device)
            spectrogram = torch.cat([spectrogram, pad], dim=-1)

        # adv_loss
        adv_loss = 0
        for p in p_gen_outs:
            adv_loss = adv_loss + torch.mean((p - 1) ** 2)
        for s in s_gen_outs:
            adv_loss = adv_loss + torch.mean((s - 1) ** 2)

        # fm_loss
        fm_loss = 0
        for real, gen in zip(p_real_feat, p_gen_feat):
            fm_loss = fm_loss + self.l1_loss(gen, real)
        for real, gen in zip(s_real_feat, s_gen_feat):
            fm_loss = fm_loss + self.l1_loss(gen, real)

        # mel_loss
        mel_loss = self.l1_loss(generated_spectrogram, spectrogram)

        G_loss = adv_loss + self.fm_coef * fm_loss + self.mel_coef * mel_loss

        return G_loss, adv_loss, fm_loss, mel_loss

class DescriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, 
        p_real_outs,
        p_gen_outs,
        s_real_outs,
        s_gen_outs,
        **kwargs):

        # D_loss
        D_loss = 0
        for p_real, p_gen in zip(p_real_outs, p_gen_outs):
            D_loss = D_loss + torch.mean((p_real - 1) ** 2) + torch.mean((p_gen - 0) ** 2)
        for s_real, s_gen in zip(s_real_outs, s_gen_outs):
            D_loss = D_loss + torch.mean((s_real - 1) ** 2) + torch.mean((s_gen - 0) ** 2)

        return D_loss
