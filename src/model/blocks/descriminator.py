from turtle import forward

import torch
from src.model.utils import get_conv_padding_size
from torch import nn
from torch.nn.utils import weight_norm


class PeriodDiscriminator(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

        self.in_conv = nn.Sequential(
            weight_norm(nn.Conv2d(1, 64, (5, 1), (3, 1),
                        padding=(get_conv_padding_size(5, 1, 1), 0))),
            nn.LeakyReLU()
        )
        mid_conv = []
        current_channels = 64
        for i in range(4):
            mid_conv.append(nn.Sequential(
                weight_norm(nn.Conv2d(current_channels, current_channels * 2,
                          (5, 1), (3, 1), padding=(get_conv_padding_size(5, 1, 1), 0))),
                nn.LeakyReLU()
            ))
            current_channels *= 2

        self.mid_conv = nn.ModuleList(mid_conv)

        self.out_conv = weight_norm(nn.Conv2d(current_channels, 1, (3, 1),
                                              padding=(get_conv_padding_size(3, 1, 1), 0)))


    def forward(self, x):
        features = []

        if x.shape[-1] % self.p != 0:
            pad_amount = self.p - x.shape[-1] % self.p
            pad = torch.zeros((x.shape[0], x.shape[1], pad_amount), device=x.device)
            x = torch.cat([x, pad], dim=-1)
        
        x = x.view(x.shape[0], 1, -1, self.p)

        x = self.in_conv(x)
        features.append(x)
        for conv in self.mid_conv:
            x = conv(x)
            features.append(x)

        x = self.out_conv(x)
        features.append(x)
        x = x.view(x.shape[0], -1)
        return x, features


class MultiPeriodDiscriminator(nn.Module):

    periods = [2, 3, 5, 7, 11]

    def __init__(self):
        super().__init__()

        period_discriminators = []
        for p in self.periods:
            period_discriminators.append(PeriodDiscriminator(p))
        
        self.period_discriminators = nn.ModuleList(period_discriminators)

    def forward(self, generated, real):
        real_features = []
        generated_features = []
        real_outs = []
        generated_outs = []
        for discriminator in self.period_discriminators:
            r_out, r_feat = discriminator(real)
            g_out, g_feat = discriminator(generated)

            real_features.extend(r_feat)
            generated_features.extend(g_feat)
            real_outs.append(r_out)
            generated_outs.append(g_out)

        return real_outs, generated_outs, real_features, generated_features


class ScaleDisctiminator(nn.Module):
    def __init__(self, norm=False):
        super().__init__()

        # https://arxiv.org/pdf/1910.06711.pdf MelGAN parameters

        convs = [
            nn.Conv1d(1, 16, 15, 1, padding=7),
            nn.Conv1d(16, 64, 41, 4, groups=4, padding=20),
            nn.Conv1d(64, 256, 41, 4, groups=16, padding=20),
            nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20),
            nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
        ]

        out_conv = nn.Conv1d(1024, 1, 3, 1, padding=1)

        if norm:
            convs = [weight_norm(module) for module in convs]
            out_conv = weight_norm(out_conv)
        
        self.convs = nn.ModuleList(convs)
        self.out_conv = out_conv

        self.activation = nn.LeakyReLU()

    def forward(self, x):
        features = []
        for conv in self.convs:
            x = self.activation(conv(x))
            features.append(x)

        x = self.out_conv(x)
        features.append(x)
        x = x.view(x.shape[0], -1)
        
        return x, features
        

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.scale_discriminators = nn.ModuleList([
            ScaleDisctiminator(norm=True),
            ScaleDisctiminator(),
            ScaleDisctiminator()
        ])

        self.avgpool1 = nn.AvgPool1d(4, 2, padding=2)
        self.avgpool2 = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, generated, real):
        real_features = []
        generated_features = []
        real_outs = []
        generated_outs = []
        for i, discriminator in enumerate(self.scale_discriminators):
            if i == 0:
                r_in = real
                g_in = generated
            elif i == 1:
                r_in = self.avgpool1(real)
                g_in = self.avgpool1(generated)
            elif i == 2:
                r_in = self.avgpool2(self.avgpool1(real))
                g_in = self.avgpool2(self.avgpool1(generated))
            r_out, r_feat = discriminator(r_in)
            g_out, g_feat = discriminator(g_in)

            real_features.extend(r_feat)
            generated_features.extend(g_feat)
            real_outs.append(r_out)
            generated_outs.append(g_out)

        return real_outs, generated_outs, real_features, generated_features


class Descriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, generated, real):
        if generated.shape[-1] > real.shape[-1]:
            pad_amount = generated.shape[-1] - real.shape[-1]
            pad = torch.zeros((real.shape[0], real.shape[1], pad_amount), device=real.device)
            real = torch.cat([real, pad], dim=-1)

        p_real_outs, p_gen_outs, p_real_feat, p_gen_feat = self.mpd(generated, real)
        s_real_outs, s_gen_outs,s_real_feat, s_gen_feat = self.msd(generated, real)
        return {
            "p_real_outs": p_real_outs,
            "p_gen_outs": p_gen_outs,
            "p_real_feat": p_real_feat,
            "p_gen_feat": p_gen_feat,
            "s_real_outs": s_real_outs,
            "s_gen_outs": s_gen_outs,
            "s_real_feat": s_real_feat,
            "s_gen_feat": s_gen_feat
        }
