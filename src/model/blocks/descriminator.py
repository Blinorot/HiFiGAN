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
            pad = torch.zeros(x.shape[0], x.shape[1], pad_amount, device=x.device)
            x = torch.cat([x, pad], dim=-1)
        
        x = x.view(x.shape[0], -1, self.p)

        x = self.in_conv(x)
        features.append(x)
        for conv in self.mid_conv:
            x = conv(x)
            features.append(x)

        x = self.out_conv(x)
        x = x.view(x.shape[0], 1, -1)
        return x, features


class MultiPeriodDiscriminator(nn.Module):

    periods = [2, 3, 5, 7, 11]

    def __init__(self):
        super().__init__()

        period_discriminators = []
        for p in self.periods:
            period_discriminators.append(PeriodDiscriminator())
        
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


class MultiScaleDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()


class Descriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.mpd = MultiPeriodDiscriminator()


    def forward(self, generated, real):
        p_real_outs, p_gen_outs, g_real_feat, g_gen_feat = self.mpd(generated, real)

        return {
            "p_real_outs": p_real_outs,
            "p_gen_outs": p_gen_outs,
            "g_real_feat": g_real_feat,
            "g_gen_feal": g_gen_feat
        }
