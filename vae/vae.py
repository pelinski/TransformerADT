# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
import numpy as np
import torch
from torch import nn
import torch.optim as opt
from torch.nn import functional as F


class VanillaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, out_x, out_y, device):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.out_x = out_x
        self.out_y = out_y
        self.device = device

        hidden_dims = [32, 64, 128, 256, 512]

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(
                            hidden_dims[i],
                            out_channels=hidden_dims[i + 1],
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.BatchNorm2d(hidden_dims[i + 1]),
                        nn.LeakyReLU(),
                    )
                )
            )
        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.LeakyReLU(),
            ),
            *modules
        )

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.decoder = nn.Sequential(
            *modules,
            nn.ConvTranspose2d(
                32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.Linear(out_x, out_y)
        )

    def encode(self, x):
        x = self.encoder(x)

        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return [mu, log_var]

    def decode(self, z):
        y = self.decoder_input(z)
        y = y.view(-1, 512, 1, 1)
        y = self.decoder(y)
        y = y.squeeze()
        y = torch.reshape(y, (y.shape[0], y.shape[1], 3, y.shape[2] // 3))
        _h = y[:, :, 0, :]
        _v = y[:, :, 1, :]
        _o = y[:, :, 2, :]

        h = torch.sigmoid(_h)
        v = torch.sigmoid(_v)
        o = torch.tanh(_o) * 0.5

        return h, v, o

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inp):
        inp = inp[:, None, :, :]
        mu, log_var = self.encode(inp)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def predict(self, x, use_thres, thres):

        return self.forward(x)
