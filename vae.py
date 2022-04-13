# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
import numpy as np
import torch
from torch import nn
import torch.optim as opt
from torch.nn import functional as F


def conv_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (
        np.floor(
            (img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1
        ).astype(int),
        np.floor(
            (img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1
        ).astype(int),
    )
    return outshape


class VanillaVAE(nn.Module):
    def __init__(self, in_channels, latent_dim, out_x, out_y):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.out_x = out_x
        self.out_y = out_y

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
        x = self.decoder_input(z)
        x = x.view(-1, 512, 1, 1)
        x = self.decoder(x)

        return x

    def reparameterize(self, mu, logvar):

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inp):
        mu, log_var = self.encode(inp)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), inp, mu, log_var]

    def sample(self, num_samples, current_device):
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x):

        return self.forward(x)[0]
