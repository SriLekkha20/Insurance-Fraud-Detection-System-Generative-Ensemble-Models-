"""
Simple GAN to generate synthetic numeric fraud data for experimentation.
This is intentionally small/simplified for demo purposes.
"""

import torch
from torch import nn, optim
import numpy as np


LATENT_DIM = 8
FEATURE_DIM = 3  # e.g., [customer_age, claim_amount, num_previous_claims]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, FEATURE_DIM),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def main():
    # Tiny "real" dataset just for training shape
    real_data = np.array(
        [
            [39, 15000, 3],
            [60, 42000, 4],
            [57, 35000, 3],
            [61, 29000, 4],
        ],
        dtype=np.float32,
    )

    real_torch = torch.tensor(real_data, device=DEVICE)

    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)

    opt_g = optim.Adam(gen.parameters(), lr=1e-3)
    opt_d = optim.Adam(disc.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    epochs = 3000
    batch_size = real_data.shape[0]

    for epoch in range(epochs):
        # Train Discriminator
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        fake = gen(z).detach()
        real_labels = torch.ones((batch_size, 1), device=DEVICE)
        fake_labels = torch.zeros((batch_size, 1), device=DEVICE)

        disc_real = disc(real_torch)
        disc_fake = disc(fake)

        loss_d_real = criterion(disc_real, real_labels)
        loss_d_fake = criterion(disc_fake, fake_labels)
        loss_d = (loss_d_real + loss_d_fake) / 2

        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # Train Generator
        z = torch.randn(batch_size, LATENT_DIM, device=DEVICE)
        gen_samples = gen(z)
        disc_pred = disc(gen_samples)
        loss_g = criterion(disc_pred, real_labels)

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch} - D loss: {loss_d.item():.4f} "
                f"G loss: {loss_g.item():.4f}"
            )

    z = torch.randn(10, LATENT_DIM, device=DEVICE)
    synthetic = gen(z).detach().cpu().numpy()
    print("Sample synthetic fraud-like points:\n", synthetic)


if __name__ == "__main__":
    main()
