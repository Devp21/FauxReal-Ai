import pandas as pd
from ctgan import CTGAN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import numpy as np

class SyntheticDataGenerator:
    def __init__(self):
        pass

    def train_and_generate(self, dataset, num_samples, output_prefix, target_column=None):
        # Train and generate using CTGAN
        ctgan = CTGAN()
        ctgan.fit(dataset, epochs=300)
        ctgan_data = ctgan.sample(num_samples)
        ctgan_data.to_csv(f"{output_prefix}_ctgan.csv", index=False)

        # Train and generate using VAE
        vae_data = self._train_vae(dataset, num_samples)
        vae_data.to_csv(f"{output_prefix}_vae.csv", index=False)

        # Train and generate using WGAN
        wgan_data = self._train_wgan(dataset, num_samples)
        wgan_data.to_csv(f"{output_prefix}_wgan.csv", index=False)

    def evaluate(self, real_data, synthetic_data, target_column):
        if target_column is None:
            raise ValueError("Target column is required for evaluation.")

        X_real = real_data.drop(columns=[target_column])
        y_real = real_data[target_column]

        X_syn = synthetic_data.drop(columns=[target_column])
        y_syn = synthetic_data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        real_f1 = f1_score(y_test, y_pred, average='weighted')

        model.fit(X_syn, y_syn)
        y_syn_pred = model.predict(X_test)
        synthetic_f1 = f1_score(y_test, y_syn_pred, average='weighted')

        return {
            "RealData_F1": real_f1,
            "SyntheticData_F1": synthetic_f1,
            "Performance_Drop": real_f1 - synthetic_f1
        }

    def _train_vae(self, dataset, num_samples):
        # Define the VAE architecture
        class VAE(nn.Module):
            def __init__(self, input_dim, latent_dim=16):
                super(VAE, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, latent_dim * 2)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim),
                    nn.Sigmoid()
                )

            def forward(self, x):
                mean_var = self.encoder(x)
                mean, log_var = mean_var[:, :latent_dim], mean_var[:, latent_dim:]
                std = torch.exp(0.5 * log_var)
                z = mean + std * torch.randn_like(std)
                reconstructed = self.decoder(z)
                return reconstructed, mean, log_var

            def loss_fn(self, x, reconstructed, mean, log_var):
                recon_loss = nn.functional.mse_loss(reconstructed, x, reduction='sum')
                kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
                return recon_loss + kl_div

        input_dim = dataset.shape[1]
        latent_dim = 16

        vae = VAE(input_dim, latent_dim)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
        data = torch.tensor(dataset.values, dtype=torch.float32)
        loader = DataLoader(data, batch_size=64, shuffle=True)

        # Train VAE
        for epoch in range(50):
            for batch in loader:
                reconstructed, mean, log_var = vae(batch)
                loss = vae.loss_fn(batch, reconstructed, mean, log_var)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Generate synthetic data
        z = torch.randn(num_samples, latent_dim)
        synthetic_data = vae.decoder(z).detach().numpy()
        return pd.DataFrame(synthetic_data, columns=dataset.columns)

    def _train_wgan(self, dataset, num_samples):
        # Define the WGAN architecture
        class Generator(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(Generator, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, output_dim),
                    nn.Tanh()
                )

            def forward(self, x):
                return self.model(x)

        class Critic(nn.Module):
            def __init__(self, input_dim):
                super(Critic, self).__init__()
                self.model = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )

            def forward(self, x):
                return self.model(x)

        input_dim = dataset.shape[1]
        latent_dim = 16

        generator = Generator(latent_dim, input_dim)
        critic = Critic(input_dim)

        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.0002)
        data = torch.tensor(dataset.values, dtype=torch.float32)
        loader = DataLoader(data, batch_size=64, shuffle=True)

        # Train WGAN
        for epoch in range(50):
            for batch in loader:
                # Update Critic
                for _ in range(5):
                    z = torch.randn(batch.size(0), latent_dim)
                    fake_data = generator(z)
                    critic_loss = -(torch.mean(critic(batch)) - torch.mean(critic(fake_data)))
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()

                # Update Generator
                z = torch.randn(batch.size(0), latent_dim)
                fake_data = generator(z)
                gen_loss = -torch.mean(critic(fake_data))
                gen_optimizer.zero_grad()
                gen_loss.backward()
                gen_optimizer.step()

        # Generate synthetic data
        z = torch.randn(num_samples, latent_dim)
        synthetic_data = generator(z).detach().numpy()
        return pd.DataFrame(synthetic_data, columns=dataset.columns)