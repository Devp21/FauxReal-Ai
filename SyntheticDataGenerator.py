import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from ctgan import CTGAN


# DataTypePreserver class for handling preprocessing and restoring types
class DataTypePreserver:
    def __init__(self):
        self.column_metadata = {}

    def analyze_dataset(self, df: pd.DataFrame):
        for column in df.columns:
            dtype = df[column].dtype
            nunique = df[column].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(dtype)

            if is_numeric:
                unique_ratio = nunique / len(df)
                is_integer = all(float(x).is_integer() for x in df[column].dropna().unique())
                is_discrete = unique_ratio < 0.05 or nunique < 20 or is_integer
            else:
                is_discrete = True

            self.column_metadata[column] = {
                'dtype': dtype,
                'is_numeric': is_numeric,
                'is_discrete': is_discrete,
                'unique_values': sorted(df[column].unique()) if is_discrete else None,
                'min_val': float(df[column].min()) if is_numeric else None,
                'max_val': float(df[column].max()) if is_numeric else None
            }

    def restore_types(self, synthetic_df: pd.DataFrame) -> pd.DataFrame:
        df = synthetic_df.copy()
        for column in df.columns:
            metadata = self.column_metadata[column]
            try:
                if metadata['is_discrete']:
                    if metadata['is_numeric']:
                        valid_values = np.array(metadata['unique_values'])
                        df[column] = df[column].apply(
                            lambda x: valid_values[np.abs(valid_values - x).argmin()]
                        )
                    else:
                        df[column] = df[column].apply(
                            lambda x: metadata['unique_values'][int(round(x)) % len(metadata['unique_values'])]
                        )
                else:
                    if metadata['is_numeric']:
                        df[column] = df[column].clip(metadata['min_val'], metadata['max_val'])
                df[column] = df[column].astype(metadata['dtype'])
            except Exception as e:
                print(f"Warning: Error processing column {column}: {str(e)}")
                continue
        return df


# WGANTrainer class
class WGANTrainer:
    def __init__(self):
        self.dtype_preserver = DataTypePreserver()

    def generate(self, dataset, num_samples):
        self.dtype_preserver.analyze_dataset(dataset)

        # Ensure all columns are numeric
        for column in dataset.select_dtypes(include=["object", "category"]).columns:
            dataset[column] = pd.Categorical(dataset[column]).codes
        dataset.fillna(0, inplace=True)

        # Define WGAN architecture
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

        synthetic_df = pd.DataFrame(synthetic_data, columns=dataset.columns)

        # Restore data types using DataTypePreserver
        synthetic_df = self.dtype_preserver.restore_types(synthetic_df)

        return synthetic_df


# VAETrainer class
class VAETrainer:
    def __init__(self):
        self.dtype_preserver = DataTypePreserver()

    def generate(self, dataset, num_samples):
        self.dtype_preserver.analyze_dataset(dataset)

        class VAE(nn.Module):
            def __init__(self, input_dim):
                super(VAE, self).__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32)
                )
                self.fc_mu = nn.Linear(32, 16)
                self.fc_logvar = nn.Linear(32, 16)
                self.decoder = nn.Sequential(
                    nn.Linear(16, 128),
                    nn.ReLU(),
                    nn.Linear(128, input_dim),
                    nn.Tanh()
                )

            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

        input_dim = dataset.shape[1]
        vae = VAE(input_dim)
        optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
        data = torch.tensor(dataset.values, dtype=torch.float32)
        loader = DataLoader(data, batch_size=64, shuffle=True)

        # Train VAE
        for epoch in range(50):
            for batch in loader:
                optimizer.zero_grad()
                recon, mu, logvar = vae(batch)
                recon_loss = nn.functional.mse_loss(recon, batch)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kld_loss
                loss.backward()
                optimizer.step()

        # Generate synthetic data
        z = torch.randn(num_samples, 16)
        synthetic_data = vae.decode(z).detach().numpy()

        synthetic_df = pd.DataFrame(synthetic_data, columns=dataset.columns)

        # Restore data types using DataTypePreserver
        synthetic_df = self.dtype_preserver.restore_types(synthetic_df)

        return synthetic_df


# CTGANTrainer class
class CTGANTrainer:
    def __init__(self):
        self.dtype_preserver = DataTypePreserver()

    def generate(self, dataset, num_samples):
        self.dtype_preserver.analyze_dataset(dataset)
        ctgan = CTGAN()
        ctgan.fit(dataset)
        synthetic_data = ctgan.sample(num_samples)
        synthetic_df = self.dtype_preserver.restore_types(synthetic_data)
        return synthetic_df


# SyntheticDataGenerator class
class SyntheticDataGenerator:
    def __init__(self):
        self.trainers = {
            "WGAN": WGANTrainer(),
            "VAE": VAETrainer(),
            "CTGAN": CTGANTrainer()
        }

    def train_and_generate(self, dataset, num_samples, output_prefix="synthetic", target_column=None):
        if target_column:
            dataset = dataset.copy()
            dataset[target_column] = dataset[target_column].astype(str)

        # Preprocess dataset: Ensure all columns are numeric
        for column in dataset.select_dtypes(include=["object", "category"]).columns:
            dataset[column] = pd.Categorical(dataset[column]).codes
        dataset.fillna(0, inplace=True)

        results = {}
        for method, trainer in self.trainers.items():
            synthetic_data = trainer.generate(dataset, num_samples)
            file_name = f"{output_prefix}_{method.lower()}.csv"
            synthetic_data.to_csv(file_name, index=False)
            results[method] = file_name
        return results
