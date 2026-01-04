"""
Variational Autoencoder (VAE) for Attack Variant Generation
Generates novel attack variations for adversarial training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, List, Dict, Optional
import numpy as np


class AttackEncoder(nn.Module):
    """Encoder network: maps attack features to latent distribution"""

    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)

        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution
        Returns: (mu, logvar)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class AttackDecoder(nn.Module):
    """Decoder network: samples latent space to generate attacks"""

    def __init__(self, latent_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        layers = []
        in_dim = latent_dim

        for h_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            in_dim = h_dim

        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to attack features
        """
        h = self.decoder(z)
        return torch.sigmoid(self.fc_out(h))


class AttackVariantVAE(nn.Module):
    """
    Variational Autoencoder for generating novel attack variants
    Learns latent attack representation for controlled generation
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 32,
        beta: float = 1.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta  # KL divergence weight (beta-VAE)

        self.encoder = AttackEncoder(input_dim, hidden_dims, latent_dim)
        self.decoder = AttackDecoder(latent_dim, hidden_dims, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE
        Returns: (reconstruction, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        VAE loss = Reconstruction Loss + β * KL Divergence
        """
        # Reconstruction loss (binary cross-entropy)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # KL divergence: D_KL(q(z|x) || p(z))
        # p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }

    def generate(
        self,
        num_samples: int = 1,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Generate new attack variants from random latent samples
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior N(0, I)
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
        return samples

    def interpolate(
        self,
        attack1: torch.Tensor,
        attack2: torch.Tensor,
        steps: int = 10,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Interpolate between two attacks in latent space
        Useful for analyzing attack evolution
        """
        self.eval()
        with torch.no_grad():
            # Encode both attacks
            mu1, _ = self.encoder(attack1.unsqueeze(0).to(device))
            mu2, _ = self.encoder(attack2.unsqueeze(0).to(device))

            # Linear interpolation in latent space
            alphas = torch.linspace(0, 1, steps).to(device)
            interpolations = []

            for alpha in alphas:
                z = (1 - alpha) * mu1 + alpha * mu2
                interpolations.append(self.decoder(z))

            return torch.cat(interpolations, dim=0)


class ConditionalAttackVAE(AttackVariantVAE):
    """
    Conditional VAE: Generate attacks conditioned on attack type/category
    Enables controlled attack variant generation
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: List[int] = [512, 256, 128],
        latent_dim: int = 32,
        num_conditions: int = 10,  # Number of attack categories
        condition_dim: int = 10,
        beta: float = 1.0
    ):
        super().__init__(input_dim, hidden_dims, latent_dim, beta)

        self.num_conditions = num_conditions
        self.condition_dim = condition_dim

        # Re-initialize encoder/decoder with conditional inputs
        self.encoder = AttackEncoder(
            input_dim + condition_dim,
            hidden_dims,
            latent_dim
        )

        self.decoder = AttackDecoder(
            latent_dim + condition_dim,
            hidden_dims,
            input_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with conditioning
        Args:
            x: Attack features [batch, input_dim]
            condition: One-hot condition [batch, num_conditions]
        """
        # Encode with condition
        x_cond = torch.cat([x, condition], dim=1)
        mu, logvar = self.encoder(x_cond)
        z = self.reparameterize(mu, logvar)

        # Decode with condition
        z_cond = torch.cat([z, condition], dim=1)
        reconstruction = self.decoder(z_cond)

        return reconstruction, mu, logvar

    def generate_conditional(
        self,
        condition: int,
        num_samples: int = 1,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Generate attacks of specific type/category
        Args:
            condition: Attack category (0 to num_conditions-1)
            num_samples: Number of samples to generate
        """
        self.eval()
        with torch.no_grad():
            # Create one-hot condition
            cond_one_hot = F.one_hot(
                torch.tensor([condition] * num_samples),
                num_classes=self.num_conditions
            ).float().to(device)

            # Sample latent vectors
            z = torch.randn(num_samples, self.latent_dim).to(device)

            # Concatenate and decode
            z_cond = torch.cat([z, cond_one_hot], dim=1)
            samples = self.decoder(z_cond)

        return samples


class AttackVAETrainer:
    """Training pipeline for Attack VAE"""

    def __init__(
        self,
        model: AttackVariantVAE,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        batch = batch.to(self.device)

        # Forward pass
        recon_batch, mu, logvar = self.model(batch)

        # Compute loss
        losses = self.model.loss_function(recon_batch, batch, mu, logvar)

        # Backward pass
        self.optimizer.zero_grad()
        losses['loss'].backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def generate_attack_variants(
        self,
        original_attack: torch.Tensor,
        num_variants: int = 10,
        diversity: float = 1.0
    ) -> torch.Tensor:
        """
        Generate variants of an existing attack
        Args:
            original_attack: Base attack features
            num_variants: Number of variants to create
            diversity: Controls variation amount (0=same, 1=diverse)
        """
        self.model.eval()
        with torch.no_grad():
            # Encode original attack
            mu, logvar = self.model.encoder(original_attack.unsqueeze(0).to(self.device))

            # Generate variants by sampling around mu
            variants = []
            for _ in range(num_variants):
                z = mu + diversity * torch.randn_like(mu) * torch.exp(0.5 * logvar)
                variant = self.model.decoder(z)
                variants.append(variant)

            return torch.cat(variants, dim=0)


# Example usage
if __name__ == "__main__":
    # Initialize VAE
    vae = AttackVariantVAE(
        input_dim=256,
        hidden_dims=[512, 256, 128],
        latent_dim=32,
        beta=1.0
    )

    # Create dummy attack data
    batch_size = 64
    attack_features = torch.randn(batch_size, 256)

    # Train
    trainer = AttackVAETrainer(vae)
    losses = trainer.train_step(attack_features)
    print(f"Training losses: {losses}")

    # Generate new attacks
    new_attacks = vae.generate(num_samples=5)
    print(f"Generated {new_attacks.shape[0]} new attack variants")

    # Interpolate between attacks
    interpolated = vae.interpolate(attack_features[0], attack_features[1], steps=10)
    print(f"Interpolated path: {interpolated.shape}")

    # Conditional VAE
    cvae = ConditionalAttackVAE(input_dim=256, num_conditions=10)
    sql_injection_attacks = cvae.generate_conditional(condition=0, num_samples=5)
    print(f"Generated SQL injection variants: {sql_injection_attacks.shape}")
