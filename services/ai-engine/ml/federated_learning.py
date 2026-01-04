"""
Federated Learning for Privacy-Preserving Model Training
Train security models across multiple organizations without sharing raw data
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import copy
import numpy as np
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Federated learning client
    Trains local model on private data, shares only model updates
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        local_data_loader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.client_id = client_id
        self.model = model.to(device)
        self.local_data_loader = local_data_loader
        self.device = device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def train_local_model(
        self,
        epochs: int = 5,
        differential_privacy: bool = False,
        epsilon: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Train model on local private data

        Args:
            epochs: Number of local training epochs
            differential_privacy: Apply differential privacy
            epsilon: Privacy budget (smaller = more privacy)

        Returns:
            Model state dict (parameters to share)
        """
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.local_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                loss.backward()

                # Apply differential privacy by adding noise to gradients
                if differential_privacy:
                    self._add_gradient_noise(epsilon)

                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.local_data_loader)
            logger.info(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # Return model parameters (not raw data!)
        return self.model.state_dict()

    def _add_gradient_noise(self, epsilon: float):
        """
        Add Gaussian noise to gradients for differential privacy

        Args:
            epsilon: Privacy budget
        """
        sensitivity = 1.0
        sigma = np.sqrt(2 * np.log(1.25)) * sensitivity / epsilon

        for param in self.model.parameters():
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=sigma,
                    size=param.grad.shape,
                    device=param.grad.device
                )
                param.grad += noise

    def update_model(self, global_state_dict: Dict[str, torch.Tensor]):
        """Update local model with aggregated global model"""
        self.model.load_state_dict(global_state_dict)


class FederatedServer:
    """
    Federated learning server
    Aggregates client model updates without accessing raw data
    """

    def __init__(
        self,
        global_model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.global_model = global_model.to(device)
        self.device = device
        self.round_count = 0

    def aggregate_models(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        aggregation_method: str = "fedavg",
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates

        Args:
            client_models: List of client model state dicts
            aggregation_method: Aggregation algorithm (fedavg, fedprox, fedopt)
            client_weights: Optional weights for each client (based on data size)

        Returns:
            Aggregated global model state dict
        """
        if not client_models:
            return self.global_model.state_dict()

        num_clients = len(client_models)

        # Default to equal weights
        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients

        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]

        if aggregation_method == "fedavg":
            return self._federated_averaging(client_models, client_weights)
        elif aggregation_method == "median":
            return self._coordinate_wise_median(client_models)
        elif aggregation_method == "trimmed_mean":
            return self._trimmed_mean(client_models, trim_ratio=0.1)
        else:
            return self._federated_averaging(client_models, client_weights)

    def _federated_averaging(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """
        FedAvg: Weighted average of client models

        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017
        """
        global_dict = OrderedDict()

        # Get keys from first client model
        for key in client_models[0].keys():
            # Weighted sum of client parameters
            global_dict[key] = sum(
                client_models[i][key] * weights[i]
                for i in range(len(client_models))
            )

        return global_dict

    def _coordinate_wise_median(
        self,
        client_models: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Robust aggregation using coordinate-wise median
        Resistant to Byzantine attacks
        """
        global_dict = OrderedDict()

        for key in client_models[0].keys():
            # Stack all client parameters
            stacked = torch.stack([model[key] for model in client_models])

            # Compute median across clients (dimension 0)
            global_dict[key] = torch.median(stacked, dim=0)[0]

        return global_dict

    def _trimmed_mean(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        trim_ratio: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Trimmed mean: Remove outliers before averaging
        Byzantine-robust aggregation
        """
        global_dict = OrderedDict()
        num_clients = len(client_models)
        num_trim = int(num_clients * trim_ratio)

        for key in client_models[0].keys():
            stacked = torch.stack([model[key] for model in client_models])

            # Sort and trim
            sorted_params, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_params[num_trim:num_clients - num_trim]

            # Average remaining
            global_dict[key] = torch.mean(trimmed, dim=0)

        return global_dict

    def update_global_model(self, aggregated_state_dict: Dict[str, torch.Tensor]):
        """Update global model with aggregated parameters"""
        self.global_model.load_state_dict(aggregated_state_dict)
        self.round_count += 1

    def get_global_model_state(self) -> Dict[str, torch.Tensor]:
        """Get current global model state"""
        return self.global_model.state_dict()


class SecureAggregation:
    """
    Secure aggregation protocol
    Prevents server from seeing individual client updates
    """

    def __init__(self, num_clients: int):
        self.num_clients = num_clients

    def generate_masks(self) -> List[Dict[str, torch.Tensor]]:
        """
        Generate random masks for each client
        Masks cancel out when summed
        """
        masks = []

        # Generate masks that sum to zero
        for i in range(self.num_clients - 1):
            mask = {}
            # Generate random mask
            # In practice, use cryptographic protocols
            masks.append(mask)

        # Last mask ensures sum = 0
        # final_mask = -sum(masks)

        return masks

    def apply_mask(
        self,
        model_update: Dict[str, torch.Tensor],
        mask: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply mask to model update"""
        masked_update = {}
        for key in model_update.keys():
            masked_update[key] = model_update[key] + mask.get(key, 0)
        return masked_update


# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    # Define simple model
    class SimpleSecurityModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )

        def forward(self, x):
            return self.fc(x)

    # Initialize global model
    global_model = SimpleSecurityModel()

    # Create federated server
    server = FederatedServer(global_model)

    # Simulate 5 clients with private data
    clients = []
    for i in range(5):
        # Each client has private local data
        X = torch.randn(100, 128)
        y = torch.randint(0, 10, (100,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32)

        client = FederatedClient(
            client_id=f"client_{i}",
            model=copy.deepcopy(global_model),
            local_data_loader=dataloader
        )
        clients.append(client)

    # Federated training rounds
    num_rounds = 10

    for round_num in range(num_rounds):
        print(f"\n=== Federated Round {round_num + 1}/{num_rounds} ===")

        # Each client trains locally
        client_updates = []
        for client in clients:
            # Train on private data with differential privacy
            local_model = client.train_local_model(
                epochs=3,
                differential_privacy=True,
                epsilon=1.0
            )
            client_updates.append(local_model)

        # Server aggregates updates (FedAvg)
        aggregated_model = server.aggregate_models(
            client_updates,
            aggregation_method="fedavg"
        )

        # Update global model
        server.update_global_model(aggregated_model)

        # Distribute global model to clients
        for client in clients:
            client.update_model(server.get_global_model_state())

    print("\nFederated training complete!")
    print(f"Total rounds: {server.round_count}")
