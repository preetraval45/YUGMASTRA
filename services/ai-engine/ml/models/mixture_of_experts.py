"""
Mixture of Experts (MoE) for Cybersecurity
Routes queries to specialized expert models for different security domains
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np


class ExpertNetwork(nn.Module):
    """Single expert network for specific security domain"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GatingNetwork(nn.Module):
    """
    Gating network that decides which experts to use
    Outputs probability distribution over experts
    """

    def __init__(
        self,
        input_dim: int,
        num_experts: int,
        hidden_dim: int = 256,
        top_k: int = 2  # Sparse MoE: use top-k experts
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)

        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts)
        )

        # Noise for load balancing (auxiliary loss)
        self.noise_epsilon = 1e-2

    def forward(
        self,
        x: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features [batch, input_dim]
            training: Whether in training mode (adds noise)

        Returns:
            gates: Expert weights [batch, num_experts]
            top_k_indices: Indices of top-k experts [batch, top_k]
            load: Load per expert for balancing
        """
        # Compute logits
        logits = self.gate(x)

        # Add noise during training for exploration
        if training:
            noise = torch.randn_like(logits) * self.noise_epsilon
            logits = logits + noise

        # Softmax over experts
        gates = F.softmax(logits, dim=-1)

        # Select top-k experts (sparse MoE)
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)

        # Renormalize top-k gates
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)

        # Compute load for load balancing loss
        load = gates.sum(dim=0)

        return top_k_gates, top_k_indices, load


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts for Cybersecurity Analysis
    Routes security queries to specialized expert models
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_names: Optional[List[str]] = None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Expert names for interpretability
        self.expert_names = expert_names or [
            "Web Security",
            "Network Security",
            "Cloud Security",
            "Malware Analysis",
            "Cryptography",
            "Application Security",
            "Infrastructure Security",
            "Threat Intelligence"
        ][:num_experts]

        # Gating network
        self.gate = GatingNetwork(input_dim, num_experts, hidden_dim, top_k)

        # Expert networks
        self.experts = nn.ModuleList([
            ExpertNetwork(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])

        # Load balancing loss weight
        self.load_balance_weight = 0.01

    def forward(
        self,
        x: torch.Tensor,
        return_expert_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through MoE
        Args:
            x: Input features [batch, input_dim]
            return_expert_info: Whether to return expert usage info

        Returns:
            output: Combined expert outputs [batch, output_dim]
            expert_info: Dictionary with expert usage (if requested)
        """
        batch_size = x.size(0)

        # Get gating decisions
        top_k_gates, top_k_indices, load = self.gate(x, training=self.training)

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        # Shape: [batch, num_experts, output_dim]

        # Gather top-k expert outputs
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, self.top_k)
        selected_outputs = expert_outputs[batch_indices, top_k_indices]
        # Shape: [batch, top_k, output_dim]

        # Weighted combination
        top_k_gates_expanded = top_k_gates.unsqueeze(-1)  # [batch, top_k, 1]
        output = (selected_outputs * top_k_gates_expanded).sum(dim=1)
        # Shape: [batch, output_dim]

        expert_info = None
        if return_expert_info:
            expert_info = {
                'expert_weights': top_k_gates,
                'expert_indices': top_k_indices,
                'expert_names': [[self.expert_names[idx] for idx in indices] for indices in top_k_indices.cpu().numpy()],
                'load': load,
                'load_balance_loss': self._load_balance_loss(load)
            }

        return output, expert_info

    def _load_balance_loss(self, load: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary loss to balance load across experts
        Encourages uniform distribution of examples to experts
        """
        # Coefficient of variation
        mean_load = load.mean()
        var_load = ((load - mean_load) ** 2).mean()
        cv = torch.sqrt(var_load) / (mean_load + 1e-10)
        return cv * self.load_balance_weight

    def get_expert_specializations(
        self,
        dataloader,
        device: str = "cpu"
    ) -> Dict[str, List[str]]:
        """
        Analyze which types of queries each expert specializes in
        Returns mapping of expert -> query types
        """
        self.eval()
        expert_query_counts = {name: {} for name in self.expert_names}

        with torch.no_grad():
            for batch, labels in dataloader:
                batch = batch.to(device)

                _, expert_info = self.forward(batch, return_expert_info=True)

                # Count which experts handle which query types
                for expert_names, label in zip(expert_info['expert_names'], labels):
                    for expert_name in expert_names:
                        label_str = str(label.item())
                        if label_str not in expert_query_counts[expert_name]:
                            expert_query_counts[expert_name][label_str] = 0
                        expert_query_counts[expert_name][label_str] += 1

        return expert_query_counts


class SecurityMoE:
    """
    Complete Mixture of Experts system for cybersecurity analysis
    Pre-configured with domain-specific experts
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        output_dim: int = 256,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device

        # Expert specializations
        expert_names = [
            "Web Application Security",      # SQLi, XSS, CSRF
            "Network Security",               # DDoS, MITM, port scans
            "Cloud Security",                 # AWS, Azure, GCP misconfigs
            "Malware Analysis",               # Ransomware, trojans, backdoors
            "Cryptography",                   # Encryption, hashing, PKI
            "API Security",                   # REST, GraphQL, authentication
            "Container Security",             # Docker, Kubernetes
            "Threat Intelligence"             # APTs, IOCs, threat actors
        ]

        self.model = MixtureOfExperts(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_experts=len(expert_names),
            top_k=2,
            expert_names=expert_names
        ).to(device)

    def analyze_threat(
        self,
        threat_features: torch.Tensor,
        return_expert_reasoning: bool = True
    ) -> Dict[str, any]:
        """
        Analyze cybersecurity threat using appropriate experts

        Args:
            threat_features: Threat representation [batch, input_dim]
            return_expert_reasoning: Include which experts were consulted

        Returns:
            Dictionary with analysis and expert info
        """
        self.model.eval()

        with torch.no_grad():
            threat_features = threat_features.to(self.device)
            output, expert_info = self.model(
                threat_features,
                return_expert_info=return_expert_reasoning
            )

            result = {
                'threat_analysis': output.cpu(),
            }

            if expert_info:
                result.update({
                    'consulted_experts': expert_info['expert_names'],
                    'expert_confidence': expert_info['expert_weights'].cpu().numpy(),
                })

            return result

    def route_query(
        self,
        query_embedding: torch.Tensor
    ) -> List[Tuple[str, float]]:
        """
        Determine which experts should handle the query
        Returns list of (expert_name, confidence) tuples
        """
        self.model.eval()

        with torch.no_grad():
            query_embedding = query_embedding.unsqueeze(0).to(self.device)
            _, expert_info = self.model(query_embedding, return_expert_info=True)

            experts = expert_info['expert_names'][0]
            weights = expert_info['expert_weights'][0].cpu().numpy()

            return list(zip(experts, weights))


# Example usage
if __name__ == "__main__":
    # Initialize MoE
    moe = SecurityMoE(input_dim=512, hidden_dim=1024, output_dim=256)

    # Simulate threat features
    batch_size = 16
    threat_features = torch.randn(batch_size, 512)

    # Analyze threats
    result = moe.analyze_threat(threat_features, return_expert_reasoning=True)

    print("Threat Analysis Shape:", result['threat_analysis'].shape)
    print("\nExpert Consultations:")
    for i, (experts, weights) in enumerate(zip(result['consulted_experts'][:3], result['expert_confidence'][:3])):
        print(f"\nThreat {i+1}:")
        for expert, weight in zip(experts, weights):
            print(f"  - {expert}: {weight:.3f}")

    # Route a specific query
    query = torch.randn(512)
    routing = moe.route_query(query)
    print("\nQuery Routing:")
    for expert, confidence in routing:
        print(f"  {expert}: {confidence:.3f}")
