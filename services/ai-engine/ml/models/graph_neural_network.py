"""
Graph Neural Network for Attack Path Prediction
Uses GraphSAGE and GAT for knowledge graph reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import List, Tuple, Dict, Optional
import numpy as np


class GraphSAGEAttackPredictor(nn.Module):
    """
    GraphSAGE model for attack path prediction
    Learns node embeddings for attack techniques, vulnerabilities, and assets
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.convs.append(SAGEConv(hidden_dim, output_dim))

        self.dropout = nn.Dropout(dropout)
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])

        # Edge prediction head (for link prediction)
        self.edge_predictor = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Node classification head (attack technique classification)
        self.node_classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 10)  # 10 attack categories
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.convs[-1](x, edge_index)
        return x

    def predict_link(
        self,
        node_embeddings: torch.Tensor,
        src_nodes: torch.Tensor,
        dst_nodes: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict likelihood of attack path between nodes
        Args:
            node_embeddings: [num_nodes, output_dim]
            src_nodes: Source node indices [num_pairs]
            dst_nodes: Destination node indices [num_pairs]
        Returns:
            Probabilities [num_pairs]
        """
        src_emb = node_embeddings[src_nodes]
        dst_emb = node_embeddings[dst_nodes]
        edge_emb = torch.cat([src_emb, dst_emb], dim=1)
        return self.edge_predictor(edge_emb).squeeze()

    def classify_attack(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Classify attack technique category
        Args:
            node_embeddings: [num_nodes, output_dim]
        Returns:
            Class logits [num_nodes, 10]
        """
        return self.node_classifier(node_embeddings)


class GATAttackPredictor(nn.Module):
    """
    Graph Attention Network for attack path prediction
    Uses multi-head attention to weight important attack relationships
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()

        # GAT layers with multi-head attention
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout))

        self.convs.append(GATConv(hidden_dim, output_dim, heads=1, concat=False, dropout=dropout))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with attention weights
        Returns:
            node_embeddings: [num_nodes, output_dim]
            attention_weights: List of attention weights per layer
        """
        attention_weights = []

        for i, conv in enumerate(self.convs[:-1]):
            x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
            attention_weights.append(attn)
            x = F.elu(x)
            x = self.dropout(x)

        x, (edge_idx, attn) = self.convs[-1](x, edge_index, return_attention_weights=True)
        attention_weights.append(attn)

        return x, attention_weights


class KnowledgeGraphGNN:
    """
    Complete GNN system for cybersecurity knowledge graph reasoning
    Combines GraphSAGE and GAT for attack path prediction
    """

    def __init__(
        self,
        node_feature_dim: int = 128,
        hidden_dim: int = 256,
        embedding_dim: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device

        # Initialize models
        self.graphsage = GraphSAGEAttackPredictor(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim
        ).to(device)

        self.gat = GATAttackPredictor(
            input_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim
        ).to(device)

    def predict_attack_paths(
        self,
        graph_data: Data,
        source_nodes: List[int],
        k: int = 5
    ) -> List[Tuple[int, int, float]]:
        """
        Predict most likely attack paths from source nodes

        Args:
            graph_data: PyTorch Geometric Data object
            source_nodes: Starting nodes (compromised assets)
            k: Number of top predictions per source

        Returns:
            List of (source, target, probability) tuples
        """
        self.graphsage.eval()

        with torch.no_grad():
            x = graph_data.x.to(self.device)
            edge_index = graph_data.edge_index.to(self.device)

            # Get node embeddings
            embeddings = self.graphsage(x, edge_index)

            predictions = []

            for src in source_nodes:
                # Predict links from source to all other nodes
                src_tensor = torch.tensor([src] * x.size(0), device=self.device)
                dst_tensor = torch.arange(x.size(0), device=self.device)

                # Get probabilities
                probs = self.graphsage.predict_link(embeddings, src_tensor, dst_tensor)

                # Get top-k predictions
                top_k = torch.topk(probs, k=min(k, x.size(0)))

                for dst, prob in zip(top_k.indices, top_k.values):
                    if dst != src:  # Skip self-loops
                        predictions.append((src, dst.item(), prob.item()))

        return sorted(predictions, key=lambda x: x[2], reverse=True)

    def get_attention_weights(
        self,
        graph_data: Data,
        edge_of_interest: Optional[Tuple[int, int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get attention weights from GAT to understand important relationships

        Args:
            graph_data: PyTorch Geometric Data object
            edge_of_interest: Specific edge to analyze (src, dst)

        Returns:
            Dictionary of attention weights per layer
        """
        self.gat.eval()

        with torch.no_grad():
            x = graph_data.x.to(self.device)
            edge_index = graph_data.edge_index.to(self.device)

            _, attention_weights = self.gat(x, edge_index)

            result = {}
            for i, attn in enumerate(attention_weights):
                result[f"layer_{i}"] = attn.cpu()

            return result

    def classify_techniques(
        self,
        graph_data: Data,
        node_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Classify attack techniques for specified nodes

        Args:
            graph_data: PyTorch Geometric Data object
            node_indices: Nodes to classify (None = all)

        Returns:
            Classification logits [num_nodes, num_classes]
        """
        self.graphsage.eval()

        with torch.no_grad():
            x = graph_data.x.to(self.device)
            edge_index = graph_data.edge_index.to(self.device)

            embeddings = self.graphsage(x, edge_index)
            logits = self.graphsage.classify_attack(embeddings)

            if node_indices is not None:
                logits = logits[node_indices]

            return logits.cpu()


# Example usage
if __name__ == "__main__":
    # Create sample knowledge graph
    # Nodes: [web_server, database, api_gateway, attacker, sqli, xss]
    num_nodes = 6
    node_features = torch.randn(num_nodes, 128)

    # Edges: attack relationships
    edge_index = torch.tensor([
        [3, 3, 4, 5, 0, 1],  # source nodes
        [4, 5, 0, 0, 1, 2]   # target nodes
    ], dtype=torch.long)

    graph = Data(x=node_features, edge_index=edge_index)

    # Initialize GNN
    gnn = KnowledgeGraphGNN(node_feature_dim=128)

    # Predict attack paths from attacker node (index 3)
    predictions = gnn.predict_attack_paths(graph, source_nodes=[3], k=3)
    print("Top attack path predictions:")
    for src, dst, prob in predictions[:5]:
        print(f"  {src} -> {dst}: {prob:.3f}")

    # Get attention weights
    attention = gnn.get_attention_weights(graph)
    print(f"\nAttention weights collected for {len(attention)} layers")

    # Classify techniques
    logits = gnn.classify_techniques(graph)
    print(f"\nTechnique classifications: {logits.shape}")
