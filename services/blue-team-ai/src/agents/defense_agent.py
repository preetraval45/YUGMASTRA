"""
Blue Team Defense Agent - Autonomous defensive AI

This agent learns to:
- Detect anomalies and attacks
- Correlate security events
- Generate detection rules
- Respond to threats adaptively
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from collections import deque


class DefenseAgent:
    """Main defense agent using custom ML/DL implementation"""

    def __init__(
        self,
        observation_space_dim: int,
        action_space_dim: int,
        config: Dict[str, Any]
    ):
        self.observation_dim = observation_space_dim
        self.action_dim = action_space_dim
        self.config = config

        # Anomaly detection module
        self.anomaly_detector = AnomalyDetectionNetwork(
            input_dim=observation_space_dim,
            latent_dim=config.get('latent_dim', 128),
            hidden_dims=config.get('hidden_dims', [256, 128])
        )

        # Temporal pattern analyzer
        self.temporal_analyzer = TemporalLSTMNetwork(
            input_dim=observation_space_dim,
            hidden_dim=config.get('lstm_hidden', 256),
            num_layers=config.get('lstm_layers', 2)
        )

        # Response policy network
        self.response_policy = ResponsePolicyNetwork(
            input_dim=observation_space_dim + 128,  # obs + anomaly features
            output_dim=action_space_dim,
            hidden_dims=config.get('policy_hidden', [256, 256])
        )

        # Alert correlation module
        self.alert_correlator = AlertCorrelator(
            max_window_size=config.get('correlation_window', 100)
        )

        # Adaptive threshold manager
        self.threshold_manager = AdaptiveThresholdManager(
            initial_threshold=config.get('initial_threshold', 0.5),
            adaptation_rate=config.get('adaptation_rate', 0.01)
        )

        # Optimizers
        self.anomaly_optimizer = torch.optim.Adam(
            self.anomaly_detector.parameters(),
            lr=config.get('learning_rate', 3e-4)
        )
        self.temporal_optimizer = torch.optim.Adam(
            self.temporal_analyzer.parameters(),
            lr=config.get('learning_rate', 3e-4)
        )
        self.policy_optimizer = torch.optim.Adam(
            self.response_policy.parameters(),
            lr=config.get('learning_rate', 3e-4)
        )

        # Memory for normal behavior patterns
        self.normal_behavior_buffer = deque(maxlen=10000)

    def detect_anomaly(
        self,
        observation: np.ndarray,
        temporal_context: Optional[List[np.ndarray]] = None
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Detect if observation is anomalous

        Args:
            observation: Current system state
            temporal_context: Previous observations for temporal analysis

        Returns:
            is_anomaly: Whether anomaly detected
            confidence: Detection confidence score
            features: Extracted anomaly features
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)

            # Get reconstruction from autoencoder
            reconstructed, latent_features = self.anomaly_detector(obs_tensor)
            reconstruction_error = torch.mean((obs_tensor - reconstructed) ** 2).item()

            # Temporal analysis if context provided
            temporal_score = 0.0
            if temporal_context:
                context_tensor = torch.FloatTensor(temporal_context).unsqueeze(0)
                temporal_features = self.temporal_analyzer(context_tensor)
                temporal_score = self._compute_temporal_anomaly_score(
                    temporal_features
                ).item()

            # Combined anomaly score
            anomaly_score = 0.7 * reconstruction_error + 0.3 * temporal_score

            # Adaptive threshold
            threshold = self.threshold_manager.get_threshold()
            is_anomaly = anomaly_score > threshold

            features = {
                'reconstruction_error': reconstruction_error,
                'temporal_score': temporal_score,
                'anomaly_score': anomaly_score,
                'threshold': threshold,
                'latent_features': latent_features.cpu().numpy()
            }

        return is_anomaly, anomaly_score, features

    def select_response_action(
        self,
        observation: np.ndarray,
        anomaly_features: Dict[str, Any]
    ) -> Tuple[int, float]:
        """
        Select appropriate response action

        Args:
            observation: Current system state
            anomaly_features: Features from anomaly detection

        Returns:
            action: Response action to take
            confidence: Action confidence
        """
        with torch.no_grad():
            # Combine observation with anomaly features
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            latent_tensor = torch.FloatTensor(anomaly_features['latent_features'])
            combined_input = torch.cat([obs_tensor, latent_tensor], dim=-1)

            # Get action probabilities
            action_probs = self.response_policy(combined_input)
            action = action_probs.argmax(dim=-1).item()
            confidence = action_probs[0, action].item()

        return action, confidence

    def train_anomaly_detector(
        self,
        normal_observations: np.ndarray,
        batch_size: int = 256
    ) -> Dict[str, float]:
        """
        Train anomaly detector on normal behavior

        Args:
            normal_observations: Dataset of normal system behavior
            batch_size: Training batch size

        Returns:
            metrics: Training metrics
        """
        # Add to normal behavior buffer
        for obs in normal_observations:
            self.normal_behavior_buffer.append(obs)

        if len(self.normal_behavior_buffer) < batch_size:
            return {}

        # Sample batch
        indices = np.random.choice(
            len(self.normal_behavior_buffer),
            batch_size,
            replace=False
        )
        batch = torch.FloatTensor([self.normal_behavior_buffer[i] for i in indices])

        # Forward pass
        reconstructed, latent = self.anomaly_detector(batch)

        # Reconstruction loss
        reconstruction_loss = nn.MSELoss()(reconstructed, batch)

        # KL divergence regularization for latent space
        kl_loss = -0.5 * torch.sum(1 + torch.log(latent.var(dim=0)) - latent.mean(dim=0).pow(2) - latent.var(dim=0))

        total_loss = reconstruction_loss + 0.01 * kl_loss

        # Optimize
        self.anomaly_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.anomaly_detector.parameters(), 1.0)
        self.anomaly_optimizer.step()

        return {
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }

    def train_response_policy(
        self,
        experiences: List[Dict[str, Any]],
        batch_size: int = 256
    ) -> Dict[str, float]:
        """
        Train response policy using collected experiences

        Args:
            experiences: List of (state, action, reward) tuples
            batch_size: Training batch size

        Returns:
            metrics: Training metrics
        """
        if len(experiences) < batch_size:
            return {}

        # Sample batch
        batch = np.random.choice(experiences, batch_size, replace=False)

        observations = torch.FloatTensor([exp['observation'] for exp in batch])
        actions = torch.LongTensor([exp['action'] for exp in batch])
        rewards = torch.FloatTensor([exp['reward'] for exp in batch])

        # Forward pass
        action_probs = self.response_policy(observations)

        # Policy gradient loss
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        policy_loss = -torch.mean(log_probs * rewards)

        # Optimize
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.response_policy.parameters(), 1.0)
        self.policy_optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'mean_reward': rewards.mean().item()
        }

    def update_thresholds(
        self,
        false_positive_rate: float,
        false_negative_rate: float
    ):
        """Update adaptive thresholds based on performance"""
        self.threshold_manager.update(false_positive_rate, false_negative_rate)

    def generate_detection_rule(
        self,
        attack_pattern: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable detection rule from learned pattern

        Args:
            attack_pattern: Detected attack pattern features

        Returns:
            rule: Detection rule in Sigma format
        """
        # This would use the knowledge distillation layer
        # Placeholder for now
        rule = f"""
title: AI-Generated Detection Rule
description: Automatically generated from learned attack pattern
detection:
    condition: anomaly_score > {attack_pattern.get('threshold', 0.5)}
    features:
        - reconstruction_error: {attack_pattern.get('reconstruction_error', 0)}
        - temporal_score: {attack_pattern.get('temporal_score', 0)}
"""
        return rule

    def _compute_temporal_anomaly_score(
        self,
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute anomaly score from temporal features"""
        # Compare with learned normal temporal patterns
        # Simplified implementation
        return torch.norm(temporal_features, p=2, dim=-1)


class AnomalyDetectionNetwork(nn.Module):
    """Autoencoder for anomaly detection"""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int]):
        super().__init__()

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


class TemporalLSTMNetwork(nn.Module):
    """LSTM for temporal pattern analysis"""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.output_layer = nn.Linear(hidden_dim, hidden_dim // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # Use last timestep output
        last_output = lstm_out[:, -1, :]
        return self.output_layer(last_output)


class ResponsePolicyNetwork(nn.Module):
    """Policy network for response actions"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.Softmax(dim=-1)
        ])

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class AlertCorrelator:
    """Correlates multiple security alerts"""

    def __init__(self, max_window_size: int):
        self.max_window_size = max_window_size
        self.alert_history = deque(maxlen=max_window_size)

    def add_alert(self, alert: Dict[str, Any]):
        """Add alert to correlation window"""
        self.alert_history.append(alert)

    def correlate(self) -> List[Dict[str, Any]]:
        """Find correlated alert chains"""
        # Simplified correlation logic
        # In full implementation, would use graph-based correlation
        correlated_chains = []
        # TODO: Implement sophisticated correlation
        return correlated_chains


class AdaptiveThresholdManager:
    """Manages adaptive detection thresholds"""

    def __init__(self, initial_threshold: float, adaptation_rate: float):
        self.threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.performance_history = deque(maxlen=1000)

    def get_threshold(self) -> float:
        """Get current threshold"""
        return self.threshold

    def update(self, false_positive_rate: float, false_negative_rate: float):
        """Update threshold based on performance"""
        self.performance_history.append({
            'fpr': false_positive_rate,
            'fnr': false_negative_rate
        })

        # Increase threshold if too many false positives
        if false_positive_rate > 0.1:
            self.threshold += self.adaptation_rate
        # Decrease threshold if too many false negatives
        elif false_negative_rate > 0.1:
            self.threshold -= self.adaptation_rate

        # Clamp to valid range
        self.threshold = np.clip(self.threshold, 0.1, 0.9)
