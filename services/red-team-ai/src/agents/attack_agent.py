"""
Red Team Attack Agent - Autonomous offensive AI

This agent learns to:
- Discover attack paths
- Evade detection systems
- Chain vulnerabilities
- Generate novel exploit strategies
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np


class AttackAgent:
    """Main attack agent using custom RL implementation"""

    def __init__(
        self,
        observation_space_dim: int,
        action_space_dim: int,
        config: Dict[str, Any]
    ):
        self.observation_dim = observation_space_dim
        self.action_dim = action_space_dim
        self.config = config

        # Initialize custom policy network
        self.policy_network = AttackPolicyNetwork(
            input_dim=observation_space_dim,
            output_dim=action_space_dim,
            hidden_dims=config.get('hidden_dims', [256, 256])
        )

        # Initialize value network for advantage estimation
        self.value_network = ValueNetwork(
            input_dim=observation_space_dim,
            hidden_dims=config.get('hidden_dims', [256, 256])
        )

        # Memory for experience replay
        self.memory = ExperienceReplayBuffer(
            capacity=config.get('memory_capacity', 100000)
        )

        # Optimizer
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=config.get('learning_rate', 3e-4)
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=config.get('learning_rate', 3e-4)
        )

    def select_action(
        self,
        observation: np.ndarray,
        explore: bool = True
    ) -> Tuple[int, float]:
        """
        Select action using policy network

        Args:
            observation: Current environment state
            explore: Whether to add exploration noise

        Returns:
            action: Selected action index
            log_prob: Log probability of selected action
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action_probs = self.policy_network(obs_tensor)

            if explore:
                # Sample from distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                # Greedy selection
                action = action_probs.argmax(dim=-1)
                log_prob = torch.log(action_probs[0, action])

        return action.item(), log_prob.item()

    def train_step(self, batch_size: int = 256) -> Dict[str, float]:
        """
        Perform one training step using PPO algorithm

        Args:
            batch_size: Number of samples for training

        Returns:
            metrics: Training metrics
        """
        if len(self.memory) < batch_size:
            return {}

        # Sample batch from memory
        batch = self.memory.sample(batch_size)

        # Compute advantages using GAE
        advantages = self._compute_gae(batch)

        # PPO update
        policy_loss = self._update_policy(batch, advantages)
        value_loss = self._update_value(batch)

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'mean_advantage': advantages.mean().item()
        }

    def _compute_gae(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        gamma = self.config.get('gamma', 0.99)
        gae_lambda = self.config.get('gae_lambda', 0.95)

        # Get value predictions
        values = self.value_network(batch['observations'])
        next_values = self.value_network(batch['next_observations'])

        # Compute TD errors
        td_errors = batch['rewards'] + gamma * next_values * (1 - batch['dones']) - values

        # Compute GAE
        advantages = torch.zeros_like(td_errors)
        gae = 0
        for t in reversed(range(len(td_errors))):
            gae = td_errors[t] + gamma * gae_lambda * (1 - batch['dones'][t]) * gae
            advantages[t] = gae

        return advantages

    def _update_policy(
        self,
        batch: Dict[str, torch.Tensor],
        advantages: torch.Tensor
    ) -> float:
        """Update policy network using PPO clipping"""
        clip_ratio = self.config.get('clip_ratio', 0.2)

        # Get current action probabilities
        action_probs = self.policy_network(batch['observations'])
        dist = torch.distributions.Categorical(action_probs)
        new_log_probs = dist.log_prob(batch['actions'])

        # Compute ratio
        ratio = torch.exp(new_log_probs - batch['old_log_probs'])

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Optimize
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5)
        self.policy_optimizer.step()

        return policy_loss.item()

    def _update_value(self, batch: Dict[str, torch.Tensor]) -> float:
        """Update value network"""
        values = self.value_network(batch['observations'])
        value_loss = nn.MSELoss()(values, batch['returns'])

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 0.5)
        self.value_optimizer.step()

        return value_loss.item()


class AttackPolicyNetwork(nn.Module):
    """Custom neural network for attack policy"""

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

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ValueNetwork(nn.Module):
    """Value network for advantage estimation"""

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class ExperienceReplayBuffer:
    """Memory buffer for storing agent experiences"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, experience: Dict[str, Any]):
        """Add experience to buffer"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch from buffer"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        return {
            'observations': torch.FloatTensor([exp['observation'] for exp in batch]),
            'actions': torch.LongTensor([exp['action'] for exp in batch]),
            'rewards': torch.FloatTensor([exp['reward'] for exp in batch]),
            'next_observations': torch.FloatTensor([exp['next_observation'] for exp in batch]),
            'dones': torch.FloatTensor([exp['done'] for exp in batch]),
            'old_log_probs': torch.FloatTensor([exp['log_prob'] for exp in batch]),
            'returns': torch.FloatTensor([exp.get('return', 0) for exp in batch])
        }

    def __len__(self) -> int:
        return len(self.buffer)
