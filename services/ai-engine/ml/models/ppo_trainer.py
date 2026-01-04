"""
Proximal Policy Optimization (PPO) for Attack Strategy Evolution
State-of-the-art RL algorithm for continuous learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PPOMemory:
    """Memory buffer for PPO training"""
    states: List[torch.Tensor]
    actions: List[int]
    rewards: List[float]
    log_probs: List[torch.Tensor]
    values: List[torch.Tensor]
    dones: List[bool]

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def clear(self):
        """Clear all memory"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self):
        return len(self.states)


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO
    Actor: Policy network (what action to take)
    Critic: Value network (how good is the current state)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            state: (batch_size, state_dim)

        Returns:
            action_probs: (batch_size, action_dim)
            state_value: (batch_size, 1)
        """
        features = self.shared(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)

        return action_probs, state_value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy

        Args:
            state: (state_dim,)
            deterministic: If True, take argmax instead of sampling

        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        action_probs, state_value = self.forward(state.unsqueeze(0))

        if deterministic:
            action = action_probs.argmax(dim=-1).item()
            dist = Categorical(action_probs)
            log_prob = dist.log_prob(torch.tensor([action]))
        else:
            dist = Categorical(action_probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)

        return action, log_prob, state_value.squeeze()


class PPOTrainer:
    """
    Proximal Policy Optimization Trainer
    Implements PPO-Clip algorithm for stable policy learning
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        epsilon_clip: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        batch_size: int = 64
    ):
        """
        Initialize PPO trainer

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            epsilon_clip: PPO clipping parameter
            value_loss_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            ppo_epochs: Number of PPO update epochs per batch
            batch_size: Batch size for training
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Actor-Critic network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Memory
        self.memory = PPOMemory()

        # Statistics
        self.total_steps = 0
        self.episodes_trained = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []

        logger.info(f"✅ PPO Trainer initialized on {self.device}")
        logger.info(f"   State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"   Gamma: {gamma}, Clip: {epsilon_clip}, LR: {lr}")

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action using current policy

        Args:
            state: Current state
            deterministic: Use deterministic policy (for evaluation)

        Returns:
            action, log_prob, value
        """
        state_tensor = torch.FloatTensor(state).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(state_tensor, deterministic)

        return action, log_prob, value

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool
    ):
        """Store transition in memory"""
        self.memory.states.append(torch.FloatTensor(state))
        self.memory.actions.append(action)
        self.memory.rewards.append(reward)
        self.memory.log_probs.append(log_prob)
        self.memory.values.append(value)
        self.memory.dones.append(done)

        self.total_steps += 1

    def compute_gae(self, rewards: List[float], values: List[torch.Tensor], dones: List[bool]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0

        # Convert to tensors
        values_tensor = torch.stack(values)
        next_values = torch.cat([values_tensor[1:], torch.tensor([0.0])])

        for t in reversed(range(len(rewards))):
            # TD error
            if dones[t]:
                delta = rewards[t] - values_tensor[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values_tensor[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values_tensor

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self):
        """
        Perform PPO update on collected experience
        """
        if len(self.memory) < self.batch_size:
            logger.warning(f"Not enough samples for update: {len(self.memory)} < {self.batch_size}")
            return

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            self.memory.rewards,
            self.memory.values,
            self.memory.dones
        )

        # Convert memory to tensors
        states = torch.stack(self.memory.states).to(self.device)
        actions = torch.tensor(self.memory.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.stack(self.memory.log_probs).to(self.device).detach()
        advantages = advantages.to(self.device)
        returns = returns.to(self.device)

        # PPO update epochs
        for epoch in range(self.ppo_epochs):
            # Generate random mini-batches
            num_samples = len(states)
            indices = np.random.permutation(num_samples)

            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_indices = indices[start:end]

                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                action_probs, state_values = self.policy(batch_states)

                # Calculate log probabilities
                dist = Categorical(action_probs)
                batch_log_probs = dist.log_prob(batch_actions)

                # Ratio for PPO-Clip
                ratio = torch.exp(batch_log_probs - batch_old_log_probs)

                # Surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * batch_advantages

                # Policy loss (maximize advantage)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (minimize TD error)
                value_loss = F.mse_loss(state_values.squeeze(), batch_returns)

                # Entropy loss (encourage exploration)
                entropy = dist.entropy().mean()
                entropy_loss = -entropy

                # Total loss
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track statistics
                self.policy_losses.append(policy_loss.item())
                self.value_losses.append(value_loss.item())
                self.entropy_losses.append(entropy_loss.item())

        self.episodes_trained += 1

        # Clear memory after update
        self.memory.clear()

        logger.info(f"PPO Update #{self.episodes_trained} | "
                   f"Policy Loss: {np.mean(self.policy_losses[-10:]):.4f} | "
                   f"Value Loss: {np.mean(self.value_losses[-10:]):.4f} | "
                   f"Entropy: {-np.mean(self.entropy_losses[-10:]):.4f}")

    def save(self, filepath: str):
        """Save model checkpoint"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'episodes_trained': self.episodes_trained
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.episodes_trained = checkpoint['episodes_trained']
        logger.info(f"Model loaded from {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            "total_steps": self.total_steps,
            "episodes_trained": self.episodes_trained,
            "avg_policy_loss": np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
            "avg_value_loss": np.mean(self.value_losses[-100:]) if self.value_losses else 0,
            "avg_entropy": -np.mean(self.entropy_losses[-100:]) if self.entropy_losses else 0
        }


# Example usage
if __name__ == "__main__":
    # Create PPO trainer
    state_dim = 10
    action_dim = 8  # Number of attack actions

    ppo = PPOTrainer(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        epsilon_clip=0.2
    )

    print("🤖 PPO Trainer initialized")
    print(f"   Device: {ppo.device}")
    print(f"   Parameters: {sum(p.numel() for p in ppo.policy.parameters()):,}")

    # Test action selection
    test_state = np.random.randn(state_dim)
    action, log_prob, value = ppo.select_action(test_state)
    print(f"\nTest action selection:")
    print(f"   State shape: {test_state.shape}")
    print(f"   Selected action: {action}")
    print(f"   Log prob: {log_prob.item():.4f}")
    print(f"   State value: {value.item():.4f}")

    print("\n✅ PPO Trainer test passed!")
