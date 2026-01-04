"""
Multi-Agent Reinforcement Learning (MARL) Engine
Enables red team and blue team to learn simultaneously through adversarial co-evolution
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

from .ppo_trainer import PPOTrainer, ActorCritic

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in MARL"""
    RED_TEAM = "red_team"  # Attacker
    BLUE_TEAM = "blue_team"  # Defender


@dataclass
class MARLState:
    """Multi-agent state representation"""
    red_state: np.ndarray  # Attacker's view
    blue_state: np.ndarray  # Defender's view
    shared_state: np.ndarray  # Shared environment state
    step: int
    episode_reward: Dict[AgentRole, float]


@dataclass
class MARLTransition:
    """Transition for multi-agent learning"""
    red_state: np.ndarray
    blue_state: np.ndarray
    red_action: int
    blue_action: int
    red_reward: float
    blue_reward: float
    next_red_state: np.ndarray
    next_blue_state: np.ndarray
    done: bool


class SelfPlayBuffer:
    """
    Replay buffer for self-play training
    Stores experiences from both red and blue agents
    """

    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, transition: MARLTransition):
        """Add transition to buffer"""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[MARLTransition]:
        """Sample random batch"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class MultiAgentRLEngine:
    """
    Multi-Agent Reinforcement Learning Engine for Adversarial Co-Evolution
    Red team (attacker) and Blue team (defender) learn simultaneously
    """

    def __init__(
        self,
        red_state_dim: int,
        blue_state_dim: int,
        red_action_dim: int,
        blue_action_dim: int,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        use_self_play: bool = True,
        nash_epsilon: float = 0.01  # Threshold for Nash equilibrium detection
    ):
        """
        Initialize MARL engine

        Args:
            red_state_dim: Red team state dimension
            blue_state_dim: Blue team state dimension
            red_action_dim: Red team action space size
            blue_action_dim: Blue team action space size
            hidden_dim: Hidden layer dimension
            lr: Learning rate
            gamma: Discount factor
            use_self_play: Enable self-play training
            nash_epsilon: Nash equilibrium detection threshold
        """
        self.use_self_play = use_self_play
        self.nash_epsilon = nash_epsilon

        # Initialize PPO trainers for both agents
        self.red_agent = PPOTrainer(
            state_dim=red_state_dim,
            action_dim=red_action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma
        )

        self.blue_agent = PPOTrainer(
            state_dim=blue_state_dim,
            action_dim=blue_action_dim,
            hidden_dim=hidden_dim,
            lr=lr,
            gamma=gamma
        )

        # Self-play buffer
        self.self_play_buffer = SelfPlayBuffer(max_size=100000)

        # Training statistics
        self.episodes_played = 0
        self.red_wins = 0
        self.blue_wins = 0
        self.draws = 0
        self.nash_equilibrium_reached = False

        # Performance tracking for Nash detection
        self.red_performance_history = deque(maxlen=100)
        self.blue_performance_history = deque(maxlen=100)

        logger.info("✅ Multi-Agent RL Engine initialized")
        logger.info(f"   Red Agent: {red_state_dim}D state, {red_action_dim} actions")
        logger.info(f"   Blue Agent: {blue_state_dim}D state, {blue_action_dim} actions")
        logger.info(f"   Self-play: {use_self_play}")

    def select_actions(
        self,
        red_state: np.ndarray,
        blue_state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, int, Dict[str, Any]]:
        """
        Select actions for both agents

        Args:
            red_state: Red team state observation
            blue_state: Blue team state observation
            deterministic: Use deterministic policy

        Returns:
            red_action: Red team action
            blue_action: Blue team action
            info: Additional information (log probs, values)
        """
        # Red team action
        red_action, red_log_prob, red_value = self.red_agent.select_action(
            red_state, deterministic
        )

        # Blue team action
        blue_action, blue_log_prob, blue_value = self.blue_agent.select_action(
            blue_state, deterministic
        )

        info = {
            "red_log_prob": red_log_prob,
            "red_value": red_value,
            "blue_log_prob": blue_log_prob,
            "blue_value": blue_value
        }

        return red_action, blue_action, info

    def store_transition(
        self,
        red_state: np.ndarray,
        blue_state: np.ndarray,
        red_action: int,
        blue_action: int,
        red_reward: float,
        blue_reward: float,
        info: Dict[str, Any],
        done: bool
    ):
        """Store transition for both agents"""
        # Store in individual agent memories
        self.red_agent.store_transition(
            red_state, red_action, red_reward,
            info["red_log_prob"], info["red_value"], done
        )

        self.blue_agent.store_transition(
            blue_state, blue_action, blue_reward,
            info["blue_log_prob"], info["blue_value"], done
        )

        # Store in self-play buffer if enabled
        if self.use_self_play:
            # Will be populated in episode_complete
            pass

    def update_agents(self):
        """Update both agents using collected experience"""
        logger.info("Updating agents...")

        # Update red team
        self.red_agent.update()

        # Update blue team
        self.blue_agent.update()

        # Check for Nash equilibrium
        self._check_nash_equilibrium()

    def episode_complete(self, winner: Optional[AgentRole], total_reward: Dict[AgentRole, float]):
        """
        Called when an episode completes

        Args:
            winner: Which agent won (None for draw)
            total_reward: Total rewards for each agent
        """
        self.episodes_played += 1

        # Track wins
        if winner == AgentRole.RED_TEAM:
            self.red_wins += 1
        elif winner == AgentRole.BLUE_TEAM:
            self.blue_wins += 1
        else:
            self.draws += 1

        # Track performance for Nash detection
        self.red_performance_history.append(total_reward[AgentRole.RED_TEAM])
        self.blue_performance_history.append(total_reward[AgentRole.BLUE_TEAM])

        # Log statistics
        if self.episodes_played % 10 == 0:
            win_rate_red = self.red_wins / self.episodes_played
            win_rate_blue = self.blue_wins / self.episodes_played
            draw_rate = self.draws / self.episodes_played

            logger.info(f"Episode {self.episodes_played} | "
                       f"Red: {win_rate_red:.2%} | "
                       f"Blue: {win_rate_blue:.2%} | "
                       f"Draw: {draw_rate:.2%}")

            if self.nash_equilibrium_reached:
                logger.info("   ⚖️  Nash Equilibrium detected!")

    def _check_nash_equilibrium(self):
        """
        Detect Nash equilibrium - when neither agent can improve by changing strategy

        Nash equilibrium is approximated by checking if:
        1. Win rates are balanced (close to 50-50)
        2. Performance variance is low (strategies are stable)
        """
        if len(self.red_performance_history) < 100:
            return

        # Calculate win rates
        recent_episodes = 100
        recent_red_wins = sum(1 for r in list(self.red_performance_history)[-recent_episodes:] if r > 0)
        recent_blue_wins = sum(1 for r in list(self.blue_performance_history)[-recent_episodes:] if r > 0)

        red_win_rate = recent_red_wins / recent_episodes
        blue_win_rate = recent_blue_wins / recent_episodes

        # Check balance (should be close to 0.5 each)
        win_rate_balance = abs(red_win_rate - 0.5)

        # Check stability (low variance in performance)
        red_variance = np.var(list(self.red_performance_history)[-recent_episodes:])
        blue_variance = np.var(list(self.blue_performance_history)[-recent_episodes:])
        avg_variance = (red_variance + blue_variance) / 2

        # Nash equilibrium criteria
        balanced = win_rate_balance < 0.1  # Within 40-60%
        stable = avg_variance < 10.0  # Low performance variance

        if balanced and stable:
            if not self.nash_equilibrium_reached:
                logger.info("🎯 Nash Equilibrium Reached!")
                logger.info(f"   Red win rate: {red_win_rate:.2%}")
                logger.info(f"   Blue win rate: {blue_win_rate:.2%}")
                logger.info(f"   Performance variance: {avg_variance:.2f}")
                self.nash_equilibrium_reached = True
        else:
            self.nash_equilibrium_reached = False

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        red_stats = self.red_agent.get_statistics()
        blue_stats = self.blue_agent.get_statistics()

        win_rate_red = self.red_wins / max(self.episodes_played, 1)
        win_rate_blue = self.blue_wins / max(self.episodes_played, 1)
        draw_rate = self.draws / max(self.episodes_played, 1)

        return {
            "episodes_played": self.episodes_played,
            "red_wins": self.red_wins,
            "blue_wins": self.blue_wins,
            "draws": self.draws,
            "red_win_rate": win_rate_red,
            "blue_win_rate": win_rate_blue,
            "draw_rate": draw_rate,
            "nash_equilibrium": self.nash_equilibrium_reached,
            "red_agent_stats": red_stats,
            "blue_agent_stats": blue_stats,
            "red_avg_reward": np.mean(list(self.red_performance_history)) if self.red_performance_history else 0,
            "blue_avg_reward": np.mean(list(self.blue_performance_history)) if self.blue_performance_history else 0
        }

    def save(self, filepath_prefix: str):
        """Save both agents"""
        self.red_agent.save(f"{filepath_prefix}_red.pt")
        self.blue_agent.save(f"{filepath_prefix}_blue.pt")

        # Save MARL statistics
        stats = {
            "episodes_played": self.episodes_played,
            "red_wins": self.red_wins,
            "blue_wins": self.blue_wins,
            "draws": self.draws,
            "nash_equilibrium_reached": self.nash_equilibrium_reached
        }

        torch.save(stats, f"{filepath_prefix}_stats.pt")
        logger.info(f"MARL engine saved to {filepath_prefix}_*.pt")

    def load(self, filepath_prefix: str):
        """Load both agents"""
        self.red_agent.load(f"{filepath_prefix}_red.pt")
        self.blue_agent.load(f"{filepath_prefix}_blue.pt")

        # Load MARL statistics
        stats = torch.load(f"{filepath_prefix}_stats.pt")
        self.episodes_played = stats["episodes_played"]
        self.red_wins = stats["red_wins"]
        self.blue_wins = stats["blue_wins"]
        self.draws = stats["draws"]
        self.nash_equilibrium_reached = stats["nash_equilibrium_reached"]

        logger.info(f"MARL engine loaded from {filepath_prefix}_*.pt")

    def curriculum_learning(self, difficulty_level: int) -> Dict[str, Any]:
        """
        Implement curriculum learning - gradually increase difficulty

        Args:
            difficulty_level: Current difficulty (1-10)

        Returns:
            Environment configuration based on difficulty
        """
        # Adjust learning parameters based on difficulty
        configs = {
            1: {"attack_complexity": 0.2, "defense_strength": 0.3, "time_limit": 100},
            3: {"attack_complexity": 0.4, "defense_strength": 0.5, "time_limit": 75},
            5: {"attack_complexity": 0.6, "defense_strength": 0.7, "time_limit": 50},
            7: {"attack_complexity": 0.8, "defense_strength": 0.9, "time_limit": 40},
            10: {"attack_complexity": 1.0, "defense_strength": 1.0, "time_limit": 30}
        }

        # Find closest difficulty level
        closest_level = min(configs.keys(), key=lambda x: abs(x - difficulty_level))
        return configs[closest_level]


# Example usage
if __name__ == "__main__":
    # Initialize MARL engine
    marl = MultiAgentRLEngine(
        red_state_dim=20,  # Red team state features
        blue_state_dim=20,  # Blue team state features
        red_action_dim=8,  # Number of attack actions
        blue_action_dim=6,  # Number of defense actions
        use_self_play=True
    )

    print("🎮 Multi-Agent RL Engine initialized")

    # Simulate episode
    red_state = np.random.randn(20)
    blue_state = np.random.randn(20)

    red_action, blue_action, info = marl.select_actions(red_state, blue_state)

    print(f"\nSimulated turn:")
    print(f"   Red action: {red_action}")
    print(f"   Blue action: {blue_action}")

    # Store transition
    marl.store_transition(
        red_state, blue_state,
        red_action, blue_action,
        red_reward=5.0, blue_reward=-3.0,
        info=info, done=False
    )

    # Get statistics
    stats = marl.get_statistics()
    print(f"\nStatistics:")
    print(f"   Episodes: {stats['episodes_played']}")
    print(f"   Nash Equilibrium: {stats['nash_equilibrium']}")

    print("\n✅ MARL Engine test passed!")
