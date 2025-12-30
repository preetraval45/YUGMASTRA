"""
Reinforcement Learning Evolution Engine with PPO
Multi-Agent RL for Red Team vs Blue Team Co-Evolution
Uses: Stable-Baselines3 (FREE), RLlib (FREE)
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import gymnasium as gym
from gymnasium import spaces
import os

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5

class CyberAttackEnv(gym.Env):
    """
    Cyber Range Environment for RL Training
    State: Network topology, service states, detection systems
    Actions: Attack techniques, defense measures
    Reward: Based on attack success vs detection
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}

        # Define observation space
        # [network_state(10), service_states(20), ids_state(5), firewall_state(5)]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(40,), dtype=np.float32
        )

        # Define action space
        # [attack_type(10), target_service(20), evasion_technique(10)]
        self.action_space = spaces.MultiDiscrete([10, 20, 10])

        self.max_steps = 100
        self.current_step = 0
        self.network_state = None

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        self.current_step = 0

        # Initialize network state randomly
        self.network_state = {
            'network': np.random.rand(10),
            'services': np.random.rand(20),
            'ids': np.random.rand(5),
            'firewall': np.random.rand(5),
            'compromised_services': set(),
            'detected_attacks': 0,
            'successful_attacks': 0,
        }

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1

        attack_type, target_service, evasion_technique = action

        # Simulate attack execution
        attack_success = self._execute_attack(attack_type, target_service, evasion_technique)

        # Calculate reward
        reward = self._calculate_reward(attack_success)

        # Check if episode is done
        done = (
            self.current_step >= self.max_steps or
            len(self.network_state['compromised_services']) >= 15
        )

        truncated = False
        obs = self._get_observation()
        info = {
            'attack_success': attack_success,
            'compromised_count': len(self.network_state['compromised_services']),
            'detected_count': self.network_state['detected_attacks']
        }

        return obs, reward, done, truncated, info

    def _execute_attack(self, attack_type: int, target: int, evasion: int) -> bool:
        """Simulate attack execution"""
        # Base success probability
        success_prob = 0.3

        # Adjust based on service vulnerability
        if self.network_state['services'][target] > 0.7:
            success_prob += 0.3

        # Adjust based on evasion technique
        evasion_bonus = evasion * 0.02
        success_prob += evasion_bonus

        # Check IDS detection
        detection_prob = np.mean(self.network_state['ids']) * (1 - evasion * 0.05)
        detected = np.random.random() < detection_prob

        if detected:
            self.network_state['detected_attacks'] += 1
            success_prob *= 0.3  # Reduced chance if detected

        # Execute attack
        success = np.random.random() < success_prob

        if success:
            self.network_state['compromised_services'].add(target)
            self.network_state['successful_attacks'] += 1
            # Reduce service security after compromise
            self.network_state['services'][target] *= 0.5

        return success

    def _calculate_reward(self, attack_success: bool) -> float:
        """Calculate reward for the agent"""
        reward = 0.0

        # Positive reward for successful attack
        if attack_success:
            reward += 10.0

        # Penalty for detection
        if self.network_state['detected_attacks'] > self.current_step * 0.5:
            reward -= 5.0

        # Bonus for staying stealthy
        if attack_success and self.network_state['detected_attacks'] == 0:
            reward += 5.0

        # Progressive reward for compromising more services
        reward += len(self.network_state['compromised_services']) * 0.5

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        return np.concatenate([
            self.network_state['network'],
            self.network_state['services'],
            self.network_state['ids'],
            self.network_state['firewall']
        ]).astype(np.float32)

class RLEvolutionEngine:
    """
    Multi-Agent RL Engine for Red Team vs Blue Team
    Uses Stable-Baselines3 (FREE) for PPO implementation
    """

    def __init__(self):
        self.red_agent = None
        self.blue_agent = None
        self.env = None
        self.training_stats = {
            'red_wins': 0,
            'blue_wins': 0,
            'draws': 0,
            'total_battles': 0
        }

        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            self.PPO = PPO
            self.DummyVecEnv = DummyVecEnv
            logger.info("✅ Stable-Baselines3 initialized")
        except ImportError:
            logger.warning("⚠️ Stable-Baselines3 not installed. Install with: pip install stable-baselines3")
            self.PPO = None

    def initialize_agents(self, config: AgentConfig = None):
        """Initialize Red and Blue team agents"""
        if not self.PPO:
            raise RuntimeError("Stable-Baselines3 not available")

        config = config or AgentConfig()

        # Create environment
        self.env = self.DummyVecEnv([lambda: CyberAttackEnv()])

        # Initialize Red Team Agent (Attacker)
        self.red_agent = self.PPO(
            "MlpPolicy",
            self.env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
            verbose=1
        )

        # Initialize Blue Team Agent (Defender) - will train separately
        self.blue_agent = self.PPO(
            "MlpPolicy",
            self.env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            verbose=1
        )

        logger.info("✅ Red and Blue agents initialized")

    async def train_red_team(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """Train Red Team agent"""
        if not self.red_agent:
            raise RuntimeError("Agents not initialized. Call initialize_agents() first.")

        logger.info(f"🔴 Training Red Team for {total_timesteps} timesteps...")

        self.red_agent.learn(total_timesteps=total_timesteps)

        # Save model
        model_path = "./models/red_team_ppo"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.red_agent.save(model_path)

        logger.info(f"✅ Red Team training complete. Model saved to {model_path}")

        return {
            "status": "completed",
            "timesteps": total_timesteps,
            "model_path": model_path
        }

    async def train_blue_team(self, total_timesteps: int = 100000) -> Dict[str, Any]:
        """Train Blue Team agent"""
        if not self.blue_agent:
            raise RuntimeError("Agents not initialized. Call initialize_agents() first.")

        logger.info(f"🔵 Training Blue Team for {total_timesteps} timesteps...")

        self.blue_agent.learn(total_timesteps=total_timesteps)

        # Save model
        model_path = "./models/blue_team_ppo"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.blue_agent.save(model_path)

        logger.info(f"✅ Blue Team training complete. Model saved to {model_path}")

        return {
            "status": "completed",
            "timesteps": total_timesteps,
            "model_path": model_path
        }

    async def self_play_training(self, num_episodes: int = 100) -> Dict[str, Any]:
        """
        Self-play training: Red vs Blue agents learn from each other
        """
        logger.info(f"🔄 Starting self-play training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            # Train red team for a while
            await self.train_red_team(total_timesteps=10000)

            # Train blue team against improved red team
            await self.train_blue_team(total_timesteps=10000)

            # Evaluate current agents
            eval_results = await self.evaluate_agents()

            logger.info(f"Episode {episode + 1}/{num_episodes}: {eval_results}")

            # Check for Nash equilibrium (both agents improving slowly)
            if eval_results['red_win_rate'] > 0.45 and eval_results['red_win_rate'] < 0.55:
                logger.info("✅ Nash equilibrium detected!")
                break

        return {
            "episodes_completed": num_episodes,
            "final_stats": self.training_stats
        }

    async def evaluate_agents(self, num_battles: int = 10) -> Dict[str, float]:
        """Evaluate current agent performance"""
        wins = 0

        for _ in range(num_battles):
            obs = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = self.red_agent.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward

            if total_reward > 0:
                wins += 1

        win_rate = wins / num_battles

        return {
            "red_win_rate": win_rate,
            "blue_win_rate": 1 - win_rate,
            "avg_reward": total_reward / num_battles
        }

    def load_agents(self, red_model_path: str, blue_model_path: str):
        """Load pre-trained agents"""
        if not self.PPO:
            raise RuntimeError("Stable-Baselines3 not available")

        self.red_agent = self.PPO.load(red_model_path)
        self.blue_agent = self.PPO.load(blue_model_path)

        logger.info("✅ Agents loaded successfully")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return self.training_stats
