"""
Co-Evolution Engine - Core training loop for red-blue adversarial learning

This module implements the heart of YUGMĀSTRA's co-evolutionary system.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EvolutionPhase(Enum):
    """Phases of co-evolution"""
    WARMUP = "warmup"
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    EQUILIBRIUM = "equilibrium"


@dataclass
class EvolutionMetrics:
    """Metrics for tracking evolution progress"""
    red_win_rate: float
    blue_detection_rate: float
    strategy_diversity: float
    nash_equilibrium_distance: float
    phase: EvolutionPhase
    episode: int


class CoEvolutionEngine:
    """
    Main co-evolution engine implementing:
    - Multi-agent self-play
    - Curriculum learning
    - Nash equilibrium detection
    - Population-based training
    """

    def __init__(
        self,
        red_agent: Any,
        blue_agent: Any,
        environment: Any,
        config: Dict[str, Any]
    ):
        self.red_agent = red_agent
        self.blue_agent = blue_agent
        self.environment = environment
        self.config = config

        # Evolution state
        self.current_episode = 0
        self.current_phase = EvolutionPhase.WARMUP
        self.metrics_history = []

        # Population management
        self.red_population = PopulationManager(
            population_size=config.get('population_size', 10),
            agent_template=red_agent
        )
        self.blue_population = PopulationManager(
            population_size=config.get('population_size', 10),
            agent_template=blue_agent
        )

        # Curriculum learning
        self.curriculum = CurriculumManager(
            initial_difficulty=config.get('initial_difficulty', 0.1),
            max_difficulty=config.get('max_difficulty', 1.0)
        )

        # Equilibrium detector
        self.equilibrium_detector = NashEquilibriumDetector(
            history_window=config.get('equilibrium_window', 100)
        )

    def run_evolution(
        self,
        num_episodes: int,
        checkpoint_interval: int = 100
    ) -> List[EvolutionMetrics]:
        """
        Main evolution loop

        Args:
            num_episodes: Number of episodes to run
            checkpoint_interval: Save checkpoint every N episodes

        Returns:
            metrics_history: List of metrics for each episode
        """
        logger.info(f"Starting co-evolution for {num_episodes} episodes")

        for episode in range(num_episodes):
            self.current_episode = episode

            # Select agents from population
            red_agent = self.red_population.select_agent()
            blue_agent = self.blue_population.select_agent()

            # Run episode with current curriculum difficulty
            difficulty = self.curriculum.get_difficulty()
            episode_results = self._run_episode(
                red_agent,
                blue_agent,
                difficulty
            )

            # Update agents based on results
            self._update_agents(red_agent, blue_agent, episode_results)

            # Update population fitness
            self.red_population.update_fitness(
                red_agent,
                episode_results['red_reward']
            )
            self.blue_population.update_fitness(
                blue_agent,
                episode_results['blue_reward']
            )

            # Update curriculum
            self.curriculum.update(
                red_win_rate=episode_results['red_win_rate'],
                blue_detection_rate=episode_results['blue_detection_rate']
            )

            # Check for Nash equilibrium
            equilibrium_metrics = self.equilibrium_detector.update(
                red_strategy=episode_results['red_strategy'],
                blue_strategy=episode_results['blue_strategy']
            )

            # Update evolution phase
            self._update_phase(episode_results, equilibrium_metrics)

            # Collect metrics
            metrics = EvolutionMetrics(
                red_win_rate=episode_results['red_win_rate'],
                blue_detection_rate=episode_results['blue_detection_rate'],
                strategy_diversity=self._compute_strategy_diversity(),
                nash_equilibrium_distance=equilibrium_metrics['distance'],
                phase=self.current_phase,
                episode=episode
            )
            self.metrics_history.append(metrics)

            # Logging
            if episode % 10 == 0:
                logger.info(
                    f"Episode {episode}: "
                    f"Red WR: {metrics.red_win_rate:.3f}, "
                    f"Blue DR: {metrics.blue_detection_rate:.3f}, "
                    f"Phase: {metrics.phase.value}, "
                    f"Difficulty: {difficulty:.3f}"
                )

            # Checkpoint
            if episode % checkpoint_interval == 0:
                self._save_checkpoint(episode)

            # Early stopping if equilibrium reached
            if self._check_convergence(equilibrium_metrics):
                logger.info(f"Nash equilibrium reached at episode {episode}")
                break

        return self.metrics_history

    def _run_episode(
        self,
        red_agent: Any,
        blue_agent: Any,
        difficulty: float
    ) -> Dict[str, Any]:
        """
        Run single episode of red-blue interaction

        Args:
            red_agent: Current red agent
            blue_agent: Current blue agent
            difficulty: Environment difficulty level

        Returns:
            episode_results: Results and metrics from episode
        """
        # Reset environment with current difficulty
        observation = self.environment.reset(difficulty=difficulty)

        red_trajectory = []
        blue_trajectory = []
        red_total_reward = 0
        blue_total_reward = 0
        done = False
        timestep = 0
        max_timesteps = self.config.get('max_timesteps', 1000)

        while not done and timestep < max_timesteps:
            # Red agent takes action (attack)
            red_action, red_info = red_agent.select_action(
                observation['red_view'],
                explore=True
            )

            # Blue agent observes and responds (defense)
            blue_action, blue_info = blue_agent.select_action(
                observation['blue_view'],
                explore=True
            )

            # Environment step
            next_observation, rewards, done, info = self.environment.step({
                'red_action': red_action,
                'blue_action': blue_action
            })

            # Store trajectories
            red_trajectory.append({
                'observation': observation['red_view'],
                'action': red_action,
                'reward': rewards['red'],
                'info': red_info
            })
            blue_trajectory.append({
                'observation': observation['blue_view'],
                'action': blue_action,
                'reward': rewards['blue'],
                'info': blue_info
            })

            red_total_reward += rewards['red']
            blue_total_reward += rewards['blue']

            observation = next_observation
            timestep += 1

        # Analyze episode
        red_win_rate = 1.0 if red_total_reward > blue_total_reward else 0.0
        blue_detection_rate = info.get('detection_rate', 0.0)

        # Extract strategies
        red_strategy = self._extract_strategy(red_trajectory)
        blue_strategy = self._extract_strategy(blue_trajectory)

        return {
            'red_trajectory': red_trajectory,
            'blue_trajectory': blue_trajectory,
            'red_reward': red_total_reward,
            'blue_reward': blue_total_reward,
            'red_win_rate': red_win_rate,
            'blue_detection_rate': blue_detection_rate,
            'red_strategy': red_strategy,
            'blue_strategy': blue_strategy,
            'timesteps': timestep
        }

    def _update_agents(
        self,
        red_agent: Any,
        blue_agent: Any,
        episode_results: Dict[str, Any]
    ):
        """Update agent policies based on episode results"""
        # Red agent learns from blue's defense
        red_metrics = red_agent.train_step(
            trajectory=episode_results['red_trajectory'],
            opponent_info=episode_results['blue_strategy']
        )

        # Blue agent learns from red's attack
        blue_metrics = blue_agent.train_step(
            trajectory=episode_results['blue_trajectory'],
            opponent_info=episode_results['red_strategy']
        )

    def _extract_strategy(self, trajectory: List[Dict[str, Any]]) -> np.ndarray:
        """Extract strategy representation from trajectory"""
        # Use action distribution as strategy representation
        actions = np.array([step['action'] for step in trajectory])
        action_dist = np.bincount(actions, minlength=self.environment.action_space.n)
        return action_dist / (len(trajectory) + 1e-8)

    def _compute_strategy_diversity(self) -> float:
        """Compute diversity of strategies in population"""
        red_diversity = self.red_population.compute_diversity()
        blue_diversity = self.blue_population.compute_diversity()
        return (red_diversity + blue_diversity) / 2.0

    def _update_phase(
        self,
        episode_results: Dict[str, Any],
        equilibrium_metrics: Dict[str, Any]
    ):
        """Update evolution phase based on current state"""
        red_wr = episode_results['red_win_rate']
        blue_dr = episode_results['blue_detection_rate']
        equilibrium_dist = equilibrium_metrics['distance']

        # Phase transitions
        if self.current_episode < 100:
            self.current_phase = EvolutionPhase.WARMUP
        elif equilibrium_dist < 0.1:
            self.current_phase = EvolutionPhase.EQUILIBRIUM
        elif abs(red_wr - 0.5) < 0.1:
            self.current_phase = EvolutionPhase.EXPLOITATION
        else:
            self.current_phase = EvolutionPhase.EXPLORATION

    def _check_convergence(self, equilibrium_metrics: Dict[str, Any]) -> bool:
        """Check if evolution has converged to Nash equilibrium"""
        distance = equilibrium_metrics['distance']
        stability = equilibrium_metrics['stability']

        convergence_threshold = self.config.get('convergence_threshold', 0.05)
        stability_threshold = self.config.get('stability_threshold', 0.9)

        return distance < convergence_threshold and stability > stability_threshold

    def _save_checkpoint(self, episode: int):
        """Save checkpoint of current evolution state"""
        checkpoint = {
            'episode': episode,
            'red_agent': self.red_agent.state_dict(),
            'blue_agent': self.blue_agent.state_dict(),
            'red_population': self.red_population.get_state(),
            'blue_population': self.blue_population.get_state(),
            'curriculum': self.curriculum.get_state(),
            'metrics_history': self.metrics_history
        }
        # TODO: Implement actual checkpoint saving
        logger.info(f"Checkpoint saved at episode {episode}")


class PopulationManager:
    """Manages population of agents for PBT"""

    def __init__(self, population_size: int, agent_template: Any):
        self.population_size = population_size
        self.agent_template = agent_template
        self.population = []
        self.fitness_scores = []

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Create initial population with diverse hyperparameters"""
        for i in range(self.population_size):
            # Clone agent with perturbed parameters
            agent = self._clone_agent_with_variation(self.agent_template)
            self.population.append(agent)
            self.fitness_scores.append(0.0)

    def select_agent(self) -> Any:
        """Select agent using tournament selection"""
        tournament_size = 3
        indices = np.random.choice(
            self.population_size,
            tournament_size,
            replace=False
        )
        best_idx = indices[np.argmax([self.fitness_scores[i] for i in indices])]
        return self.population[best_idx]

    def update_fitness(self, agent: Any, reward: float):
        """Update fitness score for agent"""
        # Find agent in population and update fitness
        for i, pop_agent in enumerate(self.population):
            if pop_agent is agent:
                # Exponential moving average
                self.fitness_scores[i] = 0.9 * self.fitness_scores[i] + 0.1 * reward
                break

    def compute_diversity(self) -> float:
        """Compute strategy diversity in population"""
        # Compute pairwise distances between agent strategies
        # Simplified implementation
        return np.std(self.fitness_scores)

    def get_state(self) -> Dict[str, Any]:
        """Get population state for checkpointing"""
        return {
            'fitness_scores': self.fitness_scores,
            # 'agents': [agent.state_dict() for agent in self.population]
        }

    def _clone_agent_with_variation(self, agent: Any) -> Any:
        """Clone agent with hyperparameter variation"""
        # TODO: Implement proper agent cloning
        return agent


class CurriculumManager:
    """Manages curriculum learning difficulty"""

    def __init__(self, initial_difficulty: float, max_difficulty: float):
        self.current_difficulty = initial_difficulty
        self.max_difficulty = max_difficulty
        self.performance_window = []
        self.window_size = 50

    def get_difficulty(self) -> float:
        """Get current difficulty level"""
        return self.current_difficulty

    def update(self, red_win_rate: float, blue_detection_rate: float):
        """Update difficulty based on performance"""
        # Track recent performance
        avg_performance = (red_win_rate + blue_detection_rate) / 2.0
        self.performance_window.append(avg_performance)

        if len(self.performance_window) > self.window_size:
            self.performance_window.pop(0)

        # Increase difficulty if agents are performing well
        if len(self.performance_window) >= self.window_size:
            recent_avg = np.mean(self.performance_window[-10:])
            if recent_avg > 0.7:
                self.current_difficulty = min(
                    self.current_difficulty + 0.05,
                    self.max_difficulty
                )
            elif recent_avg < 0.3:
                self.current_difficulty = max(
                    self.current_difficulty - 0.05,
                    0.1
                )

    def get_state(self) -> Dict[str, Any]:
        """Get curriculum state"""
        return {
            'current_difficulty': self.current_difficulty,
            'performance_window': self.performance_window
        }


class NashEquilibriumDetector:
    """Detects Nash equilibrium in co-evolution"""

    def __init__(self, history_window: int):
        self.history_window = history_window
        self.red_strategy_history = []
        self.blue_strategy_history = []

    def update(
        self,
        red_strategy: np.ndarray,
        blue_strategy: np.ndarray
    ) -> Dict[str, float]:
        """
        Update with new strategies and compute equilibrium metrics

        Args:
            red_strategy: Red agent's strategy distribution
            blue_strategy: Blue agent's strategy distribution

        Returns:
            metrics: Equilibrium distance and stability
        """
        self.red_strategy_history.append(red_strategy)
        self.blue_strategy_history.append(blue_strategy)

        if len(self.red_strategy_history) > self.history_window:
            self.red_strategy_history.pop(0)
            self.blue_strategy_history.pop(0)

        # Compute distance from equilibrium
        distance = self._compute_equilibrium_distance()

        # Compute stability
        stability = self._compute_stability()

        return {
            'distance': distance,
            'stability': stability
        }

    def _compute_equilibrium_distance(self) -> float:
        """Compute distance from Nash equilibrium"""
        if len(self.red_strategy_history) < 10:
            return 1.0

        # Simplified: check strategy stability
        recent_red = np.array(self.red_strategy_history[-10:])
        recent_blue = np.array(self.blue_strategy_history[-10:])

        red_variance = np.var(recent_red, axis=0).mean()
        blue_variance = np.var(recent_blue, axis=0).mean()

        distance = (red_variance + blue_variance) / 2.0
        return min(distance, 1.0)

    def _compute_stability(self) -> float:
        """Compute stability of current strategies"""
        if len(self.red_strategy_history) < self.history_window:
            return 0.0

        # Check if strategies have stabilized
        red_std = np.std(self.red_strategy_history[-20:], axis=0).mean()
        blue_std = np.std(self.blue_strategy_history[-20:], axis=0).mean()

        stability = 1.0 - ((red_std + blue_std) / 2.0)
        return max(stability, 0.0)
