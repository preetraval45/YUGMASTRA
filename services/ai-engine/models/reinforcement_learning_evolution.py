"""
Reinforcement Learning for Attack Evolution
Self-learning red team agents that improve from defense feedback
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque
import random


class AttackAction(Enum):
    """Possible attack actions"""
    RECONNAISSANCE = "reconnaissance"
    SCAN_PORTS = "scan_ports"
    EXPLOIT_VULNERABILITY = "exploit_vulnerability"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    DATA_EXFILTRATION = "data_exfiltration"
    ESTABLISH_PERSISTENCE = "establish_persistence"
    COVER_TRACKS = "cover_tracks"


@dataclass
class AttackState:
    """Current state of the attack"""
    compromised_systems: List[str] = field(default_factory=list)
    discovered_vulnerabilities: List[str] = field(default_factory=list)
    current_privileges: str = "user"
    network_map: Dict[str, Any] = field(default_factory=dict)
    defense_alerts_triggered: int = 0
    time_elapsed: int = 0
    data_exfiltrated: int = 0
    stealth_score: float = 1.0


@dataclass
class Experience:
    """Experience tuple for replay buffer"""
    state: AttackState
    action: AttackAction
    reward: float
    next_state: AttackState
    done: bool


class ReinforcementLearningAgent:
    """
    RL Agent that learns optimal attack strategies through trial and error
    Uses Deep Q-Learning with experience replay
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        memory_size: int = 10000
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: state -> action -> Q-value
        self.q_table: Dict[str, Dict[AttackAction, float]] = {}

        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)

        # Statistics
        self.episodes_completed = 0
        self.total_rewards = []
        self.success_rate = 0.0

    def get_state_hash(self, state: AttackState) -> str:
        """Convert state to hashable string"""
        return json.dumps({
            "systems": len(state.compromised_systems),
            "vulns": len(state.discovered_vulnerabilities),
            "privs": state.current_privileges,
            "alerts": state.defense_alerts_triggered,
            "stealth": round(state.stealth_score, 1)
        }, sort_keys=True)

    def get_q_value(self, state: AttackState, action: AttackAction) -> float:
        """Get Q-value for state-action pair"""
        state_hash = self.get_state_hash(state)
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {a: 0.0 for a in AttackAction}
        return self.q_table[state_hash][action]

    def set_q_value(self, state: AttackState, action: AttackAction, value: float):
        """Set Q-value for state-action pair"""
        state_hash = self.get_state_hash(state)
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {a: 0.0 for a in AttackAction}
        self.q_table[state_hash][action] = value

    def select_action(self, state: AttackState) -> AttackAction:
        """
        Select action using epsilon-greedy policy
        """
        # Exploration: random action
        if np.random.random() < self.epsilon:
            return random.choice(list(AttackAction))

        # Exploitation: best known action
        q_values = {action: self.get_q_value(state, action) for action in AttackAction}
        return max(q_values, key=q_values.get)

    def remember(self, experience: Experience):
        """Store experience in replay buffer"""
        self.memory.append(experience)

    def replay_and_learn(self, batch_size: int = 32):
        """
        Learn from past experiences using experience replay
        """
        if len(self.memory) < batch_size:
            return

        # Sample random batch
        batch = random.sample(self.memory, batch_size)

        for exp in batch:
            # Calculate target Q-value
            if exp.done:
                target = exp.reward
            else:
                # Q-learning update: Q(s,a) = r + γ * max(Q(s',a'))
                next_q_values = [
                    self.get_q_value(exp.next_state, a)
                    for a in AttackAction
                ]
                target = exp.reward + self.gamma * max(next_q_values)

            # Update Q-value
            current_q = self.get_q_value(exp.state, exp.action)
            new_q = current_q + self.learning_rate * (target - current_q)
            self.set_q_value(exp.state, exp.action, new_q)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def calculate_reward(
        self,
        state: AttackState,
        action: AttackAction,
        next_state: AttackState,
        success: bool
    ) -> float:
        """
        Calculate reward for an action
        Rewards successful actions, penalizes detection
        """
        reward = 0.0

        # Base reward for action success
        if success:
            reward += 10.0

            # Bonus rewards
            if action == AttackAction.EXPLOIT_VULNERABILITY:
                reward += 20.0
            elif action == AttackAction.PRIVILEGE_ESCALATION:
                reward += 30.0
            elif action == AttackAction.DATA_EXFILTRATION:
                reward += 50.0

        # Penalty for triggering alerts
        alerts_triggered = next_state.defense_alerts_triggered - state.defense_alerts_triggered
        reward -= alerts_triggered * 15.0

        # Stealth bonus
        stealth_change = next_state.stealth_score - state.stealth_score
        reward += stealth_change * 25.0

        # Time penalty (encourage efficiency)
        reward -= 0.5

        # Massive penalty for getting caught
        if next_state.stealth_score < 0.3:
            reward -= 100.0

        return reward

    def train_episode(self, environment) -> float:
        """
        Train agent for one complete attack episode
        """
        state = environment.reset()
        total_reward = 0.0
        done = False
        steps = 0
        max_steps = 50

        while not done and steps < max_steps:
            # Select and execute action
            action = self.select_action(state)
            next_state, success = environment.execute_action(action)

            # Calculate reward
            reward = self.calculate_reward(state, action, next_state, success)
            total_reward += reward

            # Check if episode is complete
            done = environment.is_terminal_state(next_state)

            # Store experience
            experience = Experience(state, action, reward, next_state, done)
            self.remember(experience)

            # Learn from experiences
            self.replay_and_learn()

            state = next_state
            steps += 1

        self.episodes_completed += 1
        self.total_rewards.append(total_reward)

        # Update success rate
        recent_rewards = self.total_rewards[-100:]
        self.success_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)

        return total_reward

    def get_best_strategy(self, initial_state: AttackState) -> List[AttackAction]:
        """
        Generate best attack strategy based on learned Q-values
        """
        strategy = []
        state = initial_state
        max_steps = 20

        for _ in range(max_steps):
            # Always exploit (no exploration)
            q_values = {action: self.get_q_value(state, action) for action in AttackAction}
            best_action = max(q_values, key=q_values.get)

            if q_values[best_action] <= 0:
                break  # No good actions found

            strategy.append(best_action)

            # Simulate state transition (simplified)
            # In practice, use actual environment
            break

        return strategy

    def save_model(self, filepath: str):
        """Save learned Q-table to file"""
        model_data = {
            "q_table": {k: {a.value: v for a, v in actions.items()} for k, actions in self.q_table.items()},
            "episodes": self.episodes_completed,
            "success_rate": self.success_rate,
            "epsilon": self.epsilon
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, filepath: str):
        """Load learned Q-table from file"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        self.q_table = {
            k: {AttackAction(a): v for a, v in actions.items()}
            for k, actions in model_data["q_table"].items()
        }
        self.episodes_completed = model_data["episodes"]
        self.success_rate = model_data["success_rate"]
        self.epsilon = model_data["epsilon"]


class GeneticAlgorithmEvolution:
    """
    Genetic Algorithm for attack mutation and optimization
    Evolves attack strategies over generations
    """

    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0

    def create_individual(self) -> List[AttackAction]:
        """Create random attack strategy"""
        length = random.randint(5, 15)
        return [random.choice(list(AttackAction)) for _ in range(length)]

    def initialize_population(self) -> List[List[AttackAction]]:
        """Create initial population of attack strategies"""
        return [self.create_individual() for _ in range(self.population_size)]

    def fitness(self, strategy: List[AttackAction], environment) -> float:
        """
        Evaluate fitness of attack strategy
        Higher fitness = more successful attack
        """
        state = environment.reset()
        score = 0.0

        for action in strategy:
            next_state, success = environment.execute_action(action)
            if success:
                score += 10.0
            score -= next_state.defense_alerts_triggered * 5.0
            state = next_state

        # Bonus for data exfiltration
        score += state.data_exfiltrated * 20.0

        return score

    def selection(self, population: List[List[AttackAction]], fitnesses: List[float]) -> List[AttackAction]:
        """Tournament selection"""
        tournament_size = 5
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        return max(tournament, key=lambda x: x[1])[0]

    def crossover(self, parent1: List[AttackAction], parent2: List[AttackAction]) -> Tuple[List[AttackAction], List[AttackAction]]:
        """Single-point crossover"""
        if len(parent1) < 2 or len(parent2) < 2:
            return parent1.copy(), parent2.copy()

        point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        return child1, child2

    def mutate(self, strategy: List[AttackAction]) -> List[AttackAction]:
        """Mutate attack strategy"""
        mutated = strategy.copy()

        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = random.choice(list(AttackAction))

        return mutated

    def evolve_generation(self, population: List[List[AttackAction]], environment) -> List[List[AttackAction]]:
        """Evolve population for one generation"""
        # Evaluate fitness
        fitnesses = [self.fitness(strategy, environment) for strategy in population]

        # Create new generation
        new_population = []

        # Elitism: keep best 10%
        elite_count = self.population_size // 10
        elite_indices = np.argsort(fitnesses)[-elite_count:]
        new_population.extend([population[i] for i in elite_indices])

        # Generate offspring
        while len(new_population) < self.population_size:
            parent1 = self.selection(population, fitnesses)
            parent2 = self.selection(population, fitnesses)

            child1, child2 = self.crossover(parent1, parent2)

            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            new_population.extend([child1, child2])

        self.generation += 1
        return new_population[:self.population_size]


# Example usage
if __name__ == "__main__":
    # Create RL agent
    agent = ReinforcementLearningAgent()

    print("🤖 Reinforcement Learning Attack Evolution Engine")
    print(f"Episodes: {agent.episodes_completed}")
    print(f"Success Rate: {agent.success_rate:.1%}")
    print(f"Exploration Rate: {agent.epsilon:.2f}")

    # Create GA evolution
    ga = GeneticAlgorithmEvolution()
    print(f"\n🧬 Genetic Algorithm Evolution")
    print(f"Population Size: {ga.population_size}")
    print(f"Mutation Rate: {ga.mutation_rate:.1%}")
