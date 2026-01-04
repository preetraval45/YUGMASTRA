"""
Cyber Range Gymnasium Environment
OpenAI Gym-compatible environment for training RL agents in simulated enterprise network
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of network nodes"""
    WEB_SERVER = "web_server"
    DATABASE = "database"
    WORKSTATION = "workstation"
    FIREWALL = "firewall"
    ROUTER = "router"
    DMZ = "dmz"
    INTERNAL = "internal"


class VulnerabilityType(Enum):
    """Types of vulnerabilities"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    BUFFER_OVERFLOW = "buffer_overflow"
    WEAK_CREDENTIALS = "weak_credentials"
    UNPATCHED_SERVICE = "unpatched_service"
    MISCONFIGURATION = "misconfiguration"


@dataclass
class NetworkNode:
    """Represents a node in the network"""
    id: int
    node_type: NodeType
    ip_address: str
    os: str
    services: List[str]
    vulnerabilities: List[VulnerabilityType] = field(default_factory=list)
    compromised: bool = False
    privilege_level: str = "none"  # none, user, admin, root
    data_value: float = 0.0  # Value of data on this node
    defense_level: float = 0.5  # 0.0 (none) to 1.0 (maximum)


class AttackAction(Enum):
    """Possible attack actions"""
    RECONNAISSANCE = 0
    PORT_SCAN = 1
    VULNERABILITY_SCAN = 2
    EXPLOIT_WEB = 3
    EXPLOIT_SQL = 4
    EXPLOIT_BUFFER_OVERFLOW = 5
    BRUTE_FORCE = 6
    PRIVILEGE_ESCALATION = 7
    LATERAL_MOVEMENT = 8
    DATA_EXFILTRATION = 9
    ESTABLISH_PERSISTENCE = 10
    COVER_TRACKS = 11


class DefenseAction(Enum):
    """Possible defense actions"""
    MONITOR = 0
    PATCH_SYSTEM = 1
    UPDATE_FIREWALL = 2
    ISOLATE_NODE = 3
    SCAN_FOR_MALWARE = 4
    RESET_CREDENTIALS = 5
    BACKUP_DATA = 6
    DEPLOY_IDS = 7
    INCREASE_LOGGING = 8


class CyberRangeEnv(gym.Env):
    """
    Cyber Range Environment for training red and blue teams

    State space: Network topology, node statuses, detected threats
    Action space: Attack actions (red) or defense actions (blue)
    Reward: Based on successful attacks/defenses, stealth, efficiency
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(
        self,
        num_nodes: int = 10,
        max_steps: int = 100,
        render_mode: Optional[str] = None,
        agent_mode: str = "red"  # "red" or "blue"
    ):
        """
        Initialize Cyber Range environment

        Args:
            num_nodes: Number of nodes in the network
            max_steps: Maximum steps per episode
            render_mode: Rendering mode
            agent_mode: Whether this env is for red team or blue team
        """
        super().__init__()

        self.num_nodes = num_nodes
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.agent_mode = agent_mode

        # Initialize network
        self.nodes: List[NetworkNode] = []
        self._create_network()

        # State space: Flattened representation of network state
        # For each node: [compromised, privilege_level_encoded, defense_level, num_services, has_vuln]
        state_features_per_node = 5
        self.state_dim = num_nodes * state_features_per_node + 3  # +3 for global features

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Action space
        if agent_mode == "red":
            self.num_actions = len(AttackAction)
        else:
            self.num_actions = len(DefenseAction)

        self.action_space = spaces.Discrete(self.num_actions)

        # Episode state
        self.current_step = 0
        self.total_reward = 0
        self.attack_success_count = 0
        self.detected_attacks = 0
        self.stealth_score = 1.0

        logger.info(f"✅ Cyber Range Environment initialized")
        logger.info(f"   Nodes: {num_nodes}, Mode: {agent_mode}")
        logger.info(f"   State dim: {self.state_dim}, Actions: {self.num_actions}")

    def _create_network(self):
        """Create simulated enterprise network"""
        self.nodes = []

        # Create different types of nodes
        node_configs = [
            (NodeType.WEB_SERVER, "10.0.1.10", "Linux", ["HTTP", "HTTPS"], 0.9),
            (NodeType.WEB_SERVER, "10.0.1.11", "Linux", ["HTTP", "HTTPS"], 0.8),
            (NodeType.DATABASE, "10.0.2.10", "Linux", ["MySQL", "SSH"], 0.95),
            (NodeType.WORKSTATION, "10.0.3.10", "Windows", ["RDP", "SMB"], 0.5),
            (NodeType.WORKSTATION, "10.0.3.11", "Windows", ["RDP", "SMB"], 0.6),
            (NodeType.WORKSTATION, "10.0.3.12", "Mac", ["SSH", "VNC"], 0.7),
            (NodeType.FIREWALL, "10.0.0.1", "Linux", ["Firewall"], 0.95),
            (NodeType.ROUTER, "10.0.0.2", "Cisco", ["SNMP"], 0.8),
            (NodeType.DMZ, "10.0.1.1", "Linux", ["HTTP", "SMTP"], 0.7),
            (NodeType.INTERNAL, "10.0.4.10", "Windows Server", ["AD", "DNS"], 1.0),
        ]

        for i, (node_type, ip, os, services, data_value) in enumerate(node_configs[:self.num_nodes]):
            node = NetworkNode(
                id=i,
                node_type=node_type,
                ip_address=ip,
                os=os,
                services=services,
                data_value=data_value,
                defense_level=np.random.uniform(0.3, 0.8)
            )

            # Randomly assign vulnerabilities
            if np.random.random() < 0.6:  # 60% have vulns
                num_vulns = np.random.randint(1, 3)
                node.vulnerabilities = list(np.random.choice(
                    list(VulnerabilityType),
                    size=min(num_vulns, len(VulnerabilityType)),
                    replace=False
                ))

            self.nodes.append(node)

    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        state = []

        # Per-node features
        for node in self.nodes:
            # Compromised status
            state.append(1.0 if node.compromised else 0.0)

            # Privilege level (normalized)
            priv_encoding = {
                "none": 0.0,
                "user": 0.33,
                "admin": 0.66,
                "root": 1.0
            }
            state.append(priv_encoding[node.privilege_level])

            # Defense level
            state.append(node.defense_level)

            # Number of services (normalized)
            state.append(len(node.services) / 10.0)

            # Has vulnerabilities
            state.append(1.0 if node.vulnerabilities else 0.0)

        # Global features
        total_compromised = sum(1 for n in self.nodes if n.compromised)
        state.append(total_compromised / len(self.nodes))  # Compromise rate
        state.append(self.stealth_score)  # Stealth
        state.append(self.current_step / self.max_steps)  # Progress

        return np.array(state, dtype=np.float32)

    def _execute_attack_action(self, action: int) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Execute attack action

        Returns:
            reward: Reward for this action
            detected: Whether attack was detected
            info: Additional information
        """
        attack = AttackAction(action)
        reward = 0.0
        detected = False
        info = {"action": attack.name, "success": False}

        # Select random target node
        target_idx = np.random.randint(0, len(self.nodes))
        target = self.nodes[target_idx]

        # Calculate success probability based on defense level and attack type
        base_success_rate = 0.7
        success_prob = base_success_rate * (1.0 - target.defense_level * 0.5)

        # Detection probability
        detection_prob = target.defense_level * 0.3

        # Execute action
        if attack == AttackAction.RECONNAISSANCE:
            # Low risk, low reward
            if np.random.random() < 0.9:
                reward = 1.0
                info["success"] = True
                info["discovered_services"] = target.services
            detected = np.random.random() < 0.05

        elif attack == AttackAction.PORT_SCAN:
            if np.random.random() < 0.8:
                reward = 2.0
                info["success"] = True
                info["open_ports"] = len(target.services)
            detected = np.random.random() < 0.1

        elif attack == AttackAction.EXPLOIT_WEB:
            if VulnerabilityType.SQL_INJECTION in target.vulnerabilities or VulnerabilityType.XSS in target.vulnerabilities:
                if np.random.random() < success_prob:
                    target.compromised = True
                    target.privilege_level = "user"
                    reward = 20.0
                    self.attack_success_count += 1
                    info["success"] = True
            detected = np.random.random() < detection_prob

        elif attack == AttackAction.PRIVILEGE_ESCALATION:
            if target.compromised and target.privilege_level != "root":
                if np.random.random() < success_prob:
                    target.privilege_level = "root"
                    reward = 30.0
                    info["success"] = True
            detected = np.random.random() < detection_prob * 1.5

        elif attack == AttackAction.LATERAL_MOVEMENT:
            if target.compromised:
                # Try to compromise adjacent node
                next_target = self.nodes[(target_idx + 1) % len(self.nodes)]
                if not next_target.compromised:
                    if np.random.random() < success_prob:
                        next_target.compromised = True
                        next_target.privilege_level = "user"
                        reward = 25.0
                        info["success"] = True
            detected = np.random.random() < detection_prob

        elif attack == AttackAction.DATA_EXFILTRATION:
            if target.compromised:
                reward = target.data_value * 40.0
                info["success"] = True
                info["data_value"] = target.data_value
            detected = np.random.random() < detection_prob * 2.0

        elif attack == AttackAction.COVER_TRACKS:
            # Reduce detection probability for future actions
            self.stealth_score = min(self.stealth_score + 0.1, 1.0)
            reward = 5.0
            detected = np.random.random() < 0.02

        # Update stealth based on detection
        if detected:
            self.detected_attacks += 1
            self.stealth_score = max(self.stealth_score - 0.2, 0.0)
            reward -= 15.0  # Penalty for detection
            info["detected"] = True

        # Time penalty
        reward -= 0.5

        return reward, detected, info

    def _execute_defense_action(self, action: int) -> Tuple[float, Dict[str, Any]]:
        """
        Execute defense action

        Returns:
            reward: Reward for this action
            info: Additional information
        """
        defense = DefenseAction(action)
        reward = 0.0
        info = {"action": defense.name, "success": False}

        # Select random node to defend
        target_idx = np.random.randint(0, len(self.nodes))
        target = self.nodes[target_idx]

        if defense == DefenseAction.PATCH_SYSTEM:
            if target.vulnerabilities:
                target.vulnerabilities.clear()
                reward = 10.0
                info["success"] = True

        elif defense == DefenseAction.UPDATE_FIREWALL:
            # Increase defense for all nodes
            for node in self.nodes:
                node.defense_level = min(node.defense_level + 0.1, 1.0)
            reward = 15.0
            info["success"] = True

        elif defense == DefenseAction.ISOLATE_NODE:
            if target.compromised:
                # Prevent lateral movement from this node
                reward = 20.0
                info["success"] = True
                info["isolated_node"] = target.id

        elif defense == DefenseAction.SCAN_FOR_MALWARE:
            # Detect compromised nodes
            compromised_nodes = [n for n in self.nodes if n.compromised]
            if compromised_nodes:
                reward = 10.0 * len(compromised_nodes)
                info["success"] = True
                info["detected_compromises"] = len(compromised_nodes)

        elif defense == DefenseAction.RESET_CREDENTIALS:
            # Remove attacker access
            if target.compromised:
                target.privilege_level = "none"
                reward = 25.0
                info["success"] = True

        return reward, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment

        Returns:
            observation: New state
            reward: Reward for this step
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.current_step += 1

        # Execute action based on agent mode
        if self.agent_mode == "red":
            reward, detected, info = self._execute_attack_action(action)
        else:
            reward, info = self._execute_defense_action(action)
            detected = False

        self.total_reward += reward

        # Get new state
        observation = self._get_state()

        # Check termination conditions
        terminated = False
        truncated = False

        if self.agent_mode == "red":
            # Red team wins if they exfiltrate enough data or compromise critical nodes
            critical_compromised = sum(1 for n in self.nodes
                                      if n.compromised and n.node_type in [NodeType.DATABASE, NodeType.INTERNAL])
            if critical_compromised >= 1:
                terminated = True
                reward += 100.0  # Bonus for mission success
                info["termination_reason"] = "mission_success"

            # Red team loses if detected too much
            if self.detected_attacks >= 5:
                terminated = True
                reward -= 50.0
                info["termination_reason"] = "detected"

        # Truncated if max steps reached
        if self.current_step >= self.max_steps:
            truncated = True

        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        # Reset network
        self._create_network()

        # Reset episode state
        self.current_step = 0
        self.total_reward = 0
        self.attack_success_count = 0
        self.detected_attacks = 0
        self.stealth_score = 1.0

        observation = self._get_state()
        info = {"reset": True}

        return observation, info

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            print(f"\n=== Cyber Range - Step {self.current_step}/{self.max_steps} ===")
            print(f"Stealth Score: {self.stealth_score:.2f}")
            print(f"Total Reward: {self.total_reward:.2f}")
            print(f"\nNetwork Status:")

            compromised_count = 0
            for node in self.nodes:
                status = "COMPROMISED" if node.compromised else "SECURE"
                priv = f" ({node.privilege_level})" if node.compromised else ""
                print(f"  [{node.id}] {node.ip_address:<15} {node.node_type.value:<15} {status}{priv}")
                if node.compromised:
                    compromised_count += 1

            print(f"\nCompromised: {compromised_count}/{len(self.nodes)}")
            print(f"Detected Attacks: {self.detected_attacks}")


# Example usage
if __name__ == "__main__":
    # Create environment for red team
    env = CyberRangeEnv(num_nodes=10, max_steps=50, agent_mode="red", render_mode="human")

    print("🎮 Cyber Range Environment Test")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Test episode
    obs, info = env.reset()
    env.render()

    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(f"\nStep {step + 1}: Action {action} | Reward: {reward:.2f} | Info: {info}")

        if terminated or truncated:
            print(f"\nEpisode ended: {'Terminated' if terminated else 'Truncated'}")
            break

        if (step + 1) % 3 == 0:
            env.render()

    print(f"\nTotal Reward: {total_reward:.2f}")
    print("\n✅ Cyber Range Environment test passed!")
