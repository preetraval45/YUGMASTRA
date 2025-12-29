"""
Cyber Range Environment - Gymnasium-compatible RL environment

This implements a realistic cyber attack/defense simulation environment.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional
import docker
import logging

logger = logging.getLogger(__name__)


class CyberRangeEnv(gym.Env):
    """
    Gymnasium environment for cybersecurity training

    Observation Space:
        - Network traffic features
        - System logs
        - Service states
        - Security events

    Action Space (for Red agent):
        - 0: Port scan
        - 1: Vulnerability scan
        - 2: Exploit web service
        - 3: Exploit database
        - 4: Privilege escalation
        - 5: Lateral movement
        - 6: Data exfiltration
        - 7: No action

    Action Space (for Blue agent):
        - 0: Update firewall
        - 1: Block IP
        - 2: Isolate host
        - 3: Update IDS rules
        - 4: Patch service
        - 5: Monitor traffic
        - 6: No action
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        difficulty: float = 0.5,
        max_steps: int = 1000,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.difficulty = difficulty
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Docker client for managing containers
        self.docker_client = docker.from_env()

        # Define observation space
        # [network_features(50), log_features(30), service_states(10), security_events(10)]
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(100,),
            dtype=np.float32
        )

        # Define action spaces
        self.red_action_space = spaces.Discrete(8)
        self.blue_action_space = spaces.Discrete(7)

        # Environment state
        self.current_step = 0
        self.network_state = None
        self.compromised_hosts = set()
        self.detected_attacks = []
        self.firewall_rules = []

        # Cyber range configuration
        self.hosts = {
            'web_server': {'ip': '10.0.1.10', 'services': ['http', 'https'], 'compromised': False},
            'db_server': {'ip': '10.0.1.11', 'services': ['mysql'], 'compromised': False},
            'file_server': {'ip': '10.0.1.12', 'services': ['smb', 'ftp'], 'compromised': False},
            'endpoint_1': {'ip': '10.0.2.10', 'services': ['rdp'], 'compromised': False},
            'endpoint_2': {'ip': '10.0.2.11', 'services': ['rdp'], 'compromised': False},
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        self.current_step = 0
        self.compromised_hosts = set()
        self.detected_attacks = []
        self.firewall_rules = []

        # Reset all hosts
        for host_name, host_info in self.hosts.items():
            host_info['compromised'] = False

        # Initialize network state
        self.network_state = self._generate_initial_network_state()

        # Get initial observation
        observation = self._get_observation()

        info = {
            'difficulty': self.difficulty,
            'compromised_hosts': len(self.compromised_hosts),
            'total_hosts': len(self.hosts)
        }

        return observation, info

    def step(
        self,
        actions: Dict[str, int]
    ) -> Tuple[np.ndarray, Dict[str, float], bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment

        Args:
            actions: Dict with 'red_action' and 'blue_action'

        Returns:
            observation, rewards, terminated, truncated, info
        """
        self.current_step += 1

        red_action = actions.get('red_action', 7)  # 7 = no action
        blue_action = actions.get('blue_action', 6)  # 6 = no action

        # Execute red team action
        red_result = self._execute_red_action(red_action)

        # Execute blue team action
        blue_result = self._execute_blue_action(blue_action)

        # Update network state
        self._update_network_state(red_result, blue_result)

        # Calculate rewards
        rewards = self._calculate_rewards(red_result, blue_result)

        # Get new observation
        observation = self._get_observation()

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps

        # Info
        info = {
            'red_action': red_action,
            'blue_action': blue_action,
            'red_success': red_result['success'],
            'blue_detection': blue_result['detected'],
            'compromised_hosts': len(self.compromised_hosts),
            'detection_rate': len(self.detected_attacks) / (self.current_step + 1)
        }

        return observation, rewards, terminated, truncated, info

    def _execute_red_action(self, action: int) -> Dict[str, Any]:
        """Execute red team attack action"""
        result = {
            'success': False,
            'detected': False,
            'impact': 0.0,
            'action_type': ''
        }

        # Determine detection probability based on action and blue defenses
        detection_prob = self._calculate_detection_probability(action)

        if action == 0:  # Port scan
            result['action_type'] = 'port_scan'
            result['success'] = True
            result['detected'] = np.random.random() < detection_prob
            result['impact'] = 0.1

        elif action == 1:  # Vulnerability scan
            result['action_type'] = 'vuln_scan'
            result['success'] = True
            result['detected'] = np.random.random() < detection_prob
            result['impact'] = 0.2

        elif action == 2:  # Exploit web service
            result['action_type'] = 'exploit_web'
            if not self.hosts['web_server']['compromised']:
                success_prob = 0.7 * (1 + self.difficulty)
                result['success'] = np.random.random() < success_prob
                if result['success']:
                    self.hosts['web_server']['compromised'] = True
                    self.compromised_hosts.add('web_server')
                    result['impact'] = 1.0
            result['detected'] = np.random.random() < detection_prob

        elif action == 3:  # Exploit database
            result['action_type'] = 'exploit_db'
            if not self.hosts['db_server']['compromised']:
                success_prob = 0.6 * (1 + self.difficulty)
                result['success'] = np.random.random() < success_prob
                if result['success']:
                    self.hosts['db_server']['compromised'] = True
                    self.compromised_hosts.add('db_server')
                    result['impact'] = 1.2
            result['detected'] = np.random.random() < detection_prob

        elif action == 4:  # Privilege escalation
            result['action_type'] = 'privesc'
            if len(self.compromised_hosts) > 0:
                success_prob = 0.5
                result['success'] = np.random.random() < success_prob
                result['impact'] = 0.8
            result['detected'] = np.random.random() < detection_prob

        elif action == 5:  # Lateral movement
            result['action_type'] = 'lateral_movement'
            if len(self.compromised_hosts) > 0:
                # Try to compromise adjacent host
                available_hosts = [
                    h for h in self.hosts.keys()
                    if not self.hosts[h]['compromised']
                ]
                if available_hosts:
                    target = np.random.choice(available_hosts)
                    success_prob = 0.4
                    result['success'] = np.random.random() < success_prob
                    if result['success']:
                        self.hosts[target]['compromised'] = True
                        self.compromised_hosts.add(target)
                        result['impact'] = 1.0
            result['detected'] = np.random.random() < detection_prob

        elif action == 6:  # Data exfiltration
            result['action_type'] = 'exfiltration'
            if len(self.compromised_hosts) > 0:
                result['success'] = True
                result['impact'] = 1.5
                result['detected'] = np.random.random() < (detection_prob * 1.5)

        # Track detected attacks
        if result['detected']:
            self.detected_attacks.append({
                'step': self.current_step,
                'action': action,
                'type': result['action_type']
            })

        return result

    def _execute_blue_action(self, action: int) -> Dict[str, Any]:
        """Execute blue team defense action"""
        result = {
            'success': False,
            'detected': False,
            'effectiveness': 0.0
        }

        if action == 0:  # Update firewall
            result['success'] = True
            self.firewall_rules.append({
                'step': self.current_step,
                'type': 'firewall_update'
            })
            result['effectiveness'] = 0.3

        elif action == 1:  # Block IP
            result['success'] = True
            result['effectiveness'] = 0.5

        elif action == 2:  # Isolate host
            # Isolate compromised host if detected
            if len(self.detected_attacks) > 0:
                result['success'] = True
                result['effectiveness'] = 0.8

        elif action == 3:  # Update IDS rules
            result['success'] = True
            result['effectiveness'] = 0.4

        elif action == 4:  # Patch service
            result['success'] = True
            result['effectiveness'] = 0.6

        elif action == 5:  # Monitor traffic
            # Increase detection rate temporarily
            result['detected'] = len(self.detected_attacks) > 0
            result['effectiveness'] = 0.2

        return result

    def _calculate_detection_probability(self, action: int) -> float:
        """Calculate probability of detecting red action"""
        base_prob = 0.3

        # Increase based on blue defenses
        defense_bonus = len(self.firewall_rules) * 0.05

        # Different actions have different detectability
        action_multipliers = {
            0: 0.8,  # Port scan - easier to detect
            1: 0.9,  # Vuln scan
            2: 1.0,  # Exploit web
            3: 1.0,  # Exploit DB
            4: 0.7,  # Privesc
            5: 0.6,  # Lateral movement
            6: 1.2,  # Exfiltration - easier to detect
        }

        multiplier = action_multipliers.get(action, 1.0)
        final_prob = min((base_prob + defense_bonus) * multiplier, 0.95)

        return final_prob

    def _update_network_state(
        self,
        red_result: Dict[str, Any],
        blue_result: Dict[str, Any]
    ):
        """Update internal network state"""
        # Simulate network traffic based on actions
        # This would integrate with actual Docker containers in full implementation
        pass

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector"""
        observation = np.zeros(100, dtype=np.float32)

        # Network traffic features (0-49)
        observation[0] = len(self.compromised_hosts) / len(self.hosts)
        observation[1] = len(self.detected_attacks) / max(self.current_step, 1)
        observation[2:12] = np.random.random(10) * 0.1  # Simulated traffic features

        # Log features (50-79)
        observation[50] = len(self.firewall_rules) / 100
        observation[51:80] = np.random.random(29) * 0.1  # Simulated log features

        # Service states (80-89)
        for i, (host_name, host_info) in enumerate(self.hosts.items()):
            if i < 10:
                observation[80 + i] = 1.0 if host_info['compromised'] else 0.0

        # Security events (90-99)
        recent_detections = len([d for d in self.detected_attacks
                                if d['step'] > self.current_step - 10])
        observation[90] = min(recent_detections / 10, 1.0)

        return observation

    def _generate_initial_network_state(self) -> Dict[str, Any]:
        """Generate initial network state"""
        return {
            'traffic': [],
            'logs': [],
            'connections': []
        }

    def _calculate_rewards(
        self,
        red_result: Dict[str, Any],
        blue_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate rewards for both agents"""

        # Red team reward
        red_reward = 0.0
        if red_result['success']:
            red_reward += red_result['impact']
        if red_result['detected']:
            red_reward -= 0.5  # Penalty for being detected

        # Blue team reward
        blue_reward = 0.0
        if blue_result['detected']:
            blue_reward += 1.0  # Reward for detection
        if blue_result['success']:
            blue_reward += blue_result['effectiveness']

        # Penalty for compromised hosts
        blue_reward -= len(self.compromised_hosts) * 0.1

        return {
            'red': red_reward,
            'blue': blue_reward
        }

    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if all hosts compromised
        if len(self.compromised_hosts) == len(self.hosts):
            return True

        # Terminate if all attacks detected and blocked
        if (len(self.detected_attacks) > 10 and
            len(self.firewall_rules) > 5):
            return False

        return False

    def render(self):
        """Render environment state"""
        if self.render_mode == "human":
            print(f"\n=== Step {self.current_step} ===")
            print(f"Compromised hosts: {self.compromised_hosts}")
            print(f"Detected attacks: {len(self.detected_attacks)}")
            print(f"Firewall rules: {len(self.firewall_rules)}")

    def close(self):
        """Clean up resources"""
        # Stop Docker containers if needed
        pass
