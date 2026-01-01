"""
Live Attack Simulation Engine
Real-time cyber warfare simulation with autonomous AI agents
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import random
from dataclasses import dataclass, field
from pydantic import BaseModel


class AgentType(str, Enum):
    """Types of cyber agents"""
    WHITE_HAT = "white_hat"  # Ethical hackers/defenders
    BLACK_HAT = "black_hat"  # Malicious attackers
    GRAY_HAT = "gray_hat"    # Ambiguous actors
    SCRIPT_KIDDIE = "script_kiddie"  # Low-skill attackers
    NATION_STATE = "nation_state"    # Advanced persistent threats
    INSIDER_THREAT = "insider_threat"  # Internal malicious actors
    SOC_ANALYST = "soc_analyst"       # Security Operations Center
    INCIDENT_RESPONDER = "incident_responder"
    THREAT_HUNTER = "threat_hunter"
    PENETRATION_TESTER = "penetration_tester"


class AttackCategory(str, Enum):
    """MITRE ATT&CK Categories"""
    RECONNAISSANCE = "reconnaissance"
    RESOURCE_DEVELOPMENT = "resource_development"
    INITIAL_ACCESS = "initial_access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DEFENSE_EVASION = "defense_evasion"
    CREDENTIAL_ACCESS = "credential_access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral_movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command_and_control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


class DefenseAction(str, Enum):
    """Defense strategies"""
    MONITOR = "monitor"
    ISOLATE_SYSTEM = "isolate_system"
    BLOCK_IP = "block_ip"
    DEPLOY_PATCH = "deploy_patch"
    ENABLE_MFA = "enable_mfa"
    ROTATE_CREDENTIALS = "rotate_credentials"
    ENABLE_FIREWALL_RULE = "enable_firewall_rule"
    DEPLOY_IDS_SIGNATURE = "deploy_ids_signature"
    QUARANTINE_FILE = "quarantine_file"
    KILL_PROCESS = "kill_process"
    SNAPSHOT_SYSTEM = "snapshot_system"
    ALERT_TEAM = "alert_team"


class AttackOutcome(str, Enum):
    """Attack result"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    DETECTED = "detected"
    BLOCKED = "blocked"
    MITIGATED = "mitigated"


@dataclass
class SimulationEvent:
    """Single event in the simulation"""
    event_id: str
    timestamp: datetime
    agent_id: str
    agent_type: AgentType
    action: str
    target: str
    description: str
    outcome: AttackOutcome
    severity: int  # 1-10
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "action": self.action,
            "target": self.target,
            "description": self.description,
            "outcome": self.outcome.value,
            "severity": self.severity,
            "metadata": self.metadata
        }


@dataclass
class CyberAgent:
    """AI-driven cyber agent"""
    agent_id: str
    agent_type: AgentType
    name: str
    skill_level: int  # 1-100
    tactics: List[str]
    active: bool = True
    actions_performed: int = 0
    successes: int = 0
    failures: int = 0

    def get_tactics_for_attack(self) -> List[str]:
        """Get available tactics based on agent type"""
        tactics_map = {
            AgentType.BLACK_HAT: [
                "phishing", "malware_deployment", "sql_injection",
                "credential_stuffing", "ransomware", "ddos"
            ],
            AgentType.NATION_STATE: [
                "zero_day_exploit", "supply_chain_attack", "advanced_persistence",
                "lateral_movement", "data_exfiltration", "infrastructure_sabotage"
            ],
            AgentType.SCRIPT_KIDDIE: [
                "port_scanning", "brute_force", "basic_dos", "known_exploit"
            ],
            AgentType.GRAY_HAT: [
                "vulnerability_scanning", "proof_of_concept_exploit", "disclosure"
            ],
            AgentType.INSIDER_THREAT: [
                "data_theft", "privilege_abuse", "sabotage", "credential_sharing"
            ],
            AgentType.WHITE_HAT: [
                "penetration_testing", "vulnerability_assessment", "code_review"
            ],
            AgentType.SOC_ANALYST: [
                "log_analysis", "threat_detection", "incident_triage", "alert_investigation"
            ],
            AgentType.INCIDENT_RESPONDER: [
                "containment", "eradication", "recovery", "forensics"
            ],
            AgentType.THREAT_HUNTER: [
                "proactive_search", "behavioral_analysis", "ioc_hunting", "anomaly_detection"
            ],
            AgentType.PENETRATION_TESTER: [
                "controlled_exploitation", "privilege_escalation_test", "social_engineering"
            ]
        }
        return tactics_map.get(self.agent_type, [])

    def calculate_success_rate(self, defense_strength: int) -> float:
        """Calculate probability of attack success"""
        base_rate = self.skill_level / 100
        defense_impact = defense_strength / 200  # Defense reduces success
        return max(0.1, min(0.9, base_rate - defense_impact))


class LiveSimulationEngine:
    """
    Real-time cyber warfare simulation engine
    Manages autonomous AI agents performing attacks and defenses
    """

    def __init__(self):
        self.agents: Dict[str, CyberAgent] = {}
        self.events: List[SimulationEvent] = []
        self.running: bool = False
        self.defense_strength: int = 50  # 0-100
        self.infrastructure_health: int = 100  # 0-100
        self.compromised_systems: List[str] = []
        self.active_defenses: List[str] = []
        self.simulation_speed: float = 1.0  # 1.0 = real-time

    async def start_simulation(self, duration: int = 300):
        """Start live simulation for duration seconds"""
        self.running = True
        self.spawn_default_agents()

        tasks = [
            self.run_attack_loop(),
            self.run_defense_loop(),
            self.run_autonomous_response_loop()
        ]

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=duration
            )
        except asyncio.TimeoutError:
            pass
        finally:
            self.running = False

    def spawn_default_agents(self):
        """Create initial set of agents"""
        default_agents = [
            CyberAgent("bh_001", AgentType.BLACK_HAT, "BlackHat_Alpha", 75, []),
            CyberAgent("bh_002", AgentType.BLACK_HAT, "BlackHat_Beta", 60, []),
            CyberAgent("ns_001", AgentType.NATION_STATE, "APT_Phantom", 95, []),
            CyberAgent("sk_001", AgentType.SCRIPT_KIDDIE, "Skid_Noob", 25, []),
            CyberAgent("it_001", AgentType.INSIDER_THREAT, "Insider_Alice", 50, []),
            CyberAgent("soc_001", AgentType.SOC_ANALYST, "SOC_Guardian", 70, []),
            CyberAgent("ir_001", AgentType.INCIDENT_RESPONDER, "IR_Rapid", 80, []),
            CyberAgent("th_001", AgentType.THREAT_HUNTER, "Hunter_Elite", 85, []),
            CyberAgent("pt_001", AgentType.PENETRATION_TESTER, "PenTest_Pro", 90, []),
        ]

        for agent in default_agents:
            agent.tactics = agent.get_tactics_for_attack()
            self.agents[agent.agent_id] = agent

    async def run_attack_loop(self):
        """Continuous loop for attack agents"""
        attack_types = [
            AgentType.BLACK_HAT,
            AgentType.NATION_STATE,
            AgentType.SCRIPT_KIDDIE,
            AgentType.GRAY_HAT,
            AgentType.INSIDER_THREAT
        ]

        while self.running:
            attackers = [a for a in self.agents.values()
                        if a.agent_type in attack_types and a.active]

            for attacker in attackers:
                if random.random() < 0.3:  # 30% chance to attack each cycle
                    await self.execute_attack(attacker)

            await asyncio.sleep(2.0 / self.simulation_speed)

    async def run_defense_loop(self):
        """Continuous loop for defense agents"""
        defense_types = [
            AgentType.WHITE_HAT,
            AgentType.SOC_ANALYST,
            AgentType.INCIDENT_RESPONDER,
            AgentType.THREAT_HUNTER,
            AgentType.PENETRATION_TESTER
        ]

        while self.running:
            defenders = [a for a in self.agents.values()
                        if a.agent_type in defense_types and a.active]

            for defender in defenders:
                if random.random() < 0.4:  # 40% chance to act each cycle
                    await self.execute_defense(defender)

            await asyncio.sleep(2.5 / self.simulation_speed)

    async def run_autonomous_response_loop(self):
        """Automated system responses"""
        while self.running:
            # Auto-heal infrastructure
            if self.infrastructure_health < 100:
                self.infrastructure_health = min(100, self.infrastructure_health + 2)

            # Random external events
            if random.random() < 0.05:  # 5% chance
                await self.inject_random_event()

            await asyncio.sleep(5.0 / self.simulation_speed)

    async def execute_attack(self, attacker: CyberAgent):
        """Execute an attack by an agent"""
        if not attacker.tactics:
            return

        tactic = random.choice(attacker.tactics)
        target = self.select_attack_target()

        # Calculate success
        success_rate = attacker.calculate_success_rate(self.defense_strength)
        success = random.random() < success_rate

        # Determine outcome
        if success:
            if random.random() < 0.3:  # 30% chance of detection even on success
                outcome = AttackOutcome.DETECTED
                self.defense_strength += 5  # Learn from detection
            else:
                outcome = AttackOutcome.SUCCESS
                self.infrastructure_health -= random.randint(5, 15)
                if target not in self.compromised_systems:
                    self.compromised_systems.append(target)
            attacker.successes += 1
        else:
            outcome = random.choice([
                AttackOutcome.FAILED,
                AttackOutcome.BLOCKED,
                AttackOutcome.MITIGATED
            ])
            self.defense_strength += 2
            attacker.failures += 1

        attacker.actions_performed += 1

        # Create event
        event = SimulationEvent(
            event_id=f"evt_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            agent_id=attacker.agent_id,
            agent_type=attacker.agent_type,
            action=tactic,
            target=target,
            description=self.generate_attack_description(attacker, tactic, outcome),
            outcome=outcome,
            severity=self.calculate_severity(outcome, attacker.agent_type),
            metadata={
                "success_rate": success_rate,
                "defense_strength": self.defense_strength,
                "infrastructure_health": self.infrastructure_health
            }
        )

        self.events.append(event)

    async def execute_defense(self, defender: CyberAgent):
        """Execute a defensive action"""
        if not defender.tactics:
            return

        action = random.choice(defender.tactics)

        # Defense actions
        if len(self.compromised_systems) > 0 and random.random() < 0.6:
            # Remediate compromised system
            target = random.choice(self.compromised_systems)
            self.compromised_systems.remove(target)
            outcome = AttackOutcome.MITIGATED
            self.infrastructure_health = min(100, self.infrastructure_health + 10)
            description = f"{defender.name} successfully remediated {target}"
        else:
            # Proactive defense
            self.defense_strength = min(100, self.defense_strength + random.randint(3, 8))
            target = self.select_defense_target()
            outcome = AttackOutcome.SUCCESS
            description = f"{defender.name} strengthened defenses on {target}"

        defender.actions_performed += 1
        defender.successes += 1

        event = SimulationEvent(
            event_id=f"evt_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            agent_id=defender.agent_id,
            agent_type=defender.agent_type,
            action=action,
            target=target,
            description=description,
            outcome=outcome,
            severity=5,
            metadata={
                "defense_strength": self.defense_strength,
                "compromised_systems": len(self.compromised_systems)
            }
        )

        self.events.append(event)

    async def inject_random_event(self):
        """Inject random external events"""
        events = [
            "Zero-day vulnerability discovered in system",
            "Suspicious traffic detected from foreign IP",
            "Employee clicked phishing link",
            "Automated security scan completed",
            "New threat intelligence received"
        ]

        event_desc = random.choice(events)
        impact = random.randint(-15, 10)
        self.infrastructure_health = max(0, min(100, self.infrastructure_health + impact))

    def select_attack_target(self) -> str:
        """Select a target for attack"""
        targets = [
            "web_server_01", "database_primary", "auth_service",
            "api_gateway", "file_server", "mail_server",
            "admin_workstation", "employee_laptop_42", "cloud_storage"
        ]
        return random.choice(targets)

    def select_defense_target(self) -> str:
        """Select a target for defense"""
        return self.select_attack_target()

    def generate_attack_description(self, attacker: CyberAgent, tactic: str, outcome: AttackOutcome) -> str:
        """Generate human-readable attack description"""
        outcome_text = {
            AttackOutcome.SUCCESS: "successfully executed",
            AttackOutcome.PARTIAL_SUCCESS: "partially succeeded in",
            AttackOutcome.FAILED: "failed to execute",
            AttackOutcome.DETECTED: "was detected attempting",
            AttackOutcome.BLOCKED: "was blocked from",
            AttackOutcome.MITIGATED: "was mitigated while attempting"
        }

        return f"{attacker.name} {outcome_text[outcome]} {tactic.replace('_', ' ')}"

    def calculate_severity(self, outcome: AttackOutcome, agent_type: AgentType) -> int:
        """Calculate event severity 1-10"""
        base_severity = {
            AgentType.SCRIPT_KIDDIE: 2,
            AgentType.GRAY_HAT: 3,
            AgentType.BLACK_HAT: 6,
            AgentType.INSIDER_THREAT: 7,
            AgentType.NATION_STATE: 9
        }

        severity = base_severity.get(agent_type, 5)

        if outcome == AttackOutcome.SUCCESS:
            severity += 2
        elif outcome in [AttackOutcome.BLOCKED, AttackOutcome.MITIGATED]:
            severity -= 2

        return max(1, min(10, severity))

    def get_simulation_state(self) -> Dict:
        """Get current simulation state"""
        return {
            "running": self.running,
            "total_events": len(self.events),
            "infrastructure_health": self.infrastructure_health,
            "defense_strength": self.defense_strength,
            "compromised_systems": len(self.compromised_systems),
            "active_agents": len([a for a in self.agents.values() if a.active]),
            "recent_events": [e.to_dict() for e in self.events[-20:]],  # Last 20 events
            "agent_stats": {
                agent_id: {
                    "name": agent.name,
                    "type": agent.agent_type.value,
                    "actions": agent.actions_performed,
                    "successes": agent.successes,
                    "failures": agent.failures,
                    "skill": agent.skill_level
                }
                for agent_id, agent in self.agents.items()
            }
        }

    def add_agent(self, agent: CyberAgent):
        """Add new agent to simulation"""
        self.agents[agent.agent_id] = agent

    def remove_agent(self, agent_id: str):
        """Remove agent from simulation"""
        if agent_id in self.agents:
            self.agents[agent_id].active = False
