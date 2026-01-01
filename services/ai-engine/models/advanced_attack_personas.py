"""
Advanced Attack Personas
Simulates Nation-state APTs, Ransomware gangs, Insider threats, and skill levels
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import random


class ThreatActorType(Enum):
    """Types of threat actors"""
    NATION_STATE = "nation_state"
    RANSOMWARE_GANG = "ransomware_gang"
    HACKTIVIST = "hacktivist"
    CYBERCRIMINAL = "cybercriminal"
    INSIDER_THREAT = "insider_threat"
    SCRIPT_KIDDIE = "script_kiddie"


class SkillLevel(Enum):
    """Attacker skill levels"""
    NOVICE = 1  # Script kiddie
    INTERMEDIATE = 2  # Skilled hacker
    ADVANCED = 3  # Professional
    EXPERT = 4  # APT-level
    ELITE = 5  # Nation-state level


@dataclass
class AttackTactic:
    """Individual attack tactic"""
    name: str
    mitre_id: str
    description: str
    success_rate: float
    stealth_level: float
    required_skill: SkillLevel
    tools: List[str]


@dataclass
class ThreatActorProfile:
    """Complete threat actor profile"""
    name: str
    actor_type: ThreatActorType
    skill_level: SkillLevel
    motivation: str
    typical_targets: List[str]
    preferred_tactics: List[AttackTactic]
    tools_arsenal: List[str]
    attribution_indicators: Dict[str, Any]
    behavior_patterns: Dict[str, Any]


class APTPersona:
    """Advanced Persistent Threat personas"""

    @staticmethod
    def lazarus_group() -> ThreatActorProfile:
        """North Korean APT group"""
        return ThreatActorProfile(
            name="Lazarus Group (APT38)",
            actor_type=ThreatActorType.NATION_STATE,
            skill_level=SkillLevel.ELITE,
            motivation="Financial gain, espionage, and political objectives",
            typical_targets=["Financial institutions", "Cryptocurrency exchanges", "Defense contractors"],
            preferred_tactics=[
                AttackTactic(
                    name="Spear Phishing",
                    mitre_id="T1566.001",
                    description="Targeted emails with malicious attachments",
                    success_rate=0.75,
                    stealth_level=0.8,
                    required_skill=SkillLevel.ADVANCED,
                    tools=["Custom malware", "Living-off-the-land binaries"]
                ),
                AttackTactic(
                    name="Watering Hole",
                    mitre_id="T1189",
                    description="Compromise frequently visited websites",
                    success_rate=0.65,
                    stealth_level=0.9,
                    required_skill=SkillLevel.EXPERT,
                    tools=["Zero-day exploits", "Browser exploits"]
                ),
            ],
            tools_arsenal=[
                "TYPEFRAME", "NACHOCHEESE", "VIVACIOUSGIFT",
                "PowerRatankba", "Brambul", "Duuzer"
            ],
            attribution_indicators={
                "language": "Korean",
                "timezone": "UTC+9",
                "code_style": "Professional, well-documented",
                "infrastructure": "Compromised servers in Asia"
            },
            behavior_patterns={
                "patience": "Very high - can wait months",
                "persistence": "Multiple backdoors, redundant C2",
                "anti_forensics": "Advanced log deletion, timestomping",
                "lateral_movement": "Slow and methodical"
            }
        )

    @staticmethod
    def apt28_fancy_bear() -> ThreatActorProfile:
        """Russian military intelligence APT"""
        return ThreatActorProfile(
            name="APT28 (Fancy Bear)",
            actor_type=ThreatActorType.NATION_STATE,
            skill_level=SkillLevel.ELITE,
            motivation="Political espionage, information warfare",
            typical_targets=["Government agencies", "Military", "Political organizations", "Media"],
            preferred_tactics=[
                AttackTactic(
                    name="Credential Harvesting",
                    mitre_id="T1589.001",
                    description="Phishing for credentials",
                    success_rate=0.70,
                    stealth_level=0.7,
                    required_skill=SkillLevel.ADVANCED,
                    tools=["Fake login pages", "OAuth token theft"]
                ),
                AttackTactic(
                    name="Zero-Day Exploitation",
                    mitre_id="T1203",
                    description="Use of unknown vulnerabilities",
                    success_rate=0.85,
                    stealth_level=0.95,
                    required_skill=SkillLevel.ELITE,
                    tools=["Custom exploits", "Commercial exploit kits"]
                ),
            ],
            tools_arsenal=[
                "X-Agent", "X-Tunnel", "CHOPSTICK", "Sofacy",
                "GAMEFISH", "SOURFACE", "VPNFilter"
            ],
            attribution_indicators={
                "language": "Russian",
                "timezone": "UTC+3",
                "code_style": "Military precision",
                "infrastructure": "Russian IPs, bulletproof hosting"
            },
            behavior_patterns={
                "aggression": "High",
                "speed": "Fast initial compromise",
                "public_exposure": "Not concerned with attribution",
                "data_destruction": "Sometimes destroys evidence"
            }
        )

    @staticmethod
    def apt29_cozy_bear() -> ThreatActorProfile:
        """Russian SVR foreign intelligence APT"""
        return ThreatActorProfile(
            name="APT29 (Cozy Bear)",
            actor_type=ThreatActorType.NATION_STATE,
            skill_level=SkillLevel.ELITE,
            motivation="Intelligence gathering, long-term espionage",
            typical_targets=["Government", "Think tanks", "Research institutions", "Healthcare"],
            preferred_tactics=[
                AttackTactic(
                    name="Supply Chain Compromise",
                    mitre_id="T1195",
                    description="Compromise software supply chain",
                    success_rate=0.90,
                    stealth_level=0.98,
                    required_skill=SkillLevel.ELITE,
                    tools=["SolarWinds compromise-style attacks"]
                ),
            ],
            tools_arsenal=[
                "HAMMERTOSS", "COZYDUKE", "COZYCAR",
                "SeaDuke", "CloudDuke", "SunBurst"
            ],
            attribution_indicators={
                "stealth_focus": "Extremely high",
                "code_quality": "Excellent",
                "opsec": "Near perfect"
            },
            behavior_patterns={
                "patience": "Extreme - years of presence",
                "stealth": "Maximum",
                "data_collection": "Comprehensive but selective exfil"
            }
        )


class RansomwareGangPersona:
    """Ransomware gang profiles"""

    @staticmethod
    def lockbit() -> ThreatActorProfile:
        """LockBit ransomware gang"""
        return ThreatActorProfile(
            name="LockBit",
            actor_type=ThreatActorType.RANSOMWARE_GANG,
            skill_level=SkillLevel.ADVANCED,
            motivation="Financial extortion",
            typical_targets=["Enterprises", "Healthcare", "Education", "Manufacturing"],
            preferred_tactics=[
                AttackTactic(
                    name="RDP Brute Force",
                    mitre_id="T1110",
                    description="Credential stuffing on exposed RDP",
                    success_rate=0.60,
                    stealth_level=0.3,
                    required_skill=SkillLevel.INTERMEDIATE,
                    tools=["Hydra", "NLBrute", "RDP scanners"]
                ),
                AttackTactic(
                    name="Data Exfiltration Before Encryption",
                    mitre_id="T1567",
                    description="Double extortion - steal then encrypt",
                    success_rate=0.80,
                    stealth_level=0.5,
                    required_skill=SkillLevel.ADVANCED,
                    tools=["Rclone", "MEGAsync", "FileZilla"]
                ),
            ],
            tools_arsenal=[
                "LockBit ransomware", "Mimikatz", "Cobalt Strike",
                "BloodHound", "SharpHound", "PSExec"
            ],
            attribution_indicators={
                "ransom_note": "Specific template and contact methods",
                "encryption": "AES + RSA hybrid",
                "speed": "Very fast encryption"
            },
            behavior_patterns={
                "automation": "Highly automated",
                "negotiation": "Professional ransom negotiation",
                "data_leaks": "Operates leak site for pressure",
                "affiliate_model": "Ransomware-as-a-Service (RaaS)"
            }
        )

    @staticmethod
    def blackcat() -> ThreatActorProfile:
        """BlackCat (ALPHV) ransomware gang"""
        return ThreatActorProfile(
            name="BlackCat (ALPHV)",
            actor_type=ThreatActorType.RANSOMWARE_GANG,
            skill_level=SkillLevel.EXPERT,
            motivation="Financial extortion",
            typical_targets=["Large enterprises", "Critical infrastructure", "Oil & Gas"],
            preferred_tactics=[
                AttackTactic(
                    name="Exploit Public-Facing Apps",
                    mitre_id="T1190",
                    description="Target vulnerabilities in web apps and VPNs",
                    success_rate=0.75,
                    stealth_level=0.6,
                    required_skill=SkillLevel.ADVANCED,
                    tools=["Exchange exploits", "VPN vulnerabilities"]
                ),
            ],
            tools_arsenal=[
                "ALPHV ransomware (Rust-based)", "Cobalt Strike",
                "Brute Ratel C4", "Mimikatz", "PsExec"
            ],
            attribution_indicators={
                "language": "Rust programming language",
                "cross_platform": "Windows and Linux variants",
                "sophistication": "Very high"
            },
            behavior_patterns={
                "triple_extortion": "Steal, encrypt, and DDoS",
                "affiliate_model": "RaaS with high affiliate cuts",
                "public_pressure": "Aggressive media campaigns"
            }
        )


class InsiderThreatPersona:
    """Insider threat profiles"""

    @staticmethod
    def disgruntled_employee() -> ThreatActorProfile:
        """Malicious insider - disgruntled employee"""
        return ThreatActorProfile(
            name="Disgruntled Employee",
            actor_type=ThreatActorType.INSIDER_THREAT,
            skill_level=SkillLevel.INTERMEDIATE,
            motivation="Revenge, sabotage, or financial gain",
            typical_targets=["Own employer", "Specific departments or individuals"],
            preferred_tactics=[
                AttackTactic(
                    name="Data Theft via Authorized Access",
                    mitre_id="T1213",
                    description="Abuse legitimate access to steal data",
                    success_rate=0.95,
                    stealth_level=0.7,
                    required_skill=SkillLevel.NOVICE,
                    tools=["USB drives", "Cloud storage", "Email"]
                ),
                AttackTactic(
                    name="Sabotage",
                    mitre_id="T1485",
                    description="Delete critical data or systems",
                    success_rate=0.90,
                    stealth_level=0.4,
                    required_skill=SkillLevel.INTERMEDIATE,
                    tools=["Admin privileges", "Scheduled tasks"]
                ),
            ],
            tools_arsenal=[
                "Legitimate company tools", "Personal devices",
                "Cloud storage accounts", "External hard drives"
            ],
            attribution_indicators={
                "timing": "Often after termination notice",
                "access_patterns": "Unusual data access before leaving",
                "behavioral_changes": "Detectable through UEBA"
            },
            behavior_patterns={
                "knowledge": "Deep internal knowledge",
                "access": "Legitimate credentials",
                "timing": "Often nights/weekends before leaving",
                "targets": "Most valuable or sensitive data"
            }
        )


class ScriptKiddiePersona:
    """Low-skill attacker using pre-made tools"""

    @staticmethod
    def amateur_hacker() -> ThreatActorProfile:
        """Script kiddie - amateur attacker"""
        return ThreatActorProfile(
            name="Script Kiddie",
            actor_type=ThreatActorType.SCRIPT_KIDDIE,
            skill_level=SkillLevel.NOVICE,
            motivation="Curiosity, bragging rights, chaos",
            typical_targets=["Opportunistic - any vulnerable target"],
            preferred_tactics=[
                AttackTactic(
                    name="Automated Scanning",
                    mitre_id="T1046",
                    description="Use automated tools to find vulnerabilities",
                    success_rate=0.30,
                    stealth_level=0.1,
                    required_skill=SkillLevel.NOVICE,
                    tools=["Nmap", "Metasploit", "SQLmap"]
                ),
                AttackTactic(
                    name="Default Credential Usage",
                    mitre_id="T1078",
                    description="Try default passwords",
                    success_rate=0.40,
                    stealth_level=0.2,
                    required_skill=SkillLevel.NOVICE,
                    tools=["Password lists", "Default cred databases"]
                ),
            ],
            tools_arsenal=[
                "Kali Linux", "Metasploit Framework", "SQLmap",
                "Burp Suite Community", "LOIC", "Shodan"
            ],
            attribution_indicators={
                "noise": "Very noisy attacks",
                "sophistication": "Low",
                "error_prone": "Many failed attempts"
            },
            behavior_patterns={
                "patience": "Very low",
                "skill": "Limited understanding of tools",
                "persistence": "Gives up easily",
                "opsec": "Poor - easily traced"
            }
        )


class PersonaSimulator:
    """Simulates behavior of different threat actor personas"""

    def __init__(self, persona: ThreatActorProfile):
        self.persona = persona

    def generate_attack_plan(self, target_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate attack plan based on persona characteristics
        """
        attack_plan = []

        # Persona-specific behavior
        if self.persona.skill_level.value >= SkillLevel.EXPERT.value:
            # Advanced actors do thorough reconnaissance
            attack_plan.append({
                "phase": "Reconnaissance",
                "duration_days": random.randint(7, 30),
                "actions": ["OSINT gathering", "Network mapping", "Employee profiling"]
            })

        # Select tactics based on persona preferences
        for tactic in self.persona.preferred_tactics:
            if random.random() < tactic.success_rate:
                attack_plan.append({
                    "phase": tactic.name,
                    "mitre_id": tactic.mitre_id,
                    "tools": random.sample(tactic.tools, min(2, len(tactic.tools))),
                    "stealth_level": tactic.stealth_level
                })

        return attack_plan

    def simulate_decision_making(self, situation: Dict[str, Any]) -> str:
        """
        Simulate how this persona would react to a situation
        """
        if situation.get("detected", False):
            if self.persona.skill_level.value >= SkillLevel.ADVANCED.value:
                return "Go dark, change TTPs, wait for things to calm down"
            else:
                return "Panic and either give up or become more aggressive"

        if situation.get("initial_access_failed", False):
            if self.persona.skill_level == SkillLevel.NOVICE:
                return "Try different target or give up"
            else:
                return "Analyze failure, adapt approach, try alternative vector"

        return "Continue with current strategy"


# Factory function
def create_persona(persona_type: str) -> ThreatActorProfile:
    """Create threat actor persona by type"""
    personas = {
        "lazarus": APTPersona.lazarus_group,
        "apt28": APTPersona.apt28_fancy_bear,
        "apt29": APTPersona.apt29_cozy_bear,
        "lockbit": RansomwareGangPersona.lockbit,
        "blackcat": RansomwareGangPersona.blackcat,
        "insider": InsiderThreatPersona.disgruntled_employee,
        "script_kiddie": ScriptKiddiePersona.amateur_hacker,
    }

    if persona_type.lower() in personas:
        return personas[persona_type.lower()]()
    else:
        raise ValueError(f"Unknown persona type: {persona_type}")


# Example usage
if __name__ == "__main__":
    print("🎭 Advanced Attack Personas\n")

    # Test each persona
    lazarus = create_persona("lazarus")
    print(f"Name: {lazarus.name}")
    print(f"Type: {lazarus.actor_type.value}")
    print(f"Skill: {lazarus.skill_level.name}")
    print(f"Tools: {', '.join(lazarus.tools_arsenal[:3])}")
    print()

    # Simulate attack
    simulator = PersonaSimulator(lazarus)
    plan = simulator.generate_attack_plan({"target": "Financial Institution"})
    print(f"Attack Plan ({len(plan)} phases):")
    for phase in plan:
        print(f"  - {phase.get('phase', 'Unknown')}")
