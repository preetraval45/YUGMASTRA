"""
MITRE ATT&CK Framework Integration
Multi-Stage Attack Chains with full 14 tactics coverage
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class MITRETactic(Enum):
    """14 MITRE ATT&CK Tactics"""
    RECONNAISSANCE = "TA0043"
    RESOURCE_DEVELOPMENT = "TA0042"
    INITIAL_ACCESS = "TA0001"
    EXECUTION = "TA0002"
    PERSISTENCE = "TA0003"
    PRIVILEGE_ESCALATION = "TA0004"
    DEFENSE_EVASION = "TA0005"
    CREDENTIAL_ACCESS = "TA0006"
    DISCOVERY = "TA0007"
    LATERAL_MOVEMENT = "TA0008"
    COLLECTION = "TA0009"
    COMMAND_AND_CONTROL = "TA0011"
    EXFILTRATION = "TA0010"
    IMPACT = "TA0040"


@dataclass
class MITRETechnique:
    """MITRE ATT&CK Technique"""
    id: str
    name: str
    tactic: MITRETactic
    description: str
    platforms: List[str]
    data_sources: List[str]
    defenses: List[str]
    examples: List[str]


class MITREAttackChain:
    """Build multi-stage attack chains"""

    def __init__(self):
        self.techniques = self._load_techniques()
        self.attack_chain: List[MITRETechnique] = []

    def _load_techniques(self) -> Dict[str, MITRETechnique]:
        """Load MITRE ATT&CK techniques database"""
        return {
            "T1595": MITRETechnique(
                id="T1595",
                name="Active Scanning",
                tactic=MITRETactic.RECONNAISSANCE,
                description="Scan target infrastructure for vulnerabilities",
                platforms=["PRE"],
                data_sources=["Network Traffic"],
                defenses=["Network Segmentation", "IDS/IPS"],
                examples=["Nmap", "Masscan", "ZMap"]
            ),
            "T1566.001": MITRETechnique(
                id="T1566.001",
                name="Spearphishing Attachment",
                tactic=MITRETactic.INITIAL_ACCESS,
                description="Send malicious email attachments",
                platforms=["Windows", "macOS", "Linux"],
                data_sources=["Email Gateway", "File Monitoring"],
                defenses=["Email Filtering", "User Training"],
                examples=["APT28", "APT29", "FIN7"]
            ),
            "T1059.001": MITRETechnique(
                id="T1059.001",
                name="PowerShell",
                tactic=MITRETactic.EXECUTION,
                description="Execute PowerShell commands",
                platforms=["Windows"],
                data_sources=["Process Monitoring", "PowerShell Logs"],
                defenses=["Application Whitelisting", "Script Block Logging"],
                examples=["Empire", "PowerSploit"]
            ),
            "T1547.001": MITRETechnique(
                id="T1547.001",
                name="Registry Run Keys",
                tactic=MITRETactic.PERSISTENCE,
                description="Add to registry autorun keys",
                platforms=["Windows"],
                data_sources=["Windows Registry"],
                defenses=["Registry Monitoring"],
                examples=["Kovter", "Emotet"]
            ),
            "T1078": MITRETechnique(
                id="T1078",
                name="Valid Accounts",
                tactic=MITRETactic.PRIVILEGE_ESCALATION,
                description="Use stolen credentials",
                platforms=["Windows", "Linux", "macOS", "Cloud"],
                data_sources=["Authentication Logs"],
                defenses=["MFA", "Privilege Access Management"],
                examples=["APT28", "APT29"]
            ),
            "T1027": MITRETechnique(
                id="T1027",
                name="Obfuscated Files or Information",
                tactic=MITRETactic.DEFENSE_EVASION,
                description="Obfuscate malware",
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["File Monitoring"],
                defenses=["Behavioral Analysis"],
                examples=["Emotet", "TrickBot"]
            ),
            "T1003.001": MITRETechnique(
                id="T1003.001",
                name="LSASS Memory",
                tactic=MITRETactic.CREDENTIAL_ACCESS,
                description="Dump credentials from LSASS",
                platforms=["Windows"],
                data_sources=["Process Monitoring"],
                defenses=["Credential Guard", "LSA Protection"],
                examples=["Mimikatz", "ProcDump"]
            ),
            "T1018": MITRETechnique(
                id="T1018",
                name="Remote System Discovery",
                tactic=MITRETactic.DISCOVERY,
                description="Identify remote systems",
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["Network Protocol Analysis"],
                defenses=["Network Segmentation"],
                examples=["BloodHound", "PowerView"]
            ),
            "T1021.001": MITRETechnique(
                id="T1021.001",
                name="Remote Desktop Protocol",
                tactic=MITRETactic.LATERAL_MOVEMENT,
                description="Use RDP for lateral movement",
                platforms=["Windows"],
                data_sources=["Authentication Logs", "Network Traffic"],
                defenses=["MFA", "Network Segmentation"],
                examples=["APT41", "Wizard Spider"]
            ),
            "T1005": MITRETechnique(
                id="T1005",
                name="Data from Local System",
                tactic=MITRETactic.COLLECTION,
                description="Collect data from local system",
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["File Monitoring"],
                defenses=["DLP"],
                examples=["APT28", "APT32"]
            ),
            "T1071.001": MITRETechnique(
                id="T1071.001",
                name="Web Protocols",
                tactic=MITRETactic.COMMAND_AND_CONTROL,
                description="C2 over HTTP/HTTPS",
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["Network Traffic", "SSL/TLS Inspection"],
                defenses=["Web Proxy", "SSL Inspection"],
                examples=["Cobalt Strike", "Metasploit"]
            ),
            "T1041": MITRETechnique(
                id="T1041",
                name="Exfiltration Over C2 Channel",
                tactic=MITRETactic.EXFILTRATION,
                description="Exfiltrate via C2",
                platforms=["Windows", "Linux", "macOS"],
                data_sources=["Network Traffic"],
                defenses=["DLP", "Network Monitoring"],
                examples=["APT28", "APT29"]
            ),
            "T1486": MITRETechnique(
                id="T1486",
                name="Data Encrypted for Impact",
                tactic=MITRETactic.IMPACT,
                description="Ransomware encryption",
                platforms=["Windows", "Linux"],
                data_sources=["File Monitoring"],
                defenses=["Backups", "EDR"],
                examples=["Ryuk", "LockBit", "BlackCat"]
            ),
        }

    def generate_attack_chain(
        self,
        start_tactic: MITRETactic = MITRETactic.RECONNAISSANCE,
        end_tactic: MITRETactic = MITRETactic.IMPACT
    ) -> List[MITRETechnique]:
        """Generate complete attack chain from reconnaissance to impact"""
        chain = []

        # Get tactics in order
        tactic_order = list(MITRETactic)
        start_idx = tactic_order.index(start_tactic)
        end_idx = tactic_order.index(end_tactic)

        for tactic in tactic_order[start_idx:end_idx + 1]:
            # Find techniques for this tactic
            techniques = [t for t in self.techniques.values() if t.tactic == tactic]
            if techniques:
                # Select first available technique (can be randomized)
                chain.append(techniques[0])

        return chain

    def get_defensive_coverage(self, chain: List[MITRETechnique]) -> Dict[str, Any]:
        """Analyze defensive coverage for attack chain"""
        all_defenses = set()
        coverage_map = {}

        for technique in chain:
            for defense in technique.defenses:
                all_defenses.add(defense)
                if defense not in coverage_map:
                    coverage_map[defense] = []
                coverage_map[defense].append(technique.id)

        return {
            "total_defenses_needed": len(all_defenses),
            "defense_techniques_map": coverage_map,
            "coverage_percentage": 0.0  # To be calculated based on deployed defenses
        }

    def export_navigator_layer(self, chain: List[MITRETechnique]) -> Dict[str, Any]:
        """Export attack chain for MITRE ATT&CK Navigator"""
        techniques = [
            {
                "techniqueID": t.id,
                "tactic": t.tactic.name.lower(),
                "color": "#ff6666",
                "comment": t.description,
                "enabled": True,
                "score": 100
            }
            for t in chain
        ]

        return {
            "name": "YUGMASTRA Attack Chain",
            "versions": {
                "attack": "14",
                "navigator": "4.9.1",
                "layer": "4.5"
            },
            "domain": "enterprise-attack",
            "description": "Generated attack chain from YUGMASTRA simulation",
            "techniques": techniques
        }


# Example usage
if __name__ == "__main__":
    framework = MITREAttackChain()

    print("🎯 MITRE ATT&CK Framework Integration\n")

    # Generate full attack chain
    chain = framework.generate_attack_chain()

    print(f"Generated Attack Chain ({len(chain)} stages):\n")
    for i, technique in enumerate(chain, 1):
        print(f"{i}. [{technique.id}] {technique.name}")
        print(f"   Tactic: {technique.tactic.name}")
        print(f"   Defenses: {', '.join(technique.defenses)}")
        print()

    # Analyze defenses
    coverage = framework.get_defensive_coverage(chain)
    print(f"\n🛡️  Defensive Requirements:")
    print(f"Total defenses needed: {coverage['total_defenses_needed']}")
