"""
Real-World Threat Intelligence Feed
Integrates with threat intelligence sources to provide real-time attack data
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import random


class ThreatSeverity(str, Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AttackPhase(str, Enum):
    """Cyber kill chain phases"""
    RECONNAISSANCE = "reconnaissance"
    WEAPONIZATION = "weaponization"
    DELIVERY = "delivery"
    EXPLOITATION = "exploitation"
    INSTALLATION = "installation"
    COMMAND_CONTROL = "command_and_control"
    ACTIONS_OBJECTIVES = "actions_on_objectives"


@dataclass
class ThreatIntelligence:
    """Real-world cyber threat incident"""
    threat_id: str
    title: str
    description: str
    severity: ThreatSeverity
    attack_vectors: List[str]
    affected_systems: List[str]
    indicators_of_compromise: List[str]
    attack_timeline: List[Dict[str, Any]]
    detection_methods: List[str]
    mitigation_steps: List[str]
    lessons_learned: List[str]
    attribution: Optional[str]
    date_discovered: datetime
    date_reported: datetime
    source: str
    tags: List[str]
    cve_ids: List[str] = None

    def to_dict(self) -> Dict:
        return {
            "threat_id": self.threat_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "attack_vectors": self.attack_vectors,
            "affected_systems": self.affected_systems,
            "indicators_of_compromise": self.indicators_of_compromise,
            "attack_timeline": self.attack_timeline,
            "detection_methods": self.detection_methods,
            "mitigation_steps": self.mitigation_steps,
            "lessons_learned": self.lessons_learned,
            "attribution": self.attribution,
            "date_discovered": self.date_discovered.isoformat(),
            "date_reported": self.date_reported.isoformat(),
            "source": self.source,
            "tags": self.tags,
            "cve_ids": self.cve_ids or []
        }


class ThreatIntelligenceFeed:
    """
    Threat Intelligence Feed Manager
    Provides real-world cyber attack information
    """

    def __init__(self):
        self.threats: List[ThreatIntelligence] = []
        self.initialize_sample_threats()

    def initialize_sample_threats(self):
        """Initialize with real-world-inspired threat scenarios"""

        # Sample threats based on real incidents
        sample_threats = [
            ThreatIntelligence(
                threat_id="TI-2024-001",
                title="SolarWinds Supply Chain Attack",
                description="Advanced persistent threat group compromised SolarWinds Orion platform, affecting thousands of organizations worldwide through trojanized software updates.",
                severity=ThreatSeverity.CRITICAL,
                attack_vectors=[
                    "Supply chain compromise",
                    "Trojanized software update",
                    "DLL side-loading",
                    "Stealth C2 communication"
                ],
                affected_systems=[
                    "Windows servers running SolarWinds Orion",
                    "Network management systems",
                    "Government agencies",
                    "Fortune 500 companies"
                ],
                indicators_of_compromise=[
                    "Suspicious DLL: SolarWinds.Orion.Core.BusinessLayer.dll",
                    "C2 domains: avsvmcloud.com, digitalcollege.org",
                    "HTTP user agent: Mozilla/5.0 (Windows NT 6.1)",
                    "Registry keys: HKEY_LOCAL_MACHINE\\SOFTWARE\\SolarWinds"
                ],
                attack_timeline=[
                    {"phase": "Reconnaissance", "date": "2019-09", "description": "Initial infrastructure setup and reconnaissance"},
                    {"phase": "Weaponization", "date": "2019-10", "description": "Development of SUNBURST backdoor"},
                    {"phase": "Delivery", "date": "2020-03", "description": "Compromised SolarWinds build system"},
                    {"phase": "Exploitation", "date": "2020-04", "description": "Trojanized updates distributed to customers"},
                    {"phase": "Command & Control", "date": "2020-05", "description": "Established C2 channels via DGA domains"},
                    {"phase": "Actions", "date": "2020-06", "description": "Data exfiltration and lateral movement"},
                    {"phase": "Discovery", "date": "2020-12", "description": "FireEye discovers breach"},
                    {"phase": "Public Disclosure", "date": "2020-12-13", "description": "SolarWinds announces compromise"}
                ],
                detection_methods=[
                    "Behavioral analysis of SolarWinds processes",
                    "Network traffic anomaly detection (DNS queries to suspicious domains)",
                    "File integrity monitoring (unexpected DLL modifications)",
                    "Memory forensics for injected code",
                    "YARA rules for SUNBURST backdoor"
                ],
                mitigation_steps=[
                    "Immediately isolate affected SolarWinds Orion servers",
                    "Block C2 domains at firewall and DNS level",
                    "Reset all credentials and deploy MFA",
                    "Apply SolarWinds hotfix 2020.2.1 HF 2 or later",
                    "Hunt for lateral movement using compromised credentials",
                    "Review firewall rules for unauthorized changes",
                    "Conduct full network compromise assessment",
                    "Implement application whitelisting",
                    "Enable enhanced logging and SIEM correlation"
                ],
                lessons_learned=[
                    "Supply chain attacks can bypass traditional perimeter defenses",
                    "Code signing does not prevent sophisticated backdoors",
                    "Long-term persistence (6+ months) is possible with advanced attackers",
                    "Third-party software requires rigorous security validation",
                    "Network segmentation limits blast radius",
                    "Behavioral analytics detect post-compromise activity",
                    "Incident response plans must include supply chain scenarios"
                ],
                attribution="APT29 (Cozy Bear) - suspected Russian intelligence",
                date_discovered=datetime(2020, 12, 8),
                date_reported=datetime(2020, 12, 13),
                source="FireEye Mandiant, Microsoft, CrowdStrike",
                tags=["APT29", "supply_chain", "backdoor", "nation_state", "espionage"],
                cve_ids=["CVE-2020-10148"]
            ),

            ThreatIntelligence(
                threat_id="TI-2024-002",
                title="Colonial Pipeline Ransomware Attack",
                description="DarkSide ransomware gang compromised Colonial Pipeline, leading to fuel supply disruption across US East Coast. Attackers used stolen VPN credentials.",
                severity=ThreatSeverity.CRITICAL,
                attack_vectors=[
                    "Stolen VPN credentials",
                    "No multi-factor authentication",
                    "Lateral movement via SMB",
                    "DarkSide ransomware deployment"
                ],
                affected_systems=[
                    "Operational Technology (OT) networks",
                    "Billing and customer systems",
                    "File servers and databases",
                    "Pipeline control systems"
                ],
                indicators_of_compromise=[
                    "DarkSide ransomware executable: ds.exe",
                    "Ransom note: README.txt",
                    "C2 IP addresses: 185.220.xxx.xxx, 23.129.xxx.xxx",
                    "Encrypted file extension: .darkside",
                    "TOR payment portal: darksidefxxxxxxxxx.onion"
                ],
                attack_timeline=[
                    {"phase": "Initial Access", "date": "2021-04-29", "description": "Compromise via stolen VPN credentials"},
                    {"phase": "Reconnaissance", "date": "2021-04-30", "description": "Network mapping and data discovery"},
                    {"phase": "Lateral Movement", "date": "2021-05-01", "description": "Spread to additional systems"},
                    {"phase": "Data Exfiltration", "date": "2021-05-06", "description": "100GB of sensitive data stolen"},
                    {"phase": "Ransomware Deployment", "date": "2021-05-07", "description": "DarkSide ransomware executed"},
                    {"phase": "Discovery", "date": "2021-05-07 06:00", "description": "IT staff discovers encryption"},
                    {"phase": "Pipeline Shutdown", "date": "2021-05-07 08:00", "description": "Operations halted preventatively"},
                    {"phase": "Ransom Payment", "date": "2021-05-08", "description": "$4.4M paid in Bitcoin"},
                    {"phase": "Recovery", "date": "2021-05-12", "description": "Pipeline operations resumed"}
                ],
                detection_methods=[
                    "Unusual VPN login from non-standard location",
                    "SMB lateral movement detection",
                    "Ransomware behavior (mass file encryption)",
                    "Data exfiltration alerts (large outbound transfers)",
                    "Endpoint Detection and Response (EDR) alerts"
                ],
                mitigation_steps=[
                    "Immediately disconnect infected systems from network",
                    "Isolate OT networks from IT networks",
                    "Reset all passwords and enforce MFA across VPN",
                    "Restore from offline backups (test backups first)",
                    "Block known DarkSide C2 infrastructure",
                    "Deploy anti-ransomware tools (behavioral detection)",
                    "Segment networks to limit lateral movement",
                    "Implement privileged access management",
                    "Conduct ransomware tabletop exercises",
                    "Review and update incident response plans"
                ],
                lessons_learned=[
                    "MFA on VPN is critical, especially for OT access",
                    "Network segmentation prevents OT impact from IT compromise",
                    "Offline backups are essential for ransomware recovery",
                    "Early detection reduces dwell time and impact",
                    "OT/IT convergence increases attack surface",
                    "Incident response must consider operational shutdowns",
                    "Paying ransom does not guarantee data deletion"
                ],
                attribution="DarkSide ransomware gang (disbanded after attack)",
                date_discovered=datetime(2021, 5, 7),
                date_reported=datetime(2021, 5, 7),
                source="Colonial Pipeline, FBI, CISA",
                tags=["ransomware", "DarkSide", "critical_infrastructure", "VPN_compromise"],
                cve_ids=[]
            ),

            ThreatIntelligence(
                threat_id="TI-2024-003",
                title="Log4Shell Zero-Day Vulnerability",
                description="Critical remote code execution vulnerability (CVE-2021-44228) in Apache Log4j library, affecting millions of systems worldwide.",
                severity=ThreatSeverity.CRITICAL,
                attack_vectors=[
                    "JNDI injection via user-controllable data",
                    "Remote code execution without authentication",
                    "Exploit via HTTP headers, form inputs, log messages"
                ],
                affected_systems=[
                    "Java applications using Log4j 2.0-2.14.1",
                    "Web servers (Apache Tomcat, etc.)",
                    "Cloud services (AWS, Azure, GCP)",
                    "Enterprise applications (VMware, Cisco, etc.)"
                ],
                indicators_of_compromise=[
                    "JNDI lookup strings in logs: ${jndi:ldap://",
                    "Outbound LDAP/RMI connections on unexpected ports",
                    "Base64 encoded payloads in HTTP headers",
                    "Cryptomining or botnet C2 connections",
                    "User-Agent headers with exploit strings"
                ],
                attack_timeline=[
                    {"phase": "Vulnerability Exists", "date": "2013-09", "description": "Vulnerable code introduced in Log4j 2.0-beta9"},
                    {"phase": "Discovery", "date": "2021-11-24", "description": "Alibaba Cloud Security team discovers vulnerability"},
                    {"phase": "Private Disclosure", "date": "2021-11-26", "description": "Reported to Apache Foundation"},
                    {"phase": "Patch Development", "date": "2021-12-05", "description": "Apache releases Log4j 2.15.0"},
                    {"phase": "Public Disclosure", "date": "2021-12-09", "description": "Vulnerability publicly disclosed"},
                    {"phase": "Mass Exploitation", "date": "2021-12-10", "description": "Widespread scanning and exploitation begins"},
                    {"phase": "Botnet Campaigns", "date": "2021-12-11", "description": "Mirai, Kinsing botnets actively exploiting"},
                    {"phase": "Emergency Patching", "date": "2021-12-13", "description": "Organizations rush to patch systems"}
                ],
                detection_methods=[
                    "WAF rules detecting JNDI injection patterns",
                    "IDS/IPS signatures for Log4Shell exploit attempts",
                    "Anomaly detection for LDAP/RMI outbound connections",
                    "Log analysis for ${jndi:ldap://} patterns",
                    "Vulnerability scanning (Nessus, Qualys, etc.)",
                    "YARA rules for malicious payloads"
                ],
                mitigation_steps=[
                    "Immediate: Upgrade to Log4j 2.17.1 or later",
                    "Workaround: Set log4j2.formatMsgNoLookups=true",
                    "Block outbound LDAP (port 389) and RMI (1099) at firewall",
                    "Deploy WAF rules to block exploitation attempts",
                    "Inventory all Java applications and dependencies",
                    "Monitor for post-exploitation activity (webshells, cryptominers)",
                    "Apply vendor-specific patches for affected products",
                    "Implement Zero Trust network architecture",
                    "Use runtime application self-protection (RASP)"
                ],
                lessons_learned=[
                    "Supply chain vulnerabilities affect countless downstream applications",
                    "Dependency management and SBOMs are critical",
                    "Zero-day response requires rapid coordination",
                    "Default-unsafe configurations amplify risk",
                    "Layered security reduces blast radius",
                    "Vulnerability disclosure timing is complex",
                    "Automated patching processes are essential"
                ],
                attribution="Multiple threat actors exploited (nation-states, cybercriminals, botnet operators)",
                date_discovered=datetime(2021, 11, 24),
                date_reported=datetime(2021, 12, 9),
                source="Apache Foundation, Alibaba Cloud, CISA",
                tags=["log4j", "zero_day", "RCE", "supply_chain", "widespread"],
                cve_ids=["CVE-2021-44228", "CVE-2021-45046", "CVE-2021-45105"]
            ),

            ThreatIntelligence(
                threat_id="TI-2024-004",
                title="Kaseya VSA Supply Chain Ransomware",
                description="REvil ransomware gang exploited Kaseya VSA zero-day to deploy ransomware to ~1,500 downstream businesses via managed service providers.",
                severity=ThreatSeverity.CRITICAL,
                attack_vectors=[
                    "Zero-day SQL injection (CVE-2021-30116)",
                    "Authentication bypass",
                    "Supply chain attack via MSP software",
                    "REvil ransomware deployment"
                ],
                affected_systems=[
                    "Kaseya VSA on-premises servers",
                    "MSP managed client systems",
                    "Swedish Coop supermarkets (800+ stores)",
                    "Schools, government agencies, businesses"
                ],
                indicators_of_compromise=[
                    "Malicious agent.crt file dropped",
                    "REvil ransomware payload: agent.exe",
                    "C2 domains: multiple TOR hidden services",
                    "Encrypted files with random extensions",
                    "Ransom demand: $70M in Bitcoin (later negotiated)"
                ],
                attack_timeline=[
                    {"phase": "Reconnaissance", "date": "2021-04", "description": "REvil identifies Kaseya VSA vulnerabilities"},
                    {"phase": "Weaponization", "date": "2021-05", "description": "Develop exploit chain for zero-days"},
                    {"phase": "Initial Compromise", "date": "2021-07-02 14:00", "description": "Exploit Kaseya VSA servers during US holiday"},
                    {"phase": "Lateral Spread", "date": "2021-07-02 15:00", "description": "Push malicious updates to MSP clients"},
                    {"phase": "Ransomware Deployment", "date": "2021-07-02 16:00", "description": "REvil encrypts ~1,500 organizations"},
                    {"phase": "Discovery", "date": "2021-07-02 17:00", "description": "Kaseyareports incident"},
                    {"phase": "Global Impact", "date": "2021-07-03", "description": "Widespread business disruptions observed"},
                    {"phase": "Decryptor Release", "date": "2021-07-21", "description": "Universal decryptor mysteriously released"}
                ],
                detection_methods=[
                    "Kaseya VSA behavior anomalies",
                    "Unexpected agent.crt file creation",
                    "Mass file encryption patterns",
                    "Suspicious PowerShell executions",
                    "Unauthorized software deployments via VSA"
                ],
                mitigation_steps=[
                    "Immediately shut down Kaseya VSA servers (per vendor guidance)",
                    "Block all VSA traffic at network perimeter",
                    "Apply Kaseya patches when available (VSA 9.5.7a)",
                    "Restore from offline backups",
                    "Hunt for REvil indicators across environment",
                    "Reset all credentials potentially accessible via VSA",
                    "Implement application whitelisting",
                    "Review MSP security practices and third-party risk",
                    "Deploy EDR on all endpoints"
                ],
                lessons_learned=[
                    "MSP compromise has massive downstream impact",
                    "Zero-day attacks can bypass traditional defenses",
                    "Attack timing (holiday weekend) amplifies damage",
                    "Third-party risk extends to software supply chain",
                    "Incident response at scale requires coordination",
                    "Offline backups are last line of defense",
                    "Vendor trust must be continuously validated"
                ],
                attribution="REvil (Sodinokibi) ransomware gang",
                date_discovered=datetime(2021, 7, 2),
                date_reported=datetime(2021, 7, 2),
                source="Kaseya, CISA, FBI",
                tags=["REvil", "supply_chain", "MSP", "zero_day", "ransomware"],
                cve_ids=["CVE-2021-30116"]
            )
        ]

        self.threats.extend(sample_threats)

    def get_recent_threats(self, days: int = 30, severity: Optional[ThreatSeverity] = None) -> List[ThreatIntelligence]:
        """Get threats from last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        threats = [t for t in self.threats if t.date_reported >= cutoff_date]

        if severity:
            threats = [t for t in threats if t.severity == severity]

        return sorted(threats, key=lambda x: x.date_reported, reverse=True)

    def get_threat_by_id(self, threat_id: str) -> Optional[ThreatIntelligence]:
        """Get specific threat by ID"""
        for threat in self.threats:
            if threat.threat_id == threat_id:
                return threat
        return None

    def search_threats(self, query: str) -> List[ThreatIntelligence]:
        """Search threats by keyword"""
        query_lower = query.lower()
        results = []

        for threat in self.threats:
            if (query_lower in threat.title.lower() or
                query_lower in threat.description.lower() or
                any(query_lower in tag for tag in threat.tags)):
                results.append(threat)

        return results

    def get_threat_statistics(self) -> Dict[str, Any]:
        """Get aggregated threat statistics"""
        return {
            "total_threats": len(self.threats),
            "by_severity": {
                severity.value: len([t for t in self.threats if t.severity == severity])
                for severity in ThreatSeverity
            },
            "recent_critical": len(self.get_recent_threats(days=7, severity=ThreatSeverity.CRITICAL)),
            "recent_high": len(self.get_recent_threats(days=7, severity=ThreatSeverity.HIGH)),
            "most_common_vectors": self._get_top_attack_vectors(10),
            "most_targeted_systems": self._get_most_targeted_systems(10)
        }

    def _get_top_attack_vectors(self, limit: int) -> List[Dict[str, Any]]:
        """Get most common attack vectors"""
        vector_counts = {}
        for threat in self.threats:
            for vector in threat.attack_vectors:
                vector_counts[vector] = vector_counts.get(vector, 0) + 1

        sorted_vectors = sorted(vector_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"vector": v, "count": c} for v, c in sorted_vectors[:limit]]

    def _get_most_targeted_systems(self, limit: int) -> List[Dict[str, Any]]:
        """Get most frequently targeted systems"""
        system_counts = {}
        for threat in self.threats:
            for system in threat.affected_systems:
                system_counts[system] = system_counts.get(system, 0) + 1

        sorted_systems = sorted(system_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"system": s, "count": c} for s, c in sorted_systems[:limit]]
