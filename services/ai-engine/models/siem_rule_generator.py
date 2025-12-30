"""
SIEM Rule Generator - Auto-generate detection rules from attack patterns
Supports: Sigma, Splunk, Elastic, Suricata, Snort
Uses: Template-based generation + LLM enhancement (FREE models only)
"""

import logging
import json
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class RuleFormat(str, Enum):
    SIGMA = "sigma"
    SPLUNK = "splunk"
    ELASTIC = "elastic"
    SURICATA = "suricata"
    SNORT = "snort"
    YARA = "yara"

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "informational"

@dataclass
class SIEMRule:
    id: str
    title: str
    description: str
    severity: Severity
    format: RuleFormat
    rule_content: str
    tags: List[str]
    mitre_attack: List[str]
    references: List[str]
    created_at: datetime
    author: str = "YUGMASTRA AI"
    false_positives: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.false_positives is None:
            self.false_positives = []
        if self.metadata is None:
            self.metadata = {}

class SigmaRuleGenerator:
    """Generate Sigma rules (universal SIEM format)"""

    @staticmethod
    def generate(attack_pattern: Dict[str, Any]) -> str:
        """
        Generate Sigma rule from attack pattern

        Sigma Format:
        - title, description, status, level
        - logsource (category, product, service)
        - detection (selection, condition, filter)
        - falsepositives, tags, references
        """
        rule = {
            "title": attack_pattern.get("name", "Unknown Attack"),
            "id": attack_pattern.get("id", ""),
            "description": attack_pattern.get("description", ""),
            "status": "experimental",
            "author": "YUGMASTRA AI",
            "date": datetime.now().strftime("%Y/%m/%d"),
            "level": SigmaRuleGenerator._map_severity(attack_pattern.get("severity", "medium")),
            "logsource": SigmaRuleGenerator._generate_logsource(attack_pattern),
            "detection": SigmaRuleGenerator._generate_detection(attack_pattern),
            "falsepositives": attack_pattern.get("false_positives", ["Unknown"]),
            "tags": attack_pattern.get("mitre_attack", []),
        }

        if attack_pattern.get("references"):
            rule["references"] = attack_pattern["references"]

        return yaml.dump(rule, default_flow_style=False, sort_keys=False)

    @staticmethod
    def _map_severity(severity: str) -> str:
        """Map severity to Sigma level"""
        severity_map = {
            "critical": "critical",
            "high": "high",
            "medium": "medium",
            "low": "low",
            "info": "informational"
        }
        return severity_map.get(severity.lower(), "medium")

    @staticmethod
    def _generate_logsource(attack_pattern: Dict[str, Any]) -> Dict[str, str]:
        """Generate logsource section"""
        attack_type = attack_pattern.get("type", "").lower()

        # Map attack types to log sources
        if "network" in attack_type or "port_scan" in attack_type:
            return {"category": "network_connection", "product": "firewall"}
        elif "web" in attack_type or "injection" in attack_type:
            return {"category": "web", "product": "web_server"}
        elif "process" in attack_type or "execution" in attack_type:
            return {"category": "process_creation", "product": "windows"}
        elif "file" in attack_type:
            return {"category": "file_event", "product": "windows"}
        else:
            return {"category": "generic", "product": "linux"}

    @staticmethod
    def _generate_detection(attack_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detection section"""
        indicators = attack_pattern.get("indicators", {})
        attack_type = attack_pattern.get("type", "").lower()

        selection = {}

        # Generate selection based on indicators
        if indicators.get("ip_addresses"):
            selection["DestinationIp"] = indicators["ip_addresses"]

        if indicators.get("ports"):
            selection["DestinationPort"] = indicators["ports"]

        if indicators.get("process_names"):
            selection["Image|endswith"] = indicators["process_names"]

        if indicators.get("command_line"):
            selection["CommandLine|contains"] = indicators["command_line"]

        if indicators.get("file_hashes"):
            selection["Hashes|contains"] = indicators["file_hashes"]

        if indicators.get("registry_keys"):
            selection["TargetObject|contains"] = indicators["registry_keys"]

        # Default selection if no indicators
        if not selection:
            if "injection" in attack_type:
                selection = {
                    "CommandLine|contains": ["' OR", "1=1", "<script>", "../../"]
                }
            elif "scan" in attack_type:
                selection = {
                    "DestinationPort": [20, 21, 22, 23, 80, 443, 3389, 8080]
                }
            else:
                selection = {"EventID": 1}

        return {
            "selection": selection,
            "condition": "selection"
        }

class SplunkRuleGenerator:
    """Generate Splunk SPL queries"""

    @staticmethod
    def generate(attack_pattern: Dict[str, Any]) -> str:
        """Generate Splunk SPL search query"""
        indicators = attack_pattern.get("indicators", {})
        attack_type = attack_pattern.get("type", "").lower()

        # Base search
        search_parts = []

        # Add index and sourcetype
        search_parts.append("index=* sourcetype=*")

        # Add conditions based on indicators
        if indicators.get("ip_addresses"):
            ips = " OR ".join([f'"{ip}"' for ip in indicators["ip_addresses"]])
            search_parts.append(f"(dest_ip IN ({ips}) OR src_ip IN ({ips}))")

        if indicators.get("ports"):
            ports = " OR ".join([str(p) for p in indicators["ports"]])
            search_parts.append(f"(dest_port IN ({ports}))")

        if indicators.get("process_names"):
            procs = " OR ".join([f'process_name="*{p}*"' for p in indicators["process_names"]])
            search_parts.append(f"({procs})")

        if indicators.get("command_line"):
            cmds = " OR ".join([f'command_line="*{c}*"' for c in indicators["command_line"]])
            search_parts.append(f"({cmds})")

        # Default patterns
        if len(search_parts) == 1:  # Only base search
            if "injection" in attack_type:
                search_parts.append('(uri="*\'*" OR uri="*<script>*" OR uri="*union*select*")')
            elif "scan" in attack_type:
                search_parts.append("| stats dc(dest_port) as ports by src_ip | where ports > 10")

        # Add stats and alerts
        spl = " ".join(search_parts)
        spl += f'\n| stats count by src_ip, dest_ip, dest_port, user\n| where count > 5'
        spl += f'\n| eval severity="{attack_pattern.get("severity", "medium")}"'
        spl += f'\n| eval description="{attack_pattern.get("description", "")}"'

        return spl

class ElasticRuleGenerator:
    """Generate Elasticsearch/Kibana detection rules"""

    @staticmethod
    def generate(attack_pattern: Dict[str, Any]) -> str:
        """Generate Elastic EQL or KQL query"""
        indicators = attack_pattern.get("indicators", {})
        attack_type = attack_pattern.get("type", "").lower()

        query_parts = []

        # Build KQL query
        if indicators.get("ip_addresses"):
            ips = " or ".join([f'destination.ip:"{ip}"' for ip in indicators["ip_addresses"]])
            query_parts.append(f"({ips})")

        if indicators.get("ports"):
            ports = " or ".join([f"destination.port:{p}" for p in indicators["ports"]])
            query_parts.append(f"({ports})")

        if indicators.get("process_names"):
            procs = " or ".join([f'process.name:*{p}*' for p in indicators["process_names"]])
            query_parts.append(f"({procs})")

        if indicators.get("command_line"):
            cmds = " or ".join([f'process.command_line:*{c}*' for c in indicators["command_line"]])
            query_parts.append(f"({cmds})")

        # Default patterns
        if not query_parts:
            if "injection" in attack_type:
                query_parts.append('(url.query:*"OR 1=1"* or url.query:*<script>*)')
            elif "scan" in attack_type:
                query_parts.append("event.category:network and event.action:connection_attempted")

        kql_query = " and ".join(query_parts) if query_parts else "event.category:*"

        # Create full rule in Kibana format
        rule = {
            "name": attack_pattern.get("name", "Unknown Attack"),
            "description": attack_pattern.get("description", ""),
            "risk_score": ElasticRuleGenerator._map_risk_score(attack_pattern.get("severity", "medium")),
            "severity": attack_pattern.get("severity", "medium"),
            "type": "query",
            "query": kql_query,
            "language": "kuery",
            "index": ["logs-*", "filebeat-*", "winlogbeat-*"],
            "interval": "5m",
            "tags": attack_pattern.get("mitre_attack", []),
            "enabled": True,
            "from": "now-6m",
            "to": "now"
        }

        return json.dumps(rule, indent=2)

    @staticmethod
    def _map_risk_score(severity: str) -> int:
        """Map severity to Elastic risk score (0-100)"""
        score_map = {
            "critical": 90,
            "high": 70,
            "medium": 50,
            "low": 30,
            "info": 10
        }
        return score_map.get(severity.lower(), 50)

class SuricataRuleGenerator:
    """Generate Suricata IDS rules"""

    @staticmethod
    def generate(attack_pattern: Dict[str, Any]) -> str:
        """
        Generate Suricata rule

        Format: action protocol src_ip src_port -> dest_ip dest_port (options)
        """
        indicators = attack_pattern.get("indicators", {})
        attack_type = attack_pattern.get("type", "").lower()

        # Determine action
        action = "alert"

        # Determine protocol
        protocol = "tcp"
        if "udp" in attack_type or indicators.get("protocol") == "udp":
            protocol = "udp"
        elif "icmp" in attack_type:
            protocol = "icmp"

        # Source and destination
        src_ip = "any"
        src_port = "any"
        dest_ip = "any"
        dest_port = "any"

        if indicators.get("ports"):
            dest_port = f"[{','.join(map(str, indicators['ports']))}]"

        if indicators.get("ip_addresses"):
            dest_ip = f"[{','.join(indicators['ip_addresses'])}]"

        # Build options
        options = []

        # Message
        msg = attack_pattern.get("name", "Suspicious Activity Detected")
        options.append(f'msg:"{msg}"')

        # Content matching
        if indicators.get("payload_patterns"):
            for pattern in indicators["payload_patterns"][:3]:  # Limit to 3 patterns
                options.append(f'content:"{pattern}"')

        if indicators.get("http_uri"):
            for uri in indicators["http_uri"][:2]:
                options.append(f'http.uri; content:"{uri}"')

        # Classification
        classtype = SuricataRuleGenerator._map_classtype(attack_type)
        options.append(f"classtype:{classtype}")

        # Severity
        sid = abs(hash(attack_pattern.get("id", msg))) % 10000000 + 1000000
        options.append(f"sid:{sid}")
        options.append("rev:1")

        # Reference to MITRE
        if attack_pattern.get("mitre_attack"):
            for technique in attack_pattern["mitre_attack"][:2]:
                options.append(f'reference:url,attack.mitre.org/techniques/{technique}')

        options_str = "; ".join(options) + ";"

        rule = f"{action} {protocol} {src_ip} {src_port} -> {dest_ip} {dest_port} ({options_str})"

        return rule

    @staticmethod
    def _map_classtype(attack_type: str) -> str:
        """Map attack type to Suricata classtype"""
        if "injection" in attack_type:
            return "web-application-attack"
        elif "scan" in attack_type:
            return "network-scan"
        elif "exploit" in attack_type:
            return "attempted-admin"
        elif "malware" in attack_type:
            return "trojan-activity"
        elif "ddos" in attack_type:
            return "denial-of-service"
        else:
            return "suspicious-traffic"

class SIEMRuleGeneratorEngine:
    """
    Main SIEM Rule Generator Engine
    Supports multiple SIEM formats
    """

    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager
        self.generated_rules = []

        # Initialize format generators
        self.generators = {
            RuleFormat.SIGMA: SigmaRuleGenerator(),
            RuleFormat.SPLUNK: SplunkRuleGenerator(),
            RuleFormat.ELASTIC: ElasticRuleGenerator(),
            RuleFormat.SURICATA: SuricataRuleGenerator(),
        }

        logger.info("✅ SIEM Rule Generator Engine initialized")

    async def generate_rule(
        self,
        attack_pattern: Dict[str, Any],
        rule_format: RuleFormat = RuleFormat.SIGMA,
        enhance_with_llm: bool = False
    ) -> SIEMRule:
        """
        Generate SIEM rule from attack pattern

        Args:
            attack_pattern: Attack pattern dictionary with indicators
            rule_format: Target SIEM format
            enhance_with_llm: Use LLM to enhance rule (optional)
        """
        try:
            # Generate base rule
            generator = self.generators.get(rule_format)
            if not generator:
                raise ValueError(f"Unsupported rule format: {rule_format}")

            rule_content = generator.generate(attack_pattern)

            # Optionally enhance with LLM
            if enhance_with_llm and self.llm_manager:
                rule_content = await self._enhance_with_llm(rule_content, attack_pattern, rule_format)

            # Create SIEMRule object
            rule = SIEMRule(
                id=f"{rule_format.value}_{attack_pattern.get('id', 'unknown')}",
                title=attack_pattern.get("name", "Unknown Attack Detection"),
                description=attack_pattern.get("description", ""),
                severity=Severity(attack_pattern.get("severity", "medium").lower()),
                format=rule_format,
                rule_content=rule_content,
                tags=attack_pattern.get("tags", []),
                mitre_attack=attack_pattern.get("mitre_attack", []),
                references=attack_pattern.get("references", []),
                created_at=datetime.now(),
                false_positives=attack_pattern.get("false_positives", []),
                metadata=attack_pattern.get("metadata", {})
            )

            self.generated_rules.append(rule)
            logger.info(f"✅ Generated {rule_format.value} rule: {rule.title}")

            return rule

        except Exception as e:
            logger.error(f"Failed to generate rule: {e}")
            raise

    async def _enhance_with_llm(
        self,
        base_rule: str,
        attack_pattern: Dict[str, Any],
        rule_format: RuleFormat
    ) -> str:
        """Use LLM to enhance and optimize the rule"""
        if not self.llm_manager:
            return base_rule

        try:
            prompt = f"""
You are a cybersecurity expert specializing in SIEM rule creation.
Enhance and optimize the following {rule_format.value} detection rule.

Attack Pattern: {attack_pattern.get('name', 'Unknown')}
Description: {attack_pattern.get('description', 'N/A')}

Current Rule:
{base_rule}

Please improve the rule by:
1. Adding more specific detection logic
2. Reducing false positives
3. Improving performance
4. Adding relevant context

Return ONLY the improved rule, no explanations.
"""

            response = await self.llm_manager.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )

            enhanced_rule = response.get("text", base_rule).strip()
            logger.info("✅ Rule enhanced with LLM")

            return enhanced_rule

        except Exception as e:
            logger.warning(f"LLM enhancement failed, using base rule: {e}")
            return base_rule

    async def batch_generate_rules(
        self,
        attack_patterns: List[Dict[str, Any]],
        formats: List[RuleFormat] = None
    ) -> Dict[str, List[SIEMRule]]:
        """
        Generate rules in multiple formats for multiple attack patterns

        Returns:
            Dictionary mapping format to list of rules
        """
        if formats is None:
            formats = [RuleFormat.SIGMA, RuleFormat.SPLUNK, RuleFormat.ELASTIC, RuleFormat.SURICATA]

        results = {fmt: [] for fmt in formats}

        for pattern in attack_patterns:
            for fmt in formats:
                try:
                    rule = await self.generate_rule(pattern, fmt)
                    results[fmt].append(rule)
                except Exception as e:
                    logger.error(f"Failed to generate {fmt} rule for {pattern.get('name')}: {e}")

        logger.info(f"✅ Batch generated {sum(len(r) for r in results.values())} rules")
        return results

    def get_rules(
        self,
        rule_format: Optional[RuleFormat] = None,
        severity: Optional[Severity] = None
    ) -> List[SIEMRule]:
        """Get generated rules with filtering"""
        rules = self.generated_rules

        if rule_format:
            rules = [r for r in rules if r.format == rule_format]

        if severity:
            rules = [r for r in rules if r.severity == severity]

        return rules

    def export_rule(self, rule: SIEMRule, file_path: str):
        """Export rule to file"""
        try:
            with open(file_path, 'w') as f:
                f.write(rule.rule_content)

            logger.info(f"✅ Rule exported to {file_path}")

        except Exception as e:
            logger.error(f"Failed to export rule: {e}")
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        if not self.generated_rules:
            return {
                "total_rules": 0,
                "by_format": {},
                "by_severity": {}
            }

        by_format = {}
        by_severity = {}

        for rule in self.generated_rules:
            by_format[rule.format.value] = by_format.get(rule.format.value, 0) + 1
            by_severity[rule.severity.value] = by_severity.get(rule.severity.value, 0) + 1

        return {
            "total_rules": len(self.generated_rules),
            "by_format": by_format,
            "by_severity": by_severity,
            "most_common_format": max(by_format, key=by_format.get) if by_format else None,
            "most_common_severity": max(by_severity, key=by_severity.get) if by_severity else None
        }
