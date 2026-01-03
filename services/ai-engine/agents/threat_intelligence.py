"""
Threat Intelligence Agent
Analyzes threat patterns, IOCs, TTPs, and provides actionable intelligence
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from models.llm_manager import LLMManager
from services.rag_service import RAGService

logger = logging.getLogger(__name__)


class ThreatIntelligenceAgent:
    """
    Advanced threat intelligence analysis agent

    Capabilities:
    - IOC (Indicators of Compromise) analysis
    - TTP (Tactics, Techniques, Procedures) mapping to MITRE ATT&CK
    - Threat actor profiling
    - Malware family identification
    - Campaign correlation
    - Threat scoring and prioritization
    """

    def __init__(self, llm_manager: LLMManager, rag_service: RAGService):
        self.llm = llm_manager
        self.rag = rag_service
        self.name = "ThreatIntelligenceAgent"
        logger.info(f"Initialized {self.name}")

    async def analyze_ioc(self, ioc: str, ioc_type: str) -> Dict[str, Any]:
        """Analyze an Indicator of Compromise"""

        prompt = f"""
        Analyze the following {ioc_type} Indicator of Compromise (IOC):

        IOC: {ioc}
        Type: {ioc_type}

        Provide detailed analysis including:
        1. Threat severity (Critical/High/Medium/Low)
        2. Known associations (malware families, threat actors, campaigns)
        3. MITRE ATT&CK TTPs
        4. Recommended actions
        5. Related IOCs to look for

        Format as structured analysis.
        """

        # Get context from RAG
        context = await self.rag.get_relevant_context(f"{ioc_type} {ioc}")

        # Generate analysis
        analysis = await self.llm.generate(
            prompt=prompt,
            system="You are a threat intelligence expert analyzing IOCs for cybersecurity defense.",
            context=context
        )

        return {
            "ioc": ioc,
            "ioc_type": ioc_type,
            "analysis": analysis,
            "severity": self._extract_severity(analysis),
            "mitre_ttps": self._extract_ttps(analysis),
            "timestamp": datetime.now().isoformat()
        }

    async def map_to_mitre(self, attack_description: str) -> Dict[str, Any]:
        """Map attack behavior to MITRE ATT&CK framework"""

        prompt = f"""
        Map the following attack behavior to MITRE ATT&CK framework:

        Attack Description: {attack_description}

        Provide:
        1. Tactics (high-level objectives)
        2. Techniques (how objectives are achieved)
        3. Sub-techniques (specific implementations)
        4. Procedure examples
        5. Detection recommendations
        6. Mitigation strategies

        Use official MITRE ATT&CK IDs (e.g., T1566, T1059).
        """

        mapping = await self.llm.generate(
            prompt=prompt,
            system="You are a MITRE ATT&CK expert mapping attack behaviors to the framework."
        )

        return {
            "description": attack_description,
            "mitre_mapping": mapping,
            "ttps": self._extract_ttps(mapping),
            "timestamp": datetime.now().isoformat()
        }

    async def profile_threat_actor(self, actor_info: str) -> Dict[str, Any]:
        """Profile a threat actor based on available information"""

        prompt = f"""
        Create a comprehensive threat actor profile based on:

        {actor_info}

        Profile should include:
        1. Attribution confidence level
        2. Known aliases
        3. Motivation (Nation-state, Financial, Hacktivism, etc.)
        4. Sophistication level
        5. Target industries/sectors
        6. Preferred TTPs
        7. Known tools and malware
        8. Historical campaigns
        9. Indicators of their operations
        10. Defensive recommendations
        """

        context = await self.rag.get_relevant_context(actor_info)

        profile = await self.llm.generate(
            prompt=prompt,
            system="You are a threat intelligence analyst specializing in threat actor profiling.",
            context=context
        )

        return {
            "actor_info": actor_info,
            "profile": profile,
            "timestamp": datetime.now().isoformat()
        }

    async def identify_malware(self, malware_data: str) -> Dict[str, Any]:
        """Identify malware family from behavior or code analysis"""

        prompt = f"""
        Identify the malware family based on the following information:

        {malware_data}

        Provide:
        1. Most likely malware family/variant
        2. Confidence level (0-100%)
        3. Key characteristics that match
        4. Known variants
        5. Typical infection vectors
        6. Persistence mechanisms
        7. C2 communication patterns
        8. Data exfiltration methods
        9. Removal/remediation steps
        10. YARA rules for detection
        """

        context = await self.rag.get_relevant_context(f"malware {malware_data}")

        identification = await self.llm.generate(
            prompt=prompt,
            system="You are a malware analyst identifying malware families and variants.",
            context=context
        )

        return {
            "malware_data": malware_data,
            "identification": identification,
            "timestamp": datetime.now().isoformat()
        }

    async def correlate_campaign(self, incidents: List[str]) -> Dict[str, Any]:
        """Correlate multiple incidents to identify potential campaigns"""

        incidents_text = "\n".join([f"Incident {i+1}: {inc}" for i, inc in enumerate(incidents)])

        prompt = f"""
        Analyze the following security incidents and determine if they're part of a coordinated campaign:

        {incidents_text}

        Provide:
        1. Campaign correlation assessment (Yes/No/Possible)
        2. Confidence level
        3. Common indicators across incidents
        4. Timeline analysis
        5. Attack pattern similarities
        6. Possible threat actor attribution
        7. Campaign objectives
        8. Predicted next targets
        9. Recommended defensive actions
        """

        correlation = await self.llm.generate(
            prompt=prompt,
            system="You are a threat intelligence analyst correlating security incidents into campaigns."
        )

        return {
            "incidents_count": len(incidents),
            "correlation": correlation,
            "timestamp": datetime.now().isoformat()
        }

    async def prioritize_threats(self, threats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize threats based on risk assessment"""

        threats_text = "\n".join([
            f"Threat {i+1}: {t.get('description', str(t))}"
            for i, t in enumerate(threats)
        ])

        prompt = f"""
        Prioritize the following threats based on risk to the organization:

        {threats_text}

        For each threat, provide:
        1. Risk score (0-100)
        2. Impact level (Critical/High/Medium/Low)
        3. Likelihood (Very High/High/Medium/Low/Very Low)
        4. Urgency (Immediate/High/Medium/Low)
        5. Recommended action priority
        6. Justification

        Order threats from highest to lowest priority.
        """

        prioritization = await self.llm.generate(
            prompt=prompt,
            system="You are a risk analyst prioritizing security threats."
        )

        return {
            "threats_count": len(threats),
            "prioritization": prioritization,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_intelligence_report(self,
                                           period: str,
                                           events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive threat intelligence report"""

        events_summary = "\n".join([
            f"- {e.get('type', 'Event')}: {e.get('description', str(e))}"
            for e in events
        ])

        prompt = f"""
        Generate a comprehensive threat intelligence report for {period}:

        Security Events:
        {events_summary}

        Report should include:
        1. Executive Summary
        2. Threat Landscape Overview
        3. Key Findings
        4. Attack Trends
        5. Emerging Threats
        6. Industry-Specific Threats
        7. Top Threat Actors
        8. Critical Vulnerabilities
        9. Recommended Actions
        10. Predictions for Next Period

        Format as professional intelligence report.
        """

        report = await self.llm.generate(
            prompt=prompt,
            system="You are a senior threat intelligence analyst writing executive reports."
        )

        return {
            "period": period,
            "events_count": len(events),
            "report": report,
            "timestamp": datetime.now().isoformat()
        }

    def _extract_severity(self, text: str) -> str:
        """Extract severity from analysis text"""
        text_lower = text.lower()
        if "critical" in text_lower:
            return "Critical"
        elif "high" in text_lower:
            return "High"
        elif "medium" in text_lower:
            return "Medium"
        elif "low" in text_lower:
            return "Low"
        return "Unknown"

    def _extract_ttps(self, text: str) -> List[str]:
        """Extract MITRE ATT&CK TTP IDs from text"""
        import re
        # Match patterns like T1566, T1059.001
        pattern = r'T\d{4}(?:\.\d{3})?'
        return list(set(re.findall(pattern, text)))
