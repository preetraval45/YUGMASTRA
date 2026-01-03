"""
Incident Response Agent
AI-powered incident detection, analysis, and response orchestration
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from models.llm_manager import LLMManager
from services.rag_service import RAGService

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class IncidentResponseAgent:
    """
    AI-powered incident response agent

    Capabilities:
    - Incident detection and classification
    - Root cause analysis
    - Impact assessment
    - Response playbook generation
    - Containment strategy
    - Evidence collection
    - Timeline reconstruction
    - Post-incident analysis
    """

    def __init__(self, llm_manager: LLMManager, rag_service: RAGService):
        self.llm = llm_manager
        self.rag = rag_service
        self.name = "IncidentResponseAgent"
        logger.info(f"Initialized {self.name}")

    async def detect_incident(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect and classify potential security incidents"""

        events_text = "\n".join([
            f"Event {i+1} [{e.get('timestamp', 'N/A')}]: {e.get('description', str(e))}"
            for i, e in enumerate(events)
        ])

        prompt = f"""
        Analyze these security events to detect potential incidents:

        {events_text}

        Determine:
        1. Is this a security incident? (Yes/No/Possible)
        2. Incident type (Malware, Data Breach, DDoS, Insider Threat, etc.)
        3. Severity (Critical/High/Medium/Low/Informational)
        4. Confidence level (0-100%)
        5. Attack phase (Reconnaissance, Initial Access, Execution, Persistence, etc.)
        6. Affected assets
        7. Potential impact
        8. Indicators of compromise
        9. Recommended immediate actions
        10. Escalation requirement

        Provide detailed incident classification.
        """

        context = await self.rag.get_relevant_context("incident detection patterns")

        detection = await self.llm.generate(
            prompt=prompt,
            system="You are a SOC analyst detecting and classifying security incidents.",
            context=context
        )

        return {
            "events_count": len(events),
            "detection": detection,
            "timestamp": datetime.now().isoformat()
        }

    async def perform_root_cause_analysis(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform root cause analysis of an incident"""

        incident_text = str(incident_data)

        prompt = f"""
        Perform root cause analysis for this security incident:

        {incident_text}

        Analyze:
        1. Initial attack vector
        2. Vulnerability exploited
        3. Security control failures
        4. Attack progression timeline
        5. Lateral movement path
        6. Privilege escalation methods
        7. Data exfiltration routes (if applicable)
        8. Detection gaps
        9. Contributing factors
        10. Root cause summary

        Use 5 Whys methodology and attack kill chain analysis.
        """

        context = await self.rag.get_relevant_context("root cause analysis incident response")

        analysis = await self.llm.generate(
            prompt=prompt,
            system="You are an incident response expert performing root cause analysis.",
            context=context
        )

        return {
            "incident": incident_data,
            "root_cause_analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }

    async def assess_impact(self, incident_details: str) -> Dict[str, Any]:
        """Assess the impact of a security incident"""

        prompt = f"""
        Assess the impact of this security incident:

        {incident_details}

        Evaluate:
        1. Confidentiality impact (data exposed, sensitivity level)
        2. Integrity impact (data/system modifications)
        3. Availability impact (service disruption)
        4. Financial impact (estimated cost ranges)
        5. Reputational impact
        6. Legal/compliance implications
        7. Customer impact
        8. Operational impact
        9. Recovery time estimate
        10. Long-term consequences

        Provide impact rating and detailed assessment.
        """

        impact = await self.llm.generate(
            prompt=prompt,
            system="You are an incident response manager assessing incident impact."
        )

        return {
            "incident": incident_details,
            "impact_assessment": impact,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_response_playbook(self,
                                         incident_type: str,
                                         severity: str) -> Dict[str, Any]:
        """Generate incident response playbook"""

        prompt = f"""
        Generate a comprehensive incident response playbook for:

        Incident Type: {incident_type}
        Severity: {severity}

        Playbook should include:

        1. PREPARATION
           - Required tools and access
           - Team roles and responsibilities
           - Communication channels

        2. DETECTION & ANALYSIS
           - Detection methods
           - Analysis procedures
           - Evidence collection

        3. CONTAINMENT
           - Short-term containment
           - Long-term containment
           - System backups

        4. ERADICATION
           - Malware removal
           - Vulnerability patching
           - Access revocation

        5. RECOVERY
           - System restoration
           - Service resumption
           - Verification testing

        6. POST-INCIDENT
           - Lessons learned
           - Documentation
           - Process improvements

        Provide step-by-step procedures with time estimates.
        """

        context = await self.rag.get_relevant_context(f"{incident_type} incident response")

        playbook = await self.llm.generate(
            prompt=prompt,
            system="You are an incident response architect creating response playbooks.",
            context=context
        )

        return {
            "incident_type": incident_type,
            "severity": severity,
            "playbook": playbook,
            "timestamp": datetime.now().isoformat()
        }

    async def recommend_containment(self, incident_info: str) -> Dict[str, Any]:
        """Recommend containment strategies"""

        prompt = f"""
        Recommend containment strategies for this incident:

        {incident_info}

        Provide:
        1. Immediate containment actions (first 15 minutes)
        2. Short-term containment strategy
        3. Long-term containment approach
        4. Network isolation recommendations
        5. Account/access restrictions
        6. System shutdown decisions
        7. Evidence preservation
        8. Business continuity considerations
        9. Risk vs. containment trade-offs
        10. Containment verification

        Prioritize by urgency and effectiveness.
        """

        recommendations = await self.llm.generate(
            prompt=prompt,
            system="You are an incident responder recommending containment strategies."
        )

        return {
            "incident": incident_info,
            "containment_strategy": recommendations,
            "timestamp": datetime.now().isoformat()
        }

    async def collect_evidence(self, incident_id: str, systems: List[str]) -> Dict[str, Any]:
        """Guide evidence collection process"""

        systems_text = "\n".join([f"- {s}" for s in systems])

        prompt = f"""
        Create evidence collection guide for incident {incident_id}:

        Affected Systems:
        {systems_text}

        Provide:
        1. Evidence collection priorities
        2. Volatile data to capture (memory, processes)
        3. Non-volatile data to preserve (logs, files)
        4. Network traffic capture
        5. Forensic imaging procedures
        6. Chain of custody requirements
        7. Tools and commands
        8. Legal considerations
        9. Timeline preservation
        10. Evidence documentation

        Follow forensic best practices and legal requirements.
        """

        context = await self.rag.get_relevant_context("digital forensics evidence collection")

        guide = await self.llm.generate(
            prompt=prompt,
            system="You are a digital forensics expert guiding evidence collection.",
            context=context
        )

        return {
            "incident_id": incident_id,
            "systems": systems,
            "evidence_guide": guide,
            "timestamp": datetime.now().isoformat()
        }

    async def reconstruct_timeline(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Reconstruct incident timeline"""

        events_text = "\n".join([
            f"{e.get('timestamp', 'N/A')}: {e.get('description', str(e))}"
            for e in events
        ])

        prompt = f"""
        Reconstruct the incident timeline from these events:

        {events_text}

        Create:
        1. Chronological timeline
        2. Attack phases identification
        3. Attacker actions
        4. System responses
        5. Detection points
        6. Critical decision moments
        7. Gaps in visibility
        8. Parallel activities
        9. Duration analysis
        10. Visual timeline representation

        Identify the complete attack progression.
        """

        timeline = await self.llm.generate(
            prompt=prompt,
            system="You are an incident analyst reconstructing attack timelines."
        )

        return {
            "events_count": len(events),
            "timeline": timeline,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_incident_report(self,
                                       incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive incident report"""

        incident_text = str(incident_data)

        prompt = f"""
        Generate a comprehensive incident report:

        Incident Data:
        {incident_text}

        Report should include:
        1. Executive Summary
        2. Incident Overview
           - Type, severity, timeline
        3. Detection and Analysis
           - How it was detected
           - Initial assessment
        4. Containment Actions
           - Steps taken
           - Effectiveness
        5. Eradication Measures
           - Root cause elimination
        6. Recovery Process
           - System restoration
           - Validation
        7. Impact Assessment
           - Business impact
           - Technical impact
        8. Root Cause Analysis
        9. Lessons Learned
        10. Recommendations
            - Immediate fixes
            - Long-term improvements
        11. Appendix
            - Technical details
            - Evidence references

        Format as professional incident report.
        """

        report = await self.llm.generate(
            prompt=prompt,
            system="You are a senior incident response manager writing incident reports."
        )

        return {
            "incident": incident_data,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }

    async def recommend_improvements(self,
                                    incident_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend security improvements based on incident"""

        analysis_text = str(incident_analysis)

        prompt = f"""
        Based on this incident analysis, recommend security improvements:

        {analysis_text}

        Provide:
        1. Process improvements
        2. Technology enhancements
        3. Training needs
        4. Policy updates
        5. Detection capability gaps
        6. Response procedure updates
        7. Tool requirements
        8. Staffing recommendations
        9. Budget considerations
        10. Implementation roadmap

        Prioritize by impact and feasibility.
        """

        recommendations = await self.llm.generate(
            prompt=prompt,
            system="You are a security architect recommending improvements after incidents."
        )

        return {
            "analysis": incident_analysis,
            "improvements": recommendations,
            "timestamp": datetime.now().isoformat()
        }
