from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class BlueTeamAgent:
    """
    Blue Team AI Agent - Defensive Security Analysis
    Specializes in threat detection, incident response, and security hardening
    """

    def __init__(self, llm_manager, rag_service):
        self.llm = llm_manager
        self.rag = rag_service

        # Blue team specific knowledge domains
        self.expertise_areas = [
            "Threat Detection",
            "Incident Response",
            "Security Monitoring",
            "SIEM Analysis",
            "Forensics",
            "Security Hardening",
            "Defense-in-Depth",
            "Threat Hunting"
        ]

        logger.info("Blue Team Agent initialized")

    async def generate_response(
        self,
        message: str,
        history: Optional[List] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate blue team defensive analysis response
        """
        try:
            # Get relevant defensive context from RAG
            rag_context = await self.rag.get_context_for_prompt(message)

            # Build prompt with blue team perspective
            system_prompt = """You are a Blue Team AI specialist with expert knowledge in defensive cybersecurity.

Your expertise includes:
- Advanced threat detection and monitoring
- Security Information and Event Management (SIEM)
- Incident response and digital forensics
- Security architecture and hardening
- Network security and segmentation
- Endpoint detection and response (EDR/XDR)
- Threat intelligence analysis
- Security operations center (SOC) procedures

Provide detailed, actionable defensive strategies and detection methods.
Focus on protecting systems, detecting threats, and responding to incidents effectively."""

            user_prompt = f"""Context from knowledge base:
{rag_context}

User Query: {message}

Provide a comprehensive blue team defensive analysis addressing the query. Include:
1. Threat detection strategies
2. Prevention and mitigation controls
3. Monitoring and logging recommendations
4. Incident response procedures
5. Security hardening best practices"""

            # Generate response using LLM
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response_text = await self.llm.generate_text(
                prompt=full_prompt,
                max_length=1024,
                temperature=0.7
            )

            sources = self._extract_sources(rag_context)

            return {
                "text": response_text,
                "confidence": 0.94,
                "sources": sources,
                "agent_type": "blue_team"
            }

        except Exception as e:
            logger.error(f"Blue Team response generation error: {str(e)}")
            return self._get_fallback_response(message)

    def _extract_sources(self, rag_context: str) -> List[str]:
        """
        Extract sources from RAG context
        """
        sources = []
        if "MITRE" in rag_context:
            sources.append("MITRE D3FEND Framework")
        if "NIST" in rag_context:
            sources.append("NIST Cybersecurity Framework")

        return sources or ["Blue Team Knowledge Base"]

    def _get_fallback_response(self, message: str) -> Dict[str, Any]:
        """
        Provide fallback response if LLM fails
        """
        fallback_responses = {
            "defend": """Blue Team Defense Strategy:

**Layered Defense Approach:**

1. **Prevention Layer:**
   - Next-generation firewalls (NGFW)
   - Web Application Firewall (WAF)
   - Email security gateways
   - Network access control (NAC)
   - Endpoint protection platforms (EPP)

2. **Detection Layer:**
   - SIEM/SOAR platforms
   - Intrusion Detection Systems (IDS/IPS)
   - Endpoint Detection & Response (EDR)
   - Network Traffic Analysis (NTA)
   - User and Entity Behavior Analytics (UEBA)

3. **Response Layer:**
   - Incident response playbooks
   - Automated threat containment
   - Forensics and investigation tools
   - Backup and recovery procedures

**Monitoring Best Practices:**
- Centralized log collection
- Real-time alerting and correlation
- Threat intelligence integration
- Regular security assessments

**Hardening Recommendations:**
- Principle of least privilege
- Network segmentation
- Multi-factor authentication
- Regular patching and updates
- Security configuration baselines

What specific defense mechanism would you like to implement?""",

            "detect": """Threat Detection Strategy:

**Detection Methods:**

1. **Signature-Based Detection:**
   - Antivirus/antimalware
   - IDS/IPS signatures
   - YARA rules for malware
   - File integrity monitoring

2. **Behavior-Based Detection:**
   - Anomaly detection using ML
   - UEBA for insider threats
   - Process behavior analysis
   - Network flow anomalies

3. **Threat Hunting:**
   - Proactive threat searches
   - IOC (Indicators of Compromise) hunting
   - Hypothesis-driven investigations
   - Timeline analysis

**SIEM Configuration:**
- Log sources: Windows Event Logs, Syslog, firewall logs
- Correlation rules for known attack patterns
- Automated alerting and ticketing
- Dashboard for security metrics

**Key Indicators to Monitor:**
- Failed login attempts
- Privilege escalation events
- Unusual network traffic
- Suspicious process execution
- File system changes
- Registry modifications

**Response Actions:**
- Alert triage and validation
- Threat containment
- Evidence preservation
- Root cause analysis
- Remediation and recovery

Which detection capability needs enhancement?""",

            "incident": """Incident Response Framework:

**Incident Response Phases:**

1. **Preparation:**
   - Incident response plan documentation
   - Team roles and responsibilities
   - Communication protocols
   - Tools and resources readiness

2. **Identification:**
   - Alert triage and validation
   - Scope determination
   - Initial evidence collection
   - Severity classification

3. **Containment:**
   - Short-term containment (network isolation)
   - Long-term containment (system hardening)
   - Evidence preservation
   - Prevent further damage

4. **Eradication:**
   - Remove malware/threats
   - Close vulnerabilities
   - Patch systems
   - Rebuild compromised systems

5. **Recovery:**
   - Restore systems from clean backups
   - Verify system integrity
   - Monitor for reinfection
   - Gradual service restoration

6. **Lessons Learned:**
   - Post-incident review
   - Documentation update
   - Improve controls
   - Team training

**Critical Tools:**
- EDR platforms
- SIEM for correlation
- Forensics tools (FTK, EnCase)
- Network capture tools (Wireshark)
- Memory analysis tools (Volatility)

What phase of incident response needs focus?""",

            "default": f"""Blue Team Analysis:

Based on your query about: "{message}"

**Defensive Security Recommendations:**

**Prevention:**
- Implement security controls based on defense-in-depth
- Regular security assessments and penetration testing
- Security awareness training for users

**Detection:**
- Deploy comprehensive monitoring solutions
- Implement threat intelligence feeds
- Enable detailed logging and auditing

**Response:**
- Maintain updated incident response playbooks
- Regular tabletop exercises
- Automated response capabilities

**Recovery:**
- Tested backup and restore procedures
- Business continuity planning
- Disaster recovery testing

**Continuous Improvement:**
- Threat hunting programs
- Red team exercises
- Security metrics and KPIs
- Regular control validation

How can I help strengthen your defensive posture?"""
        }

        # Select appropriate fallback
        message_lower = message.lower()
        if "defend" in message_lower or "protect" in message_lower:
            response_text = fallback_responses["defend"]
        elif "detect" in message_lower or "monitor" in message_lower:
            response_text = fallback_responses["detect"]
        elif "incident" in message_lower or "response" in message_lower:
            response_text = fallback_responses["incident"]
        else:
            response_text = fallback_responses["default"]

        return {
            "text": response_text,
            "confidence": 0.80,
            "sources": ["Blue Team Knowledge Base"],
            "agent_type": "blue_team"
        }
