from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EvolutionAgent:
    """
    Evolution AI Agent - Adaptive Threat Intelligence
    Combines red and blue team perspectives with evolving threat analysis
    """

    def __init__(self, llm_manager, rag_service, vector_store):
        self.llm = llm_manager
        self.rag = rag_service
        self.vector_store = vector_store

        # Evolution agent learns from both offensive and defensive perspectives
        self.expertise_areas = [
            "Emerging Threats",
            "Threat Intelligence",
            "Attack Pattern Evolution",
            "Adaptive Defenses",
            "Zero-Day Analysis",
            "APT Campaigns",
            "Threat Landscape Trends",
            "Predictive Security"
        ]

        logger.info("Evolution Agent initialized")

    async def generate_response(
        self,
        message: str,
        history: Optional[List] = None,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate adaptive intelligence response combining offensive and defensive insights
        """
        try:
            # Get context from multiple sources
            rag_context = await self.rag.get_context_for_prompt(message)
            vector_results = await self.vector_store.search_all_indices(message, top_k=2)

            # Combine contexts
            combined_context = self._combine_contexts(rag_context, vector_results)

            # Build prompt with evolution perspective
            system_prompt = """You are an Evolution AI with adaptive intelligence in cybersecurity.

Your capabilities include:
- Analyzing emerging threats and attack patterns
- Predicting future threat landscapes
- Combining offensive and defensive perspectives
- Identifying zero-day vulnerabilities and exploits
- Tracking APT (Advanced Persistent Threat) campaigns
- Providing strategic security intelligence
- Recommending adaptive defense strategies

You learn from both red team (offensive) and blue team (defensive) knowledge to provide
comprehensive, forward-looking security insights. Your responses should be strategic,
predictive, and actionable."""

            user_prompt = f"""Context from knowledge base:
{combined_context}

User Query: {message}

Provide an adaptive intelligence analysis addressing the query. Include:
1. Current threat landscape analysis
2. Emerging attack trends and techniques
3. Predictive threat modeling
4. Strategic defensive recommendations
5. Real-world threat actor TTPs (if relevant)
6. Future-proofing strategies"""

            # Generate response using LLM
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response_text = await self.llm.generate_text(
                prompt=full_prompt,
                max_length=1024,
                temperature=0.75
            )

            sources = self._extract_sources(combined_context, vector_results)

            return {
                "text": response_text,
                "confidence": 0.96,
                "sources": sources,
                "agent_type": "evolution"
            }

        except Exception as e:
            logger.error(f"Evolution response generation error: {str(e)}")
            return self._get_fallback_response(message)

    def _combine_contexts(self, rag_context: str, vector_results: Dict) -> str:
        """
        Combine RAG and vector store contexts
        """
        context_parts = [rag_context]

        for index_type, results in vector_results.items():
            if results:
                context_parts.append(f"\n--- {index_type.upper()} ---")
                for result in results[:2]:  # Top 2 from each index
                    context_parts.append(result["document"])

        return "\n\n".join(context_parts)

    def _extract_sources(self, combined_context: str, vector_results: Dict) -> List[str]:
        """
        Extract sources from combined context
        """
        sources = []

        if "MITRE" in combined_context:
            sources.append("MITRE ATT&CK Framework")
        if "CVE" in combined_context:
            sources.append("CVE Database")

        # Add vector store sources
        for index_type in vector_results.keys():
            if vector_results[index_type]:
                sources.append(f"{index_type.title()} Intelligence")

        return sources or ["Adaptive Intelligence Database"]

    def _get_fallback_response(self, message: str) -> Dict[str, Any]:
        """
        Provide fallback response if LLM fails
        """
        fallback_responses = {
            "threat": """Evolution AI - Threat Landscape Analysis:

**Current Threat Landscape (2025):**

**Emerging Threats:**
1. **AI-Powered Attacks:**
   - LLM-generated phishing (highly convincing)
   - Deepfake social engineering
   - Automated vulnerability discovery
   - Adaptive malware using ML

2. **Supply Chain Attacks:**
   - Software supply chain compromise
   - Third-party dependency exploitation
   - CI/CD pipeline attacks
   - Open-source package poisoning

3. **Cloud-Native Threats:**
   - Container escape exploits
   - Kubernetes misconfigurations
   - Serverless function abuse
   - Cloud API exploitation

4. **Ransomware Evolution:**
   - Double/triple extortion
   - Ransomware-as-a-Service (RaaS)
   - Targeted critical infrastructure
   - Cryptocurrency-based payment systems

**APT Activity Trends:**
- Nation-state actors targeting supply chains
- Increased focus on zero-day exploitation
- Living-off-the-land (LotL) techniques
- Emphasis on data exfiltration over encryption

**Defensive Evolution:**
- Zero Trust Architecture adoption
- AI-powered threat detection
- Deception technologies (honeypots, canaries)
- Continuous verification and validation

**Predictions:**
- Quantum computing threats to encryption
- IoT/OT convergence risks
- 5G network security challenges
- AI vs AI cybersecurity warfare

**Strategic Recommendations:**
- Implement adaptive security controls
- Invest in threat intelligence programs
- Develop AI-augmented security operations
- Prepare for post-quantum cryptography

What specific threat trend concerns you most?""",

            "vulnerability": """Evolution AI - Vulnerability Landscape:

**Zero-Day Trends:**

**Recent Patterns:**
1. **Web Application Frameworks:**
   - Log4Shell-style vulnerabilities
   - Deserialization flaws
   - SSRF in cloud services
   - Authentication bypasses

2. **Infrastructure:**
   - VPN/Remote access vulnerabilities
   - Container runtime exploits
   - Hypervisor escape techniques
   - Network device backdoors

3. **Supply Chain:**
   - Dependency confusion attacks
   - Typosquatting in package managers
   - Compromised build tools
   - Backdoored libraries

**Vulnerability Evolution:**
- Complexity increasing in cloud environments
- Chaining multiple vulnerabilities
- Exploiting trust relationships
- Targeting zero-trust gaps

**Detection Strategies:**
- Continuous vulnerability scanning
- Threat hunting for exploitation indicators
- Behavioral analytics for anomaly detection
- Honeypot/canary deployments

**Patch Management Evolution:**
- Risk-based prioritization (CVSS + EPSS)
- Virtual patching via WAF/IPS
- Automated patch deployment
- Vulnerability disclosure coordination

**Future Considerations:**
- AI-discovered vulnerabilities
- Quantum-resistant cryptography needs
- IoT firmware vulnerability explosion
- Hardware-level exploits (Spectre/Meltdown variants)

Which vulnerability class needs immediate attention?""",

            "intelligence": """Evolution AI - Strategic Intelligence:

**Threat Intelligence Fusion:**

**Intelligence Sources:**
1. **Open Source Intelligence (OSINT):**
   - Public vulnerability databases
   - Security researcher disclosures
   - Dark web monitoring
   - Social media threat indicators

2. **Technical Intelligence:**
   - Malware analysis and reverse engineering
   - Network traffic pattern analysis
   - Endpoint telemetry data
   - Honeypot observations

3. **Human Intelligence (HUMINT):**
   - Industry information sharing (ISACs)
   - Security community collaboration
   - Insider threat indicators
   - Social engineering attempt reports

**Threat Actor Tracking:**
- APT groups and their TTPs
- Cybercriminal syndicates
- Hacktivist collectives
- Insider threat profiles

**Predictive Analytics:**
- Attack pattern forecasting
- Vulnerability trend analysis
- Threat actor capability assessment
- Industry-specific risk modeling

**Intelligence-Driven Defense:**
- IOC (Indicators of Compromise) integration
- TTP-based detection rules
- Threat hunting hypotheses
- Proactive security posture adjustments

**Evolution Cycle:**
1. Collect intelligence from multiple sources
2. Analyze and correlate threat data
3. Predict emerging threats and trends
4. Adapt defensive strategies
5. Share intelligence with community
6. Measure effectiveness and iterate

**Strategic Initiatives:**
- Establish threat intelligence platform
- Build cyber threat intelligence team
- Participate in information sharing
- Develop predictive security models

What intelligence aspect would you like to enhance?""",

            "default": f"""Evolution AI - Adaptive Security Analysis:

Based on your query: "{message}"

**Adaptive Intelligence Assessment:**

**Current State Analysis:**
- Threat landscape is constantly evolving
- Attackers are becoming more sophisticated
- Traditional defenses are increasingly bypassed
- Zero-day exploits are more common

**Emerging Patterns:**
1. **Attack Evolution:**
   - Automation and AI augmentation
   - Multi-stage, low-and-slow campaigns
   - Living-off-the-land techniques
   - Supply chain targeting

2. **Defense Evolution:**
   - AI/ML-powered detection
   - Zero Trust Architecture
   - Deception technologies
   - Automated response systems

**Strategic Recommendations:**

**Immediate Actions:**
- Assess current security posture
- Identify critical assets and threats
- Implement threat intelligence feeds
- Enhance monitoring and detection

**Medium-Term:**
- Adopt adaptive security architecture
- Develop threat hunting capabilities
- Implement SOAR platforms
- Regular red team exercises

**Long-Term:**
- Build predictive security models
- Invest in AI-powered defenses
- Prepare for quantum threats
- Continuous security evolution

**Key Principles:**
- Assume breach mentality
- Defense in depth
- Continuous adaptation
- Intelligence-driven security

How can I help you adapt to evolving threats?"""
        }

        # Select appropriate fallback
        message_lower = message.lower()
        if "threat" in message_lower or "attack" in message_lower:
            response_text = fallback_responses["threat"]
        elif "vulnerability" in message_lower or "vuln" in message_lower:
            response_text = fallback_responses["vulnerability"]
        elif "intelligence" in message_lower or "intel" in message_lower:
            response_text = fallback_responses["intelligence"]
        else:
            response_text = fallback_responses["default"]

        return {
            "text": response_text,
            "confidence": 0.85,
            "sources": ["Adaptive Intelligence Database"],
            "agent_type": "evolution"
        }
