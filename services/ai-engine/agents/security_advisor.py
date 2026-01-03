"""
Security Advisor Agent
AI-powered security consulting and strategic guidance
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from models.llm_manager import LLMManager
from services.rag_service import RAGService

logger = logging.getLogger(__name__)


class SecurityAdvisorAgent:
    """
    AI-powered security advisor

    Capabilities:
    - Security architecture review
    - Compliance guidance
    - Risk assessment
    - Security roadmap planning
    - Best practices recommendations
    - Technology selection
    - Security maturity assessment
    - Strategic planning
    """

    def __init__(self, llm_manager: LLMManager, rag_service: RAGService):
        self.llm = llm_manager
        self.rag = rag_service
        self.name = "SecurityAdvisorAgent"
        logger.info(f"Initialized {self.name}")

    async def review_architecture(self, architecture: str) -> Dict[str, Any]:
        """Review security architecture"""

        prompt = f"""
        Review this security architecture:

        {architecture}

        Evaluate:
        1. Security design principles (Defense in Depth, Least Privilege, etc.)
        2. Trust boundaries and segmentation
        3. Authentication and authorization
        4. Encryption (at rest, in transit)
        5. Network security controls
        6. Data protection mechanisms
        7. Logging and monitoring
        8. Incident response capabilities
        9. Business continuity/disaster recovery
        10. Scalability and performance considerations

        Provide:
        - Strengths
        - Weaknesses
        - Security gaps
        - Recommendations
        - Best practices violations
        - Compliance considerations

        Format as detailed architecture review.
        """

        context = await self.rag.get_relevant_context("security architecture best practices")

        review = await self.llm.generate(
            prompt=prompt,
            system="You are a senior security architect reviewing system designs.",
            context=context
        )

        return {
            "architecture": architecture,
            "review": review,
            "timestamp": datetime.now().isoformat()
        }

    async def assess_compliance(self, framework: str, current_state: str) -> Dict[str, Any]:
        """Assess compliance with security frameworks"""

        prompt = f"""
        Assess compliance with {framework}:

        Current State:
        {current_state}

        Provide:
        1. Compliance overview (% compliant)
        2. Framework requirements mapping
        3. Compliant controls
        4. Non-compliant controls
        5. Partially compliant controls
        6. Gap analysis
        7. Risk areas
        8. Remediation priorities
        9. Implementation guidance
        10. Timeline recommendations

        Common frameworks: NIST CSF, ISO 27001, SOC 2, PCI DSS, HIPAA, GDPR
        """

        context = await self.rag.get_relevant_context(f"{framework} compliance requirements")

        assessment = await self.llm.generate(
            prompt=prompt,
            system=f"You are a compliance expert specializing in {framework}.",
            context=context
        )

        return {
            "framework": framework,
            "assessment": assessment,
            "timestamp": datetime.now().isoformat()
        }

    async def perform_risk_assessment(self, assets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""

        assets_text = "\n".join([
            f"Asset: {a.get('name', 'Unknown')}\n"
            f"Type: {a.get('type', 'N/A')}\n"
            f"Value: {a.get('value', 'N/A')}\n"
            f"Threats: {a.get('threats', 'N/A')}\n"
            for a in assets
        ])

        prompt = f"""
        Perform risk assessment for these assets:

        {assets_text}

        For each asset, assess:
        1. Threat identification
        2. Vulnerability analysis
        3. Impact assessment (Confidentiality, Integrity, Availability)
        4. Likelihood estimation
        5. Risk rating (Risk = Impact × Likelihood)
        6. Existing controls
        7. Residual risk
        8. Risk treatment options (Accept/Mitigate/Transfer/Avoid)
        9. Control recommendations
        10. Cost-benefit analysis

        Use qualitative and quantitative risk analysis.
        Provide risk matrix and prioritized risk register.
        """

        context = await self.rag.get_relevant_context("risk assessment methodology")

        assessment = await self.llm.generate(
            prompt=prompt,
            system="You are a risk management expert performing security risk assessments.",
            context=context
        )

        return {
            "assets_count": len(assets),
            "risk_assessment": assessment,
            "timestamp": datetime.now().isoformat()
        }

    async def create_security_roadmap(self,
                                      current_state: str,
                                      target_state: str,
                                      timeline: str) -> Dict[str, Any]:
        """Create security improvement roadmap"""

        prompt = f"""
        Create a security roadmap:

        Current State: {current_state}
        Target State: {target_state}
        Timeline: {timeline}

        Roadmap should include:
        1. Vision and objectives
        2. Gap analysis (current vs. target)
        3. Strategic initiatives
        4. Phased implementation plan
           - Phase 1: Quick wins (0-3 months)
           - Phase 2: Foundation building (3-6 months)
           - Phase 3: Advanced capabilities (6-12 months)
           - Phase 4: Optimization (12+ months)
        5. Key milestones
        6. Resource requirements (people, budget, tools)
        7. Success metrics and KPIs
        8. Dependencies and prerequisites
        9. Risk and mitigation strategies
        10. Governance and oversight

        Format as strategic roadmap with timeline.
        """

        roadmap = await self.llm.generate(
            prompt=prompt,
            system="You are a CISO creating strategic security roadmaps."
        )

        return {
            "timeline": timeline,
            "roadmap": roadmap,
            "timestamp": datetime.now().isoformat()
        }

    async def recommend_best_practices(self, domain: str) -> Dict[str, Any]:
        """Recommend security best practices for a domain"""

        prompt = f"""
        Provide security best practices for: {domain}

        Include:
        1. Industry standards and frameworks
        2. Essential security controls
        3. Configuration hardening
        4. Secure development practices
        5. Access management
        6. Data protection
        7. Network security
        8. Monitoring and logging
        9. Incident response
        10. Common pitfalls to avoid

        Provide actionable, prioritized recommendations.
        """

        context = await self.rag.get_relevant_context(f"{domain} security best practices")

        recommendations = await self.llm.generate(
            prompt=prompt,
            system="You are a security consultant providing best practice guidance.",
            context=context
        )

        return {
            "domain": domain,
            "best_practices": recommendations,
            "timestamp": datetime.now().isoformat()
        }

    async def evaluate_security_tools(self,
                                      requirements: str,
                                      options: List[str]) -> Dict[str, Any]:
        """Evaluate and compare security tools"""

        options_text = "\n".join([f"- {opt}" for opt in options])

        prompt = f"""
        Evaluate security tools for these requirements:

        Requirements:
        {requirements}

        Tool Options:
        {options_text}

        For each tool, assess:
        1. Feature coverage
        2. Integration capabilities
        3. Deployment complexity
        4. Operational overhead
        5. Cost (licensing, implementation, maintenance)
        6. Scalability
        7. Performance impact
        8. Vendor reputation and support
        9. Community and ecosystem
        10. Future roadmap

        Provide:
        - Comparison matrix
        - Pros and cons for each
        - Recommendation with justification
        - Implementation considerations
        """

        context = await self.rag.get_relevant_context("security tools comparison")

        evaluation = await self.llm.generate(
            prompt=prompt,
            system="You are a security engineer evaluating security tools.",
            context=context
        )

        return {
            "requirements": requirements,
            "options": options,
            "evaluation": evaluation,
            "timestamp": datetime.now().isoformat()
        }

    async def assess_security_maturity(self, organization_info: str) -> Dict[str, Any]:
        """Assess security maturity level"""

        prompt = f"""
        Assess security maturity for this organization:

        {organization_info}

        Use maturity levels:
        Level 1 - Initial/Ad-hoc
        Level 2 - Managed
        Level 3 - Defined
        Level 4 - Quantitatively Managed
        Level 5 - Optimizing

        Assess maturity across domains:
        1. Governance
        2. Risk Management
        3. Asset Management
        4. Access Control
        5. Vulnerability Management
        6. Incident Response
        7. Business Continuity
        8. Security Operations
        9. Application Security
        10. Security Awareness

        For each domain provide:
        - Current maturity level
        - Justification
        - Gap to next level
        - Improvement recommendations

        Overall maturity score and roadmap to advance.
        """

        context = await self.rag.get_relevant_context("security maturity model")

        assessment = await self.llm.generate(
            prompt=prompt,
            system="You are a security consultant assessing organizational maturity.",
            context=context
        )

        return {
            "organization": organization_info,
            "maturity_assessment": assessment,
            "timestamp": datetime.now().isoformat()
        }

    async def provide_strategic_guidance(self, situation: str) -> Dict[str, Any]:
        """Provide strategic security guidance"""

        prompt = f"""
        Provide strategic security guidance for this situation:

        {situation}

        Address:
        1. Situation analysis
        2. Security implications
        3. Strategic options
        4. Recommended approach
        5. Implementation strategy
        6. Quick wins
        7. Long-term vision
        8. Resource allocation
        9. Success metrics
        10. Risk management

        Think as a CISO providing executive-level guidance.
        """

        guidance = await self.llm.generate(
            prompt=prompt,
            system="You are a CISO providing strategic security guidance to executives."
        )

        return {
            "situation": situation,
            "guidance": guidance,
            "timestamp": datetime.now().isoformat()
        }

    async def calculate_security_roi(self,
                                     investment: Dict[str, Any],
                                     benefits: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ROI for security investments"""

        investment_text = str(investment)
        benefits_text = str(benefits)

        prompt = f"""
        Calculate ROI for this security investment:

        Investment:
        {investment_text}

        Expected Benefits:
        {benefits_text}

        Provide:
        1. Total Cost of Ownership (TCO)
           - Initial costs
           - Ongoing costs
           - Hidden costs
        2. Quantifiable benefits
           - Risk reduction value
           - Incident cost avoidance
           - Operational efficiency gains
           - Compliance cost savings
        3. Intangible benefits
           - Brand protection
           - Customer trust
           - Competitive advantage
        4. ROI calculation
           - Payback period
           - Net Present Value (NPV)
           - Internal Rate of Return (IRR)
        5. Risk-adjusted ROI
        6. Sensitivity analysis
        7. Recommendation (invest/defer/alternatives)

        Use security economics principles.
        """

        roi_analysis = await self.llm.generate(
            prompt=prompt,
            system="You are a security financial analyst calculating security ROI."
        )

        return {
            "investment": investment,
            "roi_analysis": roi_analysis,
            "timestamp": datetime.now().isoformat()
        }
