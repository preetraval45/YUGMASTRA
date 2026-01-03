"""
Security Code Generator Agent
AI-powered secure code generation and security automation
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from models.llm_manager import LLMManager
from services.rag_service import RAGService

logger = logging.getLogger(__name__)


class SecurityCodeGenerator:
    """
    AI-powered security code generator

    Capabilities:
    - Secure API endpoint generation
    - Authentication/authorization code
    - Encryption implementation
    - Input validation functions
    - Security test generation
    - WAF rules creation
    - IDS/IPS signatures
    - Security policy as code
    """

    def __init__(self, llm_manager: LLMManager, rag_service: RAGService):
        self.llm = llm_manager
        self.rag = rag_service
        self.name = "SecurityCodeGenerator"
        logger.info(f"Initialized {self.name}")

    async def generate_secure_api(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate secure API endpoint code"""

        prompt = f"""
        Generate secure API endpoint code based on this specification:

        {spec}

        Requirements:
        1. Input validation (all parameters)
        2. Authentication (JWT/API key)
        3. Authorization (RBAC/ABAC)
        4. Rate limiting
        5. SQL injection prevention
        6. XSS prevention
        7. CSRF protection
        8. Proper error handling (no info leakage)
        9. Security logging
        10. API documentation

        Language: {spec.get('language', 'python')}
        Framework: {spec.get('framework', 'fastapi')}

        Generate complete, production-ready code with security best practices.
        Include inline comments explaining security measures.
        """

        context_data = await self.rag.get_relevant_context(f"secure API {spec.get('framework')}")

        code = await self.llm.generate(
            prompt=prompt,
            system="You are a senior security engineer generating secure API code.",
            context=context_data
        )

        return {
            "specification": spec,
            "generated_code": code,
            "language": spec.get('language', 'python'),
            "framework": spec.get('framework', 'fastapi'),
            "timestamp": datetime.now().isoformat()
        }

    async def generate_auth_system(self, requirements: str) -> Dict[str, Any]:
        """Generate authentication and authorization system"""

        prompt = f"""
        Generate complete authentication/authorization system:

        Requirements: {requirements}

        Implement:
        1. User registration with email verification
        2. Secure password hashing (bcrypt/argon2)
        3. JWT token generation and validation
        4. Refresh token mechanism
        5. Password reset flow
        6. Multi-factor authentication (2FA/TOTP)
        7. Session management
        8. Role-based access control (RBAC)
        9. OAuth 2.0 integration (Google, GitHub)
        10. Security audit logging

        Security features:
        - Rate limiting on auth endpoints
        - Account lockout after failed attempts
        - Secure token storage
        - CSRF protection
        - XSS prevention
        - SQL injection prevention

        Generate complete, modular code with proper error handling.
        """

        context_data = await self.rag.get_relevant_context("secure authentication system")

        code = await self.llm.generate(
            prompt=prompt,
            system="You are an identity and access management expert generating secure auth systems.",
            context=context_data
        )

        return {
            "requirements": requirements,
            "auth_system_code": code,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_encryption_code(self, use_case: str, language: str = "python") -> Dict[str, Any]:
        """Generate encryption/decryption code"""

        prompt = f"""
        Generate secure encryption code for: {use_case}

        Language: {language}

        Implement:
        1. Symmetric encryption (AES-256-GCM)
        2. Key derivation (PBKDF2/scrypt)
        3. Secure key storage
        4. IV generation
        5. Data encryption
        6. Data decryption
        7. Key rotation support
        8. Error handling

        Security requirements:
        - Use cryptographically secure random numbers
        - Proper key management
        - Secure defaults (no weak ciphers)
        - Timing attack prevention
        - Memory cleanup after use

        Include both encryption and decryption functions.
        Add comprehensive error handling and logging.
        """

        context_data = await self.rag.get_relevant_context(f"encryption {language}")

        code = await self.llm.generate(
            prompt=prompt,
            system="You are a cryptography expert generating secure encryption code.",
            context=context_data
        )

        return {
            "use_case": use_case,
            "language": language,
            "encryption_code": code,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_input_validation(self, fields: List[Dict[str, str]], language: str = "python") -> Dict[str, Any]:
        """Generate input validation functions"""

        fields_spec = "\n".join([
            f"- {f['name']} ({f['type']}): {f.get('rules', 'No specific rules')}"
            for f in fields
        ])

        prompt = f"""
        Generate input validation functions for these fields:

        {fields_spec}

        Language: {language}

        Implement validation for:
        1. Data type checking
        2. Length limits
        3. Format validation (email, URL, etc.)
        4. Whitelist/blacklist checking
        5. SQL injection prevention
        6. XSS prevention
        7. Path traversal prevention
        8. Command injection prevention
        9. LDAP injection prevention
        10. NoSQL injection prevention

        Generate:
        - Individual field validators
        - Combined form validator
        - Sanitization functions
        - Custom validation rules
        - Error message generation

        Use modern validation libraries where applicable.
        """

        code = await self.llm.generate(
            prompt=prompt,
            system="You are an input validation expert preventing injection attacks."
        )

        return {
            "fields": fields,
            "language": language,
            "validation_code": code,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_security_tests(self, code: str, language: str) -> Dict[str, Any]:
        """Generate security tests for given code"""

        prompt = f"""
        Generate comprehensive security tests for this code:

        ```{language}
        {code}
        ```

        Test cases should cover:
        1. SQL injection attempts
        2. XSS attack vectors
        3. CSRF vulnerabilities
        4. Authentication bypass
        5. Authorization bypass
        6. Input validation bypass
        7. Path traversal
        8. Command injection
        9. XML/XXE injection
        10. Insecure deserialization

        Generate:
        - Unit tests for security functions
        - Integration tests for security flows
        - Penetration testing scripts
        - Fuzzing test cases
        - Negative test cases

        Framework: pytest (Python) / Jest (JavaScript) / JUnit (Java)
        """

        tests = await self.llm.generate(
            prompt=prompt,
            system="You are a security testing expert creating comprehensive test suites."
        )

        return {
            "original_code": code,
            "language": language,
            "security_tests": tests,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_waf_rules(self, attack_patterns: List[str]) -> Dict[str, Any]:
        """Generate WAF (Web Application Firewall) rules"""

        patterns_list = "\n".join([f"- {p}" for p in attack_patterns])

        prompt = f"""
        Generate WAF rules to block these attack patterns:

        {patterns_list}

        Create rules for:
        1. ModSecurity format
        2. AWS WAF format
        3. Cloudflare WAF format
        4. NGINX WAF format

        Rule components:
        - Pattern matching (regex)
        - Threat severity
        - Action (block/log/challenge)
        - False positive prevention
        - Performance optimization

        For each pattern:
        - Multiple detection methods
        - Bypass prevention
        - Logging configuration
        - Alert triggers

        Include rule testing methodology.
        """

        rules = await self.llm.generate(
            prompt=prompt,
            system="You are a WAF engineer creating detection and prevention rules."
        )

        return {
            "attack_patterns": attack_patterns,
            "waf_rules": rules,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_ids_signatures(self, threat_description: str) -> Dict[str, Any]:
        """Generate IDS/IPS signatures"""

        prompt = f"""
        Generate IDS/IPS signatures for this threat:

        {threat_description}

        Create signatures for:
        1. Snort format
        2. Suricata format
        3. YARA format (for malware)

        Signature components:
        - Network traffic patterns
        - Payload inspection
        - Protocol analysis
        - Behavioral indicators
        - File signatures (hashes, patterns)

        Include:
        - Multiple detection methods
        - False positive reduction
        - Performance considerations
        - Rule testing examples

        Provide both alert and block rules.
        """

        signatures = await self.llm.generate(
            prompt=prompt,
            system="You are an intrusion detection expert creating detection signatures."
        )

        return {
            "threat_description": threat_description,
            "ids_signatures": signatures,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_security_policy_code(self, policy_description: str, format: str = "terraform") -> Dict[str, Any]:
        """Generate security policy as code"""

        prompt = f"""
        Convert this security policy to {format} code:

        Policy: {policy_description}

        Generate Infrastructure as Code (IaC) with security policies for:
        1. Network security groups
        2. IAM policies
        3. Encryption requirements
        4. Logging and monitoring
        5. Backup policies
        6. Access controls
        7. Compliance rules
        8. Data classification
        9. Incident response automation
        10. Security scanning integration

        Format: {format} (Terraform/CloudFormation/Pulumi)

        Include:
        - Least privilege access
        - Defense in depth
        - Zero trust principles
        - Compliance alignment
        - Automated remediation

        Add inline comments explaining security rationale.
        """

        context_data = await self.rag.get_relevant_context(f"security policy {format}")

        code = await self.llm.generate(
            prompt=prompt,
            system="You are a security architect implementing policy as code.",
            context=context_data
        )

        return {
            "policy_description": policy_description,
            "format": format,
            "policy_code": code,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_secure_dockerfile(self, base_requirements: str) -> Dict[str, Any]:
        """Generate secure Dockerfile"""

        prompt = f"""
        Generate secure Dockerfile for: {base_requirements}

        Security hardening:
        1. Use minimal base images (alpine/distroless)
        2. Run as non-root user
        3. Multi-stage builds
        4. Vulnerability scanning
        5. No secrets in layers
        6. Read-only filesystem
        7. Drop unnecessary capabilities
        8. Health checks
        9. Resource limits
        10. Security labels

        Include:
        - .dockerignore file
        - Security scanning commands
        - Build best practices
        - Runtime security configs

        Explain each security measure.
        """

        dockerfile = await self.llm.generate(
            prompt=prompt,
            system="You are a container security expert creating secure Dockerfiles."
        )

        return {
            "requirements": base_requirements,
            "secure_dockerfile": dockerfile,
            "timestamp": datetime.now().isoformat()
        }

    async def generate_kubernetes_security(self, app_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Generate secure Kubernetes manifests"""

        prompt = f"""
        Generate secure Kubernetes manifests for: {app_spec}

        Security configurations:
        1. Pod Security Policy / Pod Security Standards
        2. Network Policies
        3. RBAC (Roles, RoleBindings)
        4. Security Context (non-root, read-only fs)
        5. Resource limits
        6. Secrets management
        7. Service Mesh (Istio/Linkerd)
        8. Admission Controllers
        9. Image pull policies
        10. Security scanning

        Generate:
        - Deployment manifest
        - Service manifest
        - NetworkPolicy
        - RBAC manifests
        - PodSecurityPolicy
        - ConfigMap/Secret
        - Ingress with TLS

        Follow CIS Kubernetes Benchmark.
        """

        manifests = await self.llm.generate(
            prompt=prompt,
            system="You are a Kubernetes security expert creating secure manifests."
        )

        return {
            "app_specification": app_spec,
            "kubernetes_manifests": manifests,
            "timestamp": datetime.now().isoformat()
        }
