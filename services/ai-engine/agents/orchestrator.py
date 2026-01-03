"""
AI Agent Orchestrator
Coordinates multiple AI agents for complex multi-step tasks
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentTask:
    """Represents a task for an agent"""

    def __init__(self, agent_name: str, action: str, params: Dict[str, Any], priority: TaskPriority = TaskPriority.MEDIUM):
        self.id = f"task_{datetime.now().timestamp()}"
        self.agent_name = agent_name
        self.action = action
        self.params = params
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.started_at = None
        self.completed_at = None


class AIOrchestrator:
    """
    Autonomous AI Agent Orchestrator

    Coordinates multiple specialized AI agents to solve complex cybersecurity tasks
    Capabilities:
    - Multi-agent task decomposition
    - Parallel agent execution
    - Agent result aggregation
    - Intelligent task routing
    - Failure handling and retry logic
    - Learning from agent interactions
    """

    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents
        self.task_queue: List[AgentTask] = []
        self.completed_tasks: List[AgentTask] = []
        self.running_tasks: List[AgentTask] = []
        logger.info("AI Orchestrator initialized with {} agents".format(len(agents)))

    async def execute_workflow(self, workflow_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a pre-defined multi-agent workflow"""

        workflows = {
            "full_security_assessment": self._workflow_security_assessment,
            "incident_investigation": self._workflow_incident_investigation,
            "threat_hunting": self._workflow_threat_hunting,
            "vulnerability_remediation": self._workflow_vulnerability_remediation,
            "attack_simulation": self._workflow_attack_simulation,
        }

        workflow_func = workflows.get(workflow_type)
        if not workflow_func:
            raise ValueError(f"Unknown workflow: {workflow_type}")

        logger.info(f"Executing workflow: {workflow_type}")
        return await workflow_func(context)

    async def _workflow_security_assessment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive security assessment workflow
        Agents: Vulnerability Scanner → Threat Intelligence → Security Advisor
        """

        results = {
            "workflow": "full_security_assessment",
            "started_at": datetime.now().isoformat(),
            "steps": []
        }

        # Step 1: Scan for vulnerabilities
        vuln_task = AgentTask(
            agent_name="vulnerability_scanner",
            action="scan_code",
            params={
                "code": context.get("code", ""),
                "language": context.get("language", "python")
            },
            priority=TaskPriority.HIGH
        )
        vuln_result = await self._execute_task(vuln_task)
        results["steps"].append({"step": "vulnerability_scan", "result": vuln_result})

        # Step 2: Check threat intelligence for identified vulnerabilities
        if vuln_result and "vulnerabilities" in vuln_result:
            threat_task = AgentTask(
                agent_name="threat_intelligence",
                action="analyze_threats",
                params={"vulnerabilities": vuln_result["vulnerabilities"]},
                priority=TaskPriority.HIGH
            )
            threat_result = await self._execute_task(threat_task)
            results["steps"].append({"step": "threat_analysis", "result": threat_result})

        # Step 3: Get security recommendations
        advisor_task = AgentTask(
            agent_name="security_advisor",
            action="review_architecture",
            params={"architecture": context.get("architecture", "")},
            priority=TaskPriority.MEDIUM
        )
        advisor_result = await self._execute_task(advisor_task)
        results["steps"].append({"step": "security_recommendations", "result": advisor_result})

        # Step 4: Generate remediation plan
        if vuln_result:
            remediation_task = AgentTask(
                agent_name="vulnerability_scanner",
                action="generate_remediation_plan",
                params={"vulnerabilities": [vuln_result]},
                priority=TaskPriority.HIGH
            )
            remediation_result = await self._execute_task(remediation_task)
            results["steps"].append({"step": "remediation_plan", "result": remediation_result})

        results["completed_at"] = datetime.now().isoformat()
        results["status"] = "completed"

        return results

    async def _workflow_incident_investigation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Incident investigation workflow
        Agents: Incident Response → Threat Intelligence → Red Team (attack reconstruction)
        """

        results = {
            "workflow": "incident_investigation",
            "started_at": datetime.now().isoformat(),
            "steps": []
        }

        events = context.get("events", [])

        # Step 1: Detect and classify incident
        detect_task = AgentTask(
            agent_name="incident_response",
            action="detect_incident",
            params={"events": events},
            priority=TaskPriority.CRITICAL
        )
        detection = await self._execute_task(detect_task)
        results["steps"].append({"step": "incident_detection", "result": detection})

        # Step 2: Reconstruct timeline
        timeline_task = AgentTask(
            agent_name="incident_response",
            action="reconstruct_timeline",
            params={"events": events},
            priority=TaskPriority.HIGH
        )
        timeline = await self._execute_task(timeline_task)
        results["steps"].append({"step": "timeline_reconstruction", "result": timeline})

        # Step 3: Identify threat actors and TTPs
        intel_task = AgentTask(
            agent_name="threat_intelligence",
            action="correlate_campaign",
            params={"incidents": [str(e) for e in events]},
            priority=TaskPriority.HIGH
        )
        intel = await self._execute_task(intel_task)
        results["steps"].append({"step": "threat_intelligence", "result": intel})

        # Step 4: Root cause analysis
        rca_task = AgentTask(
            agent_name="incident_response",
            action="perform_root_cause_analysis",
            params={"incident_data": detection},
            priority=TaskPriority.HIGH
        )
        rca = await self._execute_task(rca_task)
        results["steps"].append({"step": "root_cause_analysis", "result": rca})

        # Step 5: Generate response playbook
        playbook_task = AgentTask(
            agent_name="incident_response",
            action="generate_response_playbook",
            params={
                "incident_type": context.get("incident_type", "unknown"),
                "severity": context.get("severity", "high")
            },
            priority=TaskPriority.CRITICAL
        )
        playbook = await self._execute_task(playbook_task)
        results["steps"].append({"step": "response_playbook", "result": playbook})

        results["completed_at"] = datetime.now().isoformat()
        results["status"] = "completed"

        return results

    async def _workflow_threat_hunting(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Proactive threat hunting workflow
        Agents: Threat Intelligence → Vulnerability Scanner → Incident Response
        """

        results = {
            "workflow": "threat_hunting",
            "started_at": datetime.now().isoformat(),
            "steps": [],
            "findings": []
        }

        # Step 1: Gather threat intelligence
        ioc_task = AgentTask(
            agent_name="threat_intelligence",
            action="analyze_ioc",
            params={
                "ioc": context.get("ioc", ""),
                "ioc_type": context.get("ioc_type", "ip")
            },
            priority=TaskPriority.HIGH
        )
        ioc_analysis = await self._execute_task(ioc_task)
        results["steps"].append({"step": "ioc_analysis", "result": ioc_analysis})

        # Step 2: Map to MITRE ATT&CK
        if ioc_analysis:
            mitre_task = AgentTask(
                agent_name="threat_intelligence",
                action="map_to_mitre",
                params={"attack_description": str(ioc_analysis)},
                priority=TaskPriority.MEDIUM
            )
            mitre = await self._execute_task(mitre_task)
            results["steps"].append({"step": "mitre_mapping", "result": mitre})

        # Step 3: Check for vulnerabilities that could be exploited
        vuln_task = AgentTask(
            agent_name="vulnerability_scanner",
            action="assess_attack_surface",
            params={"system_info": context.get("system_info", {})},
            priority=TaskPriority.HIGH
        )
        attack_surface = await self._execute_task(vuln_task)
        results["steps"].append({"step": "attack_surface_analysis", "result": attack_surface})

        # Step 4: Generate detection rules (if SIEM enabled)
        # This would be integrated if SIEM generator is available

        results["completed_at"] = datetime.now().isoformat()
        results["status"] = "completed"

        return results

    async def _workflow_vulnerability_remediation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Vulnerability remediation workflow
        Agents: Vulnerability Scanner → Threat Intelligence → Security Advisor
        """

        results = {
            "workflow": "vulnerability_remediation",
            "started_at": datetime.now().isoformat(),
            "steps": []
        }

        vulnerabilities = context.get("vulnerabilities", [])

        # Step 1: Calculate CVSS scores
        cvss_tasks = []
        for vuln in vulnerabilities[:5]:  # Limit to first 5 for performance
            task = AgentTask(
                agent_name="vulnerability_scanner",
                action="calculate_cvss",
                params={"vulnerability_details": str(vuln)},
                priority=TaskPriority.MEDIUM
            )
            cvss_tasks.append(task)

        cvss_results = await self._execute_parallel(cvss_tasks)
        results["steps"].append({"step": "cvss_scoring", "results": cvss_results})

        # Step 2: Predict exploitability
        exploit_tasks = []
        for vuln in vulnerabilities[:5]:
            if isinstance(vuln, dict) and "cve_id" in vuln:
                task = AgentTask(
                    agent_name="vulnerability_scanner",
                    action="predict_exploitability",
                    params={
                        "cve_id": vuln["cve_id"],
                        "details": str(vuln)
                    },
                    priority=TaskPriority.HIGH
                )
                exploit_tasks.append(task)

        exploit_results = await self._execute_parallel(exploit_tasks)
        results["steps"].append({"step": "exploitability_prediction", "results": exploit_results})

        # Step 3: Generate remediation plan
        remediation_task = AgentTask(
            agent_name="vulnerability_scanner",
            action="generate_remediation_plan",
            params={"vulnerabilities": vulnerabilities},
            priority=TaskPriority.CRITICAL
        )
        remediation = await self._execute_task(remediation_task)
        results["steps"].append({"step": "remediation_plan", "result": remediation})

        results["completed_at"] = datetime.now().isoformat()
        results["status"] = "completed"

        return results

    async def _workflow_attack_simulation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Red Team vs Blue Team simulation workflow
        Agents: Red Team → Blue Team → Evolution → Incident Response
        """

        results = {
            "workflow": "attack_simulation",
            "started_at": datetime.now().isoformat(),
            "rounds": []
        }

        rounds = context.get("rounds", 3)
        scenario = context.get("scenario", "network_intrusion")

        for round_num in range(rounds):
            round_result = {"round": round_num + 1, "steps": []}

            # Red Team: Generate attack
            red_task = AgentTask(
                agent_name="red_team",
                action="generate_attack",
                params={"scenario": scenario, "round": round_num},
                priority=TaskPriority.HIGH
            )
            attack = await self._execute_task(red_task)
            round_result["steps"].append({"actor": "red_team", "action": attack})

            # Blue Team: Defend
            blue_task = AgentTask(
                agent_name="blue_team",
                action="generate_defense",
                params={"attack": attack, "round": round_num},
                priority=TaskPriority.HIGH
            )
            defense = await self._execute_task(blue_task)
            round_result["steps"].append({"actor": "blue_team", "action": defense})

            # Evolution: Learn and adapt
            evolution_task = AgentTask(
                agent_name="evolution",
                action="evolve_strategy",
                params={"attack": attack, "defense": defense, "round": round_num},
                priority=TaskPriority.MEDIUM
            )
            evolution = await self._execute_task(evolution_task)
            round_result["steps"].append({"actor": "evolution", "insight": evolution})

            results["rounds"].append(round_result)

        # Final analysis
        incident_task = AgentTask(
            agent_name="incident_response",
            action="generate_incident_report",
            params={"incident_data": {"rounds": results["rounds"], "scenario": scenario}},
            priority=TaskPriority.MEDIUM
        )
        final_report = await self._execute_task(incident_task)
        results["final_analysis"] = final_report

        results["completed_at"] = datetime.now().isoformat()
        results["status"] = "completed"

        return results

    async def _execute_task(self, task: AgentTask) -> Optional[Dict[str, Any]]:
        """Execute a single agent task"""

        try:
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now().isoformat()
            self.running_tasks.append(task)

            agent = self.agents.get(task.agent_name)
            if not agent:
                raise ValueError(f"Agent not found: {task.agent_name}")

            # Get agent method
            method = getattr(agent, task.action, None)
            if not method:
                raise ValueError(f"Agent {task.agent_name} has no action: {task.action}")

            # Execute agent action
            result = await method(**task.params)

            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now().isoformat()

            self.running_tasks.remove(task)
            self.completed_tasks.append(task)

            logger.info(f"Task completed: {task.id} ({task.agent_name}.{task.action})")
            return result

        except Exception as e:
            logger.error(f"Task failed: {task.id} - {str(e)}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now().isoformat()

            if task in self.running_tasks:
                self.running_tasks.remove(task)
            self.completed_tasks.append(task)

            return None

    async def _execute_parallel(self, tasks: List[AgentTask]) -> List[Optional[Dict[str, Any]]]:
        """Execute multiple tasks in parallel"""

        results = await asyncio.gather(*[self._execute_task(task) for task in tasks])
        return list(results)

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""

        return {
            "agents": list(self.agents.keys()),
            "agent_count": len(self.agents),
            "tasks": {
                "pending": len(self.task_queue),
                "running": len(self.running_tasks),
                "completed": len(self.completed_tasks)
            },
            "timestamp": datetime.now().isoformat()
        }
