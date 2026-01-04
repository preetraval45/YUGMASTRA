"""
Microsoft Sentinel (Azure Sentinel) SIEM Integration
Query logs, create incidents, manage analytics rules with KQL
"""

import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class SentinelConnector:
    """
    Microsoft Sentinel connector for YUGMASTRA
    - Query logs using KQL (Kusto Query Language)
    - Create and manage security incidents
    - Deploy analytics rules
    - Integrate with Azure Monitor
    """

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        tenant_id: str,
        client_id: str,
        client_secret: str
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret

        self.access_token = None
        self._authenticate()

    def _authenticate(self):
        """Authenticate to Azure AD and get access token"""
        auth_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

        try:
            response = requests.post(
                auth_url,
                data={
                    'client_id': self.client_id,
                    'scope': 'https://management.azure.com/.default',
                    'client_secret': self.client_secret,
                    'grant_type': 'client_credentials'
                }
            )
            response.raise_for_status()

            self.access_token = response.json()['access_token']
            logger.info("Successfully authenticated to Azure AD")

        except Exception as e:
            logger.error(f"Azure AD authentication failed: {e}")
            raise

    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers"""
        return {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json'
        }

    def _get_base_url(self) -> str:
        """Get base URL for Sentinel API"""
        return (
            f"https://management.azure.com/subscriptions/{self.subscription_id}"
            f"/resourceGroups/{self.resource_group}"
            f"/providers/Microsoft.OperationalInsights/workspaces/{self.workspace_name}"
        )

    def query_logs(
        self,
        kql_query: str,
        timespan: str = "PT24H"  # ISO 8601 duration (PT24H = 24 hours)
    ) -> List[Dict[str, Any]]:
        """
        Execute KQL query against Sentinel workspace

        Args:
            kql_query: Kusto Query Language query
            timespan: Time range in ISO 8601 format (PT1H, PT24H, P7D)

        Returns:
            Query results
        """
        query_url = (
            f"https://api.loganalytics.io/v1/workspaces/{self.workspace_name}/query"
        )

        try:
            response = requests.post(
                query_url,
                headers=self._get_headers(),
                json={
                    'query': kql_query,
                    'timespan': timespan
                }
            )
            response.raise_for_status()

            data = response.json()
            tables = data.get('tables', [])

            if not tables:
                return []

            # Extract rows and column names
            columns = tables[0]['columns']
            rows = tables[0]['rows']

            # Convert to list of dictionaries
            results = []
            for row in rows:
                result = {col['name']: val for col, val in zip(columns, row)}
                results.append(result)

            logger.info(f"KQL query returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"KQL query failed: {e}")
            return []

    def get_security_incidents(
        self,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        top: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get security incidents from Sentinel

        Args:
            severity: Filter by severity (High, Medium, Low, Informational)
            status: Filter by status (New, Active, Closed)
            top: Maximum results

        Returns:
            List of incidents
        """
        api_version = "2023-02-01"
        incidents_url = (
            f"{self._get_base_url()}"
            f"/providers/Microsoft.SecurityInsights/incidents"
            f"?api-version={api_version}"
            f"&$top={top}"
        )

        # Build filter
        filters = []
        if severity:
            filters.append(f"properties/severity eq '{severity}'")
        if status:
            filters.append(f"properties/status eq '{status}'")

        if filters:
            incidents_url += f"&$filter={' and '.join(filters)}"

        try:
            response = requests.get(
                incidents_url,
                headers=self._get_headers()
            )
            response.raise_for_status()

            incidents = response.json().get('value', [])
            logger.info(f"Retrieved {len(incidents)} incidents")
            return incidents

        except Exception as e:
            logger.error(f"Failed to get incidents: {e}")
            return []

    def create_incident(
        self,
        title: str,
        description: str,
        severity: str = "Medium",
        status: str = "New",
        classification: Optional[str] = None,
        labels: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Create security incident in Sentinel

        Args:
            title: Incident title
            description: Incident description
            severity: High, Medium, Low, Informational
            status: New, Active, Closed
            classification: TruePositive, BenignPositive, FalsePositive, Undetermined
            labels: List of labels/tags

        Returns:
            Incident ID if successful
        """
        api_version = "2023-02-01"
        incident_name = f"yugmastra-{int(datetime.now().timestamp())}"
        incident_url = (
            f"{self._get_base_url()}"
            f"/providers/Microsoft.SecurityInsights/incidents/{incident_name}"
            f"?api-version={api_version}"
        )

        incident_data = {
            "properties": {
                "title": title,
                "description": description,
                "severity": severity,
                "status": status,
                "owner": {
                    "assignedTo": "YUGMASTRA AI",
                    "email": "ai@yugmastra.com",
                    "objectId": None
                }
            }
        }

        if classification:
            incident_data["properties"]["classification"] = classification

        if labels:
            incident_data["properties"]["labels"] = [{"labelName": label} for label in labels]

        try:
            response = requests.put(
                incident_url,
                headers=self._get_headers(),
                json=incident_data
            )
            response.raise_for_status()

            result = response.json()
            incident_id = result.get('name')

            logger.info(f"Created Sentinel incident: {incident_id}")
            return incident_id

        except Exception as e:
            logger.error(f"Failed to create incident: {e}")
            return None

    def create_analytics_rule(
        self,
        rule_name: str,
        display_name: str,
        description: str,
        kql_query: str,
        severity: str = "Medium",
        tactics: Optional[List[str]] = None,
        techniques: Optional[List[str]] = None,
        frequency: str = "PT5M",  # 5 minutes
        period: str = "PT10M"  # 10 minutes lookback
    ) -> bool:
        """
        Create scheduled analytics rule in Sentinel

        Args:
            rule_name: Unique rule name
            display_name: Display name
            description: Rule description
            kql_query: Detection query (KQL)
            severity: High, Medium, Low, Informational
            tactics: MITRE tactics (InitialAccess, Execution, etc.)
            techniques: MITRE technique IDs (T1566, T1059, etc.)
            frequency: Check frequency (PT5M, PT1H)
            period: Lookback period (PT10M, PT24H)

        Returns:
            Success status
        """
        api_version = "2023-02-01"
        rule_url = (
            f"{self._get_base_url()}"
            f"/providers/Microsoft.SecurityInsights/alertRules/{rule_name}"
            f"?api-version={api_version}"
        )

        rule_data = {
            "kind": "Scheduled",
            "properties": {
                "displayName": display_name,
                "description": description,
                "severity": severity,
                "enabled": True,
                "query": kql_query,
                "queryFrequency": frequency,
                "queryPeriod": period,
                "triggerOperator": "GreaterThan",
                "triggerThreshold": 0,
                "suppressionDuration": "PT1H",
                "suppressionEnabled": False,
                "tactics": tactics or [],
                "techniques": techniques or [],
                "alertRuleTemplateName": None,
                "incidentConfiguration": {
                    "createIncident": True,
                    "groupingConfiguration": {
                        "enabled": True,
                        "reopenClosedIncident": False,
                        "lookbackDuration": "PT5H",
                        "matchingMethod": "AllEntities"
                    }
                },
                "eventGroupingSettings": {
                    "aggregationKind": "AlertPerResult"
                }
            }
        }

        try:
            response = requests.put(
                rule_url,
                headers=self._get_headers(),
                json=rule_data
            )
            response.raise_for_status()

            logger.info(f"Created analytics rule: {rule_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create analytics rule: {e}")
            return False

    def add_comment_to_incident(
        self,
        incident_id: str,
        message: str
    ) -> bool:
        """Add comment to existing incident"""
        api_version = "2023-02-01"
        comment_name = f"comment-{int(datetime.now().timestamp())}"
        comment_url = (
            f"{self._get_base_url()}"
            f"/providers/Microsoft.SecurityInsights/incidents/{incident_id}"
            f"/comments/{comment_name}"
            f"?api-version={api_version}"
        )

        comment_data = {
            "properties": {
                "message": message
            }
        }

        try:
            response = requests.put(
                comment_url,
                headers=self._get_headers(),
                json=comment_data
            )
            response.raise_for_status()

            logger.info(f"Added comment to incident {incident_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to add comment: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize connector
    sentinel = SentinelConnector(
        subscription_id="00000000-0000-0000-0000-000000000000",
        resource_group="security-rg",
        workspace_name="yugmastra-sentinel",
        tenant_id="your-tenant-id",
        client_id="your-client-id",
        client_secret="your-client-secret"
    )

    # Query logs with KQL
    kql_query = """
    SecurityEvent
    | where EventID == 4625
    | where TimeGenerated > ago(1h)
    | summarize FailedAttempts=count() by Account, Computer
    | where FailedAttempts > 5
    | order by FailedAttempts desc
    """
    failed_logins = sentinel.query_logs(kql_query, timespan="PT1H")
    print(f"Found {len(failed_logins)} accounts with failed login attempts")

    # Get high-severity active incidents
    incidents = sentinel.get_security_incidents(
        severity="High",
        status="Active"
    )
    print(f"Found {len(incidents)} high-severity active incidents")

    # Create incident
    incident_id = sentinel.create_incident(
        title="YUGMASTRA: Potential Zero-Day Detected",
        description="AI model identified anomalous behavior indicating possible zero-day exploitation",
        severity="High",
        status="New",
        classification="TruePositive",
        labels=["yugmastra", "ai-detection", "zero-day"]
    )

    if incident_id:
        # Add AI analysis as comment
        sentinel.add_comment_to_incident(
            incident_id,
            "YUGMASTRA AI Analysis:\n"
            "- Attack vector: Web application\n"
            "- Confidence: 94%\n"
            "- MITRE Technique: T1203 (Exploitation for Client Execution)\n"
            "- Recommended actions: Isolate affected system, collect memory dump, analyze network traffic"
        )

    # Create analytics rule
    sentinel.create_analytics_rule(
        rule_name="yugmastra-lateral-movement-001",
        display_name="YUGMASTRA - Lateral Movement Detection",
        description="Detects lateral movement using AI correlation across multiple data sources",
        kql_query="""
        SecurityEvent
        | where EventID in (4624, 4648, 4672)
        | where LogonType in (3, 10)
        | summarize LogonCount=count() by Account, DestinationHost=Computer
        | where LogonCount > 3
        """,
        severity="High",
        tactics=["LateralMovement", "PrivilegeEscalation"],
        techniques=["T1021", "T1078"],
        frequency="PT5M",
        period="PT15M"
    )
