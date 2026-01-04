"""
Splunk SIEM Integration Connector
Real-time alert ingestion and detection push to Splunk
"""

import requests
import json
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SplunkConnector:
    """
    Splunk connector for bidirectional integration
    - Query Splunk for alerts and logs
    - Push YUGMASTRA detections to Splunk
    - Execute SPL queries
    """

    def __init__(
        self,
        host: str,
        port: int = 8089,
        username: str = "",
        password: str = "",
        token: Optional[str] = None,
        verify_ssl: bool = True
    ):
        self.base_url = f"https://{host}:{port}"
        self.username = username
        self.password = password
        self.token = token
        self.verify_ssl = verify_ssl
        self.session_key = None

        if not token:
            self._authenticate()

    def _authenticate(self):
        """Authenticate and get session key"""
        auth_url = f"{self.base_url}/services/auth/login"

        try:
            response = requests.post(
                auth_url,
                data={
                    'username': self.username,
                    'password': self.password
                },
                verify=self.verify_ssl
            )
            response.raise_for_status()

            # Extract session key from XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)
            self.session_key = root.findtext(".//sessionKey")

            logger.info("Successfully authenticated to Splunk")

        except Exception as e:
            logger.error(f"Splunk authentication failed: {e}")
            raise

    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if self.token:
            return {
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json'
            }
        else:
            return {
                'Authorization': f'Splunk {self.session_key}',
                'Content-Type': 'application/json'
            }

    def search(
        self,
        spl_query: str,
        earliest_time: str = "-24h",
        latest_time: str = "now",
        max_results: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Execute SPL search query

        Args:
            spl_query: Splunk Processing Language query
            earliest_time: Start time (e.g., "-24h", "2024-01-01T00:00:00")
            latest_time: End time (e.g., "now", "2024-01-02T00:00:00")
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        search_url = f"{self.base_url}/services/search/jobs"

        try:
            # Create search job
            response = requests.post(
                search_url,
                headers=self._get_headers(),
                data={
                    'search': f'search {spl_query}',
                    'earliest_time': earliest_time,
                    'latest_time': latest_time,
                    'output_mode': 'json'
                },
                verify=self.verify_ssl
            )
            response.raise_for_status()

            # Get search job ID
            job_data = response.json()
            sid = job_data.get('sid')

            if not sid:
                logger.error("No search job ID returned")
                return []

            # Wait for results
            results_url = f"{self.base_url}/services/search/jobs/{sid}/results"

            while True:
                status_response = requests.get(
                    f"{self.base_url}/services/search/jobs/{sid}",
                    headers=self._get_headers(),
                    params={'output_mode': 'json'},
                    verify=self.verify_ssl
                )

                status_data = status_response.json()
                if status_data['entry'][0]['content']['isDone']:
                    break

            # Fetch results
            results_response = requests.get(
                results_url,
                headers=self._get_headers(),
                params={
                    'output_mode': 'json',
                    'count': max_results
                },
                verify=self.verify_ssl
            )
            results_response.raise_for_status()

            results = results_response.json().get('results', [])
            logger.info(f"Splunk search returned {len(results)} results")

            return results

        except Exception as e:
            logger.error(f"Splunk search failed: {e}")
            return []

    def get_notable_events(
        self,
        time_range: str = "-1h",
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch notable events (Splunk Enterprise Security)

        Args:
            time_range: Time range (e.g., "-1h", "-24h")
            severity: Filter by severity (critical, high, medium, low)

        Returns:
            List of notable events
        """
        spl_query = 'index=notable'

        if severity:
            spl_query += f' severity={severity}'

        spl_query += ' | head 1000'

        return self.search(spl_query, earliest_time=time_range)

    def push_detection(
        self,
        detection: Dict[str, Any],
        index: str = "main",
        sourcetype: str = "yugmastra:detection"
    ) -> bool:
        """
        Push YUGMASTRA detection to Splunk

        Args:
            detection: Detection data
            index: Target Splunk index
            sourcetype: Event sourcetype

        Returns:
            Success status
        """
        event_url = f"{self.base_url}/services/receivers/simple"

        try:
            # Format event
            event_data = {
                'time': int(datetime.now().timestamp()),
                'source': 'yugmastra',
                'sourcetype': sourcetype,
                'index': index,
                'event': json.dumps(detection)
            }

            response = requests.post(
                event_url,
                headers=self._get_headers(),
                data=event_data,
                verify=self.verify_ssl
            )
            response.raise_for_status()

            logger.info(f"Successfully pushed detection to Splunk index={index}")
            return True

        except Exception as e:
            logger.error(f"Failed to push detection to Splunk: {e}")
            return False

    def create_notable_event(
        self,
        title: str,
        description: str,
        severity: str = "medium",
        urgency: str = "medium",
        owner: str = "unassigned",
        additional_fields: Optional[Dict] = None
    ) -> bool:
        """
        Create notable event in Splunk ES

        Args:
            title: Event title
            description: Event description
            severity: critical, high, medium, low
            urgency: critical, high, medium, low
            owner: Assigned owner
            additional_fields: Extra fields

        Returns:
            Success status
        """
        notable_data = {
            'title': title,
            'description': description,
            'severity': severity,
            'urgency': urgency,
            'owner': owner,
            'time': int(datetime.now().timestamp()),
            'source': 'yugmastra_ai'
        }

        if additional_fields:
            notable_data.update(additional_fields)

        return self.push_detection(
            notable_data,
            index='notable',
            sourcetype='stash'
        )

    def create_correlation_search(
        self,
        name: str,
        spl_search: str,
        description: str = "",
        cron_schedule: str = "*/5 * * * *"
    ) -> bool:
        """
        Create saved search / correlation rule in Splunk

        Args:
            name: Search name
            spl_search: SPL query
            description: Search description
            cron_schedule: Cron schedule for search

        Returns:
            Success status
        """
        saved_search_url = f"{self.base_url}/services/saved/searches"

        try:
            response = requests.post(
                saved_search_url,
                headers=self._get_headers(),
                data={
                    'name': name,
                    'search': spl_search,
                    'description': description,
                    'cron_schedule': cron_schedule,
                    'is_scheduled': 1,
                    'actions': 'notable',
                    'action.notable.param.severity': 'high',
                    'action.notable.param.rule_title': name
                },
                verify=self.verify_ssl
            )
            response.raise_for_status()

            logger.info(f"Created correlation search: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to create correlation search: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Initialize connector
    splunk = SplunkConnector(
        host="splunk.example.com",
        port=8089,
        username="admin",
        password="changeme"
    )

    # Search for security events
    results = splunk.search(
        spl_query='index=security sourcetype=firewall action=blocked',
        earliest_time="-1h",
        max_results=100
    )
    print(f"Found {len(results)} blocked firewall events")

    # Get notable events
    notables = splunk.get_notable_events(time_range="-24h", severity="critical")
    print(f"Found {len(notables)} critical notable events")

    # Push YUGMASTRA detection
    detection = {
        'attack_type': 'SQL Injection',
        'target': 'web_server',
        'severity': 'high',
        'mitre_technique': 'T1190',
        'confidence': 0.95,
        'details': 'ML model detected SQL injection attempt'
    }
    splunk.push_detection(detection)

    # Create notable event
    splunk.create_notable_event(
        title="YUGMASTRA: Zero-Day Detected",
        description="AI model identified potential zero-day exploitation attempt",
        severity="critical",
        urgency="high",
        additional_fields={
            'mitre_technique': 'T1203',
            'cve': 'CVE-2024-XXXXX',
            'affected_asset': '10.0.1.50'
        }
    )

    # Create correlation search
    splunk.create_correlation_search(
        name="YUGMASTRA - Brute Force Detection",
        spl_search='index=security sourcetype=auth failed_attempts>5 | stats count by src_ip',
        description="Detects brute force attacks using YUGMASTRA intelligence",
        cron_schedule="*/5 * * * *"
    )
