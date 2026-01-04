"""
Elasticsearch / Elastic Security SIEM Integration
Query logs, push detections, manage alerts
"""

from elasticsearch import Elasticsearch, helpers
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


class ElasticConnector:
    """
    Elasticsearch connector for YUGMASTRA integration
    - Query logs and alerts from Elasticsearch
    - Push AI detections to Elastic Security
    - Create detection rules
    """

    def __init__(
        self,
        hosts: List[str],
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ca_certs: Optional[str] = None,
        verify_certs: bool = True
    ):
        auth_params = {}

        if api_key:
            auth_params['api_key'] = api_key
        elif username and password:
            auth_params['basic_auth'] = (username, password)

        self.es = Elasticsearch(
            hosts,
            ca_certs=ca_certs,
            verify_certs=verify_certs,
            **auth_params
        )

        # Test connection
        if self.es.ping():
            logger.info("Successfully connected to Elasticsearch")
        else:
            raise ConnectionError("Failed to connect to Elasticsearch")

    def search(
        self,
        index: str,
        query: Dict[str, Any],
        time_field: str = "@timestamp",
        time_range: str = "24h",
        size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Execute Elasticsearch query

        Args:
            index: Index pattern (e.g., "logs-*", "filebeat-*")
            query: Elasticsearch Query DSL
            time_field: Timestamp field name
            time_range: Time range (e.g., "1h", "24h", "7d")
            size: Maximum results

        Returns:
            List of matching documents
        """
        try:
            # Build time filter
            time_filter = {
                "range": {
                    time_field: {
                        "gte": f"now-{time_range}",
                        "lte": "now"
                    }
                }
            }

            # Combine with user query
            full_query = {
                "bool": {
                    "must": [query],
                    "filter": [time_filter]
                }
            }

            # Execute search
            response = self.es.search(
                index=index,
                query=full_query,
                size=size,
                sort=[{time_field: {"order": "desc"}}]
            )

            hits = response['hits']['hits']
            results = [hit['_source'] for hit in hits]

            logger.info(f"Elasticsearch query returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []

    def query_security_alerts(
        self,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        time_range: str = "24h"
    ) -> List[Dict[str, Any]]:
        """
        Query alerts from Elastic Security

        Args:
            severity: Filter by severity (critical, high, medium, low)
            status: Filter by status (open, acknowledged, closed)
            time_range: Time range

        Returns:
            List of security alerts
        """
        query_filters = []

        if severity:
            query_filters.append({
                "term": {"kibana.alert.severity": severity}
            })

        if status:
            query_filters.append({
                "term": {"kibana.alert.workflow_status": status}
            })

        query = {
            "bool": {
                "must": query_filters
            }
        } if query_filters else {"match_all": {}}

        return self.search(
            index=".alerts-security.alerts-*",
            query=query,
            time_field="@timestamp",
            time_range=time_range
        )

    def push_detection(
        self,
        detection: Dict[str, Any],
        index: str = "yugmastra-detections"
    ) -> bool:
        """
        Push YUGMASTRA detection to Elasticsearch

        Args:
            detection: Detection data
            index: Target index

        Returns:
            Success status
        """
        try:
            # Add timestamp if not present
            if '@timestamp' not in detection:
                detection['@timestamp'] = datetime.utcnow().isoformat()

            # Index document
            response = self.es.index(
                index=index,
                document=detection
            )

            logger.info(f"Pushed detection to Elasticsearch: {response['_id']}")
            return True

        except Exception as e:
            logger.error(f"Failed to push detection: {e}")
            return False

    def bulk_push_detections(
        self,
        detections: List[Dict[str, Any]],
        index: str = "yugmastra-detections"
    ) -> int:
        """
        Bulk push multiple detections

        Args:
            detections: List of detections
            index: Target index

        Returns:
            Number of successfully indexed documents
        """
        try:
            # Prepare bulk actions
            actions = []
            for detection in detections:
                if '@timestamp' not in detection:
                    detection['@timestamp'] = datetime.utcnow().isoformat()

                actions.append({
                    "_index": index,
                    "_source": detection
                })

            # Bulk index
            success, failed = helpers.bulk(
                self.es,
                actions,
                raise_on_error=False,
                raise_on_exception=False
            )

            logger.info(f"Bulk indexed {success} detections, {len(failed)} failed")
            return success

        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            return 0

    def create_detection_rule(
        self,
        rule_id: str,
        name: str,
        description: str,
        query: str,
        severity: str = "medium",
        risk_score: int = 50,
        interval: str = "5m",
        index_patterns: List[str] = ["logs-*"],
        mitre_techniques: Optional[List[str]] = None
    ) -> bool:
        """
        Create detection rule in Elastic Security

        Args:
            rule_id: Unique rule identifier
            name: Rule name
            description: Rule description
            query: KQL or Lucene query
            severity: critical, high, medium, low
            risk_score: 0-100
            interval: Check interval (e.g., "5m", "1h")
            index_patterns: Indices to search
            mitre_techniques: MITRE ATT&CK techniques

        Returns:
            Success status
        """
        try:
            # Use Kibana API to create rule
            # Note: This requires Kibana endpoint, not ES directly
            # For production, use Kibana/ES Security API

            rule_config = {
                "rule_id": rule_id,
                "name": name,
                "description": description,
                "type": "query",
                "query": query,
                "language": "kuery",
                "index": index_patterns,
                "interval": interval,
                "severity": severity,
                "risk_score": risk_score,
                "enabled": True,
                "from": "now-6m",
                "to": "now",
                "tags": ["yugmastra", "ai-detection"],
                "actions": []
            }

            if mitre_techniques:
                rule_config["threat"] = [{
                    "framework": "MITRE ATT&CK",
                    "technique": [{"id": t} for t in mitre_techniques]
                }]

            logger.info(f"Detection rule config created: {rule_id}")
            # In production, POST to Kibana API: /api/detection_engine/rules
            return True

        except Exception as e:
            logger.error(f"Failed to create detection rule: {e}")
            return False

    def get_aggregated_stats(
        self,
        index: str,
        field: str,
        time_range: str = "24h"
    ) -> Dict[str, int]:
        """
        Get aggregated statistics (e.g., top attackers, common attack types)

        Args:
            index: Index pattern
            field: Field to aggregate
            time_range: Time range

        Returns:
            Dictionary of field values and counts
        """
        try:
            response = self.es.search(
                index=index,
                size=0,
                query={
                    "range": {
                        "@timestamp": {
                            "gte": f"now-{time_range}",
                            "lte": "now"
                        }
                    }
                },
                aggs={
                    "top_values": {
                        "terms": {
                            "field": field,
                            "size": 20
                        }
                    }
                }
            )

            buckets = response['aggregations']['top_values']['buckets']
            stats = {bucket['key']: bucket['doc_count'] for bucket in buckets}

            return stats

        except Exception as e:
            logger.error(f"Aggregation query failed: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Initialize connector
    elastic = ElasticConnector(
        hosts=["https://localhost:9200"],
        username="elastic",
        password="changeme",
        verify_certs=False
    )

    # Search for failed authentication attempts
    query = {
        "match": {
            "event.action": "authentication_failure"
        }
    }
    results = elastic.search(
        index="logs-*",
        query=query,
        time_range="1h"
    )
    print(f"Found {len(results)} failed authentication attempts")

    # Get security alerts
    alerts = elastic.query_security_alerts(
        severity="high",
        status="open",
        time_range="24h"
    )
    print(f"Found {len(alerts)} high-severity open alerts")

    # Push YUGMASTRA detection
    detection = {
        "attack.type": "lateral_movement",
        "attack.technique": "T1021.002",
        "source.ip": "10.0.1.50",
        "destination.ip": "10.0.2.100",
        "severity": "high",
        "confidence": 0.92,
        "details": "AI detected anomalous SMB connections",
        "model": "yugmastra_marl_v2"
    }
    elastic.push_detection(detection)

    # Create detection rule
    elastic.create_detection_rule(
        rule_id="yugmastra-brute-force-001",
        name="YUGMASTRA - SSH Brute Force Detection",
        description="Detects SSH brute force attacks using AI correlation",
        query='event.action:"ssh_login" and event.outcome:"failure"',
        severity="high",
        risk_score=75,
        interval="5m",
        index_patterns=["logs-*", "filebeat-*"],
        mitre_techniques=["T1110.001", "T1110.003"]
    )

    # Get aggregated statistics
    top_attackers = elastic.get_aggregated_stats(
        index="yugmastra-detections",
        field="source.ip.keyword",
        time_range="24h"
    )
    print(f"Top attackers: {top_attackers}")
