# YUGMASTRA API & Webhook Ecosystem

## REST API Endpoints

### Authentication
- `POST /api/v1/auth/login` - User authentication
- `POST /api/v1/auth/logout` - User logout
- `POST /api/v1/auth/refresh` - Refresh access token

### Simulations
- `GET /api/v1/simulations` - List all simulations
- `POST /api/v1/simulations` - Create new simulation
- `GET /api/v1/simulations/{id}` - Get simulation details
- `PUT /api/v1/simulations/{id}` - Update simulation
- `DELETE /api/v1/simulations/{id}` - Delete simulation
- `POST /api/v1/simulations/{id}/start` - Start simulation
- `POST /api/v1/simulations/{id}/stop` - Stop simulation
- `GET /api/v1/simulations/{id}/status` - Get real-time status

### Threat Intelligence
- `GET /api/v1/threats` - Get threat intelligence feed
- `GET /api/v1/threats/{id}` - Get specific threat
- `POST /api/v1/threats/search` - Search threats
- `GET /api/v1/iocs` - Get indicators of compromise
- `POST /api/v1/iocs/correlate` - Correlate observed IOCs

### AI Models
- `GET /api/v1/models` - List AI models
- `POST /api/v1/models/query` - Query AI ensemble
- `GET /api/v1/models/stats` - Model statistics

### Attack Chains
- `POST /api/v1/attack-chains/generate` - Generate MITRE ATT&CK chain
- `GET /api/v1/attack-chains/{id}` - Get attack chain details
- `POST /api/v1/attack-chains/export` - Export for Navigator

### Analytics
- `GET /api/v1/analytics/dashboard` - Dashboard metrics
- `GET /api/v1/analytics/predictions` - Predictive analytics
- `GET /api/v1/analytics/risks` - Risk assessments

### Webhooks
- `POST /api/v1/webhooks` - Create webhook
- `GET /api/v1/webhooks` - List webhooks
- `DELETE /api/v1/webhooks/{id}` - Delete webhook
- `POST /api/v1/webhooks/{id}/test` - Test webhook

## Webhook Events

### Simulation Events
- `simulation.started` - Simulation has started
- `simulation.completed` - Simulation finished
- `simulation.failed` - Simulation encountered error
- `simulation.attack_detected` - Attack was detected
- `simulation.breach_occurred` - Successful breach

### Threat Events
- `threat.critical_detected` - Critical threat identified
- `threat.ioc_matched` - IOC correlation match
- `threat.actor_identified` - Threat actor profiled

### Alert Events
- `alert.high_severity` - High severity alert
- `alert.anomaly_detected` - Anomaly detected
- `alert.compliance_violation` - Compliance issue

## Example Webhook Payload

```json
{
  "event": "simulation.breach_occurred",
  "timestamp": "2025-12-31T21:00:00Z",
  "simulation_id": "sim_12345",
  "data": {
    "attack_type": "SQL Injection",
    "affected_system": "Web Server 1",
    "severity": "high",
    "threat_actor": "APT28",
    "mitre_techniques": ["T1190", "T1059.001"]
  }
}
```

## Integration Examples

### Slack Integration
```bash
curl -X POST https://hooks.slack.com/services/YOUR/WEBHOOK/URL \
  -H 'Content-Type: application/json' \
  -d '{"text": "🚨 Critical threat detected in YUGMASTRA simulation!"}'
```

### PagerDuty Integration
```bash
curl -X POST https://api.pagerduty.com/incidents \
  -H 'Authorization: Token token=YOUR_API_KEY' \
  -H 'Content-Type: application/json' \
  -d '{"incident": {"type": "incident", "title": "YUGMASTRA Alert"}}'
```

### Jira Integration
```bash
curl -X POST https://your-domain.atlassian.net/rest/api/2/issue \
  -H 'Authorization: Basic YOUR_AUTH' \
  -H 'Content-Type: application/json' \
  -d '{"fields": {"project": {"key": "SEC"}, "summary": "Security Incident"}}'
```

## Rate Limits
- 1000 requests per hour per API key
- Burst limit: 100 requests per minute
- WebSocket connections: 10 concurrent per user

## Authentication
Use Bearer token in Authorization header:
```
Authorization: Bearer YOUR_API_TOKEN
```

## STIX/TAXII Support
- STIX 2.1 format for threat intelligence
- TAXII 2.1 server for automated sharing
- Endpoint: `/taxii2/` for TAXII root
