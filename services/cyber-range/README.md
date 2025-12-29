# Cyber Range Simulation Environment

Docker-based simulated enterprise network for training AI agents.

## Architecture

```
cyber-range/
├── src/
│   ├── environment/     # Simulation environment
│   ├── network/         # Virtual network setup
│   ├── services/        # Simulated services
│   ├── monitoring/      # Traffic/log capture
│   └── scenarios/       # Attack scenarios
├── docker/              # Docker configurations
│   ├── web-server/     # Vulnerable web server
│   ├── database/       # Database server
│   ├── endpoints/      # User endpoints
│   └── siem/           # SIEM collector
└── configs/            # Configuration files
```

## Components

### 1. Virtual Network
- Isolated Docker network
- Multiple subnets (DMZ, internal, management)
- Realistic network topology

### 2. Simulated Services
- **Web Server**: Vulnerable web application
- **Database**: MySQL with test data
- **File Server**: SMB/FTP services
- **Mail Server**: SMTP server
- **User Endpoints**: Simulated workstations

### 3. Monitoring
- Packet capture (tcpdump)
- Log aggregation (Fluentd)
- SIEM integration
- Metrics collection

### 4. RL Environment Interface
- Gymnasium-compatible interface
- Observation space: Network state
- Action space: Attack/defense actions
- Reward function: Success/detection metrics

## Usage

```bash
# Start cyber range
docker-compose -f docker/docker-compose.yml up -d

# Reset environment
python src/environment/reset.py

# Run training
python -m src.train_agents
```

## Observation Space

```python
{
    'network_traffic': [packet_features],
    'system_logs': [log_entries],
    'service_states': [service_status],
    'user_behavior': [user_actions],
    'security_events': [siem_alerts]
}
```

## Action Space

### Red Team Actions
- Port scan
- Vulnerability scan
- Exploit execution
- Privilege escalation
- Lateral movement
- Data exfiltration

### Blue Team Actions
- Update firewall rules
- Block IP address
- Isolate host
- Update detection rules
- Patch vulnerability
- Monitor suspicious activity

## Scenarios

1. **Web Application Attack**
   - SQL injection
   - XSS
   - CSRF

2. **Network Intrusion**
   - Port scanning
   - Service exploitation
   - Lateral movement

3. **Phishing Campaign**
   - Email delivery
   - Credential harvesting
   - Post-exploitation

4. **Ransomware Simulation**
   - Initial access
   - Encryption
   - C2 communication
