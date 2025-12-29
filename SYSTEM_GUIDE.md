# YUGMĀSTRA - System Guide for Preet Raval

## System Overview

YUGMĀSTRA is your personal Autonomous Adversary-Defender Co-Evolution Platform where AI agents battle in real-time to improve cybersecurity defenses.

**System Owner:** Preet Raval
**Email:** preetraval45@gmail.com
**Access URL:** http://localhost:200

---

## What's Happening?

This system simulates a real cybersecurity environment where:

1. **Red Team AI** continuously attacks your system using various techniques
2. **Blue Team AI** learns to defend against these attacks in real-time
3. Both teams evolve through adversarial competition
4. You can watch everything happen live!

---

## Key Features

### 🔥 Live Battle Arena (NEW!)
**URL:** http://localhost:200/live-battle

Watch in real-time as:
- Red Team AI launches attacks (SQL Injection, XSS, RCE, etc.)
- Blue Team AI detects and blocks threats
- System health fluctuates based on attack success
- Battle statistics update every second

**Features:**
- Real-time attack feed showing all incoming threats
- Defense action log showing Blue Team responses
- Live scoring: Red Team vs Blue Team
- System health monitoring
- Pause/Resume battle functionality

### 📊 Dashboard
**URL:** http://localhost:200/dashboard

Your personalized control center showing:
- Total training episodes
- Win rates for both teams
- Real-time activity feed
- System status for all services
- Performance charts

### ⚔️ Attack Analytics
**URL:** http://localhost:200/attacks

Detailed breakdown of:
- All attack attempts
- Success rates
- Attack type distribution
- Recent attack history with impact metrics

### 🛡️ Defense Analytics
**URL:** http://localhost:200/defenses

Monitor your defenses:
- Detection rates
- AI-generated defense rules
- False positive analysis
- Adaptive threshold tuning
- Learning progress metrics

### 🧬 Evolution Tracking
**URL:** http://localhost:200/evolution

Track how both AIs improve over time through adversarial co-evolution.

### 🕸️ Knowledge Graph
**URL:** http://localhost:200/knowledge-graph

Visualize attack patterns and defense strategies discovered by the AI.

### ⚙️ Settings
**URL:** http://localhost:200/settings

Manage:
- Your profile (Preet Raval's info is pre-filled)
- Notification preferences
- Training configuration
- Data management

---

## How It Works

### Attack Flow
1. Red Team AI chooses an attack strategy
2. Attack is launched against a target (web_server, database, api_gateway, etc.)
3. Blue Team AI analyzes the attack
4. If detected: Blue Team blocks it (Blue scores +1)
5. If successful: Red Team wins (Red scores +1, system health decreases)

### Defense Learning
- Blue Team AI learns from every attack
- Generates new detection rules automatically
- Adapts thresholds based on false positive rates
- Improves detection accuracy over time

### Co-Evolution
- Both teams push each other to improve
- Red Team discovers new attack patterns
- Blue Team develops better defenses
- System reaches Nash equilibrium

---

## Understanding the Live Battle

### Attack Severity Levels
- **CRITICAL** (Red): Maximum damage potential
- **HIGH** (Orange): Significant threat
- **MEDIUM** (Yellow): Moderate risk
- **LOW** (Blue): Minor concern

### Attack Status
- **Attacking** (Pulsing Red): Attack in progress
- **Detected** (Yellow): Blue Team spotted it
- **Blocked** (Green): Successfully defended
- **Successful** (Red X): Attack succeeded

### System Health
- **Green (70-100%)**: System secure
- **Yellow (30-70%)**: Under stress
- **Red (0-30%)**: Critical state
- Auto-heals slowly over time (0.5% per second)

---

## Docker Services Running

All 17 services are containerized and running:

### Infrastructure
- **PostgreSQL** - Main database
- **Redis** - Caching and queuing
- **Neo4j** - Knowledge graph storage
- **Elasticsearch** - Search and analytics
- **Kibana** - Log visualization
- **Kafka** - Event streaming
- **Prometheus** - Metrics collection
- **Grafana** - Monitoring dashboards
- **MinIO** - Object storage

### Application Services
- **API Gateway** - http://localhost:200/api
- **Web App** - http://localhost:200
- **Red Team AI** - Attack generation service
- **Blue Team AI** - Defense service
- **Evolution Engine** - Co-evolution orchestration
- **Cyber Range** - Training environment
- **Nginx** - Reverse proxy (single entry point on port 200)

---

## Quick Start Guide

1. **Access the system:**
   ```
   Open browser: http://localhost:200
   ```

2. **Watch the live battle:**
   ```
   Click "🔥 Watch Live Battle" or go to /live-battle
   ```

3. **Monitor your system:**
   ```
   Go to Dashboard to see overall metrics
   ```

4. **Analyze attacks:**
   ```
   Visit /attacks to see what Red Team is trying
   ```

5. **Review defenses:**
   ```
   Visit /defenses to see how Blue Team is learning
   ```

---

## Managing Docker Containers

### View all containers:
```bash
docker-compose ps
```

### Stop all services:
```bash
docker-compose down
```

### Start all services:
```bash
docker-compose up -d
```

### View logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker logs yugmastra-web -f
docker logs yugmastra-red-team-ai -f
docker logs yugmastra-blue-team-ai -f
```

### Rebuild after code changes:
```bash
# Rebuild specific service
docker-compose build web
docker-compose up -d web

# Rebuild all
docker-compose build
docker-compose up -d
```

---

## Understanding the Metrics

### Detection Rate
Percentage of attacks that Blue Team successfully detects. Higher is better.

### Win Rate
Percentage of total engagements won by each team. Balanced (~50%) indicates Nash equilibrium.

### System Health
Real-time indicator of system integrity. Decreases with successful attacks, slowly auto-heals.

### False Positive Rate
Percentage of detections that weren't real attacks. Lower is better.

### Effectiveness
How well a defense action worked (0-100%). Higher means better blocking.

---

## Tips for Understanding the System

1. **The battle is continuous** - Attacks happen every 0.8-2 seconds
2. **Blue Team learns from failures** - Every successful Red Team attack makes Blue Team smarter
3. **System auto-heals** - Don't worry if health drops, it recovers over time
4. **Both teams evolve** - Watch win rates converge as they get equally skilled
5. **Detection ≠ Blocking** - Detecting an attack doesn't always mean blocking it

---

## Personalization

The system displays your information:
- Name: **Preet Raval**
- Email: **preetraval45@gmail.com**
- Shown on: Home page, Dashboard, Settings, and Live Battle

---

## Future Enhancements

The system is designed to be extended with:
- Real machine learning models (currently simulated)
- Actual vulnerability testing
- Custom attack scenarios
- Defense strategy export
- Integration with real security tools
- Historical replay of battles
- Team comparison analytics

---

## Troubleshooting

### If containers don't start:
```bash
# Check logs
docker-compose logs

# Restart specific service
docker-compose restart web
```

### If web app shows errors:
```bash
# Check web logs
docker logs yugmastra-web --tail 50

# Rebuild web
docker-compose build web
docker-compose up -d web
```

### If port 200 is busy:
Edit `docker-compose.yml` and change:
```yaml
ports:
  - "200:80"  # Change 200 to another port like 300
```

---

## Support

Created for: **Preet Raval**
Contact: **preetraval45@gmail.com**
System: **YUGMĀSTRA - Autonomous Adversary-Defender Co-Evolution Platform**

---

Enjoy watching the AI battle unfold! 🔥⚔️🛡️
