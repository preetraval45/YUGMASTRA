# 🚀 YUGMĀSTRA Quick Start - Preet Raval

## One-Minute Setup

Your system is **READY TO GO**! All Docker containers are running.

---

## 🔗 Access Your System

### Main URL
```
http://localhost:200
```

---

## 🎯 Top 3 Things to Do Now

### 1️⃣ Watch the Live Battle (MOST EXCITING!)
```
http://localhost:200/live-battle
```
See Red Team AI attack your system in real-time while Blue Team AI defends it!

### 2️⃣ Check the Dashboard
```
http://localhost:200/dashboard
```
Overview of all metrics and system status.

### 3️⃣ View Your Profile
```
http://localhost:200/settings
```
Your info: Preet Raval | preetraval45@gmail.com

---

## 📊 All Available Pages

| Page | URL | What It Shows |
|------|-----|---------------|
| **Home** | `/` | Welcome page with your info |
| **🔥 Live Battle** | `/live-battle` | Real-time attack/defense action |
| **Dashboard** | `/dashboard` | Metrics and system overview |
| **Attacks** | `/attacks` | Red Team attack analytics |
| **Defenses** | `/defenses` | Blue Team defense performance |
| **Evolution** | `/evolution` | How AIs are improving |
| **Knowledge Graph** | `/knowledge-graph` | Attack patterns visualization |
| **Settings** | `/settings` | Your profile and preferences |

---

## 🎮 What You'll See in Live Battle

### Attack Types
- SQL Injection
- Cross-Site Scripting (XSS)
- Remote Code Execution (RCE)
- Privilege Escalation
- Lateral Movement
- Data Exfiltration
- DDoS Attacks
- Phishing
- Brute Force

### How to Read the Battle
- **Red numbers** = Red Team (attackers) score
- **Blue numbers** = Blue Team (defenders) score
- **Green bar** = System health
- **Left panel** = Incoming attacks
- **Right panel** = Defense responses

---

## 🐳 Docker Commands

### Check Status
```bash
docker-compose ps
```

### View Logs
```bash
docker-compose logs -f web
```

### Stop Everything
```bash
docker-compose down
```

### Start Everything
```bash
docker-compose up -d
```

### Restart a Service
```bash
docker-compose restart web
```

---

## 💡 Pro Tips

1. **Leave it running** - The longer it runs, the more interesting the battles become
2. **Watch the scores** - They should balance out over time (~50/50)
3. **System health fluctuates** - That's normal! It auto-heals
4. **Check different pages** - Each shows different insights
5. **Pause when needed** - Use the "Pause Battle" button in Live Battle

---

## 🎨 UI Color Guide

### Attack Severity
- 🔴 **Red** = Critical
- 🟠 **Orange** = High
- 🟡 **Yellow** = Medium
- 🔵 **Blue** = Low

### Status Colors
- 🟢 **Green** = Success/Healthy
- 🟡 **Yellow** = Warning/Detected
- 🔴 **Red** = Failed/Critical
- ⚫ **Gray** = Neutral

---

## 📈 Understanding Scores

### Red Team Score
Number of successful attacks that penetrated defenses.

### Blue Team Score
Number of attacks successfully blocked.

### System Health
- Starts at 100%
- Decreases when attacks succeed
- Auto-heals at 0.5% per second
- Visual: Green (safe) → Yellow (stressed) → Red (critical)

---

## 🔧 Customization

Want to change port from 200?

Edit `docker-compose.yml` line 290:
```yaml
ports:
  - "YOUR_PORT:80"  # Change to 300, 8080, etc.
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

---

## 📞 System Info

- **Owner:** Preet Raval
- **Email:** preetraval45@gmail.com
- **Port:** 200
- **Services:** 17 containers
- **Status:** ✅ All running

---

## 🎯 Your Mission

Watch as AI agents learn to attack and defend your system through continuous adversarial training. The goal is to reach Nash equilibrium where both teams are equally skilled!

**Have fun exploring!** 🚀

---

*Need more details? Check [SYSTEM_GUIDE.md](SYSTEM_GUIDE.md) for the complete guide.*
