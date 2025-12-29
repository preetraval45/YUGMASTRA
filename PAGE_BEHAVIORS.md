# 📄 What Happens on Each Page - YUGMĀSTRA

**System Owner:** Preet Raval (preetraval45@gmail.com)

---

## 🎮 Live Battle Page - The Main Event

**URL:** http://localhost:200/live-battle

### What Happens When Battle is Running

#### Real-Time Actions (Every 0.8-2 seconds)
1. **New Attack Generated**
   - Red Team AI selects random attack type
   - Chooses target system
   - Attack appears in "Incoming Attacks" feed
   - Status: "attacking" (red pulsing icon)

2. **Defense Response (1-3 seconds later)**
   - Blue Team AI analyzes the attack
   - 70% chance of detection
   - If detected:
     - Calculates effectiveness (60-100%)
     - If effectiveness > 70%: Attack BLOCKED ✅
     - If effectiveness ≤ 70%: Attack DETECTED but succeeded ⚠️
   - If not detected: Attack SUCCESSFUL ❌

3. **Score Updates**
   - Blue Team blocks attack → Blue +1 point
   - Red Team succeeds → Red +1 point

4. **System Health Changes**
   - Successful attack → Health -12%
   - Detected but not blocked → Health -3% to -8%
   - Blocked attack → No damage
   - Auto-heal → +0.5% per second

5. **Battle Timer**
   - Increments every second when running
   - Displays in MM:SS format
   - Pauses when battle paused

#### Visual Updates
- Attack feed scrolls automatically (shows last 20 attacks)
- Defense action feed updates (shows last 15 defenses)
- Health bar animates smoothly
- Scores animate on change
- Colors change based on status

### Controls Available

#### While Battle is Active
- **⏸ Pause Battle** - Stops attack generation, freezes timer
- **⏹ End Battle** - Stops everything, shows final results

#### When Battle is Paused
- **▶ Resume Battle** - Continues from where paused
- **⏹ End Battle** - Stops and shows results

#### When Battle Has Ended
- **🔄 Start New Battle** - Resets everything:
  - Clears all attacks and defenses
  - Resets health to 100%
  - Resets scores to 0-0
  - Resets timer to 0:00
  - Starts fresh battle

### What You See When Battle Ends
```
🏁 Battle Ended!
┌─────────────┬──────────┬─────────────┐
│  Duration   │  Winner  │ Final Score │
│   5:32      │ Blue Team│   45 - 38   │
└─────────────┴──────────┴─────────────┘
```

Winner determined by:
- Red score > Blue score → 🔴 Red Team wins
- Blue score > Red score → 🔵 Blue Team wins
- Red score = Blue score → 🤝 Draw

---

## 📊 Dashboard Page

**URL:** http://localhost:200/dashboard

### What Happens When You're On This Page

#### Static Display (Simulated Data)
Currently shows **hardcoded metrics**:
- Total Episodes: 523
- Red Wins: 271
- Blue Wins: 252
- Active Attacks: 12
- Blocked Attacks: 34

#### Real-Time Updates Feed
- New message every 5 seconds
- Random selection from predefined updates:
  - "Red agent discovered new attack path"
  - "Blue agent updated detection rule"
  - "Nash equilibrium distance decreased to 0.23"
  - "New vulnerability chain detected"
  - "Defense strategy adapted successfully"

#### What Displays
1. **Metrics Grid** (4 cards)
   - Total Episodes
   - Red Team Wins (with win rate %)
   - Blue Team Wins (with detection rate %)
   - Evolution Phase status

2. **Real-time Activity Feed**
   - Shows last 10 updates
   - Auto-scrolls
   - Each update timestamped "Just now"

3. **System Status Panel**
   - Cyber Range: Online
   - Red Team AI: Training
   - Blue Team AI: Training
   - Knowledge Graph: Indexing
   - API Gateway: Healthy

4. **Performance Charts**
   - Win rate trend (7-day bars)
   - Detection rate trend
   - Equilibrium distance

### Connection to Live Battle
❌ **NOT CONNECTED** - Dashboard metrics don't update from Live Battle
- Dashboard shows simulated static data
- Live Battle runs independently
- **Future improvement:** Sync Live Battle stats to Dashboard

---

## ⚔️ Attacks Page

**URL:** http://localhost:200/attacks

### What Happens On This Page

#### Static Display
Shows **hardcoded attack analytics**:

1. **Stats Cards**
   - Total Attacks: 1,523
   - Successful: 891 (58.5%)
   - Detected: 642 (42%)
   - Avg Time to Detect: 45.3s

2. **Attack Type Distribution**
   - Web Exploit: 342 (22%)
   - Phishing: 298 (20%)
   - Lateral Movement: 267 (18%)
   - Privilege Escalation: 245 (16%)
   - Data Exfiltration: 189 (12%)
   - Port Scanning: 182 (12%)

3. **Recent Attacks Table**
   - 5 sample attacks
   - Shows: Type, Target, Status, Detection, Impact
   - Static data (doesn't update)

### Connection to Live Battle
❌ **NOT CONNECTED** - Shows separate simulated data
- **Future improvement:** Show actual attacks from Live Battle

---

## 🛡️ Defenses Page

**URL:** http://localhost:200/defenses

### What Happens On This Page

#### Static Display
Shows **hardcoded defense analytics**:

1. **Stats Cards**
   - Total Detections: 642
   - True Positives: 588 (91.6%)
   - False Positives: 54 (8.4%)
   - Avg Response Time: 12.4s

2. **Detection Rate Trend**
   - 30-day chart
   - Simulated sine wave pattern
   - Shows improvement over time

3. **AI-Generated Detection Rules**
   - 5 sample rules:
     - SQL Injection Pattern (92% confidence, 3% FP rate)
     - Lateral Movement Detector (87% confidence, 8% FP)
     - Anomalous Traffic Pattern (79% confidence, 12% FP)
     - Privilege Escalation Alert (94% confidence, 2% FP)
     - Data Exfiltration Monitor (88% confidence, 6% FP)

4. **Adaptive Thresholds**
   - Shows 3 auto-adjusting thresholds
   - Displays adjustment direction (↑↓→)

5. **Learning Progress**
   - Improvement Rate: +18.2%
   - Strategies Learned: 127

### Connection to Live Battle
❌ **NOT CONNECTED** - Separate simulated data
- **Future improvement:** Show actual defense stats from Live Battle

---

## 🧬 Evolution Page

**URL:** http://localhost:200/evolution

### What Happens On This Page

#### Static Display
Shows **simulated co-evolution metrics**

Currently displays:
- Evolution phases
- Strategy convergence
- Skill progression graphs
- Nash equilibrium tracking

### Connection to Live Battle
❌ **NOT CONNECTED**
- **Future improvement:** Track actual strategy evolution from battles

---

## 🕸️ Knowledge Graph Page

**URL:** http://localhost:200/knowledge-graph

### What Happens On This Page

#### Static Display
Shows **visualization concept**

### Connection to Live Battle
❌ **NOT CONNECTED**
- **Future improvement:** Build graph from actual attack/defense patterns

---

## ⚙️ Settings Page

**URL:** http://localhost:200/settings

### What Happens On This Page

#### Your Profile Section
Shows **your information**:
- Full Name: Preet Raval
- Email: preetraval45@gmail.com
- Organization: YUGMĀSTRA Research Lab

**System Owner badge** displays:
- "This system is actively defending against Red Team AI attacks"

#### Configuration Options
1. **Notifications**
   - Email for new attacks (checkbox)
   - Push for critical alerts (checkbox)
   - Weekly reports (checkbox)

2. **Training Configuration**
   - Population Size: 10
   - Initial Difficulty: slider
   - Learning Rate: 0.0003

3. **Data & Privacy**
   - Export Training Data
   - Clear Cache
   - Delete All Data

### Functionality
⚠️ **NOT FUNCTIONAL** - Checkboxes and inputs work but don't save
- **Future improvement:** Save to database, apply to Live Battle

---

## 🏠 Home Page

**URL:** http://localhost:200

### What Happens On This Page

#### Static Display
- Welcome page with your name and email
- System description
- Three feature cards (Red Team, Blue Team, Co-Evolution)
- Key features grid

#### Action Buttons
1. **🔥 Watch Live Battle** → Goes to /live-battle
2. **Launch Dashboard** → Goes to /dashboard
3. **View Evolution** → Goes to /evolution

### No Active Processes
Just a landing page, no real-time updates

---

## 📡 WHAT'S CONNECTED VS WHAT'S NOT

### ✅ Fully Functional (Self-Contained)
| Page | Status | What Works |
|------|--------|------------|
| **Live Battle** | ✅ WORKS | Complete battle simulation, pause/resume/end, timer |
| **Home** | ✅ WORKS | Navigation, displays your info |

### ⚠️ Static Display Only
| Page | Status | What Shows |
|------|--------|------------|
| **Dashboard** | ⚠️ STATIC | Hardcoded metrics, simulated updates |
| **Attacks** | ⚠️ STATIC | Sample attack data |
| **Defenses** | ⚠️ STATIC | Sample defense data |
| **Evolution** | ⚠️ STATIC | Simulated evolution |
| **Knowledge Graph** | ⚠️ STATIC | Placeholder |
| **Settings** | ⚠️ STATIC | Form display only, no saving |

### ❌ Not Connected Between Pages
- Live Battle ❌➡️ Dashboard
- Live Battle ❌➡️ Attacks
- Live Battle ❌➡️ Defenses
- Live Battle ❌➡️ Evolution
- Settings ❌➡️ Live Battle

---

## 🔮 FUTURE: What SHOULD Happen

### When Live Battle is Running

#### Dashboard Should Show:
- Total Attacks: (count from Live Battle)
- Red/Blue Wins: (actual scores)
- Real-time activity from battle
- Live system health

#### Attacks Page Should Show:
- All attacks from Live Battle
- Real-time statistics
- Actual attack distribution
- Live attack feed

#### Defenses Page Should Show:
- All defenses from Live Battle
- Actual detection rate
- Real false positive tracking
- Live defense rules generated

#### Evolution Page Should Show:
- Actual strategy changes
- Real win rate trends
- True Nash equilibrium distance

#### Knowledge Graph Should Show:
- Attack chains from battles
- Defense patterns discovered
- Vulnerability relationships

#### Settings Should:
- Actually save preferences
- Apply to battle parameters
- Configure AI behavior

---

## 🎯 CURRENT REALITY

**Only Live Battle page is truly interactive and functional.**

**Other pages are:**
- Beautiful UI mockups
- Static demonstrations
- Educational displays
- Simulated data

**To make them all work together:**
Need to implement:
1. WebSocket for real-time data streaming
2. Database to persist battle data
3. API endpoints to serve data
4. State management across pages
5. Real backend services

---

## 💡 QUICK SUMMARY

### What Works Now
- ✅ Live Battle: Full simulation with pause/resume/end/reset
- ✅ Navigation between pages
- ✅ UI looks professional on all pages
- ✅ Your personal info displayed throughout

### What's Simulated
- ⚠️ All metrics on other pages
- ⚠️ Real-time updates (random, not from battle)
- ⚠️ Settings don't actually change behavior
- ⚠️ No data persistence

### What Doesn't Connect
- ❌ Pages don't share data
- ❌ Battle stats don't feed dashboard
- ❌ No database storage
- ❌ No real WebSocket connections

---

**Bottom Line:** Live Battle is a complete, working simulation. Other pages are beautiful mockups waiting to be connected to real data sources!

Ready to connect everything? Just say "implement" and specify what you want! 🚀
