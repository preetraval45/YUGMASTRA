"use client"

import { useState, useEffect } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { AlertTriangle, Shield, Zap, CheckCircle, Clock, PlayCircle, StopCircle, Download, Terminal, BookOpen, Lightbulb, Code2, Activity } from 'lucide-react'

interface Incident {
  id: string
  title: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  status: 'new' | 'analyzing' | 'containing' | 'eradicating' | 'recovering' | 'resolved'
  detectedAt: string
  affectedAssets: string[]
  indicators: string[]
  aiConfidence: number
}

interface PlaybookStep {
  id: string
  action: string
  status: 'pending' | 'executing' | 'completed' | 'failed'
  result?: string
  timestamp?: string
}

interface PlaybookExample {
  title: string
  incidentType: string
  description: string
  technicalDetails: string
  nistPhases: {
    preparation: string[]
    detection: string[]
    containment: string[]
    eradication: string[]
    recovery: string[]
    postIncident: string[]
  }
  automatedSteps: string[]
  estimatedTimeline: string
  severity: 'critical' | 'high' | 'medium'
}

const PLAYBOOK_EXAMPLES: PlaybookExample[] = [
  {
    title: 'Ransomware Attack Response',
    incidentType: 'Ransomware',
    description: 'Comprehensive playbook for responding to ransomware infections that encrypt files and demand payment.',
    technicalDetails: 'Ransomware typically spreads through phishing emails, exploit kits, or compromised Remote Desktop Protocol (RDP) connections. Modern ransomware families like Ryuk, Sodinokibi, and LockBit use sophisticated encryption algorithms (AES-256, RSA-2048) and often exfiltrate data before encryption for double extortion.',
    nistPhases: {
      preparation: [
        'Maintain offline, immutable backups tested for restoration',
        'Deploy endpoint detection and response (EDR) with behavioral analysis',
        'Segment network to limit lateral movement',
        'Implement application whitelisting and disable macros',
        'Train users on phishing recognition'
      ],
      detection: [
        'Monitor for mass file encryption events (high volume .encrypted, .locked extensions)',
        'Detect unusual process creation chains (e.g., WMIC, PowerShell spawning encryption tools)',
        'Alert on ransom note file creation (README.txt, HOW_TO_DECRYPT.html)',
        'Identify spike in file system I/O operations',
        'Correlate with threat intelligence feeds for known ransomware indicators'
      ],
      containment: [
        'Immediately isolate infected systems from network (pull cables, disable network adapters)',
        'Disable user accounts showing suspicious activity',
        'Block command and control (C2) IPs/domains at firewall and DNS',
        'Prevent spread to backup systems - verify backup integrity',
        'Capture memory dumps and disk images for forensic analysis'
      ],
      eradication: [
        'Identify ransomware variant using hash analysis and behavioral signatures',
        'Remove malware executables, registry keys, and persistence mechanisms',
        'Patch vulnerabilities exploited for initial access (CVE remediation)',
        'Reset credentials for all potentially compromised accounts',
        'Review and remove unauthorized remote access tools'
      ],
      recovery: [
        'Restore systems from clean, verified backups (oldest safe backup)',
        'Rebuild compromised systems from known-good gold images',
        'Restore data incrementally, scanning each file for malware',
        'Monitor restored systems for 72+ hours for re-infection',
        'Validate business-critical applications before full production'
      ],
      postIncident: [
        'Document attack timeline, tactics, techniques, and procedures (TTPs)',
        'Identify root cause - phishing, unpatched VPN, weak RDP credentials',
        'Calculate financial impact (downtime, recovery costs, ransom demand)',
        'Update incident response plan based on lessons learned',
        'Report to authorities (FBI IC3, CISA) and insurance providers',
        'Conduct tabletop exercise to validate improved defenses'
      ]
    },
    automatedSteps: [
      'Auto-isolate infected hosts via EDR/SOAR integration',
      'Automated backup verification and integrity checks',
      'Dynamic C2 blocking via threat intelligence feeds',
      'Automated credential rotation for affected systems',
      'Orchestrated malware removal via EDR platforms'
    ],
    estimatedTimeline: '24-72 hours for containment and eradication, 1-2 weeks for full recovery',
    severity: 'critical'
  },
  {
    title: 'Data Breach & Exfiltration',
    incidentType: 'Data Breach',
    description: 'Response playbook for unauthorized access to sensitive data with evidence of exfiltration to external systems.',
    technicalDetails: 'Data breaches often involve multi-stage attacks: initial compromise via spear-phishing or credential stuffing, privilege escalation using tools like Mimikatz or Bloodhound, lateral movement to database servers, and exfiltration via DNS tunneling, HTTPS uploads to cloud storage, or encrypted channels.',
    nistPhases: {
      preparation: [
        'Implement Data Loss Prevention (DLP) solutions with content inspection',
        'Deploy database activity monitoring (DAM) and file integrity monitoring (FIM)',
        'Enable logging for all data access events (S3, databases, file shares)',
        'Classify data and apply appropriate access controls (least privilege)',
        'Encrypt sensitive data at rest and in transit (AES-256, TLS 1.3)'
      ],
      detection: [
        'Alert on large data transfers to external IPs or cloud storage',
        'Detect database queries returning abnormally large result sets',
        'Monitor for use of compression tools (7zip, WinRAR) before transfer',
        'Identify authentication from unusual geolocations or VPN exit nodes',
        'Correlate user behavior analytics (UBA) for anomalous data access patterns'
      ],
      containment: [
        'Revoke access credentials for compromised accounts immediately',
        'Block egress traffic to identified exfiltration destinations',
        'Isolate affected database and file servers from production network',
        'Enable enhanced logging and full packet capture for forensics',
        'Preserve evidence: memory dumps, network traffic, authentication logs'
      ],
      eradication: [
        'Identify all systems accessed by attacker using lateral movement analysis',
        'Remove persistent backdoors, web shells, and remote access tools',
        'Patch vulnerabilities used for privilege escalation (e.g., PrintNightmare, ZeroLogon)',
        'Reset all privileged credentials and enforce MFA',
        'Review and remove any unauthorized user accounts or service principals'
      ],
      recovery: [
        'Restore database from clean backup if data integrity is compromised',
        'Rebuild compromised servers from trusted baselines',
        'Implement enhanced monitoring on restored systems',
        'Conduct security validation testing before reconnecting to production',
        'Re-enable business processes with heightened monitoring'
      ],
      postIncident: [
        'Determine scope of data exposed - PII, PHI, payment cards, trade secrets',
        'Notify affected individuals per breach notification laws (GDPR 72hr, state laws)',
        'File regulatory reports (HHS for HIPAA, state AGs, credit bureaus)',
        'Engage forensic investigators to determine full attack scope',
        'Implement data access controls and DLP policies to prevent recurrence',
        'Consider credit monitoring services for affected individuals'
      ]
    },
    automatedSteps: [
      'Automated credential revocation via IAM integration',
      'Dynamic firewall rules to block exfiltration destinations',
      'SOAR playbook for evidence preservation and chain of custody',
      'Automated notification workflows for legal and compliance teams',
      'Integration with threat intelligence for attacker infrastructure tracking'
    ],
    estimatedTimeline: '12-48 hours for containment, 2-4 weeks for full investigation and remediation',
    severity: 'critical'
  },
  {
    title: 'Advanced Persistent Threat (APT) Campaign',
    incidentType: 'APT',
    description: 'Sophisticated, long-term intrusion by nation-state or organized cybercrime groups with strategic objectives.',
    technicalDetails: 'APT actors use advanced tradecraft including zero-day exploits, custom malware, living-off-the-land binaries (LOLBins), supply chain compromises, and encrypted C2 channels. They maintain persistent access through multiple backdoors, rootkits, and compromised legitimate accounts. Common APT groups include APT29 (Cozy Bear), APT28 (Fancy Bear), Lazarus Group, and APT41.',
    nistPhases: {
      preparation: [
        'Deploy network detection and response (NDR) with full packet capture',
        'Implement zero trust architecture with micro-segmentation',
        'Enable enhanced logging for PowerShell, WMI, and administrative activities',
        'Conduct regular threat hunting exercises and red team assessments',
        'Maintain threat intelligence feeds for APT indicators and TTPs'
      ],
      detection: [
        'Identify unusual lateral movement patterns (WMI, PsExec, remote scheduled tasks)',
        'Detect beaconing to external IPs with regular time intervals',
        'Monitor for credential dumping (LSASS access, DCSync, Golden Ticket)',
        'Alert on rare or suspicious parent-child process relationships',
        'Correlate events across multiple systems for attack chain reconstruction'
      ],
      containment: [
        'DO NOT alert attackers - maintain normal operations while investigating',
        'Identify all compromised systems through artifact analysis and network telemetry',
        'Map attacker infrastructure (C2 servers, staging servers, exfiltration points)',
        'Coordinate simultaneous remediation across all affected systems',
        'Engage external incident response firm with APT experience'
      ],
      eradication: [
        'Remove all identified malware, backdoors, and persistence mechanisms simultaneously',
        'Rebuild compromised domain controllers and critical infrastructure',
        'Reset all credentials including service accounts, krbtgt, and admin passwords',
        'Revoke and reissue certificates if PKI compromise suspected',
        'Replace hardware if firmware-level compromise detected (UEFI/BIOS)'
      ],
      recovery: [
        'Restore systems from pre-compromise backups validated as clean',
        'Implement enhanced detection rules for APT TTPs observed',
        'Deploy deception technology (honeypots, honeyfiles) to detect re-intrusion',
        'Monitor for 90+ days for signs of persistent access or re-compromise',
        'Conduct continuous threat hunting for residual attacker presence'
      ],
      postIncident: [
        'Conduct comprehensive threat intelligence analysis of APT campaign',
        'Attribute attack to threat actor group using MITRE ATT&CK framework mapping',
        'Share indicators of compromise (IOCs) with information sharing groups (ISACs)',
        'Report to national CERT/CSIRT and law enforcement (FBI, Secret Service)',
        'Implement security architecture changes to address identified gaps',
        'Consider retainer with specialized APT response team'
      ]
    },
    automatedSteps: [
      'Automated IOC sweeping across enterprise using EDR platforms',
      'Behavioral analytics for lateral movement detection',
      'Automated network segmentation to contain spread',
      'SOAR orchestration for coordinated cross-system remediation',
      'Continuous threat hunting automation using ML models'
    ],
    estimatedTimeline: '1-2 weeks for investigation, 2-4 weeks for coordinated eradication, 3-6 months monitoring',
    severity: 'critical'
  },
  {
    title: 'Distributed Denial of Service (DDoS) Attack',
    incidentType: 'DDoS',
    description: 'Large-scale attack overwhelming network infrastructure or applications with malicious traffic from distributed sources.',
    technicalDetails: 'Modern DDoS attacks use botnets of compromised IoT devices (Mirai), amplification techniques (DNS, NTP, memcached reflection), and application-layer attacks (HTTP floods, Slowloris). Attack volumes can exceed 1 Tbps using techniques like UDP/TCP SYN floods, amplification factor attacks, and sophisticated application-layer attacks that mimic legitimate traffic.',
    nistPhases: {
      preparation: [
        'Deploy DDoS mitigation service (Cloudflare, Akamai, AWS Shield)',
        'Implement rate limiting and connection limits at application and network layers',
        'Configure anycast routing for traffic distribution',
        'Establish bandwidth capacity baselines and alerting thresholds',
        'Create communication plan for stakeholders during outages'
      ],
      detection: [
        'Monitor for sudden traffic spikes (10x+ baseline)',
        'Detect high rates of SYN packets, UDP floods, or ICMP traffic',
        'Identify traffic from known botnet IP ranges',
        'Alert on application slowness, timeout errors, and resource exhaustion',
        'Analyze traffic patterns for amplification attack signatures'
      ],
      containment: [
        'Activate DDoS mitigation service (enable scrubbing centers)',
        'Implement upstream filtering with ISP support',
        'Enable geo-blocking if attack originates from specific regions',
        'Apply rate limiting and connection throttling',
        'Failover to redundant infrastructure if available'
      ],
      eradication: [
        'Work with DDoS mitigation provider to identify attack vectors',
        'Block attacking IP addresses and ASNs at network edge',
        'Implement behavioral-based filtering to distinguish legitimate from attack traffic',
        'Update firewall and IPS rules to drop malicious packets',
        'Coordinate with law enforcement and ISP to trace attack source'
      ],
      recovery: [
        'Gradually restore normal traffic routing after attack subsides',
        'Monitor for secondary attacks or attack vector changes',
        'Validate application performance and functionality',
        'Assess infrastructure damage and capacity limits tested',
        'Restore any systems that were taken offline for protection'
      ],
      postIncident: [
        'Analyze attack telemetry to identify tactics and peak volumes',
        'Calculate business impact (downtime, revenue loss, mitigation costs)',
        'Optimize DDoS defenses based on observed attack patterns',
        'Consider increasing bandwidth capacity or CDN coverage',
        'Report attack to abuse contacts and law enforcement',
        'Update incident response procedures for faster activation'
      ]
    },
    automatedSteps: [
      'Automated traffic analysis and anomaly detection',
      'Auto-activation of DDoS mitigation upon threshold breach',
      'Dynamic rate limiting based on traffic patterns',
      'Automated failover to backup infrastructure',
      'Real-time traffic scrubbing via cloud providers'
    ],
    estimatedTimeline: '2-12 hours for mitigation activation, attack duration varies (minutes to days)',
    severity: 'high'
  }
]

export default function IncidentResponsePage() {
  const [activeTab, setActiveTab] = useState('respond')
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null)
  const [playbook, setPlaybook] = useState<PlaybookStep[]>([])
  const [autoMode, setAutoMode] = useState(false)

  useEffect(() => {
    // Load mock incidents
    const mockIncidents: Incident[] = [
      {
        id: 'INC-2025-001',
        title: 'Ransomware Detected - Encryption Activity',
        severity: 'critical',
        status: 'analyzing',
        detectedAt: new Date().toISOString(),
        affectedAssets: ['WKS-001', 'WKS-002', 'FILE-SRV-01'],
        indicators: ['suspicious_encryption', 'mass_file_modification', 'c2_beaconing'],
        aiConfidence: 94
      },
      {
        id: 'INC-2025-002',
        title: 'Lateral Movement - Pass-the-Hash Attack',
        severity: 'high',
        status: 'containing',
        detectedAt: new Date(Date.now() - 3600000).toISOString(),
        affectedAssets: ['DC-01', 'WKS-005'],
        indicators: ['abnormal_auth', 'ntlm_relay', 'privilege_escalation'],
        aiConfidence: 87
      },
      {
        id: 'INC-2025-003',
        title: 'Data Exfiltration Attempt via DNS Tunneling',
        severity: 'high',
        status: 'new',
        detectedAt: new Date(Date.now() - 7200000).toISOString(),
        affectedAssets: ['WEB-SRV-01'],
        indicators: ['dns_tunneling', 'large_dns_queries', 'data_exfiltration'],
        aiConfidence: 91
      }
    ]
    setIncidents(mockIncidents)
  }, [])

  const generatePlaybook = async (incident: Incident) => {
    setSelectedIncident(incident)

    // AI-generated playbook steps
    const steps: PlaybookStep[] = [
      { id: '1', action: 'Isolate affected hosts from network', status: 'pending' },
      { id: '2', action: 'Suspend user accounts with suspicious activity', status: 'pending' },
      { id: '3', action: 'Collect memory dumps from infected systems', status: 'pending' },
      { id: '4', action: 'Analyze malware samples and IOCs', status: 'pending' },
      { id: '5', action: 'Block malicious IPs at firewall', status: 'pending' },
      { id: '6', action: 'Deploy detection rules for similar attacks', status: 'pending' },
      { id: '7', action: 'Restore systems from clean backups', status: 'pending' },
      { id: '8', action: 'Conduct post-incident review', status: 'pending' }
    ]

    setPlaybook(steps)
  }

  const executePlaybook = async () => {
    if (!autoMode) return

    for (let i = 0; i < playbook.length; i++) {
      // Update step to executing
      setPlaybook(prev => prev.map((step, idx) =>
        idx === i ? { ...step, status: 'executing' } : step
      ))

      // Simulate execution
      await new Promise(resolve => setTimeout(resolve, 2000))

      // Mark as completed
      setPlaybook(prev => prev.map((step, idx) =>
        idx === i ? {
          ...step,
          status: 'completed',
          result: 'Success',
          timestamp: new Date().toISOString()
        } : step
      ))
    }

    // Update incident status
    if (selectedIncident) {
      setIncidents(prev => prev.map(inc =>
        inc.id === selectedIncident.id
          ? { ...inc, status: 'resolved' }
          : inc
      ))
    }
  }

  useEffect(() => {
    if (autoMode && playbook.length > 0) {
      executePlaybook()
    }
  }, [autoMode])

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/50'
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/50'
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50'
      case 'low': return 'bg-blue-500/20 text-blue-400 border-blue-500/50'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50'
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'new': return 'bg-purple-500/20 text-purple-400 border-purple-500/50'
      case 'analyzing': return 'bg-blue-500/20 text-blue-400 border-blue-500/50'
      case 'containing': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50'
      case 'eradicating': return 'bg-orange-500/20 text-orange-400 border-orange-500/50'
      case 'recovering': return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/50'
      case 'resolved': return 'bg-green-500/20 text-green-400 border-green-500/50'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50'
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-red-400 to-orange-600 bg-clip-text text-transparent">
            Automated Incident Response
          </h1>
          <p className="text-gray-400 mt-2">AI-powered incident detection, analysis, and orchestrated response</p>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="bg-black/40 border-red-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Active Incidents</p>
              <p className="text-3xl font-bold text-white mt-1">
                {incidents.filter(i => i.status !== 'resolved').length}
              </p>
            </div>
            <AlertTriangle className="h-8 w-8 text-red-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-orange-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Critical</p>
              <p className="text-3xl font-bold text-orange-400 mt-1">
                {incidents.filter(i => i.severity === 'critical').length}
              </p>
            </div>
            <Zap className="h-8 w-8 text-orange-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-green-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Resolved</p>
              <p className="text-3xl font-bold text-green-400 mt-1">
                {incidents.filter(i => i.status === 'resolved').length}
              </p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-400" />
          </div>
        </Card>

        <Card className="bg-black/40 border-blue-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Avg Response Time</p>
              <p className="text-3xl font-bold text-blue-400 mt-1">4.2m</p>
            </div>
            <Clock className="h-8 w-8 text-blue-400" />
          </div>
        </Card>
      </div>

      {/* Main Tabs */}
      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 h-auto p-1 bg-muted">
          <TabsTrigger value="respond" className="data-[state=active]:bg-primary">
            <Activity className="mr-2 h-4 w-4" />
            Respond
          </TabsTrigger>
          <TabsTrigger value="learn" className="data-[state=active]:bg-primary">
            <BookOpen className="mr-2 h-4 w-4" />
            Learn
          </TabsTrigger>
          <TabsTrigger value="examples" className="data-[state=active]:bg-primary">
            <Lightbulb className="mr-2 h-4 w-4" />
            Playbooks
          </TabsTrigger>
        </TabsList>

        {/* Respond Tab */}
        <TabsContent value="respond">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Incidents List */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-white">Active Incidents</h2>
                <Button
                  onClick={() => setAutoMode(!autoMode)}
                  className={autoMode ? 'bg-red-500 hover:bg-red-600' : 'bg-green-500 hover:bg-green-600'}
                  size="sm"
                >
                  {autoMode ? <StopCircle className="mr-2 h-4 w-4" /> : <PlayCircle className="mr-2 h-4 w-4" />}
                  {autoMode ? 'Auto Mode ON' : 'Manual Mode'}
                </Button>
              </div>
              {incidents.map((incident) => (
                <Card
                  key={incident.id}
                  className={`bg-black/40 border-red-500/20 p-6 cursor-pointer hover:border-red-500/40 transition-all ${
                    selectedIncident?.id === incident.id ? 'border-red-500 ring-2 ring-red-500/20' : ''
                  }`}
                  onClick={() => generatePlaybook(incident)}
                >
                  <div className="space-y-3">
                    <div className="flex items-start justify-between">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <h3 className="text-lg font-semibold text-white">{incident.title}</h3>
                        </div>
                        <p className="text-gray-400 text-sm">{incident.id}</p>
                      </div>
                      <div className="flex flex-col gap-2 items-end">
                        <Badge className={getSeverityColor(incident.severity)}>
                          {incident.severity.toUpperCase()}
                        </Badge>
                        <Badge className={getStatusColor(incident.status)}>
                          {incident.status}
                        </Badge>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-gray-400">AI Confidence</p>
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-2 bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-blue-500 to-cyan-500"
                              style={{ width: `${incident.aiConfidence}%` }}
                            />
                          </div>
                          <span className="text-white font-semibold">{incident.aiConfidence}%</span>
                        </div>
                      </div>
                      <div>
                        <p className="text-gray-400">Detected</p>
                        <p className="text-white">{new Date(incident.detectedAt).toLocaleTimeString()}</p>
                      </div>
                    </div>

                    <div>
                      <p className="text-gray-400 text-sm mb-1">Affected Assets ({incident.affectedAssets.length})</p>
                      <div className="flex flex-wrap gap-1">
                        {incident.affectedAssets.map((asset) => (
                          <Badge key={asset} className="bg-gray-500/20 text-gray-300 border-gray-500/50 text-xs">
                            {asset}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div>
                      <p className="text-gray-400 text-sm mb-1">Indicators</p>
                      <div className="flex flex-wrap gap-1">
                        {incident.indicators.map((indicator) => (
                          <Badge key={indicator} className="bg-red-500/20 text-red-400 border-red-500/50 text-xs">
                            {indicator}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>

            {/* Response Playbook */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-white">AI-Generated Response Playbook</h2>
                {playbook.length > 0 && (
                  <Button size="sm" className="bg-blue-500 hover:bg-blue-600">
                    <Download className="mr-2 h-4 w-4" />
                    Export
                  </Button>
                )}
              </div>

              {!selectedIncident ? (
                <Card className="bg-black/40 border-gray-500/20 p-12 text-center">
                  <Shield className="h-12 w-12 text-gray-500 mx-auto mb-4" />
                  <p className="text-gray-400">Select an incident to generate response playbook</p>
                </Card>
              ) : (
                <Card className="bg-black/40 border-blue-500/20 p-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between pb-4 border-b border-gray-700">
                      <div>
                        <h3 className="text-lg font-semibold text-white">Response Plan</h3>
                        <p className="text-gray-400 text-sm">for {selectedIncident.id}</p>
                      </div>
                      <Button
                        onClick={() => setAutoMode(true)}
                        disabled={autoMode}
                        className="bg-green-500 hover:bg-green-600"
                      >
                        <PlayCircle className="mr-2 h-4 w-4" />
                        Execute Playbook
                      </Button>
                    </div>

                    <div className="space-y-3 max-h-[600px] overflow-y-auto">
                      {playbook.map((step, index) => (
                        <div
                          key={step.id}
                          className={`p-4 rounded-lg border ${
                            step.status === 'completed'
                              ? 'bg-green-500/10 border-green-500/50'
                              : step.status === 'executing'
                              ? 'bg-blue-500/10 border-blue-500/50 animate-pulse'
                              : step.status === 'failed'
                              ? 'bg-red-500/10 border-red-500/50'
                              : 'bg-gray-500/10 border-gray-500/50'
                          }`}
                        >
                          <div className="flex items-start gap-3">
                            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
                              <span className="text-blue-400 font-semibold text-sm">{index + 1}</span>
                            </div>
                            <div className="flex-1">
                              <div className="flex items-start justify-between">
                                <p className="text-white font-medium">{step.action}</p>
                                <div className="flex items-center gap-2">
                                  {step.status === 'completed' && (
                                    <CheckCircle className="h-5 w-5 text-green-400" />
                                  )}
                                  {step.status === 'executing' && (
                                    <Terminal className="h-5 w-5 text-blue-400 animate-pulse" />
                                  )}
                                  {step.status === 'pending' && (
                                    <Clock className="h-5 w-5 text-gray-400" />
                                  )}
                                </div>
                              </div>
                              {step.result && (
                                <p className="text-green-400 text-sm mt-1">✓ {step.result}</p>
                              )}
                              {step.timestamp && (
                                <p className="text-gray-500 text-xs mt-1">
                                  {new Date(step.timestamp).toLocaleTimeString()}
                                </p>
                              )}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        {/* Learn Tab */}
        <TabsContent value="learn" className="space-y-6">
          <Card className="bg-black/40 border-red-500/20 p-8">
            <div className="space-y-8">
              <div>
                <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
                  <Shield className="h-7 w-7 text-red-400" />
                  NIST Incident Response Framework
                </h2>
                <p className="text-gray-300 text-lg leading-relaxed">
                  The NIST Incident Response Lifecycle (NIST SP 800-61) provides a structured approach to handling security incidents.
                  This framework ensures consistent, effective responses that minimize damage, reduce recovery time and costs, and prevent future incidents.
                </p>
              </div>

              <div className="border-t border-gray-700 pt-8">
                <h3 className="text-xl font-semibold text-white mb-6">The 6 Phases of Incident Response</h3>
                <div className="space-y-6">
                  <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border border-blue-500/30 rounded-lg p-6">
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0 w-12 h-12 rounded-full bg-blue-500/20 flex items-center justify-center">
                        <span className="text-blue-400 font-bold text-xl">1</span>
                      </div>
                      <div className="flex-1">
                        <h4 className="text-xl font-semibold text-blue-400 mb-3">Preparation</h4>
                        <p className="text-gray-300 mb-4">
                          Establishing and training an incident response team, acquiring necessary tools and resources, and implementing
                          security controls to prevent incidents.
                        </p>
                        <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                          <p className="text-sm text-gray-400 mb-2"><strong>Key Activities:</strong></p>
                          <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                            <li>Develop and document incident response policies and procedures</li>
                            <li>Build and train Computer Security Incident Response Team (CSIRT)</li>
                            <li>Deploy security tools: SIEM, EDR, NDR, forensic tools</li>
                            <li>Establish communication channels and escalation procedures</li>
                            <li>Create incident classification and severity matrix</li>
                            <li>Conduct tabletop exercises and simulations</li>
                            <li>Maintain jump bags with forensic hardware and software</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/30 rounded-lg p-6">
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0 w-12 h-12 rounded-full bg-purple-500/20 flex items-center justify-center">
                        <span className="text-purple-400 font-bold text-xl">2</span>
                      </div>
                      <div className="flex-1">
                        <h4 className="text-xl font-semibold text-purple-400 mb-3">Detection & Analysis</h4>
                        <p className="text-gray-300 mb-4">
                          Identifying potential security incidents through monitoring, alerts, and threat intelligence, then analyzing
                          the scope and impact.
                        </p>
                        <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                          <p className="text-sm text-gray-400 mb-2"><strong>Detection Methods:</strong></p>
                          <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                            <li>SIEM alerts and correlation rules</li>
                            <li>Intrusion Detection/Prevention Systems (IDS/IPS)</li>
                            <li>Endpoint Detection and Response (EDR) alerts</li>
                            <li>Network traffic analysis and anomaly detection</li>
                            <li>Threat intelligence feeds and IOC matching</li>
                            <li>User-reported suspicious activity</li>
                            <li>Third-party notifications (law enforcement, security researchers)</li>
                          </ul>
                        </div>
                        <div className="mt-3 p-3 bg-purple-500/10 border border-purple-500/30 rounded">
                          <p className="text-sm text-purple-300">
                            <strong>Analysis Goal:</strong> Determine if an incident occurred, its scope, affected systems, attack vectors,
                            and classify severity (P1-Critical to P4-Low) for appropriate response.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-yellow-500/10 to-orange-500/10 border border-yellow-500/30 rounded-lg p-6">
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0 w-12 h-12 rounded-full bg-yellow-500/20 flex items-center justify-center">
                        <span className="text-yellow-400 font-bold text-xl">3</span>
                      </div>
                      <div className="flex-1">
                        <h4 className="text-xl font-semibold text-yellow-400 mb-3">Containment</h4>
                        <p className="text-gray-300 mb-4">
                          Preventing the incident from spreading and causing additional damage while preserving evidence for investigation.
                        </p>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                            <p className="text-sm text-gray-400 mb-2"><strong>Short-term Containment:</strong></p>
                            <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                              <li>Network isolation (disconnect infected systems)</li>
                              <li>Disable compromised accounts</li>
                              <li>Block malicious IPs/domains at firewall</li>
                              <li>Take forensic images before changes</li>
                              <li>Segment network to limit spread</li>
                            </ul>
                          </div>
                          <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                            <p className="text-sm text-gray-400 mb-2"><strong>Long-term Containment:</strong></p>
                            <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                              <li>Rebuild systems from clean backups</li>
                              <li>Apply patches and hardening</li>
                              <li>Change all passwords and rotate keys</li>
                              <li>Implement enhanced monitoring</li>
                              <li>Deploy temporary compensating controls</li>
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-red-500/10 to-orange-500/10 border border-red-500/30 rounded-lg p-6">
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0 w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center">
                        <span className="text-red-400 font-bold text-xl">4</span>
                      </div>
                      <div className="flex-1">
                        <h4 className="text-xl font-semibold text-red-400 mb-3">Eradication</h4>
                        <p className="text-gray-300 mb-4">
                          Removing the threat actor's presence from the environment, including malware, backdoors, and vulnerabilities exploited.
                        </p>
                        <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                          <p className="text-sm text-gray-400 mb-2"><strong>Eradication Steps:</strong></p>
                          <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                            <li>Remove malware from all infected systems</li>
                            <li>Delete unauthorized user accounts and access</li>
                            <li>Eliminate backdoors and persistent access mechanisms</li>
                            <li>Patch vulnerabilities exploited in the attack</li>
                            <li>Strengthen security controls that failed</li>
                            <li>Update security tools with new IOCs and signatures</li>
                            <li>Verify complete removal through hunting and scanning</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-lg p-6">
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0 w-12 h-12 rounded-full bg-green-500/20 flex items-center justify-center">
                        <span className="text-green-400 font-bold text-xl">5</span>
                      </div>
                      <div className="flex-1">
                        <h4 className="text-xl font-semibold text-green-400 mb-3">Recovery</h4>
                        <p className="text-gray-300 mb-4">
                          Restoring affected systems to normal operations while monitoring for signs of attacker persistence or re-compromise.
                        </p>
                        <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                          <p className="text-sm text-gray-400 mb-2"><strong>Recovery Process:</strong></p>
                          <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                            <li>Restore systems from clean, verified backups</li>
                            <li>Rebuild compromised systems from secure baselines</li>
                            <li>Test system functionality before production</li>
                            <li>Gradually return to normal operations</li>
                            <li>Monitor restored systems intensively for 2-4 weeks</li>
                            <li>Validate security controls are functioning</li>
                            <li>Communicate restoration status to stakeholders</li>
                          </ul>
                        </div>
                        <div className="mt-3 p-3 bg-green-500/10 border border-green-500/30 rounded">
                          <p className="text-sm text-green-300">
                            <strong>Validation:</strong> Conduct penetration testing and vulnerability assessments post-recovery
                            to ensure the attack vector has been closed and no residual compromise exists.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/30 rounded-lg p-6">
                    <div className="flex items-start gap-4">
                      <div className="flex-shrink-0 w-12 h-12 rounded-full bg-cyan-500/20 flex items-center justify-center">
                        <span className="text-cyan-400 font-bold text-xl">6</span>
                      </div>
                      <div className="flex-1">
                        <h4 className="text-xl font-semibold text-cyan-400 mb-3">Post-Incident Activity</h4>
                        <p className="text-gray-300 mb-4">
                          Conducting lessons learned sessions, updating security controls, and documenting the incident for future reference.
                        </p>
                        <div className="bg-black/40 rounded-lg p-4 border border-gray-700">
                          <p className="text-sm text-gray-400 mb-2"><strong>Post-Incident Tasks:</strong></p>
                          <ul className="list-disc list-inside text-gray-300 text-sm space-y-1">
                            <li>Hold post-mortem meeting with incident response team</li>
                            <li>Document timeline, actions taken, and outcomes</li>
                            <li>Calculate total cost and business impact</li>
                            <li>Identify security gaps and create remediation plan</li>
                            <li>Update incident response procedures based on lessons learned</li>
                            <li>Share IOCs with threat intelligence communities</li>
                            <li>File reports with regulators, law enforcement, insurance</li>
                            <li>Conduct training on new attack techniques observed</li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="border-t border-gray-700 pt-8">
                <h3 className="text-xl font-semibold text-white mb-4">AI-Enhanced Incident Response</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-5">
                    <h4 className="font-semibold text-blue-400 mb-3 flex items-center gap-2">
                      <Code2 className="h-5 w-5" />
                      Automated Detection
                    </h4>
                    <ul className="space-y-2 text-gray-300 text-sm">
                      <li className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span><strong>Behavioral Analytics:</strong> ML models detect anomalous user and entity behavior</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span><strong>Threat Correlation:</strong> AI correlates events across systems for attack chain visibility</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-blue-400">•</span>
                        <span><strong>Zero-Day Detection:</strong> Identifies unknown threats through anomaly detection</span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-5">
                    <h4 className="font-semibold text-purple-400 mb-3 flex items-center gap-2">
                      <Activity className="h-5 w-5" />
                      Automated Response
                    </h4>
                    <ul className="space-y-2 text-gray-300 text-sm">
                      <li className="flex gap-2">
                        <span className="text-purple-400">•</span>
                        <span><strong>SOAR Integration:</strong> Security Orchestration automates playbook execution</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-purple-400">•</span>
                        <span><strong>Dynamic Containment:</strong> AI decides optimal isolation strategy per incident</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-purple-400">•</span>
                        <span><strong>Predictive Remediation:</strong> Suggests fixes based on similar historical incidents</span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-5">
                    <h4 className="font-semibold text-green-400 mb-3 flex items-center gap-2">
                      <CheckCircle className="h-5 w-5" />
                      Intelligent Triage
                    </h4>
                    <ul className="space-y-2 text-gray-300 text-sm">
                      <li className="flex gap-2">
                        <span className="text-green-400">•</span>
                        <span><strong>Priority Scoring:</strong> AI ranks incidents by business impact and threat severity</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-green-400">•</span>
                        <span><strong>False Positive Reduction:</strong> ML learns from analyst feedback to improve accuracy</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-green-400">•</span>
                        <span><strong>Context Enrichment:</strong> Auto-gathers threat intel, asset info, user context</span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-5">
                    <h4 className="font-semibold text-orange-400 mb-3 flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Continuous Learning
                    </h4>
                    <ul className="space-y-2 text-gray-300 text-sm">
                      <li className="flex gap-2">
                        <span className="text-orange-400">•</span>
                        <span><strong>Playbook Optimization:</strong> AI improves response procedures from outcomes</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-orange-400">•</span>
                        <span><strong>Threat Hunting:</strong> ML generates hypotheses for proactive hunting</span>
                      </li>
                      <li className="flex gap-2">
                        <span className="text-orange-400">•</span>
                        <span><strong>Root Cause Analysis:</strong> AI identifies systemic vulnerabilities from incidents</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </Card>
        </TabsContent>

        {/* Playbooks Tab */}
        <TabsContent value="examples" className="space-y-6">
          <div className="grid grid-cols-1 gap-6">
            {PLAYBOOK_EXAMPLES.map((playbook, index) => (
              <Card key={index} className="bg-black/40 border-red-500/20 p-8">
                <div className="space-y-6">
                  {/* Header */}
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-2xl font-bold text-white">{playbook.title}</h3>
                        <Badge className={getSeverityColor(playbook.severity)}>
                          {playbook.severity.toUpperCase()}
                        </Badge>
                        <Badge className="bg-blue-500/20 text-blue-400 border-blue-500/50">
                          {playbook.incidentType}
                        </Badge>
                      </div>
                      <p className="text-gray-300 text-lg">{playbook.description}</p>
                    </div>
                  </div>

                  {/* Technical Details */}
                  <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-5">
                    <h4 className="text-lg font-semibold text-blue-400 mb-3 flex items-center gap-2">
                      <Code2 className="h-5 w-5" />
                      Technical Overview
                    </h4>
                    <p className="text-gray-300 leading-relaxed">{playbook.technicalDetails}</p>
                  </div>

                  {/* Timeline */}
                  <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-5">
                    <h4 className="text-lg font-semibold text-purple-400 mb-2 flex items-center gap-2">
                      <Clock className="h-5 w-5" />
                      Estimated Timeline
                    </h4>
                    <p className="text-gray-300">{playbook.estimatedTimeline}</p>
                  </div>

                  {/* NIST Phases */}
                  <div>
                    <h4 className="text-xl font-semibold text-white mb-4">NIST Response Phases</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {Object.entries(playbook.nistPhases).map(([phase, steps]) => (
                        <div key={phase} className="bg-gradient-to-br from-gray-800/50 to-gray-900/50 border border-gray-700 rounded-lg p-5">
                          <h5 className="text-lg font-semibold text-cyan-400 mb-3 capitalize">
                            {phase.replace(/([A-Z])/g, ' $1').trim()}
                          </h5>
                          <ul className="space-y-2">
                            {steps.map((step, idx) => (
                              <li key={idx} className="flex items-start gap-2 text-gray-300 text-sm">
                                <span className="text-cyan-400 mt-1">▸</span>
                                <span>{step}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Automated Steps */}
                  <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-5">
                    <h4 className="text-lg font-semibold text-green-400 mb-3 flex items-center gap-2">
                      <Activity className="h-5 w-5" />
                      AI-Automated Response Steps
                    </h4>
                    <ul className="space-y-2">
                      {playbook.automatedSteps.map((step, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-gray-300">
                          <span className="text-green-400 mt-1">✓</span>
                          <span>{step}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
