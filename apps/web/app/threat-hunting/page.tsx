"use client"

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Search, Target, TrendingUp, AlertCircle, CheckCircle, Brain, Download, Play, BookOpen, Lightbulb, Copy } from 'lucide-react'

interface Hunt {
  id: string
  hypothesis: string
  status: 'active' | 'completed' | 'investigating'
  findings: number
  iocs: string[]
  techniques: string[]
  confidence: number
  created: string
  queries: string[]
}

const SAMPLE_HYPOTHESES = [
  {
    title: 'Lateral Movement Detection',
    description: 'Detect Pass-the-Hash attacks using abnormal NTLM authentication patterns',
    query: 'Detect lateral movement using Pass-the-Hash attacks',
    mitreAttack: ['T1550.002', 'T1021.002'],
    explanation: 'This hunt searches for unusual NTLM authentication events that may indicate an attacker using stolen password hashes to move laterally across your network.'
  },
  {
    title: 'Data Exfiltration via DNS',
    description: 'Identify data exfiltration attempts through DNS tunneling',
    query: 'Identify data exfiltration via DNS tunneling',
    mitreAttack: ['T1048.003', 'T1071.004'],
    explanation: 'DNS tunneling is a covert channel technique where attackers encode data into DNS queries to bypass firewalls and exfiltrate sensitive information.'
  },
  {
    title: 'Living off the Land (LOLBins)',
    description: 'Hunt for suspicious use of legitimate Windows binaries for malicious purposes',
    query: 'Hunt for living-off-the-land binaries (LOLBins) execution',
    mitreAttack: ['T1218', 'T1059.001'],
    explanation: 'Attackers use built-in Windows tools like PowerShell, certutil, and mshta to evade detection. This hunt identifies abnormal usage patterns of these legitimate tools.'
  },
  {
    title: 'PowerShell Remoting Abuse',
    description: 'Find unauthorized PowerShell remoting sessions indicating potential compromise',
    query: 'Find unauthorized PowerShell remoting sessions',
    mitreAttack: ['T1021.006', 'T1059.001'],
    explanation: 'PowerShell remoting (WinRM) can be abused by attackers for remote command execution. This hunt detects unusual or unauthorized remote PowerShell sessions.'
  }
]

export default function ThreatHuntingPage() {
  const [hypothesis, setHypothesis] = useState('')
  const [hunts, setHunts] = useState<Hunt[]>([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('hunt')

  const generateHunt = async () => {
    if (!hypothesis) return
    setLoading(true)

    await new Promise(r => setTimeout(r, 2000))

    const newHunt: Hunt = {
      id: 'HUNT-' + Date.now(),
      hypothesis,
      status: 'active',
      findings: Math.floor(Math.random() * 20),
      iocs: ['192.168.1.100', 'malicious.com', 'hash:abc123'],
      techniques: ['T1059.001', 'T1071.001', 'T1055'],
      confidence: 70 + Math.floor(Math.random() * 25),
      created: new Date().toISOString(),
      queries: [
        'SELECT * FROM processes WHERE name LIKE "%powershell%" AND parent != "explorer.exe"',
        'SELECT * FROM network_connections WHERE remote_address NOT IN (whitelist) AND protocol = "HTTPS"'
      ]
    }

    setHunts([newHunt, ...hunts])
    setHypothesis('')
    setLoading(false)
  }

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <div className="min-h-screen bg-background p-8 pt-32">
      <div className="mb-8">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
          <div>
            <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold bg-gradient-to-r from-purple-400 to-pink-600 bg-clip-text text-transparent mb-3">
              AI Threat Hunting
            </h1>
            <p className="text-lg sm:text-xl text-muted-foreground">
              Proactive threat detection using AI-powered hypothesis generation and behavioral analysis
            </p>
          </div>
        <Button className="bg-gradient-to-r from-purple-500 to-pink-600 text-base sm:text-lg px-6 py-6">
          <Download className="mr-2 h-5 w-5" />
          Export Report
        </Button>
        </div>

        {/* Description Banner */}
        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-4 flex items-start gap-3">
          <Search className="w-5 h-5 text-purple-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong className="text-foreground">What this page does:</strong> AI Threat Hunting enables proactive threat detection by generating intelligent hypotheses about potential security compromises in your environment. Unlike reactive detection, this tool helps you actively search for threats that haven't triggered alerts yet. The AI analyzes behavioral patterns, generates hunt queries, and provides sample hypotheses for common attack scenarios like lateral movement (Pass-the-Hash), data exfiltration via DNS tunneling, Living-off-the-Land binaries (LOLBins) abuse, and PowerShell remoting attacks. Each hypothesis includes MITRE ATT&CK technique mappings, detection queries, and educational content explaining the attack method and why it matters. Perfect for SOC analysts and security researchers conducting threat hunting campaigns.
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <Card className="card-responsive bg-card border-purple-500/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-base sm:text-lg text-muted-foreground">Active Hunts</p>
              <p className="text-4xl sm:text-5xl font-bold mt-2">{hunts.filter(h => h.status === 'active').length}</p>
            </div>
            <Target className="h-10 w-10 sm:h-12 sm:w-12 text-purple-400" />
          </div>
        </Card>
        <Card className="card-responsive bg-card border-green-500/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-base sm:text-lg text-muted-foreground">Total Findings</p>
              <p className="text-4xl sm:text-5xl font-bold text-green-400 mt-2">
                {hunts.reduce((acc, h) => acc + h.findings, 0)}
              </p>
            </div>
            <CheckCircle className="h-10 w-10 sm:h-12 sm:w-12 text-green-400" />
          </div>
        </Card>
        <Card className="card-responsive bg-card border-blue-500/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-base sm:text-lg text-muted-foreground">Avg Confidence</p>
              <p className="text-4xl sm:text-5xl font-bold text-blue-400 mt-2">
                {hunts.length > 0 ? Math.round(hunts.reduce((acc, h) => acc + h.confidence, 0) / hunts.length) : 0}%
              </p>
            </div>
            <TrendingUp className="h-10 w-10 sm:h-12 sm:w-12 text-blue-400" />
          </div>
        </Card>
        <Card className="card-responsive bg-card border-orange-500/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-base sm:text-lg text-muted-foreground">IOCs Found</p>
              <p className="text-4xl sm:text-5xl font-bold text-orange-400 mt-2">
                {hunts.reduce((acc, h) => acc + h.iocs.length, 0)}
              </p>
            </div>
            <AlertCircle className="h-10 w-10 sm:h-12 sm:w-12 text-orange-400" />
          </div>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 h-auto p-1 bg-muted">
          <TabsTrigger value="hunt" className="text-base sm:text-lg py-3 sm:py-4">
            <Target className="mr-2 h-5 w-5" />
            Hunt
          </TabsTrigger>
          <TabsTrigger value="learn" className="text-base sm:text-lg py-3 sm:py-4">
            <BookOpen className="mr-2 h-5 w-5" />
            Learn
          </TabsTrigger>
          <TabsTrigger value="examples" className="text-base sm:text-lg py-3 sm:py-4">
            <Lightbulb className="mr-2 h-5 w-5" />
            Examples
          </TabsTrigger>
        </TabsList>

        <TabsContent value="hunt" className="space-y-6">
          <Card className="card-responsive bg-card border-purple-500/20">
            <h2 className="text-2xl sm:text-3xl font-semibold mb-4 sm:mb-6">Generate Hunt Hypothesis</h2>
            <div className="space-y-4 sm:space-y-6">
              <div className="flex flex-col sm:flex-row gap-3 sm:gap-4">
                <Input
                  placeholder="Describe the threat behavior you want to hunt for..."
                  value={hypothesis}
                  onChange={(e) => setHypothesis(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && generateHunt()}
                  className="flex-1 text-base sm:text-lg py-6 bg-background border-purple-500/20"
                />
                <Button onClick={generateHunt} disabled={loading || !hypothesis} className="bg-purple-500 hover:bg-purple-600 text-base sm:text-lg px-6 py-6">
                  <Brain className="mr-2 h-5 w-5" />
                  {loading ? 'Generating...' : 'Start Hunt'}
                </Button>
              </div>
              <div>
                <p className="text-base sm:text-lg text-muted-foreground mb-3">Quick Hypotheses:</p>
                <div className="flex flex-wrap gap-2 sm:gap-3">
                  {SAMPLE_HYPOTHESES.map((hyp, idx) => (
                    <button
                      key={idx}
                      onClick={() => setHypothesis(hyp.query)}
                      className="px-4 py-2 text-sm sm:text-base bg-purple-500/20 text-purple-400 border border-purple-500/50 rounded-lg hover:bg-purple-500/30 transition-all"
                    >
                      {hyp.query}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          <div className="space-y-4 sm:space-y-6">
            <h2 className="text-2xl sm:text-3xl font-semibold">Hunt Results ({hunts.length})</h2>
            {hunts.length === 0 ? (
              <Card className="card-responsive bg-card border-purple-500/20 text-center py-16">
                <Search className="h-16 w-16 sm:h-20 sm:w-20 text-muted-foreground mx-auto mb-4" />
                <p className="text-lg sm:text-xl text-muted-foreground">No active hunts. Create a hypothesis above to start hunting!</p>
              </Card>
            ) : (
              hunts.map((hunt) => (
                <Card key={hunt.id} className="card-responsive bg-card border-purple-500/20 hover:border-purple-500/40 transition-all">
                  <div className="space-y-4 sm:space-y-6">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex flex-wrap items-center gap-3 mb-2">
                          <h3 className="text-xl sm:text-2xl font-semibold">{hunt.hypothesis}</h3>
                          <Badge className={
                            hunt.status === 'active' ? 'bg-green-500/20 text-green-400 border-green-500/50 text-sm sm:text-base' :
                            hunt.status === 'investigating' ? 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50 text-sm sm:text-base' :
                            'bg-blue-500/20 text-blue-400 border-blue-500/50 text-sm sm:text-base'
                          }>
                            {hunt.status}
                          </Badge>
                        </div>
                        <p className="text-base sm:text-lg text-muted-foreground">{hunt.id} • Started {new Date(hunt.created).toLocaleString()}</p>
                      </div>
                      <Button size="sm" className="bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 px-4 py-2">
                        <Play className="h-5 w-5" />
                      </Button>
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                      <div className="bg-purple-500/10 rounded-lg p-4">
                        <p className="text-base sm:text-lg text-muted-foreground mb-1">Findings</p>
                        <p className="text-3xl sm:text-4xl font-bold">{hunt.findings}</p>
                      </div>
                      <div className="bg-blue-500/10 rounded-lg p-4">
                        <p className="text-base sm:text-lg text-muted-foreground mb-1">Confidence</p>
                        <p className="text-3xl sm:text-4xl font-bold text-blue-400">{hunt.confidence}%</p>
                      </div>
                      <div className="bg-green-500/10 rounded-lg p-4">
                        <p className="text-base sm:text-lg text-muted-foreground mb-1">IOCs</p>
                        <p className="text-3xl sm:text-4xl font-bold text-green-400">{hunt.iocs.length}</p>
                      </div>
                    </div>

                    <div>
                      <p className="text-base sm:text-lg text-muted-foreground mb-3">MITRE ATT&CK Techniques:</p>
                      <div className="flex flex-wrap gap-2">
                        {hunt.techniques.map((t) => (
                          <Badge key={t} className="bg-purple-500/20 text-purple-400 border-purple-500/50 text-sm sm:text-base px-3 py-1">
                            {t}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div>
                      <p className="text-base sm:text-lg text-muted-foreground mb-3">Indicators of Compromise:</p>
                      <div className="flex flex-wrap gap-2">
                        {hunt.iocs.map((ioc, idx) => (
                          <Badge key={idx} className="bg-red-500/20 text-red-400 border-red-500/50 font-mono text-sm sm:text-base px-3 py-1">
                            {ioc}
                          </Badge>
                        ))}
                      </div>
                    </div>

                    <div>
                      <p className="text-base sm:text-lg text-muted-foreground mb-3">AI-Generated Queries:</p>
                      <div className="space-y-3">
                        {hunt.queries.map((query, idx) => (
                          <div key={idx} className="bg-black/60 border border-border rounded-lg p-4 relative group">
                            <code className="text-sm sm:text-base text-cyan-400 break-all">{query}</code>
                            <button
                              onClick={() => copyToClipboard(query)}
                              className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-2 bg-background rounded hover:bg-muted"
                            >
                              <Copy className="h-4 w-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-2 sm:gap-3">
                      <Button size="sm" className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 text-sm sm:text-base px-4 py-2">
                        View Details
                      </Button>
                      <Button size="sm" className="bg-green-500/20 hover:bg-green-500/30 text-green-400 text-sm sm:text-base px-4 py-2">
                        Export IOCs
                      </Button>
                      <Button size="sm" className="bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 text-sm sm:text-base px-4 py-2">
                        Create Alert
                      </Button>
                    </div>
                  </div>
                </Card>
              ))
            )}
          </div>
        </TabsContent>

        <TabsContent value="learn" className="space-y-6">
          <Card className="card-responsive bg-card border-blue-500/20">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4 sm:mb-6">What is Threat Hunting?</h2>
            <div className="space-y-4 sm:space-y-6 text-base sm:text-lg">
              <p className="leading-relaxed">
                Threat hunting is a proactive cybersecurity approach where security analysts actively search for threats that have evaded traditional security defenses. Unlike reactive security measures that wait for alerts, threat hunting assumes that adversaries are already inside your network and seeks to find them before they cause damage.
              </p>
              <p className="leading-relaxed">
                Our AI-powered threat hunting platform uses machine learning and behavioral analysis to generate hypotheses about potential threats, then automatically searches your environment for evidence supporting those hypotheses.
              </p>
            </div>
          </Card>

          <Card className="card-responsive bg-card border-purple-500/20">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4 sm:mb-6">How It Works</h2>
            <div className="space-y-4 sm:space-y-6">
              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-purple-500/20 flex items-center justify-center text-xl sm:text-2xl font-bold text-purple-400">1</div>
                <div>
                  <h3 className="text-xl sm:text-2xl font-semibold mb-2">Hypothesis Generation</h3>
                  <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
                    Describe a potential threat behavior or attack technique. Our AI analyzes the hypothesis and generates detection queries based on MITRE ATT&CK framework and known adversary tactics.
                  </p>
                </div>
              </div>
              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-blue-500/20 flex items-center justify-center text-xl sm:text-2xl font-bold text-blue-400">2</div>
                <div>
                  <h3 className="text-xl sm:text-2xl font-semibold mb-2">Data Collection</h3>
                  <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
                    The system automatically executes queries across your security data sources including endpoint logs, network traffic, authentication events, and process execution data.
                  </p>
                </div>
              </div>
              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-green-500/20 flex items-center justify-center text-xl sm:text-2xl font-bold text-green-400">3</div>
                <div>
                  <h3 className="text-xl sm:text-2xl font-semibold mb-2">Analysis & Results</h3>
                  <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
                    AI analyzes the results, identifies Indicators of Compromise (IOCs), maps findings to MITRE ATT&CK techniques, and calculates confidence scores for each detected behavior.
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <Card className="card-responsive bg-card border-green-500/20">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4 sm:mb-6">Key Benefits</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
              <div className="p-4 sm:p-6 bg-green-500/10 rounded-lg">
                <h3 className="text-xl sm:text-2xl font-semibold mb-2">Proactive Defense</h3>
                <p className="text-base sm:text-lg text-muted-foreground">Find threats before they cause damage</p>
              </div>
              <div className="p-4 sm:p-6 bg-blue-500/10 rounded-lg">
                <h3 className="text-xl sm:text-2xl font-semibold mb-2">AI-Powered</h3>
                <p className="text-base sm:text-lg text-muted-foreground">Automated hypothesis and query generation</p>
              </div>
              <div className="p-4 sm:p-6 bg-purple-500/10 rounded-lg">
                <h3 className="text-xl sm:text-2xl font-semibold mb-2">MITRE Mapped</h3>
                <p className="text-base sm:text-lg text-muted-foreground">All findings mapped to ATT&CK framework</p>
              </div>
              <div className="p-4 sm:p-6 bg-orange-500/10 rounded-lg">
                <h3 className="text-xl sm:text-2xl font-semibold mb-2">IOC Extraction</h3>
                <p className="text-base sm:text-lg text-muted-foreground">Automatic indicator of compromise identification</p>
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="examples" className="space-y-6">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">Sample Threat Hunt Scenarios</h2>
          {SAMPLE_HYPOTHESES.map((sample, idx) => (
            <Card key={idx} className="card-responsive bg-card border-purple-500/20">
              <div className="space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <h3 className="text-2xl sm:text-3xl font-semibold">{sample.title}</h3>
                  <Button
                    onClick={() => {
                      setHypothesis(sample.query)
                      setActiveTab('hunt')
                    }}
                    className="bg-purple-500 hover:bg-purple-600 text-base sm:text-lg px-6 py-3"
                  >
                    Use This Hypothesis
                  </Button>
                </div>
                <p className="text-lg sm:text-xl text-muted-foreground leading-relaxed">{sample.description}</p>
                <div>
                  <p className="text-base sm:text-lg text-muted-foreground mb-2">MITRE ATT&CK Techniques:</p>
                  <div className="flex flex-wrap gap-2">
                    {sample.mitreAttack.map((technique) => (
                      <Badge key={technique} className="bg-purple-500/20 text-purple-400 border-purple-500/50 text-sm sm:text-base px-3 py-1">
                        {technique}
                      </Badge>
                    ))}
                  </div>
                </div>
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 sm:p-6">
                  <p className="text-base sm:text-lg font-semibold text-blue-400 mb-2">How This Works:</p>
                  <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">{sample.explanation}</p>
                </div>
              </div>
            </Card>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  )
}
