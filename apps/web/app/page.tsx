'use client';

import Link from 'next/link';
import { useState } from 'react';
import { Shield, Swords, Brain, Target, Zap, Network, HelpCircle, BookOpen, PlayCircle, TrendingUp, Info } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

export default function Home() {
  const [showWelcome, setShowWelcome] = useState(false);

  const features = [
    {
      icon: Target,
      title: 'Zero-Day Discovery',
      description: 'AI-powered vulnerability detection using behavioral analysis and pattern recognition',
      details: 'Our AI agents analyze system behavior in real-time to identify previously unknown vulnerabilities. Using machine learning and anomaly detection, the system discovers zero-day exploits before they can be weaponized.',
      learnMore: 'Zero-day vulnerabilities are security flaws unknown to software vendors. YUGMASTRA uses autonomous AI to find these critical vulnerabilities through continuous security testing and behavioral monitoring.'
    },
    {
      icon: Brain,
      title: 'Self-Play MARL',
      description: 'Multi-Agent Reinforcement Learning through adversarial co-evolution',
      details: 'Red and Blue Team AI agents compete in thousands of simulated attack scenarios. Each agent learns from wins and losses, continuously improving attack and defense strategies through self-play.',
      learnMore: 'Multi-Agent Reinforcement Learning (MARL) allows AI agents to learn complex strategies by competing against each other. This creates more sophisticated defenses than traditional rule-based systems.'
    },
    {
      icon: Network,
      title: 'Knowledge Graph',
      description: 'Attack pattern visualization and relationship mapping via MITRE ATT&CK',
      details: 'Interactive visualization of attack techniques, tactics, and procedures. Maps relationships between different attack patterns and shows how threats evolve through the MITRE ATT&CK framework.',
      learnMore: 'The knowledge graph connects attack patterns, vulnerabilities, and defenses. This helps security teams understand how attackers chain techniques together for maximum impact.'
    },
    {
      icon: TrendingUp,
      title: 'Real-time Analytics',
      description: 'Live performance metrics and AI model training visualization',
      details: 'Monitor AI model performance, training progress, win rates, and strategy effectiveness in real-time. Track how defensive capabilities improve over thousands of training episodes.',
      learnMore: 'Analytics show how AI models learn and improve. Watch defense strategies evolve as the system processes more attack scenarios and refines detection patterns.'
    },
  ];

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 sm:p-8 md:p-12 lg:p-24 bg-gradient-to-br from-background via-accent to-accent/50">
      <div className="z-10 max-w-5xl w-full items-center justify-center">
        {/* Header with Help */}
        <div className="text-center mb-8 sm:mb-12">
          <div className="flex items-center justify-center gap-3 mb-3 sm:mb-4">
            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-bold bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 bg-clip-text text-transparent tracking-tight">
              YUGMĀSTRA
            </h1>
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                  <HelpCircle className="h-6 w-6 text-primary" />
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
                <DialogHeader>
                  <DialogTitle className="text-2xl">Welcome to YUGMĀSTRA</DialogTitle>
                  <DialogDescription className="text-base">
                    Understanding Autonomous Cyber Defense
                  </DialogDescription>
                </DialogHeader>
                <div className="space-y-6 pt-4">
                  <div>
                    <h3 className="text-lg font-semibold mb-2 flex items-center gap-2">
                      <Info className="h-5 w-5 text-blue-500" />
                      What is YUGMĀSTRA?
                    </h3>
                    <p className="text-muted-foreground mb-3">
                      YUGMĀSTRA is an autonomous adversary-defender co-evolution platform where cybersecurity defenses are not manually engineered—they emerge naturally through adversarial self-play between AI agents.
                    </p>
                    <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                      <p className="text-sm text-muted-foreground">
                        <strong className="text-blue-400">The Core Concept:</strong> Instead of programming security rules, we let Red Team AI (attackers) and Blue Team AI (defenders) compete against each other in thousands of simulated battles. Through this competition, both sides evolve sophisticated strategies that would be difficult or impossible to code manually.
                      </p>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold mb-3">How It Works</h3>
                    <div className="space-y-3">
                      <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Swords className="h-5 w-5 text-red-400" />
                          <h4 className="font-semibold text-red-400">1. Red Team AI Attacks</h4>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Autonomous offensive AI explores thousands of attack strategies—from SQL injection and XSS to advanced persistent threats (APTs) and zero-day exploits. It learns which techniques work best against different defenses.
                        </p>
                      </div>

                      <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Shield className="h-5 w-5 text-blue-400" />
                          <h4 className="font-semibold text-blue-400">2. Blue Team AI Defends</h4>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Defensive AI develops detection patterns, response strategies, and automated countermeasures. It learns to identify attack signatures, behavioral anomalies, and novel exploitation techniques.
                        </p>
                      </div>

                      <div className="bg-green-500/10 border border-green-500/20 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <Brain className="h-5 w-5 text-green-400" />
                          <h4 className="font-semibold text-green-400">3. Continuous Evolution</h4>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          Both agents learn from every battle. When Red Team finds a way past defenses, Blue Team adapts. When Blue Team blocks an attack, Red Team evolves new tactics. This creates an arms race that produces increasingly sophisticated security.
                        </p>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold mb-3">Real-World Applications</h3>
                    <ul className="space-y-2 text-sm text-muted-foreground">
                      <li className="flex items-start gap-2">
                        <span className="text-primary mt-1">▸</span>
                        <span><strong>Zero-Day Discovery:</strong> Find vulnerabilities before attackers do by simulating novel attack patterns</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-primary mt-1">▸</span>
                        <span><strong>Automated Threat Hunting:</strong> Generate hypotheses and detection rules for emerging threats</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-primary mt-1">▸</span>
                        <span><strong>Incident Response:</strong> Automated playbook generation and response orchestration</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-primary mt-1">▸</span>
                        <span><strong>Security Training:</strong> Practice defending against realistic AI-generated attacks in a safe cyber range</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="text-primary mt-1">▸</span>
                        <span><strong>Code Review:</strong> AI-powered vulnerability scanning with contextual security recommendations</span>
                      </li>
                    </ul>
                  </div>

                  <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-4">
                    <h4 className="font-semibold text-purple-400 mb-2 flex items-center gap-2">
                      <PlayCircle className="h-4 w-4" />
                      Getting Started
                    </h4>
                    <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                      <li>Click <strong>"Watch Live Battle"</strong> to see AI agents compete in real-time</li>
                      <li>Visit the <strong>Dashboard</strong> to explore all security tools and features</li>
                      <li>Try the <strong>AI Assistant</strong> to ask cybersecurity questions and get expert guidance</li>
                      <li>Use <strong>Threat Hunting</strong> or <strong>Code Review</strong> tools for practical security tasks</li>
                      <li>Check <strong>Analytics</strong> to see how AI models improve over time</li>
                    </ol>
                  </div>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          <p className="text-lg sm:text-xl md:text-2xl text-primary mb-3 sm:mb-4 px-4">
            Autonomous Adversary-Defender Co-Evolution Platform
          </p>

          <div className="inline-block bg-card/50 backdrop-blur-lg border rounded-lg px-3 sm:px-4 md:px-6 py-2 sm:py-3 mb-4 sm:mb-6 md:mb-8 mx-2">
            <p className="text-sm sm:text-base md:text-lg text-foreground">
              <span className="font-semibold text-primary">System Owner:</span> Preet Raval
            </p>
            <p className="text-xs sm:text-sm text-muted-foreground">
              <span className="font-semibold">Email:</span> preetraval45@gmail.com
            </p>
          </div>

          <p className="text-sm sm:text-base md:text-lg text-muted-foreground max-w-3xl mx-auto mb-6 sm:mb-8 md:mb-12 px-4">
            Where cybersecurity defenses are not engineered—they emerge through
            adversarial self-play between autonomous AI agents. Watch as Red Team AI attacks
            your system in real-time while Blue Team AI learns to defend it.
          </p>

          <div className="flex gap-2 sm:gap-3 md:gap-4 justify-center flex-wrap px-4">
            <Link
              href="/live-battle"
              className="bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-700 hover:to-orange-700 text-white px-4 sm:px-6 md:px-8 py-2.5 sm:py-3 md:py-4 rounded-lg text-sm sm:text-base md:text-lg font-semibold transition-all transform hover:scale-105 shadow-lg"
            >
              🔥 Watch Live Battle
            </Link>
            <Link
              href="/dashboard"
              className="bg-blue-600 hover:bg-blue-700 text-white px-4 sm:px-6 md:px-8 py-2.5 sm:py-3 md:py-4 rounded-lg text-sm sm:text-base md:text-lg font-semibold transition-colors"
            >
              Launch Dashboard
            </Link>
            <Link
              href="/evolution"
              className="bg-purple-600 hover:bg-purple-700 text-white px-4 sm:px-6 md:px-8 py-2.5 sm:py-3 md:py-4 rounded-lg text-sm sm:text-base md:text-lg font-semibold transition-colors"
            >
              View Evolution
            </Link>
          </div>
        </div>

        {/* AI Teams Grid with Educational Content */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-16">
          <Dialog>
            <DialogTrigger asChild>
              <div className="bg-card/50 backdrop-blur-lg rounded-lg p-6 border hover:border-red-500/50 transition-all cursor-pointer group">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xl font-bold text-foreground flex items-center gap-2">
                    <Swords className="h-5 w-5 text-red-500" />
                    Red Team AI
                  </h3>
                  <Info className="h-4 w-4 text-muted-foreground group-hover:text-red-400 transition-colors" />
                </div>
                <p className="text-muted-foreground">
                  Autonomous attack agent that discovers novel exploitation strategies
                  through reinforcement learning and self-play.
                </p>
              </div>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Swords className="h-6 w-6 text-red-500" />
                  Red Team AI - Offensive Agent
                </DialogTitle>
                <DialogDescription>Autonomous attack strategy discovery through self-play</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <p className="text-foreground">
                  The Red Team AI is an autonomous offensive agent that explores the attack surface of systems to discover vulnerabilities and develop exploitation strategies.
                </p>

                <div>
                  <h4 className="font-semibold mb-2">What It Does</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <span className="text-red-400">•</span>
                      <span><strong>Discovers Vulnerabilities:</strong> Probes systems for weaknesses including SQL injection, XSS, authentication bypasses, and zero-day exploits</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-red-400">•</span>
                      <span><strong>Learns Attack Chains:</strong> Combines multiple techniques (MITRE ATT&CK) to achieve objectives like privilege escalation and data exfiltration</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-red-400">•</span>
                      <span><strong>Adapts to Defenses:</strong> When blocked, evolves new tactics to bypass security controls</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-red-400">•</span>
                      <span><strong>Generates Intelligence:</strong> Creates attack patterns that inform defensive improvements</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-3">
                  <p className="text-sm text-muted-foreground">
                    <strong className="text-red-400">Key Insight:</strong> Red Team AI doesn't just run pre-programmed attacks. It uses reinforcement learning to discover novel exploitation paths that human security testers might miss.
                  </p>
                </div>

                <Button asChild className="w-full bg-gradient-to-r from-red-500 to-orange-500">
                  <Link href="/live-battle">Watch Red Team in Action</Link>
                </Button>
              </div>
            </DialogContent>
          </Dialog>

          <Dialog>
            <DialogTrigger asChild>
              <div className="bg-card/50 backdrop-blur-lg rounded-lg p-6 border hover:border-blue-500/50 transition-all cursor-pointer group">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xl font-bold text-foreground flex items-center gap-2">
                    <Shield className="h-5 w-5 text-blue-500" />
                    Blue Team AI
                  </h3>
                  <Info className="h-4 w-4 text-muted-foreground group-hover:text-blue-400 transition-colors" />
                </div>
                <p className="text-muted-foreground">
                  Adaptive defense system that learns detection patterns and generates
                  countermeasures automatically.
                </p>
              </div>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Shield className="h-6 w-6 text-blue-500" />
                  Blue Team AI - Defensive Agent
                </DialogTitle>
                <DialogDescription>Adaptive defense through continuous learning</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <p className="text-foreground">
                  The Blue Team AI is an adaptive defensive agent that learns to detect, analyze, and respond to security threats through adversarial training.
                </p>

                <div>
                  <h4 className="font-semibold mb-2">What It Does</h4>
                  <ul className="space-y-2 text-sm text-muted-foreground">
                    <li className="flex items-start gap-2">
                      <span className="text-blue-400">•</span>
                      <span><strong>Detects Threats:</strong> Identifies malicious activity through behavioral analysis, anomaly detection, and pattern recognition</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-400">•</span>
                      <span><strong>Generates Rules:</strong> Automatically creates detection signatures (SIEM rules, IDS/IPS rules) from observed attacks</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-400">•</span>
                      <span><strong>Responds Automatically:</strong> Executes incident response playbooks including isolation, blocking, and remediation</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <span className="text-blue-400">•</span>
                      <span><strong>Improves Continuously:</strong> Learns from every attack attempt to strengthen future defenses</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-3">
                  <p className="text-sm text-muted-foreground">
                    <strong className="text-blue-400">Key Insight:</strong> Blue Team AI adapts faster than traditional security systems. When Red Team discovers a new attack vector, Blue Team immediately learns to detect it and shares that knowledge across all defenses.
                  </p>
                </div>

                <Button asChild className="w-full bg-gradient-to-r from-blue-500 to-cyan-500">
                  <Link href="/incident-response">Explore Incident Response</Link>
                </Button>
              </div>
            </DialogContent>
          </Dialog>

          <Dialog>
            <DialogTrigger asChild>
              <div className="bg-card/50 backdrop-blur-lg rounded-lg p-6 border hover:border-purple-500/50 transition-all cursor-pointer group">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xl font-bold text-foreground flex items-center gap-2">
                    <Brain className="h-5 w-5 text-purple-500" />
                    Co-Evolution
                  </h3>
                  <Info className="h-4 w-4 text-muted-foreground group-hover:text-purple-400 transition-colors" />
                </div>
                <p className="text-muted-foreground">
                  Multi-agent system where strategies emerge through adversarial
                  competition, reaching Nash equilibrium.
                </p>
              </div>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Brain className="h-6 w-6 text-purple-500" />
                  Adversarial Co-Evolution
                </DialogTitle>
                <DialogDescription>How competition drives innovation in security</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 pt-4">
                <p className="text-foreground">
                  Co-evolution is the process where two competing AI agents (Red and Blue Teams) drive each other to become increasingly sophisticated through continuous competition.
                </p>

                <div>
                  <h4 className="font-semibold mb-2">The Evolution Cycle</h4>
                  <div className="space-y-3">
                    <div className="bg-gradient-to-r from-red-500/10 to-blue-500/10 border border-purple-500/20 rounded-lg p-3">
                      <p className="text-sm text-muted-foreground mb-2">
                        <strong className="text-purple-400">Phase 1:</strong> Red Team finds a vulnerability and successfully exploits it
                      </p>
                      <p className="text-sm text-muted-foreground mb-2">
                        <strong className="text-purple-400">Phase 2:</strong> Blue Team learns from the attack and develops a countermeasure
                      </p>
                      <p className="text-sm text-muted-foreground mb-2">
                        <strong className="text-purple-400">Phase 3:</strong> Red Team evolves new tactics to bypass the improved defenses
                      </p>
                      <p className="text-sm text-muted-foreground">
                        <strong className="text-purple-400">Phase 4:</strong> Blue Team adapts again, creating an ongoing arms race
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Nash Equilibrium in Security</h4>
                  <p className="text-sm text-muted-foreground">
                    The system reaches Nash equilibrium when neither agent can improve their win rate without the other adapting. This represents the optimal balance between attack sophistication and defensive capability—the best possible security posture for the given environment.
                  </p>
                </div>

                <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-3">
                  <p className="text-sm text-muted-foreground">
                    <strong className="text-purple-400">Why This Matters:</strong> Traditional security relies on humans writing rules after attacks are discovered. Co-evolution discovers vulnerabilities proactively and develops optimal defenses automatically, staying ahead of real-world threats.
                  </p>
                </div>

                <Button asChild className="w-full bg-gradient-to-r from-purple-500 to-pink-500">
                  <Link href="/evolution">View Evolution Timeline</Link>
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>

        {/* Key Features with Educational Dialogs */}
        <div className="mt-16 text-center">
          <div className="flex items-center justify-center gap-2 mb-6">
            <h2 className="text-3xl font-bold text-foreground">Key Features</h2>
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
                  <HelpCircle className="h-5 w-5 text-muted-foreground" />
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Platform Capabilities</DialogTitle>
                  <DialogDescription>Advanced security features powered by AI</DialogDescription>
                </DialogHeader>
                <div className="space-y-3 pt-4">
                  {features.map((feature, idx) => (
                    <div key={idx} className="bg-card/50 border rounded-lg p-3">
                      <div className="flex items-center gap-2 mb-1">
                        <feature.icon className="h-4 w-4 text-primary" />
                        <p className="font-semibold">{feature.title}</p>
                      </div>
                      <p className="text-sm text-muted-foreground">{feature.learnMore}</p>
                    </div>
                  ))}
                </div>
              </DialogContent>
            </Dialog>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-muted-foreground">
            {features.map((feature, index) => (
              <Dialog key={index}>
                <DialogTrigger asChild>
                  <div className="bg-card/30 p-4 rounded-lg border hover:border-primary/50 transition-all cursor-pointer group">
                    <div className="flex items-center justify-center mb-2">
                      <feature.icon className="h-8 w-8 text-primary group-hover:scale-110 transition-transform" />
                    </div>
                    <div className="font-semibold mb-1">{feature.title}</div>
                    <Info className="h-3 w-3 mx-auto text-muted-foreground group-hover:text-primary transition-colors" />
                  </div>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle className="flex items-center gap-2">
                      <feature.icon className="h-6 w-6 text-primary" />
                      {feature.title}
                    </DialogTitle>
                    <DialogDescription>{feature.description}</DialogDescription>
                  </DialogHeader>
                  <div className="space-y-4 pt-4">
                    <div>
                      <h4 className="font-semibold mb-2">How It Works</h4>
                      <p className="text-sm text-muted-foreground">{feature.details}</p>
                    </div>
                    <div className="bg-primary/10 border border-primary/20 rounded-lg p-3">
                      <p className="text-sm text-muted-foreground">
                        <strong className="text-primary">Learn More:</strong> {feature.learnMore}
                      </p>
                    </div>
                    <Button asChild className="w-full">
                      <Link href="/dashboard">Explore on Dashboard</Link>
                    </Button>
                  </div>
                </DialogContent>
              </Dialog>
            ))}
          </div>
        </div>

        {/* Getting Started Section */}
        <div className="mt-16 bg-card/50 backdrop-blur-lg border rounded-xl p-6 sm:p-8">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-foreground flex items-center gap-2">
              <BookOpen className="h-6 w-6 text-primary" />
              Quick Start Guide
            </h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <h3 className="font-semibold text-lg">For Security Professionals</h3>
              <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                <li>Start with <Link href="/threat-hunting" className="text-primary hover:underline">Threat Hunting</Link> to detect advanced threats</li>
                <li>Use <Link href="/code-review" className="text-primary hover:underline">Code Review</Link> to scan applications for vulnerabilities</li>
                <li>Generate detection rules with <Link href="/siem-rules" className="text-primary hover:underline">SIEM Rules</Link></li>
                <li>Monitor attacks in <Link href="/live-battle" className="text-primary hover:underline">Live Battle</Link> arena</li>
              </ol>
            </div>
            <div className="space-y-3">
              <h3 className="font-semibold text-lg">For Learning & Research</h3>
              <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                <li>Watch <Link href="/live-battle" className="text-primary hover:underline">Live Battles</Link> to see AI attack strategies</li>
                <li>Practice in the <Link href="/cyber-range" className="text-primary hover:underline">Cyber Range</Link> safely</li>
                <li>Ask the <Link href="/ai-assistant" className="text-primary hover:underline">AI Assistant</Link> cybersecurity questions</li>
                <li>Study <Link href="/zero-day" className="text-primary hover:underline">Zero-Day cases</Link> and learn from history</li>
              </ol>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}
