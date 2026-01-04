'use client';

import { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle2, TrendingUp, Lightbulb, Target, Lock, Eye } from 'lucide-react';

interface Recommendation {
  id: string;
  title: string;
  description: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  category: string;
  implemented: boolean;
  attacksBlocked: number;
  effectiveness: number;
}

export default function RecommendationsPage() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([
    {
      id: '1',
      title: 'Implement Multi-Factor Authentication (MFA)',
      description: 'Your system is vulnerable to credential stuffing and brute force attacks. MFA would block 99% of these attacks by requiring a second form of authentication beyond passwords.',
      severity: 'critical',
      category: 'Authentication',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 99,
    },
    {
      id: '2',
      title: 'Deploy Web Application Firewall (WAF)',
      description: 'SQL Injection and XSS attacks are succeeding at a 45% rate. A WAF would filter malicious HTTP traffic and block common web exploits before they reach your application.',
      severity: 'critical',
      category: 'Network Security',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 85,
    },
    {
      id: '3',
      title: 'Enable Rate Limiting on API Endpoints',
      description: 'Brute force attacks are overwhelming your authentication endpoints. Rate limiting would restrict the number of login attempts, making brute force attacks impractical.',
      severity: 'high',
      category: 'API Security',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 92,
    },
    {
      id: '4',
      title: 'Implement Network Segmentation',
      description: 'Lateral movement attacks are spreading across your network. Segmenting your network would contain breaches and prevent attackers from moving freely between systems.',
      severity: 'high',
      category: 'Network Security',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 78,
    },
    {
      id: '5',
      title: 'Deploy Intrusion Detection System (IDS)',
      description: '30% of attacks go undetected. An IDS would monitor network traffic for suspicious patterns and alert on potential threats in real-time.',
      severity: 'high',
      category: 'Monitoring',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 75,
    },
    {
      id: '6',
      title: 'Implement Input Validation & Sanitization',
      description: 'XSS and injection attacks exploit poor input handling. Strict validation and sanitization would prevent malicious code execution.',
      severity: 'high',
      category: 'Application Security',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 88,
    },
    {
      id: '7',
      title: 'Enable Database Encryption at Rest',
      description: 'If data exfiltration succeeds, attackers get plain-text data. Encryption would render stolen data useless without decryption keys.',
      severity: 'medium',
      category: 'Data Protection',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 95,
    },
    {
      id: '8',
      title: 'Implement Security Headers (CSP, HSTS, X-Frame-Options)',
      description: 'Missing security headers make your app vulnerable to clickjacking, XSS, and man-in-the-middle attacks. Proper headers provide browser-level protection.',
      severity: 'medium',
      category: 'Application Security',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 70,
    },
    {
      id: '9',
      title: 'Deploy Honeypots & Deception Technology',
      description: 'Attract and trap attackers with decoy systems. Honeypots waste attacker time, provide early warning, and gather threat intelligence.',
      severity: 'medium',
      category: 'Defense Strategy',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 65,
    },
    {
      id: '10',
      title: 'Implement Automated Patch Management',
      description: 'Privilege escalation attacks exploit unpatched vulnerabilities. Automated patching keeps systems up-to-date and closes security holes.',
      severity: 'high',
      category: 'System Hardening',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 82,
    },
    {
      id: '11',
      title: 'Enable Comprehensive Logging & SIEM Integration',
      description: 'Insufficient logging makes forensics impossible. Centralized logging with SIEM correlation would improve detection and investigation capabilities.',
      severity: 'medium',
      category: 'Monitoring',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 73,
    },
    {
      id: '12',
      title: 'Implement Zero Trust Architecture',
      description: 'Current perimeter-based security fails against insider threats and lateral movement. Zero Trust verifies every access request regardless of origin.',
      severity: 'high',
      category: 'Architecture',
      implemented: false,
      attacksBlocked: 0,
      effectiveness: 90,
    },
  ]);

  const [filter, setFilter] = useState<string>('all');
  const [stats, setStats] = useState({
    totalRecommendations: 12,
    implemented: 0,
    pending: 12,
    criticalPending: 2,
    potentialBlockRate: 0,
  });

  useEffect(() => {
    const implemented = recommendations.filter(r => r.implemented).length;
    const criticalPending = recommendations.filter(r => r.severity === 'critical' && !r.implemented).length;
    const potentialBlockRate = recommendations
      .filter(r => !r.implemented)
      .reduce((acc, r) => acc + r.effectiveness, 0) / recommendations.length;

    setStats({
      totalRecommendations: recommendations.length,
      implemented,
      pending: recommendations.length - implemented,
      criticalPending,
      potentialBlockRate: Math.round(potentialBlockRate),
    });
  }, [recommendations]);

  const toggleImplementation = (id: string) => {
    setRecommendations(prev =>
      prev.map(r =>
        r.id === id
          ? {
              ...r,
              implemented: !r.implemented,
              attacksBlocked: !r.implemented ? Math.floor(Math.random() * 50) + 10 : 0,
            }
          : r
      )
    );
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500/20 text-red-400 border-red-500/30';
      case 'high':
        return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      case 'medium':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'low':
        return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    }
  };

  const filteredRecommendations = recommendations.filter(r => {
    if (filter === 'all') return true;
    if (filter === 'implemented') return r.implemented;
    if (filter === 'pending') return !r.implemented;
    if (filter === 'critical') return r.severity === 'critical';
    return true;
  });

  return (
    <div className="min-h-screen bg-background p-8">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
              <Lightbulb className="w-8 h-8 text-yellow-500" />
              Security Recommendations
            </h1>
            <p className="text-muted-foreground">
              AI-generated suggestions to improve Preet Raval's system defenses
            </p>
          </div>
        </div>

        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 flex items-start gap-3 mb-6">
          <Lightbulb className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-muted-foreground leading-relaxed">
              <strong className="text-foreground">What this page does:</strong> Receive AI-generated security recommendations based on observed attack patterns from Live Battle simulations. Blue Team AI analyzes which attacks succeed most frequently and suggests prioritized defenses (MFA, WAF, rate limiting, network segmentation). View effectiveness ratings, implementation guides, and estimated improvement to your overall security posture.
            </p>
          </div>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
        <div className="bg-card rounded-lg p-6 border">
          <div className="flex items-center gap-2 mb-2">
            <Target className="w-5 h-5 text-primary" />
            <h3 className="text-sm text-muted-foreground">Total Recommendations</h3>
          </div>
          <p className="text-3xl font-bold">{stats.totalRecommendations}</p>
        </div>

        <div className="bg-card rounded-lg p-6 border border-green-500/30">
          <div className="flex items-center gap-2 mb-2">
            <CheckCircle2 className="w-5 h-5 text-green-500" />
            <h3 className="text-sm text-muted-foreground">Implemented</h3>
          </div>
          <p className="text-3xl font-bold text-green-500">{stats.implemented}</p>
        </div>

        <div className="bg-card rounded-lg p-6 border border-yellow-500/30">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-yellow-500" />
            <h3 className="text-sm text-muted-foreground">Pending</h3>
          </div>
          <p className="text-3xl font-bold text-yellow-500">{stats.pending}</p>
        </div>

        <div className="bg-card rounded-lg p-6 border border-red-500/30">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            <h3 className="text-sm text-muted-foreground">Critical Pending</h3>
          </div>
          <p className="text-3xl font-bold text-red-500">{stats.criticalPending}</p>
        </div>

        <div className="bg-card rounded-lg p-6 border border-purple-500/30">
          <div className="flex items-center gap-2 mb-2">
            <TrendingUp className="w-5 h-5 text-purple-500" />
            <h3 className="text-sm text-muted-foreground">Potential Block Rate</h3>
          </div>
          <p className="text-3xl font-bold text-purple-500">{stats.potentialBlockRate}%</p>
        </div>
      </div>

      {/* Filters */}
      <div className="flex gap-2 mb-6">
        {['all', 'pending', 'implemented', 'critical'].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`px-4 py-2 rounded-lg font-medium transition-all ${
              filter === f
                ? 'bg-blue-600 text-white'
                : 'bg-white/10 text-gray-300 hover:bg-white/20'
            }`}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* Recommendations List */}
      <div className="space-y-4">
        {filteredRecommendations.map((rec) => (
          <div
            key={rec.id}
            className={`bg-card rounded-lg p-6 border transition-all ${
              rec.implemented
                ? 'border-green-500/30 bg-green-500/5'
                : 'hover:bg-accent'
            }`}
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-2">
                  <h3 className="text-xl font-bold">{rec.title}</h3>
                  <span className={`text-xs px-3 py-1 rounded-full border ${getSeverityColor(rec.severity)}`}>
                    {rec.severity.toUpperCase()}
                  </span>
                  <span className="text-xs bg-blue-500/20 text-blue-400 px-3 py-1 rounded-full">
                    {rec.category}
                  </span>
                  {rec.implemented && (
                    <span className="text-xs bg-green-500/20 text-green-400 px-3 py-1 rounded-full flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      IMPLEMENTED
                    </span>
                  )}
                </div>
                <p className="text-muted-foreground mb-3">{rec.description}</p>
                <div className="flex items-center gap-6 text-sm">
                  <div>
                    <span className="text-muted-foreground">Effectiveness: </span>
                    <span className="font-semibold">{rec.effectiveness}%</span>
                  </div>
                  {rec.implemented && (
                    <div>
                      <span className="text-muted-foreground">Attacks Blocked: </span>
                      <span className="font-semibold text-green-500">+{rec.attacksBlocked}</span>
                    </div>
                  )}
                </div>
              </div>
              <button
                onClick={() => toggleImplementation(rec.id)}
                className={`ml-4 px-6 py-2 rounded-lg font-semibold transition-all ${
                  rec.implemented
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-green-600 hover:bg-green-700 text-white'
                }`}
              >
                {rec.implemented ? 'Remove' : 'Implement'}
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Implementation Guide */}
      <div className="mt-8 bg-card rounded-lg p-6 border border-purple-500/30">
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Lock className="w-6 h-6 text-purple-500" />
          Priority Implementation Guide
        </h2>
        <div className="space-y-3">
          <div className="flex items-start gap-3 p-3 bg-red-500/10 rounded-lg border border-red-500/20">
            <span className="text-2xl">1️⃣</span>
            <div>
              <h4 className="font-semibold">Start with CRITICAL severity items</h4>
              <p className="text-sm text-muted-foreground">These address the most dangerous vulnerabilities with highest impact.</p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-3 bg-orange-500/10 rounded-lg border border-orange-500/20">
            <span className="text-2xl">2️⃣</span>
            <div>
              <h4 className="font-semibold">Implement HIGH severity recommendations</h4>
              <p className="text-sm text-muted-foreground">Significant security improvements with good ROI.</p>
            </div>
          </div>
          <div className="flex items-start gap-3 p-3 bg-yellow-500/10 rounded-lg border border-yellow-500/20">
            <span className="text-2xl">3️⃣</span>
            <div>
              <h4 className="font-semibold">Add MEDIUM severity enhancements</h4>
              <p className="text-sm text-muted-foreground">Defense-in-depth layers for comprehensive protection.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
