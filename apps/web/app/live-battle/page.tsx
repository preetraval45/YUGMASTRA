'use client';

import { useEffect, useState, useRef } from 'react';
import { Shield, Swords, AlertTriangle, CheckCircle2, XCircle, Zap, Target, Activity } from 'lucide-react';
import { sendNotification } from '@/hooks/use-notifications';

interface Attack {
  id: string;
  timestamp: number;
  type: string;
  target: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'attacking' | 'detected' | 'blocked' | 'successful';
  technique: string;
  x: number;
  y: number;
}

interface Defense {
  id: string;
  timestamp: number;
  action: string;
  attackId: string;
  effectiveness: number;
}

export default function LiveBattlePage() {
  const [attacks, setAttacks] = useState<Attack[]>([]);
  const [defenses, setDefenses] = useState<Defense[]>([]);
  const [systemHealth, setSystemHealth] = useState(100);
  const [score, setScore] = useState({ red: 0, blue: 0 });
  const [isRunning, setIsRunning] = useState(true);
  const [battleEnded, setBattleEnded] = useState(false);
  const [battleDuration, setBattleDuration] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const attackTypes = [
    { type: 'SQL Injection', technique: 'UNION-based SQLi', severity: 'high' as const },
    { type: 'XSS Attack', technique: 'Stored XSS', severity: 'medium' as const },
    { type: 'CSRF', technique: 'Token forgery', severity: 'medium' as const },
    { type: 'Remote Code Execution', technique: 'Deserialization attack', severity: 'critical' as const },
    { type: 'Privilege Escalation', technique: 'Kernel exploit', severity: 'high' as const },
    { type: 'Lateral Movement', technique: 'Pass-the-Hash', severity: 'high' as const },
    { type: 'Data Exfiltration', technique: 'DNS tunneling', severity: 'critical' as const },
    { type: 'DDoS', technique: 'SYN flood', severity: 'high' as const },
    { type: 'Phishing', technique: 'Spear phishing', severity: 'medium' as const },
    { type: 'Brute Force', technique: 'Credential stuffing', severity: 'low' as const },
  ];

  const targets = ['web_server', 'database', 'api_gateway', 'auth_service', 'file_storage', 'network_router'];

  const defenseActions = [
    'Blocked by firewall rule',
    'Detected anomalous pattern',
    'IDS signature match',
    'Rate limiting applied',
    'Connection terminated',
    'Traffic redirected to honeypot',
    'Machine learning model detected',
    'Behavioral analysis flagged',
  ];

  // Generate attacks
  useEffect(() => {
    if (!isRunning) return;

    const interval = setInterval(() => {
      const attackType = attackTypes[Math.floor(Math.random() * attackTypes.length)];
      const target = targets[Math.floor(Math.random() * targets.length)];

      const newAttack: Attack = {
        id: `attack-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        type: attackType.type,
        target,
        severity: attackType.severity,
        status: 'attacking',
        technique: attackType.technique,
        x: Math.random() * 100,
        y: Math.random() * 100,
      };

      setAttacks((prev) => [...prev.slice(-20), newAttack]);

      // Send notification for critical attacks
      if (attackType.severity === 'critical' || attackType.severity === 'high') {
        sendNotification(
          'attack',
          `${attackType.severity.toUpperCase()}: ${attackType.type}`,
          `Red Team attacking ${target.replace('_', ' ')} using ${attackType.technique}`,
          attackType.severity
        );
      }

      // Simulate defense response (70% detection rate)
      setTimeout(() => {
        const detected = Math.random() > 0.3;

        if (detected) {
          const effectiveness = 0.6 + Math.random() * 0.4;
          const blocked = effectiveness > 0.7;

          setAttacks((prev) =>
            prev.map((a) =>
              a.id === newAttack.id
                ? { ...a, status: blocked ? 'blocked' : 'detected' }
                : a
            )
          );

          const defense: Defense = {
            id: `defense-${Date.now()}`,
            timestamp: Date.now(),
            action: defenseActions[Math.floor(Math.random() * defenseActions.length)],
            attackId: newAttack.id,
            effectiveness,
          };

          setDefenses((prev) => [...prev.slice(-15), defense]);

          if (blocked) {
            setScore((prev) => ({ ...prev, blue: prev.blue + 1 }));
            // Send notification for successful blocks on critical attacks
            if (attackType.severity === 'critical') {
              sendNotification(
                'defense',
                'Critical Attack Blocked!',
                `Blue Team successfully blocked ${attackType.type} on ${target.replace('_', ' ')}`,
                'high'
              );
            }
          } else {
            setScore((prev) => ({ ...prev, red: prev.red + 1 }));
            setSystemHealth((prev) => Math.max(0, prev - (effectiveness > 0.5 ? 3 : 8)));
            // Send notification for detected but not blocked attacks
            if (attackType.severity === 'critical' || attackType.severity === 'high') {
              sendNotification(
                'attack',
                'Attack Detected but Not Blocked',
                `${attackType.type} was detected but succeeded on ${target.replace('_', ' ')}`,
                attackType.severity
              );
            }
          }
        } else {
          setAttacks((prev) =>
            prev.map((a) =>
              a.id === newAttack.id ? { ...a, status: 'successful' } : a
            )
          );
          setScore((prev) => ({ ...prev, red: prev.red + 1 }));
          setSystemHealth((prev) => Math.max(0, prev - 12));
          // Send notification for successful undetected attacks
          sendNotification(
            'attack',
            'Undetected Attack Successful!',
            `${attackType.type} bypassed defenses on ${target.replace('_', ' ')}`,
            'critical'
          );
        }
      }, 1000 + Math.random() * 2000);
    }, 800 + Math.random() * 1200);

    return () => clearInterval(interval);
  }, [isRunning]);

  // Auto-heal system and battle timer
  useEffect(() => {
    if (battleEnded) return;

    const interval = setInterval(() => {
      setSystemHealth((prev) => Math.min(100, prev + 0.5));
      if (isRunning) {
        setBattleDuration((prev) => prev + 1);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [isRunning, battleEnded]);

  // Handle end battle
  const handleEndBattle = () => {
    setIsRunning(false);
    setBattleEnded(true);
  };

  // Handle reset battle
  const handleResetBattle = () => {
    setAttacks([]);
    setDefenses([]);
    setSystemHealth(100);
    setScore({ red: 0, blue: 0 });
    setBattleDuration(0);
    setBattleEnded(false);
    setIsRunning(true);
  };

  // Format battle duration
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-600 bg-red-500/20 border-red-500/30';
      case 'high':
        return 'text-orange-600 bg-orange-500/20 border-orange-500/30';
      case 'medium':
        return 'text-yellow-600 bg-yellow-500/20 border-yellow-500/30';
      case 'low':
        return 'text-blue-600 bg-blue-500/20 border-blue-500/30';
      default:
        return 'text-gray-600 bg-gray-500/20 border-gray-500/30';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'attacking':
        return <Swords className="w-4 h-4 text-red-500 animate-pulse" />;
      case 'detected':
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'blocked':
        return <Shield className="w-4 h-4 text-green-500" />;
      case 'successful':
        return <XCircle className="w-4 h-4 text-red-500" />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-red-950 to-gray-900 p-6">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-white mb-2 flex items-center gap-3">
              <Zap className="w-8 h-8 text-yellow-500 animate-pulse" />
              Live Battle Arena
            </h1>
            <p className="text-blue-200">
              Red Team AI vs Blue Team AI - Real-time Defense of Preet Raval's System
            </p>
            <p className="text-sm text-gray-400 mt-1">
              Battle Duration: <span className="font-mono text-white">{formatDuration(battleDuration)}</span>
            </p>
          </div>
          <div className="flex gap-3">
            {!battleEnded ? (
              <>
                <button
                  onClick={() => setIsRunning(!isRunning)}
                  className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                    isRunning
                      ? 'bg-yellow-600 hover:bg-yellow-700 text-white'
                      : 'bg-green-600 hover:bg-green-700 text-white'
                  }`}
                >
                  {isRunning ? '⏸ Pause Battle' : '▶ Resume Battle'}
                </button>
                <button
                  onClick={handleEndBattle}
                  className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white rounded-lg font-semibold transition-all"
                >
                  ⏹ End Battle
                </button>
              </>
            ) : (
              <button
                onClick={handleResetBattle}
                className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-all"
              >
                🔄 Start New Battle
              </button>
            )}
          </div>
        </div>
        {battleEnded && (
          <div className="mt-4 p-4 bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/30 rounded-lg">
            <h2 className="text-2xl font-bold text-white mb-2">🏁 Battle Ended!</h2>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-gray-300 text-sm">Duration</p>
                <p className="text-2xl font-bold text-white">{formatDuration(battleDuration)}</p>
              </div>
              <div>
                <p className="text-gray-300 text-sm">Winner</p>
                <p className="text-2xl font-bold text-white">
                  {score.red > score.blue ? '🔴 Red Team' : score.blue > score.red ? '🔵 Blue Team' : '🤝 Draw'}
                </p>
              </div>
              <div>
                <p className="text-gray-300 text-sm">Final Score</p>
                <p className="text-2xl font-bold text-white">{score.red} - {score.blue}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Score and Health */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-red-500/30">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Swords className="w-6 h-6 text-red-500" />
              <h3 className="text-xl font-bold text-white">Red Team</h3>
            </div>
            <span className="text-3xl font-bold text-red-500">{score.red}</span>
          </div>
          <p className="text-sm text-gray-300">Successful Attacks</p>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-blue-500/30">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Shield className="w-6 h-6 text-blue-500" />
              <h3 className="text-xl font-bold text-white">Blue Team</h3>
            </div>
            <span className="text-3xl font-bold text-blue-500">{score.blue}</span>
          </div>
          <p className="text-sm text-gray-300">Blocked Attacks</p>
        </div>

        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-green-500/30">
          <div className="mb-2">
            <h3 className="text-xl font-bold text-white mb-2">System Health</h3>
            <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  systemHealth > 70
                    ? 'bg-green-500'
                    : systemHealth > 30
                    ? 'bg-yellow-500'
                    : 'bg-red-500'
                }`}
                style={{ width: `${systemHealth}%` }}
              />
            </div>
          </div>
          <p className="text-sm text-gray-300">{systemHealth.toFixed(1)}%</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Live Attacks Feed */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
          <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
            <Target className="w-6 h-6 text-red-500" />
            Incoming Attacks
          </h2>
          <div className="space-y-3 max-h-[600px] overflow-y-auto">
            {attacks
              .slice()
              .reverse()
              .map((attack) => (
                <div
                  key={attack.id}
                  className="p-4 bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-all"
                >
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(attack.status)}
                      <span className="font-semibold text-white">{attack.type}</span>
                    </div>
                    <span className={`text-xs px-2 py-1 rounded border ${getSeverityColor(attack.severity)}`}>
                      {attack.severity.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-sm text-gray-300 space-y-1">
                    <p>
                      <strong>Technique:</strong> {attack.technique}
                    </p>
                    <p>
                      <strong>Target:</strong>{' '}
                      <code className="bg-black/30 px-2 py-0.5 rounded">{attack.target}</code>
                    </p>
                    <p>
                      <strong>Status:</strong>{' '}
                      <span
                        className={
                          attack.status === 'blocked'
                            ? 'text-green-400'
                            : attack.status === 'successful'
                            ? 'text-red-400'
                            : 'text-yellow-400'
                        }
                      >
                        {attack.status}
                      </span>
                    </p>
                  </div>
                  <div className="mt-2 text-xs text-gray-500">
                    {new Date(attack.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              ))}
          </div>
        </div>

        {/* Defense Actions Feed */}
        <div className="bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
          <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
            <Shield className="w-6 h-6 text-blue-500" />
            Defense Actions
          </h2>
          <div className="space-y-3 max-h-[600px] overflow-y-auto">
            {defenses
              .slice()
              .reverse()
              .map((defense) => {
                const attack = attacks.find((a) => a.id === defense.attackId);
                return (
                  <div
                    key={defense.id}
                    className="p-4 bg-white/5 rounded-lg border border-blue-500/20 hover:bg-white/10 transition-all"
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <CheckCircle2 className="w-4 h-4 text-blue-500" />
                        <span className="font-semibold text-white">{defense.action}</span>
                      </div>
                      <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-1 rounded">
                        {(defense.effectiveness * 100).toFixed(0)}% effective
                      </span>
                    </div>
                    {attack && (
                      <div className="text-sm text-gray-300">
                        <p>
                          Defending against:{' '}
                          <span className="text-red-400">{attack.type}</span>
                        </p>
                        <p>
                          Target: <code className="bg-black/30 px-2 py-0.5 rounded">{attack.target}</code>
                        </p>
                      </div>
                    )}
                    <div className="mt-2 text-xs text-gray-500">
                      {new Date(defense.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      </div>

      {/* Battle Stats */}
      <div className="mt-6 bg-white/10 backdrop-blur-lg rounded-lg p-6 border border-white/20">
        <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
          <Activity className="w-6 h-6 text-purple-500" />
          Battle Statistics
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-white/5 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">Total Attacks</p>
            <p className="text-2xl font-bold text-white">{attacks.length}</p>
          </div>
          <div className="p-4 bg-white/5 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">Detection Rate</p>
            <p className="text-2xl font-bold text-white">
              {attacks.length > 0
                ? ((defenses.length / attacks.length) * 100).toFixed(1)
                : 0}%
            </p>
          </div>
          <div className="p-4 bg-white/5 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">Blue Team Win Rate</p>
            <p className="text-2xl font-bold text-white">
              {score.red + score.blue > 0
                ? ((score.blue / (score.red + score.blue)) * 100).toFixed(1)
                : 0}%
            </p>
          </div>
          <div className="p-4 bg-white/5 rounded-lg">
            <p className="text-sm text-gray-400 mb-1">System Owner</p>
            <p className="text-lg font-bold text-blue-400">Preet Raval</p>
          </div>
        </div>
      </div>
    </div>
  );
}
