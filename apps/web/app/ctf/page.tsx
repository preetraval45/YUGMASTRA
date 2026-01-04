'use client';

/**
 * Capture The Flag (CTF) Challenge Platform
 * Educational cybersecurity challenges with progressive difficulty
 */

import { useState } from 'react';
import { Flag, Lock, CheckCircle, Star, Trophy, Code, Shield, AlertTriangle } from 'lucide-react';

interface Challenge {
  id: string;
  title: string;
  category: 'web' | 'crypto' | 'forensics' | 'pwn' | 'reverse' | 'misc';
  difficulty: 'easy' | 'medium' | 'hard' | 'expert';
  points: number;
  description: string;
  hints: string[];
  flag: string;
  solved: boolean;
}

const challenges: Challenge[] = [
  {
    id: 'web-1',
    title: 'SQL Injection 101',
    category: 'web',
    difficulty: 'easy',
    points: 100,
    description: 'Exploit a vulnerable login form to bypass authentication. The application uses a simple SQL query without prepared statements.',
    hints: [
      'Try common SQL injection payloads',
      'Use OR 1=1 to bypass authentication',
      'The flag is in the database table "secrets"'
    ],
    flag: 'YUGMASTRA{sql_1nj3ct10n_b4s1cs}',
    solved: false,
  },
  {
    id: 'web-2',
    title: 'XSS Cookie Stealer',
    category: 'web',
    difficulty: 'medium',
    points: 200,
    description: 'Find and exploit a reflected XSS vulnerability to steal admin cookies. The application reflects user input without sanitization.',
    hints: [
      'Look for user input that gets reflected in the HTML',
      'Use <script> tags to execute JavaScript',
      'document.cookie contains the session'
    ],
    flag: 'YUGMASTRA{xss_c00k13_st34l3r}',
    solved: false,
  },
  {
    id: 'crypto-1',
    title: 'Caesar Cipher Decoded',
    category: 'crypto',
    difficulty: 'easy',
    points: 50,
    description: 'Decrypt the following message encrypted with a Caesar cipher (ROT13): LHTZNFGEN{p3nfne_pvcure_vf_r4fl}',
    hints: [
      'Caesar cipher shifts each letter by a fixed amount',
      'Try ROT13 (shift of 13)',
      'Online tools can help'
    ],
    flag: 'YUGMASTRA{c3asar_cipher_is_e4sy}',
    solved: false,
  },
  {
    id: 'forensics-1',
    title: 'Hidden in Plain Sight',
    category: 'forensics',
    difficulty: 'medium',
    points: 150,
    description: 'Analyze the provided network packet capture (PCAP) file to find the exfiltrated flag. The attacker used DNS tunneling.',
    hints: [
      'Use Wireshark to analyze the PCAP',
      'Filter for DNS queries',
      'Look for suspicious subdomain names'
    ],
    flag: 'YUGMASTRA{dns_tunn3l1ng_d3t3ct3d}',
    solved: false,
  },
  {
    id: 'pwn-1',
    title: 'Buffer Overflow Basics',
    category: 'pwn',
    difficulty: 'hard',
    points: 300,
    description: 'Exploit a buffer overflow vulnerability in a C program to gain shell access. ASLR is disabled.',
    hints: [
      'Overflow the buffer to overwrite the return address',
      'Use pattern_create to find offset',
      'Inject shellcode or use ROP'
    ],
    flag: 'YUGMASTRA{buff3r_0v3rfl0w_pwn3d}',
    solved: false,
  },
  {
    id: 'reverse-1',
    title: 'Keygen Challenge',
    category: 'reverse',
    difficulty: 'hard',
    points: 350,
    description: 'Reverse engineer the provided binary to understand the license key validation algorithm and generate a valid key.',
    hints: [
      'Use Ghidra or IDA Pro for static analysis',
      'Look for the key validation function',
      'The algorithm uses simple XOR operations'
    ],
    flag: 'YUGMASTRA{r3v3rs3_3ng1n33r1ng}',
    solved: false,
  },
  {
    id: 'crypto-2',
    title: 'RSA Weak Keys',
    category: 'crypto',
    difficulty: 'expert',
    points: 500,
    description: 'Factor the RSA modulus N = 1234567890123456789 to decrypt the encrypted flag. The primes are small enough to factor.',
    hints: [
      'Use factorization tools like yafu or msieve',
      'Once you have p and q, calculate d',
      'Decrypt using m = c^d mod N'
    ],
    flag: 'YUGMASTRA{rsa_w34k_k3ys_br0k3n}',
    solved: false,
  },
];

export default function CTFPage() {
  const [solvedChallenges, setSolvedChallenges] = useState<Set<string>>(new Set());
  const [flagInput, setFlagInput] = useState<{ [key: string]: string }>({});
  const [selectedChallenge, setSelectedChallenge] = useState<Challenge | null>(null);
  const [showHints, setShowHints] = useState<{ [key: string]: number }>({});

  const handleFlagSubmit = (challenge: Challenge) => {
    const userFlag = flagInput[challenge.id]?.trim();

    if (userFlag === challenge.flag) {
      setSolvedChallenges(new Set([...solvedChallenges, challenge.id]));
      setFlagInput({ ...flagInput, [challenge.id]: '' });
      alert(`🎉 Correct! You've earned ${challenge.points} points!`);
    } else {
      alert('❌ Incorrect flag. Try again!');
    }
  };

  const revealHint = (challengeId: string) => {
    const currentHints = showHints[challengeId] || 0;
    setShowHints({ ...showHints, [challengeId]: currentHints + 1 });
  };

  const totalPoints = challenges.reduce((sum, c) => sum + c.points, 0);
  const earnedPoints = Array.from(solvedChallenges).reduce((sum, id) => {
    const challenge = challenges.find((c) => c.id === id);
    return sum + (challenge?.points || 0);
  }, 0);

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'web':
        return <Code className="w-5 h-5" />;
      case 'crypto':
        return <Lock className="w-5 h-5" />;
      case 'forensics':
        return <AlertTriangle className="w-5 h-5" />;
      case 'pwn':
        return <Shield className="w-5 h-5" />;
      case 'reverse':
        return <Code className="w-5 h-5" />;
      default:
        return <Flag className="w-5 h-5" />;
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'easy':
        return 'text-green-400 bg-green-500/20 border-green-500/30';
      case 'medium':
        return 'text-yellow-400 bg-yellow-500/20 border-yellow-500/30';
      case 'hard':
        return 'text-orange-400 bg-orange-500/20 border-orange-500/30';
      case 'expert':
        return 'text-red-400 bg-red-500/20 border-red-500/30';
      default:
        return 'text-gray-400 bg-gray-500/20 border-gray-500/30';
    }
  };

  return (
    <div className="min-h-screen bg-background p-8 pt-32">
      <div className="mb-8">
        <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
          <Trophy className="w-8 h-8 text-yellow-500" />
          YUGMĀSTRA CTF
        </h1>
        <p className="text-muted-foreground">
          Capture The Flag - Educational Cybersecurity Challenges
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <div className="bg-card rounded-lg p-6 border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Challenges Solved</p>
              <p className="text-3xl font-bold">{solvedChallenges.size}/{challenges.length}</p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-card rounded-lg p-6 border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Points Earned</p>
              <p className="text-3xl font-bold">{earnedPoints}/{totalPoints}</p>
            </div>
            <Star className="w-8 h-8 text-yellow-500" />
          </div>
        </div>

        <div className="bg-card rounded-lg p-6 border">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-muted-foreground">Progress</p>
              <p className="text-3xl font-bold">{((earnedPoints / totalPoints) * 100).toFixed(0)}%</p>
            </div>
            <Trophy className="w-8 h-8 text-purple-500" />
          </div>
        </div>
      </div>

      {/* Challenges Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {challenges.map((challenge) => {
          const isSolved = solvedChallenges.has(challenge.id);
          const revealedHints = showHints[challenge.id] || 0;

          return (
            <div
              key={challenge.id}
              className={`bg-card rounded-lg p-6 border ${
                isSolved ? 'border-green-500/50' : 'border-border'
              } hover:border-primary/50 transition-all`}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="text-primary">{getCategoryIcon(challenge.category)}</div>
                  <div>
                    <h3 className="text-xl font-bold">{challenge.title}</h3>
                    <p className="text-sm text-muted-foreground capitalize">{challenge.category}</p>
                  </div>
                </div>
                {isSolved && <CheckCircle className="w-6 h-6 text-green-500" />}
              </div>

              <div className="flex items-center gap-2 mb-4">
                <span className={`text-xs px-2 py-1 rounded border ${getDifficultyColor(challenge.difficulty)}`}>
                  {challenge.difficulty.toUpperCase()}
                </span>
                <span className="text-xs bg-purple-500/20 text-purple-400 px-2 py-1 rounded border border-purple-500/30">
                  {challenge.points} points
                </span>
              </div>

              <p className="text-sm text-muted-foreground mb-4">{challenge.description}</p>

              {/* Hints */}
              {challenge.hints.length > 0 && (
                <div className="mb-4">
                  <button
                    onClick={() => revealHint(challenge.id)}
                    disabled={revealedHints >= challenge.hints.length}
                    className="text-xs text-yellow-400 hover:text-yellow-300 disabled:opacity-50"
                  >
                    💡 Reveal Hint ({revealedHints}/{challenge.hints.length})
                  </button>
                  {revealedHints > 0 && (
                    <div className="mt-2 space-y-1">
                      {challenge.hints.slice(0, revealedHints).map((hint, idx) => (
                        <p key={idx} className="text-xs text-yellow-400 bg-yellow-500/10 p-2 rounded">
                          {idx + 1}. {hint}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Flag Submission */}
              {!isSolved && (
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={flagInput[challenge.id] || ''}
                    onChange={(e) => setFlagInput({ ...flagInput, [challenge.id]: e.target.value })}
                    placeholder="YUGMASTRA{flag_here}"
                    className="flex-1 px-3 py-2 bg-background border border-border rounded text-sm"
                  />
                  <button
                    onClick={() => handleFlagSubmit(challenge)}
                    className="px-4 py-2 bg-primary hover:bg-primary/80 text-white rounded text-sm font-semibold"
                  >
                    Submit
                  </button>
                </div>
              )}

              {isSolved && (
                <div className="bg-green-500/20 border border-green-500/30 rounded p-3 text-green-400 text-sm">
                  ✓ Solved! Flag: <code className="bg-black/30 px-2 py-1 rounded">{challenge.flag}</code>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
