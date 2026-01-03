"use client"

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Code, AlertTriangle, CheckCircle, Shield, Download, Copy, Sparkles } from 'lucide-react'

interface Vulnerability {
  id: string
  title: string
  severity: 'critical' | 'high' | 'medium' | 'low'
  cwe: string
  line: number
  description: string
  vulnerable_code: string
  fixed_code: string
  impact: string
}

export default function CodeReviewPage() {
  const [code, setCode] = useState('')
  const [language, setLanguage] = useState('javascript')
  const [vulnerabilities, setVulnerabilities] = useState<Vulnerability[]>([])
  const [analyzing, setAnalyzing] = useState(false)

  const analyzeCode = async () => {
    if (!code) return
    setAnalyzing(true)

    await new Promise(r => setTimeout(r, 2000))

    const mockVulnerabilities: Vulnerability[] = [
      {
        id: 'V1',
        title: 'SQL Injection Vulnerability',
        severity: 'critical',
        cwe: 'CWE-89',
        line: 5,
        description: 'User input is directly concatenated into SQL query without sanitization',
        vulnerable_code: 'const query = "SELECT * FROM users WHERE id = " + userId;',
        fixed_code: 'const query = "SELECT * FROM users WHERE id = ?"; \ndb.query(query, [userId]);',
        impact: 'Attackers can execute arbitrary SQL commands, potentially accessing or modifying database'
      },
      {
        id: 'V2',
        title: 'Cross-Site Scripting (XSS)',
        severity: 'high',
        cwe: 'CWE-79',
        line: 12,
        description: 'User input rendered without escaping, allowing script injection',
        vulnerable_code: 'element.innerHTML = userInput;',
        fixed_code: 'element.textContent = userInput; // or use DOMPurify.sanitize(userInput)',
        impact: 'Attackers can inject malicious scripts to steal cookies or perform actions as the user'
      }
    ]

    setVulnerabilities(mockVulnerabilities)
    setAnalyzing(false)
  }

  const exampleCode = {
    javascript: `// Login endpoint with vulnerabilities
app.post('/login', (req, res) => {
  const { username, password } = req.body;
  const query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'";
  db.query(query, (err, results) => {
    if (results.length > 0) {
      res.send("<h1>Welcome " + username + "</h1>");
    }
  });
});`,
    python: `# API endpoint with security issues
@app.route('/user/<id>')
def get_user(id):
    query = f"SELECT * FROM users WHERE id = {id}"
    result = db.execute(query)
    return result`,
    java: `// Unsafe deserialization
public void processData(String data) {
    ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(data.getBytes()));
    Object obj = ois.readObject();
    return obj;
}`
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/50'
      case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/50'
      case 'medium': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50'
      case 'low': return 'bg-blue-500/20 text-blue-400 border-blue-500/50'
      default: return 'bg-gray-500/20 text-gray-400 border-gray-500/50'
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent">
            AI Security Code Reviewer
          </h1>
          <p className="text-gray-400 mt-2">Paste code to find vulnerabilities and get secure fixes</p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card className="bg-black/40 border-red-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Critical</p>
              <p className="text-3xl font-bold text-red-400 mt-1">
                {vulnerabilities.filter(v => v.severity === 'critical').length}
              </p>
            </div>
            <AlertTriangle className="h-8 w-8 text-red-400" />
          </div>
        </Card>
        <Card className="bg-black/40 border-orange-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">High</p>
              <p className="text-3xl font-bold text-orange-400 mt-1">
                {vulnerabilities.filter(v => v.severity === 'high').length}
              </p>
            </div>
            <Shield className="h-8 w-8 text-orange-400" />
          </div>
        </Card>
        <Card className="bg-black/40 border-yellow-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Medium</p>
              <p className="text-3xl font-bold text-yellow-400 mt-1">
                {vulnerabilities.filter(v => v.severity === 'medium').length}
              </p>
            </div>
            <CheckCircle className="h-8 w-8 text-yellow-400" />
          </div>
        </Card>
        <Card className="bg-black/40 border-green-500/20 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-gray-400 text-sm">Security Score</p>
              <p className="text-3xl font-bold text-green-400 mt-1">
                {vulnerabilities.length > 0 ? Math.max(0, 100 - (vulnerabilities.length * 15)) : 100}
              </p>
            </div>
            <Code className="h-8 w-8 text-green-400" />
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="bg-black/40 border-cyan-500/20 p-6">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-white">Code to Review</h2>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="px-3 py-1 bg-black/40 border border-cyan-500/20 rounded-lg text-white text-sm"
              >
                <option value="javascript">JavaScript</option>
                <option value="python">Python</option>
                <option value="java">Java</option>
                <option value="csharp">C#</option>
                <option value="php">PHP</option>
              </select>
            </div>

            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              placeholder="Paste your code here..."
              className="w-full h-96 bg-black/60 border border-cyan-500/20 rounded-lg p-4 text-gray-300 font-mono text-sm focus:outline-none focus:border-cyan-500/50"
            />

            <div className="flex gap-2">
              <Button onClick={analyzeCode} disabled={analyzing || !code} className="flex-1 bg-cyan-500 hover:bg-cyan-600">
                <Sparkles className="mr-2 h-4 w-4" />
                {analyzing ? 'Analyzing...' : 'Analyze Code'}
              </Button>
              <Button
                onClick={() => setCode(exampleCode[language as keyof typeof exampleCode])}
                variant="outline"
                className="bg-black/40 border-cyan-500/20 hover:bg-cyan-500/10"
              >
                Load Example
              </Button>
            </div>
          </div>
        </Card>

        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-white">Vulnerabilities Found ({vulnerabilities.length})</h2>
          {vulnerabilities.length === 0 ? (
            <Card className="bg-black/40 border-cyan-500/20 p-12 text-center">
              <Shield className="h-12 w-12 text-gray-500 mx-auto mb-4" />
              <p className="text-gray-400">No analysis yet. Paste code and click Analyze.</p>
            </Card>
          ) : (
            vulnerabilities.map((vuln) => (
              <Card key={vuln.id} className="bg-black/40 border-red-500/20 p-6">
                <div className="space-y-4">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-3 mb-1">
                        <h3 className="text-lg font-semibold text-white">{vuln.title}</h3>
                        <Badge className={getSeverityColor(vuln.severity)}>
                          {vuln.severity.toUpperCase()}
                        </Badge>
                      </div>
                      <p className="text-gray-400 text-sm">Line {vuln.line} • {vuln.cwe}</p>
                    </div>
                  </div>

                  <p className="text-gray-300">{vuln.description}</p>

                  <div>
                    <p className="text-red-400 text-sm font-semibold mb-2">❌ Vulnerable Code:</p>
                    <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-3">
                      <code className="text-sm text-gray-300">{vuln.vulnerable_code}</code>
                    </div>
                  </div>

                  <div>
                    <p className="text-green-400 text-sm font-semibold mb-2">✅ Secure Fix:</p>
                    <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-3">
                      <code className="text-sm text-gray-300 whitespace-pre-wrap">{vuln.fixed_code}</code>
                    </div>
                  </div>

                  <div className="bg-blue-500/10 border border-blue-500/50 rounded-lg p-3">
                    <p className="text-blue-400 text-sm font-semibold mb-1">Impact:</p>
                    <p className="text-gray-300 text-sm">{vuln.impact}</p>
                  </div>

                  <div className="flex gap-2">
                    <Button size="sm" className="bg-green-500/20 hover:bg-green-500/30 text-green-400">
                      <Copy className="mr-2 h-4 w-4" />
                      Copy Fix
                    </Button>
                    <Button size="sm" className="bg-blue-500/20 hover:bg-blue-500/30 text-blue-400">
                      Learn More
                    </Button>
                  </div>
                </div>
              </Card>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
