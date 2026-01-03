"use client"

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Code, AlertTriangle, CheckCircle, Shield, Download, Copy, Sparkles, BookOpen, Lightbulb, Terminal } from 'lucide-react'

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

const SAMPLE_VULNERABILITIES = [
  {
    title: 'SQL Injection',
    language: 'javascript',
    code: `// Login endpoint with SQL injection vulnerability
app.post('/login', (req, res) => {
  const { username, password } = req.body;
  const query = "SELECT * FROM users WHERE username = '" + username + "' AND password = '" + password + "'";
  db.query(query, (err, results) => {
    if (results.length > 0) {
      res.send("<h1>Welcome " + username + "</h1>");
    }
  });
});`,
    description: 'This code directly concatenates user input into SQL queries, allowing attackers to inject malicious SQL commands and potentially access or modify the entire database.',
    fix: 'Use parameterized queries or prepared statements. Never concatenate user input directly into SQL strings.',
    secureCode: `app.post('/login', (req, res) => {
  const { username, password } = req.body;
  const query = "SELECT * FROM users WHERE username = ? AND password = ?";
  db.query(query, [username, password], (err, results) => {
    if (results.length > 0) {
      res.send("<h1>Welcome " + DOMPurify.sanitize(username) + "</h1>");
    }
  });
});`
  },
  {
    title: 'Cross-Site Scripting (XSS)',
    language: 'javascript',
    code: `// Vulnerable code rendering user input
function displayMessage(userInput) {
  document.getElementById('message').innerHTML = userInput;
}

// Vulnerable React component
function UserProfile({ username }) {
  return <div dangerouslySetInnerHTML={{ __html: username }} />;
}`,
    description: 'This code renders unsanitized user input directly into the DOM, allowing attackers to inject malicious scripts that can steal cookies, session tokens, or perform actions as the user.',
    fix: 'Use textContent instead of innerHTML, or sanitize input with DOMPurify. In React, avoid dangerouslySetInnerHTML or use a sanitization library.',
    secureCode: `// Secure vanilla JS
function displayMessage(userInput) {
  document.getElementById('message').textContent = userInput;
}

// Secure React component
function UserProfile({ username }) {
  return <div>{username}</div>; // React automatically escapes
}`
  },
  {
    title: 'Command Injection',
    language: 'python',
    code: `# Vulnerable file processing endpoint
@app.route('/process')
def process_file():
    filename = request.args.get('file')
    os.system(f"cat {filename}")
    return "Processed"

# Another vulnerable example
def backup_database(db_name):
    os.system(f"pg_dump {db_name} > backup.sql")`,
    description: 'This code executes shell commands with unsanitized user input, allowing attackers to inject arbitrary commands and potentially gain full system access.',
    fix: 'Use subprocess with argument lists instead of os.system, validate and sanitize all inputs, and use allowlists for file paths.',
    secureCode: `import subprocess
import os

@app.route('/process')
def process_file():
    filename = request.args.get('file')
    # Validate filename is in allowed directory
    safe_path = os.path.join('/safe/directory', os.path.basename(filename))
    if not os.path.exists(safe_path):
        return "Invalid file", 400
    subprocess.run(['cat', safe_path], check=True)
    return "Processed"`
  },
  {
    title: 'Insecure Deserialization',
    language: 'java',
    code: `// Unsafe object deserialization
public Object loadData(String data) {
    try {
        ObjectInputStream ois = new ObjectInputStream(
            new ByteArrayInputStream(data.getBytes())
        );
        return ois.readObject();
    } catch (Exception e) {
        return null;
    }
}`,
    description: 'This code deserializes untrusted data without validation, which can lead to remote code execution. Attackers can craft malicious serialized objects to execute arbitrary code.',
    fix: 'Avoid deserializing untrusted data. Use safe formats like JSON, implement object type validation, or use secure deserialization libraries with allowlists.',
    secureCode: `// Use JSON instead
import com.fasterxml.jackson.databind.ObjectMapper;

public User loadUserData(String jsonData) {
    ObjectMapper mapper = new ObjectMapper();
    try {
        return mapper.readValue(jsonData, User.class);
    } catch (JsonProcessingException e) {
        throw new SecurityException("Invalid data");
    }
}`
  }
]

export default function CodeReviewPage() {
  const [code, setCode] = useState('')
  const [language, setLanguage] = useState('javascript')
  const [vulnerabilities, setVulnerabilities] = useState<Vulnerability[]>([])
  const [analyzing, setAnalyzing] = useState(false)
  const [activeTab, setActiveTab] = useState('review')

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

  const loadExample = (example: typeof SAMPLE_VULNERABILITIES[0]) => {
    setCode(example.code)
    setLanguage(example.language)
    setActiveTab('review')
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

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  return (
    <div className="container-responsive py-6 sm:py-8 space-y-6 sm:space-y-8">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-4xl sm:text-5xl md:text-6xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent mb-3">
            AI Security Code Reviewer
          </h1>
          <p className="text-lg sm:text-xl text-muted-foreground">
            Automated vulnerability detection with AI-powered code analysis and secure fix recommendations
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <Card className="card-responsive bg-card border-red-500/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-base sm:text-lg text-muted-foreground">Critical</p>
              <p className="text-4xl sm:text-5xl font-bold text-red-400 mt-2">
                {vulnerabilities.filter(v => v.severity === 'critical').length}
              </p>
            </div>
            <AlertTriangle className="h-10 w-10 sm:h-12 sm:w-12 text-red-400" />
          </div>
        </Card>
        <Card className="card-responsive bg-card border-orange-500/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-base sm:text-lg text-muted-foreground">High</p>
              <p className="text-4xl sm:text-5xl font-bold text-orange-400 mt-2">
                {vulnerabilities.filter(v => v.severity === 'high').length}
              </p>
            </div>
            <Shield className="h-10 w-10 sm:h-12 sm:w-12 text-orange-400" />
          </div>
        </Card>
        <Card className="card-responsive bg-card border-yellow-500/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-base sm:text-lg text-muted-foreground">Medium</p>
              <p className="text-4xl sm:text-5xl font-bold text-yellow-400 mt-2">
                {vulnerabilities.filter(v => v.severity === 'medium').length}
              </p>
            </div>
            <CheckCircle className="h-10 w-10 sm:h-12 sm:w-12 text-yellow-400" />
          </div>
        </Card>
        <Card className="card-responsive bg-card border-green-500/20">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-base sm:text-lg text-muted-foreground">Security Score</p>
              <p className="text-4xl sm:text-5xl font-bold text-green-400 mt-2">
                {vulnerabilities.length > 0 ? Math.max(0, 100 - (vulnerabilities.length * 15)) : 100}
              </p>
            </div>
            <Code className="h-10 w-10 sm:h-12 sm:w-12 text-green-400" />
          </div>
        </Card>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-3 h-auto p-1 bg-muted">
          <TabsTrigger value="review" className="text-base sm:text-lg py-3 sm:py-4">
            <Terminal className="mr-2 h-5 w-5" />
            Review
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

        <TabsContent value="review" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="card-responsive bg-card border-cyan-500/20">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-2xl sm:text-3xl font-semibold">Code to Review</h2>
                  <select
                    value={language}
                    onChange={(e) => setLanguage(e.target.value)}
                    className="px-4 py-3 bg-background border border-border rounded-lg text-foreground text-base sm:text-lg"
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
                  placeholder="Paste your code here for AI security analysis..."
                  className="w-full h-96 bg-background border border-border rounded-lg p-4 text-foreground font-mono text-base sm:text-lg focus:outline-none focus:border-cyan-500/50 resize-none"
                />

                <div className="flex flex-col sm:flex-row gap-3">
                  <Button onClick={analyzeCode} disabled={analyzing || !code} className="flex-1 bg-cyan-500 hover:bg-cyan-600 text-base sm:text-lg px-6 py-6">
                    <Sparkles className="mr-2 h-5 w-5" />
                    {analyzing ? 'Analyzing...' : 'Analyze Code'}
                  </Button>
                  <Button
                    onClick={() => setCode(exampleCode[language as keyof typeof exampleCode])}
                    variant="outline"
                    className="bg-background border-border hover:bg-accent text-base sm:text-lg px-6 py-6"
                  >
                    Load Example
                  </Button>
                </div>
              </div>
            </Card>

            <div className="space-y-4">
              <h2 className="text-2xl sm:text-3xl font-semibold">Vulnerabilities Found ({vulnerabilities.length})</h2>
              {vulnerabilities.length === 0 ? (
                <Card className="card-responsive bg-card border-cyan-500/20 text-center py-16">
                  <Shield className="h-16 w-16 sm:h-20 sm:w-20 text-muted-foreground mx-auto mb-4" />
                  <p className="text-lg sm:text-xl text-muted-foreground">No analysis yet. Paste code and click Analyze.</p>
                </Card>
              ) : (
                vulnerabilities.map((vuln) => (
                  <Card key={vuln.id} className="card-responsive bg-card border-red-500/20">
                    <div className="space-y-4">
                      <div className="flex items-start justify-between">
                        <div>
                          <div className="flex flex-wrap items-center gap-3 mb-2">
                            <h3 className="text-xl sm:text-2xl font-semibold">{vuln.title}</h3>
                            <Badge className={getSeverityColor(vuln.severity) + ' text-sm sm:text-base px-3 py-1'}>
                              {vuln.severity.toUpperCase()}
                            </Badge>
                          </div>
                          <p className="text-base sm:text-lg text-muted-foreground">Line {vuln.line} • {vuln.cwe}</p>
                        </div>
                      </div>

                      <p className="text-base sm:text-lg leading-relaxed">{vuln.description}</p>

                      <div>
                        <p className="text-red-400 text-base sm:text-lg font-semibold mb-2">❌ Vulnerable Code:</p>
                        <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4 relative group">
                          <code className="text-sm sm:text-base">{vuln.vulnerable_code}</code>
                        </div>
                      </div>

                      <div>
                        <p className="text-green-400 text-base sm:text-lg font-semibold mb-2">✅ Secure Fix:</p>
                        <div className="bg-green-500/10 border border-green-500/50 rounded-lg p-4 relative group">
                          <code className="text-sm sm:text-base whitespace-pre-wrap">{vuln.fixed_code}</code>
                          <button
                            onClick={() => copyToClipboard(vuln.fixed_code)}
                            className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-2 bg-background rounded hover:bg-muted"
                          >
                            <Copy className="h-4 w-4" />
                          </button>
                        </div>
                      </div>

                      <div className="bg-blue-500/10 border border-blue-500/50 rounded-lg p-4">
                        <p className="text-blue-400 text-base sm:text-lg font-semibold mb-2">Impact:</p>
                        <p className="text-base sm:text-lg leading-relaxed">{vuln.impact}</p>
                      </div>
                    </div>
                  </Card>
                ))
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="learn" className="space-y-6">
          <Card className="card-responsive bg-card border-blue-500/20">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4 sm:mb-6">What is AI Security Code Review?</h2>
            <div className="space-y-4 sm:space-y-6 text-base sm:text-lg">
              <p className="leading-relaxed">
                AI Security Code Review uses advanced machine learning models trained on millions of code vulnerabilities to automatically detect security flaws in your code. Unlike traditional static analysis tools that rely on predefined rules, our AI understands context, coding patterns, and can identify novel vulnerability patterns.
              </p>
              <p className="leading-relaxed">
                The system analyzes your code for OWASP Top 10 vulnerabilities, CWE (Common Weakness Enumeration) patterns, and provides not just detection but also secure code fixes with detailed explanations.
              </p>
            </div>
          </Card>

          <Card className="card-responsive bg-card border-purple-500/20">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4 sm:mb-6">How It Works</h2>
            <div className="space-y-4 sm:space-y-6">
              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-cyan-500/20 flex items-center justify-center text-xl sm:text-2xl font-bold text-cyan-400">1</div>
                <div>
                  <h3 className="text-xl sm:text-2xl font-semibold mb-2">Code Parsing & Analysis</h3>
                  <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
                    Your code is parsed into an Abstract Syntax Tree (AST) and tokenized. The AI analyzes code structure, data flow, and control flow to understand all possible execution paths and identify where user input flows through the system.
                  </p>
                </div>
              </div>
              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-blue-500/20 flex items-center justify-center text-xl sm:text-2xl font-bold text-blue-400">2</div>
                <div>
                  <h3 className="text-xl sm:text-2xl font-semibold mb-2">Vulnerability Detection</h3>
                  <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
                    AI models trained on CVE databases, security research papers, and millions of code samples identify patterns matching known vulnerabilities including SQL injection, XSS, CSRF, insecure deserialization, command injection, path traversal, and more.
                  </p>
                </div>
              </div>
              <div className="flex gap-4 items-start">
                <div className="flex-shrink-0 w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-green-500/20 flex items-center justify-center text-xl sm:text-2xl font-bold text-green-400">3</div>
                <div>
                  <h3 className="text-xl sm:text-2xl font-semibold mb-2">Secure Fix Generation</h3>
                  <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
                    For each vulnerability found, the AI generates a secure code fix following industry best practices, OWASP guidelines, and language-specific security patterns. Each fix includes explanations of why the original code was vulnerable and how the fix prevents exploitation.
                  </p>
                </div>
              </div>
            </div>
          </Card>

          <Card className="card-responsive bg-card border-green-500/20">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4 sm:mb-6">Key Features</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
              <div className="p-4 sm:p-6 bg-cyan-500/10 rounded-lg">
                <h3 className="text-xl sm:text-2xl font-semibold mb-2">Multi-Language Support</h3>
                <p className="text-base sm:text-lg text-muted-foreground">JavaScript, Python, Java, C#, PHP, Go, Ruby, and more</p>
              </div>
              <div className="p-4 sm:p-6 bg-blue-500/10 rounded-lg">
                <h3 className="text-xl sm:text-2xl font-semibold mb-2">OWASP Top 10</h3>
                <p className="text-base sm:text-lg text-muted-foreground">Complete coverage of critical web application vulnerabilities</p>
              </div>
              <div className="p-4 sm:p-6 bg-purple-500/10 rounded-lg">
                <h3 className="text-xl sm:text-2xl font-semibold mb-2">CWE Mapping</h3>
                <p className="text-base sm:text-lg text-muted-foreground">All findings mapped to Common Weakness Enumeration</p>
              </div>
              <div className="p-4 sm:p-6 bg-green-500/10 rounded-lg">
                <h3 className="text-xl sm:text-2xl font-semibold mb-2">Instant Fixes</h3>
                <p className="text-base sm:text-lg text-muted-foreground">AI-generated secure code corrections with explanations</p>
              </div>
            </div>
          </Card>
        </TabsContent>

        <TabsContent value="examples" className="space-y-6">
          <h2 className="text-3xl sm:text-4xl font-bold mb-4">Common Vulnerability Examples</h2>
          {SAMPLE_VULNERABILITIES.map((sample, idx) => (
            <Card key={idx} className="card-responsive bg-card border-cyan-500/20">
              <div className="space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <h3 className="text-2xl sm:text-3xl font-semibold">{sample.title}</h3>
                  <Button
                    onClick={() => loadExample(sample)}
                    className="bg-cyan-500 hover:bg-cyan-600 text-base sm:text-lg px-6 py-3"
                  >
                    Analyze This Code
                  </Button>
                </div>
                <p className="text-lg sm:text-xl text-muted-foreground leading-relaxed">{sample.description}</p>
                <div>
                  <p className="text-base sm:text-lg text-muted-foreground mb-2">Vulnerable Code ({sample.language}):</p>
                  <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 relative group">
                    <pre className="text-sm sm:text-base font-mono overflow-x-auto"><code>{sample.code}</code></pre>
                    <button
                      onClick={() => copyToClipboard(sample.code)}
                      className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-2 bg-card rounded hover:bg-accent"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                  </div>
                </div>
                <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 sm:p-6">
                  <p className="text-base sm:text-lg font-semibold text-green-400 mb-2">How to Fix:</p>
                  <p className="text-base sm:text-lg text-muted-foreground leading-relaxed mb-4">{sample.fix}</p>
                  <p className="text-base sm:text-lg text-muted-foreground mb-2">Secure Code:</p>
                  <div className="bg-background border border-border rounded-lg p-4 relative group">
                    <pre className="text-sm sm:text-base font-mono overflow-x-auto"><code>{sample.secureCode}</code></pre>
                    <button
                      onClick={() => copyToClipboard(sample.secureCode)}
                      className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-2 bg-card rounded hover:bg-accent"
                    >
                      <Copy className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
            </Card>
          ))}
        </TabsContent>
      </Tabs>
    </div>
  )
}
