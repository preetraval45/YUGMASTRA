import { NextRequest, NextResponse } from 'next/server';
import { getSession } from '@/lib/auth';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface AIRequest {
  message: string;
  mode: 'red-team' | 'blue-team' | 'evolution';
  history?: Message[];
  conversationId?: string;
}

// AI Engine URL - from environment variable
const AI_ENGINE_URL = process.env.AI_ENGINE_URL || 'http://localhost:8001';

// Map frontend modes to backend agent types
const modeToAgent: Record<string, string> = {
  'red-team': 'red_team',
  'blue-team': 'blue_team',
  'evolution': 'evolution',
};

// Call RAG + AI Engine with enhanced capabilities
const callAIEngine = async (request: AIRequest): Promise<string> => {
  const { message, mode, history } = request;
  const agentType = modeToAgent[mode] || 'red_team';

  try {
    // Step 1: Query RAG system for relevant context (Vercel serverless)
    const baseUrl = process.env.NEXT_PUBLIC_APP_URL || (typeof window !== 'undefined' ? window.location.origin : 'http://localhost:3000');
    const ragResponse = await fetch(`${baseUrl}/api/rag/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: message,
        mode: agentType,
        k: 3,
        alpha: 0.5
      }),
      signal: AbortSignal.timeout(10000),
    });

    let ragContext = '';
    if (ragResponse.ok) {
      const ragData = await ragResponse.json();
      ragContext = ragData.context || '';
    }

    // Step 2: Try orchestrator endpoint for complex queries with RAG context
    const isComplexQuery = message.length > 200 ||
                          message.includes('analyze') ||
                          message.includes('recommend') ||
                          message.includes('detect');

    const endpoint = isComplexQuery
      ? `${AI_ENGINE_URL}/api/orchestrator/analyze`
      : `${AI_ENGINE_URL}/api/${agentType}/analyze`;

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: message,
        query: message,
        context: {
          history: history || [],
          mode: agentType,
          rag_context: ragContext
        }
      }),
      signal: AbortSignal.timeout(30000),
    });

    if (!response.ok) {
      console.error(`AI Engine responded with status: ${response.status}`);
      throw new Error(`AI Engine error: ${response.status}`);
    }

    const data = await response.json();
    return data.analysis || data.response || data.result || 'Analysis completed.';

  } catch (error) {
    console.error('AI Engine connection error:', error);
    throw error;
  }
};

// Fallback responses if AI Engine is unavailable
const getFallbackResponse = (message: string, mode: string): string => {
  const lowerMessage = message.toLowerCase();

  if (lowerMessage.includes('attack') || lowerMessage.includes('exploit')) {
    if (mode === 'red-team') {
      return `Based on my analysis, I've identified several potential attack vectors:\n\n1. **Common Vulnerabilities**: SQL injection, XSS, CSRF, and authentication bypass are frequent entry points.\n\n2. **Advanced Techniques**: Consider supply chain attacks, zero-day exploits, and social engineering vectors.\n\n3. **Reconnaissance**: OSINT gathering, network scanning, and vulnerability assessment are critical first steps.\n\nWould you like me to elaborate on any specific attack technique or provide mitigation strategies?`;
    } else if (mode === 'blue-team') {
      return `To defend against attacks, I recommend:\n\n1. **Detection**: Implement SIEM, IDS/IPS, and behavioral analytics to identify anomalies.\n\n2. **Prevention**: Use WAF, rate limiting, input validation, and security headers.\n\n3. **Response**: Have an incident response plan, maintain logs, and implement automated threat response.\n\n4. **Hardening**: Regular patching, principle of least privilege, and network segmentation.\n\nWhat specific defense mechanism would you like to explore?`;
    }
  }

  if (lowerMessage.includes('defend') || lowerMessage.includes('protect')) {
    return `Defense strategies for modern cyber threats:\n\n**Detection Layer**:\n- Real-time monitoring with SIEM\n- Behavioral analytics and ML-based anomaly detection\n- Threat intelligence integration\n\n**Prevention Layer**:\n- Zero-trust architecture\n- Multi-factor authentication\n- Application security controls\n\n**Response Layer**:\n- Automated incident response\n- Forensics and root cause analysis\n- Threat hunting capabilities\n\nWhich layer would you like to strengthen first?`;
  }

  if (lowerMessage.includes('vulnerability') || lowerMessage.includes('cve')) {
    return `Vulnerability management best practices:\n\n1. **Discovery**: Continuous vulnerability scanning and asset inventory\n2. **Prioritization**: Risk-based vulnerability scoring (CVSS, EPSS)\n3. **Remediation**: Patch management and compensating controls\n4. **Validation**: Penetration testing and security assessments\n\nRecent critical vulnerabilities to monitor:\n- Log4Shell (CVE-2021-44228)\n- ProxyShell/ProxyLogon (Exchange vulnerabilities)\n- Spring4Shell (CVE-2022-22965)\n\nWould you like specific remediation guidance?`;
  }

  if (lowerMessage.includes('threat') || lowerMessage.includes('intelligence')) {
    if (mode === 'evolution') {
      return `Current threat landscape analysis:\n\n**Emerging Threats**:\n- AI-powered phishing and deepfakes\n- Supply chain attacks (SolarWinds, Log4j)\n- Ransomware-as-a-Service (RaaS)\n- Cloud-native threats\n\n**Evolving Tactics**:\n- Living-off-the-land techniques\n- Fileless malware\n- API abuse and business logic flaws\n\n**Threat Actors**:\n- Nation-state APT groups\n- Cybercriminal syndicates\n- Hacktivists and insider threats\n\nWhat specific threat would you like to investigate?`;
    }
  }

  // Default intelligent response based on mode
  const defaultResponses: Record<string, string> = {
    'red-team': `From a Red Team perspective: I'm analyzing your query for potential security implications. Could you provide more specific details about:\n\n- The target system or environment?\n- The attack surface you're interested in?\n- Whether this is for authorized penetration testing?\n\nThis will help me provide more targeted offensive security insights.`,
    'blue-team': `From a Blue Team perspective: I'm ready to help strengthen your defenses. To provide the most relevant guidance:\n\n- What assets are you protecting?\n- What threats are you most concerned about?\n- What security controls do you currently have in place?\n\nThis context will help me recommend appropriate defensive measures.`,
    'evolution': `From an adaptive intelligence perspective: I'm continuously learning from both offensive and defensive strategies. To provide strategic insights:\n\n- What's your current security maturity level?\n- Are you focusing on prevention, detection, or response?\n- What industry or threat landscape are you operating in?\n\nThis will help me provide evolving threat intelligence tailored to your needs.`,
  };

  return defaultResponses[mode] || defaultResponses['evolution'];
};

export async function POST(request: NextRequest) {
  try {
    // Verify authentication
    const user = await getSession();
    if (!user) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body: AIRequest = await request.json();
    const { message, mode, history } = body;

    if (!message || !mode) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    let response: string;

    try {
      // Try to call the AI Engine
      response = await callAIEngine({ message, mode, history });
    } catch (error) {
      // Fallback to simulated responses if AI Engine is unavailable
      console.log('Using fallback AI responses');
      response = getFallbackResponse(message, mode);
    }

    return NextResponse.json({
      response,
      mode,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    console.error('AI Assistant API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
