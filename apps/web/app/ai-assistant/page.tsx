'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Sparkles, Zap, Shield, Swords, Copy, RefreshCw, Brain } from 'lucide-react';
import { cn } from '@/lib/utils';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  mode?: 'red-team' | 'blue-team' | 'evolution';
}

const AI_MODES = [
  { id: 'red-team', label: 'Red Team AI', icon: Swords, color: 'text-red-500', description: 'Offensive security analysis' },
  { id: 'blue-team', label: 'Blue Team AI', icon: Shield, color: 'text-blue-500', description: 'Defensive security analysis' },
  { id: 'evolution', label: 'Evolution AI', icon: Zap, color: 'text-purple-500', description: 'Adaptive threat intelligence' },
] as const;

const SAMPLE_PROMPTS = {
  'red-team': [
    "Analyze potential attack vectors for a web application using React and Node.js",
    "Generate a penetration testing strategy for a cloud infrastructure on AWS",
    "Identify OWASP Top 10 vulnerabilities in a typical e-commerce platform",
    "Create an attack scenario simulating APT28 tactics against a corporate network",
    "Develop a social engineering campaign for security awareness training",
    "Analyze lateral movement techniques in a Windows Active Directory environment"
  ],
  'blue-team': [
    "Design a layered defense strategy for protecting critical infrastructure",
    "Create SIEM rules to detect ransomware behavior patterns",
    "Develop an incident response playbook for a data breach scenario",
    "Recommend security controls for a zero-trust architecture implementation",
    "Analyze this network traffic log for potential threats: [paste logs]",
    "Generate detection rules for detecting Cobalt Strike beacons"
  ],
  'evolution': [
    "Predict emerging attack trends for 2026 based on current threat landscape",
    "Analyze the co-evolution of ransomware tactics and defensive countermeasures",
    "Identify gaps in our security posture that attackers might exploit",
    "Recommend adaptive security controls that evolve with threat patterns",
    "Analyze the effectiveness of our red-blue team training cycles",
    "Generate threat intelligence on nation-state APT groups targeting our industry"
  ]
};

export default function AIAssistantPage() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content: 'Welcome to YUGMĀSTRA AI Assistant. I am your intelligent companion for cybersecurity analysis.\n\n🔴 Red Team AI: Offensive security, penetration testing, attack simulation\n🔵 Blue Team AI: Defensive strategies, threat detection, incident response\n⚡ Evolution AI: Adaptive intelligence, threat prediction, co-evolution analysis\n\nChoose an AI mode below and ask me anything, or use the sample prompts to get started!',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedMode, setSelectedMode] = useState<'red-team' | 'blue-team' | 'evolution'>('evolution');
  const [showSamples, setShowSamples] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setShowSamples(false);

    try {
      const response = await fetch('/api/ai-assistant', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          mode: selectedMode,
          history: messages.slice(-5),
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get AI response');
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.response,
        timestamp: new Date(),
        mode: selectedMode,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('AI Assistant error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handlePromptClick = (prompt: string) => {
    setInput(prompt);
    setShowSamples(false);
  };

  const handleReset = () => {
    setMessages([{
      id: '1',
      role: 'assistant',
      content: 'Conversation reset. How can I help you with cybersecurity analysis?',
      timestamp: new Date(),
    }]);
    setShowSamples(true);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="flex h-screen flex-col bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="px-8">
          <div className="flex items-center justify-between p-6">
            <div className="flex items-center gap-4">
              <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-purple-500/20">
                <Sparkles className="h-7 w-7 text-white" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">AI Security Assistant</h1>
                <p className="text-sm text-gray-400 mt-1">Multi-agent AI powered by YUGMĀSTRA Intelligence Engine</p>
              </div>
            </div>
            <button
              onClick={handleReset}
              className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg border border-gray-700 transition-all"
            >
              <RefreshCw className="h-4 w-4" />
              Reset
            </button>
          </div>

          {/* AI Mode Selection */}
          <div className="px-6 pb-6">
            <div className="grid grid-cols-3 gap-4">
              {AI_MODES.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setSelectedMode(mode.id)}
                  className={cn(
                    'flex items-start gap-3 px-5 py-4 rounded-xl border transition-all',
                    selectedMode === mode.id
                      ? 'bg-gradient-to-br from-blue-600/20 to-purple-600/20 border-blue-500/50 shadow-lg shadow-blue-500/10'
                      : 'bg-[#0a0e1a] hover:bg-[#12172a] border-gray-800 hover:border-gray-700'
                  )}
                >
                  <mode.icon className={cn(
                    'h-6 w-6 mt-0.5',
                    selectedMode === mode.id ? mode.color : 'text-gray-500'
                  )} />
                  <div className="text-left flex-1">
                    <div className={cn(
                      'font-semibold text-base mb-1',
                      selectedMode === mode.id ? 'text-white' : 'text-gray-300'
                    )}>
                      {mode.label}
                    </div>
                    <div className="text-xs text-gray-500">
                      {mode.description}
                    </div>
                  </div>
                </button>
              ))}
            </div>

            {/* Description Banner */}
            <div className="px-6 pb-6">
              <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 flex items-start gap-3">
                <Brain className="w-5 h-5 text-blue-400 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    <strong className="text-foreground">What this page does:</strong> This AI Security Assistant is powered by multiple specialized agents (Red Team, Blue Team, Evolution) that provide expert cybersecurity guidance. Ask questions about vulnerabilities, get attack strategy suggestions, receive defense recommendations, or learn about emerging threats. The Red Team mode helps analyze attack vectors for penetration testing, Blue Team mode provides defense strategies and SIEM rules, and Evolution mode predicts future attack trends. Includes 18 sample prompts covering topics like React/Node.js security, cloud infrastructure pentesting, ransomware defense, zero-day detection, ML-powered threat hunting, and supply chain security. Perfect for security professionals, pentesters, SOC analysts, and anyone learning offensive/defensive security.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="px-8 py-6 space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                'flex gap-4',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              {message.role === 'assistant' && (
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center flex-shrink-0 shadow-lg shadow-purple-500/20">
                  <Bot className="h-6 w-6 text-white" />
                </div>
              )}
              <div
                className={cn(
                  'max-w-[75%] rounded-xl p-5 border',
                  message.role === 'user'
                    ? 'bg-gradient-to-br from-blue-600 to-purple-600 text-white border-blue-500/30 shadow-lg shadow-blue-500/10'
                    : 'bg-[#0f1419] border-gray-800'
                )}
              >
                {message.mode && message.role === 'assistant' && (
                  <div className="flex items-center gap-2 mb-3 pb-3 border-b border-gray-800">
                    {AI_MODES.find(m => m.id === message.mode)?.icon && (
                      <span className={AI_MODES.find(m => m.id === message.mode)?.color}>
                        {(() => {
                          const Icon = AI_MODES.find(m => m.id === message.mode)!.icon;
                          return <Icon className="h-4 w-4" />;
                        })()}
                      </span>
                    )}
                    <span className="text-sm font-medium text-gray-400">
                      {AI_MODES.find(m => m.id === message.mode)?.label}
                    </span>
                  </div>
                )}
                <p className={cn(
                  'whitespace-pre-wrap leading-relaxed',
                  message.role === 'user' ? 'text-white' : 'text-gray-300'
                )}>
                  {message.content}
                </p>
                <div className="flex items-center justify-between mt-4 pt-3 border-t border-gray-800/50">
                  <p className={cn(
                    'text-xs',
                    message.role === 'user' ? 'text-blue-200' : 'text-gray-500'
                  )}>
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                  {message.role === 'assistant' && (
                    <button
                      onClick={() => copyToClipboard(message.content)}
                      className="text-gray-500 hover:text-gray-300 transition-colors"
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>
              </div>
              {message.role === 'user' && (
                <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center flex-shrink-0 shadow-lg shadow-green-500/20">
                  <User className="h-6 w-6 text-white" />
                </div>
              )}
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-4 justify-start">
              <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 flex items-center justify-center flex-shrink-0 shadow-lg shadow-purple-500/20">
                <Bot className="h-6 w-6 text-white" />
              </div>
              <div className="bg-[#0f1419] border border-gray-800 rounded-xl p-5">
                <Loader2 className="h-5 w-5 animate-spin text-purple-500" />
              </div>
            </div>
          )}

          {/* Sample Prompts */}
          {showSamples && messages.length === 1 && (
            <div className="mt-8 space-y-4">
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-purple-500" />
                Try these sample prompts:
              </h3>
              <div className="grid gap-3">
                {SAMPLE_PROMPTS[selectedMode].map((prompt) => (
                  <button
                    key={prompt}
                    onClick={() => handlePromptClick(prompt)}
                    className="text-left p-4 bg-[#0f1419] hover:bg-[#12172a] border border-gray-800 hover:border-gray-700 rounded-xl transition-all group"
                  >
                    <p className="text-sm text-gray-300 group-hover:text-white transition-colors">
                      {prompt}
                    </p>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-border bg-card">
        <div className="px-8 py-6">
          <div className="flex gap-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Ask ${AI_MODES.find(m => m.id === selectedMode)?.label} anything about cybersecurity...`}
              className="flex-1 px-5 py-4 bg-[#0a0e1a] border border-gray-800 rounded-xl focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-white placeholder-gray-500 transition-all"
              disabled={isLoading}
            />
            <button
              onClick={handleSendMessage}
              disabled={isLoading || !input.trim()}
              className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-lg shadow-purple-500/20 hover:shadow-purple-500/30"
            >
              {isLoading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Send className="h-5 w-5" />
              )}
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-3 text-center">
            AI responses are powered by advanced LLM models and may not always be 100% accurate. Use for research and training purposes.
          </p>
        </div>
      </div>
    </div>
  );
}
