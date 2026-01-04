'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import {
  Shield,
  Swords,
  Brain,
  Target,
  Zap,
  Database,
  Search,
  FileCode,
  AlertTriangle,
  Bug,
  Activity,
  Network,
  ChevronRight,
  ChevronLeft,
  X,
  Sparkles
} from 'lucide-react';

interface TourStep {
  title: string;
  description: string;
  icon: any;
  action?: {
    label: string;
    href: string;
  };
  tips?: string[];
}

const tourSteps: TourStep[] = [
  {
    title: 'Welcome to YUGMĀSTRA',
    description: 'An autonomous adversary-defender co-evolution platform where AI agents battle to improve cybersecurity. Red Team AI discovers attacks while Blue Team AI learns to defend—both evolving together through self-play reinforcement learning.',
    icon: Shield,
    tips: [
      'This platform runs 24/7 AI battles to discover new threats',
      'Both offensive and defensive AI improve automatically',
      'Real-world security insights from every battle'
    ]
  },
  {
    title: 'Command Center - Dashboard',
    description: 'Your central hub for monitoring AI training progress. Track total episodes, win rates, and system status. Click any metric card to learn more about what it represents.',
    icon: Target,
    action: {
      label: 'Go to Dashboard',
      href: '/dashboard'
    },
    tips: [
      'Dashboard updates in real-time as battles occur',
      'Click metrics for detailed educational explanations',
      'Quick actions provide instant access to key tools'
    ]
  },
  {
    title: 'Live Battle Arena',
    description: 'Watch Red Team and Blue Team AI compete in real-time. Observe attack strategies, defense responses, and see both agents adapt their tactics dynamically.',
    icon: Swords,
    action: {
      label: 'Watch Live Battles',
      href: '/live-battle'
    },
    tips: [
      'See actual attack vectors being discovered',
      'Watch defense mechanisms activate in real-time',
      'Learn from AI decision-making processes'
    ]
  },
  {
    title: 'AI Security Tools',
    description: 'Powerful AI-driven tools for cybersecurity professionals. Each tool includes educational content to help you understand the techniques and frameworks.',
    icon: Brain,
    tips: [
      'Threat Hunting - MITRE ATT&CK based detection',
      'Code Review - OWASP vulnerability scanning',
      'Incident Response - NIST framework automation',
      'Zero-Day Discovery - CVE/CVSS analysis',
      'SIEM Rules - Generate detection rules for Sigma, Splunk, Elastic'
    ]
  },
  {
    title: 'Threat Hunting',
    description: 'Hunt for advanced persistent threats using MITRE ATT&CK framework scenarios. Learn about tactics, techniques, and procedures (TTPs) while the AI helps detect anomalies.',
    icon: Search,
    action: {
      label: 'Start Hunting',
      href: '/threat-hunting'
    },
    tips: [
      'Pre-loaded with real-world APT scenarios',
      'MITRE ATT&CK mapped tactics and techniques',
      'Educational content explains each threat pattern'
    ]
  },
  {
    title: 'Code Review & Vulnerability Scanning',
    description: 'AI-powered code analysis for OWASP Top 10 vulnerabilities. Upload code or use examples to learn about SQL injection, XSS, authentication flaws, and more.',
    icon: FileCode,
    action: {
      label: 'Review Code',
      href: '/code-review'
    },
    tips: [
      'Detects OWASP Top 10 vulnerabilities',
      'Detailed examples for each vulnerability type',
      'Learn secure coding best practices'
    ]
  },
  {
    title: 'Incident Response Playbooks',
    description: 'Automated incident handling using NIST framework. Access playbooks for ransomware, data breaches, APTs, and DDoS attacks with step-by-step guidance.',
    icon: AlertTriangle,
    action: {
      label: 'View Playbooks',
      href: '/incident-response'
    },
    tips: [
      'Complete NIST IR lifecycle coverage',
      'Pre-built playbooks for common incidents',
      'AI-enhanced response automation'
    ]
  },
  {
    title: 'Zero-Day Discovery',
    description: 'Discover unknown vulnerabilities using AI analysis. Learn about the CVE system, CVSS scoring, and study historical zero-day exploits.',
    icon: Bug,
    action: {
      label: 'Discover Vulnerabilities',
      href: '/zero-day'
    },
    tips: [
      'Understand CVE and CVSS scoring systems',
      'Study historical zero-days like Heartbleed, Log4Shell',
      'EPSS prediction for exploit probability'
    ]
  },
  {
    title: 'SIEM Rules Generation',
    description: 'Generate detection rules for multiple SIEM platforms. Learn detection engineering for Sigma, Splunk SPL, Elasticsearch, Suricata, and Snort.',
    icon: Shield,
    action: {
      label: 'Generate Rules',
      href: '/siem-rules'
    },
    tips: [
      'Supports 5 major rule formats',
      'Detailed rule examples with explanations',
      'Best practices for detection quality'
    ]
  },
  {
    title: 'Evolution Timeline',
    description: 'Track AI training progress over time. See how Red Team and Blue Team capabilities evolve, with metrics showing improvement in attack sophistication and defense effectiveness.',
    icon: Activity,
    action: {
      label: 'View Evolution',
      href: '/evolution'
    },
    tips: [
      'Visualize AI learning curves',
      'Compare Red vs Blue progression',
      'Identify performance breakthroughs'
    ]
  },
  {
    title: 'Knowledge Graph',
    description: 'Visualize attack patterns and defense strategies as an interactive graph. Explore relationships between MITRE ATT&CK techniques, vulnerabilities, and defensive measures.',
    icon: Network,
    action: {
      label: 'Explore Graph',
      href: '/knowledge-graph'
    },
    tips: [
      'Interactive MITRE ATT&CK visualization',
      'Discover technique relationships',
      'Map attacks to defenses'
    ]
  },
  {
    title: 'Quick Navigation',
    description: 'Use Cmd/Ctrl+K to open the command palette for instant navigation. On mobile, use the bottom navigation bar to quickly access key features.',
    icon: Zap,
    tips: [
      'Press Cmd/Ctrl+K for global search',
      'Mobile bottom nav for quick access',
      'Click help icons throughout for educational content',
      'All pages include Learn tabs with detailed explanations'
    ]
  },
  {
    title: 'You\'re All Set!',
    description: 'You\'re ready to explore YUGMĀSTRA. Remember: every page has educational content accessible via help icons and Learn tabs. Start with the Dashboard to see live AI training in action.',
    icon: Sparkles,
    action: {
      label: 'Start Exploring',
      href: '/dashboard'
    },
    tips: [
      'Click any ? icon for help',
      'Use Learn tabs on each tool page',
      'Watch live battles to see AI in action',
      'All features are ready to use immediately'
    ]
  }
];

export function OnboardingTour() {
  const [open, setOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const router = useRouter();

  useEffect(() => {
    // Check if user has completed onboarding
    const hasCompletedOnboarding = localStorage.getItem('yugmastra_onboarding_completed');

    if (!hasCompletedOnboarding) {
      // Show onboarding after a short delay
      const timer = setTimeout(() => {
        setOpen(true);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, []);

  const handleNext = () => {
    if (currentStep < tourSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSkip = () => {
    localStorage.setItem('yugmastra_onboarding_completed', 'true');
    setOpen(false);
  };

  const handleComplete = () => {
    localStorage.setItem('yugmastra_onboarding_completed', 'true');
    setOpen(false);

    // Navigate to dashboard if there's an action
    const step = tourSteps[currentStep];
    if (step.action) {
      router.push(step.action.href);
    } else {
      router.push('/dashboard');
    }
  };

  const handleActionClick = () => {
    const step = tourSteps[currentStep];
    if (step.action) {
      localStorage.setItem('yugmastra_onboarding_completed', 'true');
      router.push(step.action.href);
      setOpen(false);
    }
  };

  const step = tourSteps[currentStep];
  const progress = ((currentStep + 1) / tourSteps.length) * 100;
  const StepIcon = step.icon;

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <button
          onClick={handleSkip}
          className="absolute top-4 right-4 p-2 rounded-full hover:bg-accent transition-colors"
          aria-label="Skip tour"
        >
          <X className="h-5 w-5" />
        </button>

        <DialogHeader className="space-y-4">
          <div className="flex items-center gap-4">
            <div className="flex-shrink-0 w-16 h-16 rounded-2xl bg-primary/15 dark:bg-primary/10 flex items-center justify-center">
              <StepIcon className="h-8 w-8 text-primary" />
            </div>
            <div className="flex-1">
              <DialogTitle className="text-2xl mb-1">{step.title}</DialogTitle>
              <DialogDescription className="text-base">
                Step {currentStep + 1} of {tourSteps.length}
              </DialogDescription>
            </div>
          </div>

          <Progress value={progress} className="h-2" />
        </DialogHeader>

        <div className="space-y-6 pt-4">
          <p className="text-lg text-foreground leading-relaxed">
            {step.description}
          </p>

          {step.tips && step.tips.length > 0 && (
            <div className="bg-blue-500/10 dark:bg-blue-500/5 border border-blue-500/20 rounded-xl p-4">
              <h4 className="text-base font-semibold text-blue-600 dark:text-blue-400 mb-3 flex items-center gap-2">
                <Sparkles className="h-4 w-4" />
                Key Points
              </h4>
              <ul className="space-y-2">
                {step.tips.map((tip, index) => (
                  <li key={index} className="text-sm text-muted-foreground flex items-start gap-2">
                    <span className="text-blue-500 mt-1">•</span>
                    <span>{tip}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {step.action && (
            <Button
              onClick={handleActionClick}
              className="w-full touch-target-lg text-base font-semibold"
              size="lg"
            >
              {step.action.label}
              <ChevronRight className="h-5 w-5 ml-2" />
            </Button>
          )}
        </div>

        <div className="flex items-center justify-between gap-3 pt-6 border-t">
          <Button
            onClick={handlePrevious}
            disabled={currentStep === 0}
            variant="outline"
            className="touch-target-lg"
            size="lg"
          >
            <ChevronLeft className="h-5 w-5 mr-2" />
            Previous
          </Button>

          <button
            onClick={handleSkip}
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Skip tour
          </button>

          <Button
            onClick={handleNext}
            className="touch-target-lg"
            size="lg"
          >
            {currentStep === tourSteps.length - 1 ? 'Get Started' : 'Next'}
            <ChevronRight className="h-5 w-5 ml-2" />
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
