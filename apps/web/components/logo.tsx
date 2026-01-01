'use client';

import React from 'react';
import { cn } from '@/lib/utils';

interface LogoProps {
  className?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  showText?: boolean;
  variant?: 'default' | 'minimal';
}

export function Logo({
  className,
  size = 'md',
  showText = true,
  variant = 'default',
}: LogoProps) {
  const sizeClasses = {
    sm: 'h-8 w-8',
    md: 'h-12 w-12',
    lg: 'h-16 w-16',
    xl: 'h-24 w-24',
  };

  const textSizeClasses = {
    sm: 'text-xl',
    md: 'text-2xl',
    lg: 'text-3xl',
    xl: 'text-4xl',
  };

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {/* Logo Icon - Stylized Y with Shield and Swords */}
      <div className={cn('relative', sizeClasses[size])}>
        <svg
          viewBox="0 0 100 100"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="w-full h-full"
        >
          {/* Gradient Definitions */}
          <defs>
            <linearGradient id="yugmastra-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#3b82f6" />
              <stop offset="50%" stopColor="#8b5cf6" />
              <stop offset="100%" stopColor="#ec4899" />
            </linearGradient>
            <linearGradient id="yugmastra-accent" x1="100%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#60a5fa" />
              <stop offset="100%" stopColor="#a78bfa" />
            </linearGradient>
            <radialGradient id="yugmastra-glow" cx="50%" cy="50%">
              <stop offset="0%" stopColor="#a78bfa" stopOpacity="0.25" />
              <stop offset="100%" stopColor="transparent" stopOpacity="0" />
            </radialGradient>
            <filter id="yugmastra-shadow">
              <feDropShadow dx="0" dy="2" stdDeviation="4" floodOpacity="0.3" />
            </filter>
            <filter id="yugmastra-glow-filter">
              <feGaussianBlur stdDeviation="2" result="blur"/>
              <feMerge>
                <feMergeNode in="blur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          {variant === 'default' ? (
            <>
              {/* Subtle Background Glow */}
              <circle cx="50" cy="50" r="40" fill="url(#yugmastra-glow)"/>

              {/* Main Shield - Clean */}
              <path
                d="M50 10 L72 22 L75 50 C75 67, 64 80, 50 88 C36 80, 25 67, 25 50 L28 22 Z"
                fill="none"
                stroke="url(#yugmastra-gradient)"
                strokeWidth="3.5"
                opacity="0.7"
                filter="url(#yugmastra-shadow)"
              />

              {/* Y Letter - Bold and Unique */}
              <g transform="translate(50, 50)">
                {/* Y Glow */}
                <path
                  d="M -13 -22 L 0 -6 L 13 -22 M 0 -6 L 0 24"
                  stroke="url(#yugmastra-accent)"
                  strokeWidth="10"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  opacity="0.25"
                />

                {/* Y Main */}
                <path
                  d="M -13 -22 L 0 -6 L 13 -22 M 0 -6 L 0 24"
                  stroke="url(#yugmastra-gradient)"
                  strokeWidth="6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  filter="url(#yugmastra-shadow)"
                />

                {/* Inner Highlights */}
                <path
                  d="M -11 -19 L -2 -8"
                  stroke="#ffffff"
                  strokeWidth="2"
                  strokeLinecap="round"
                  opacity="0.5"
                />
                <path
                  d="M 11 -19 L 2 -8"
                  stroke="#ffffff"
                  strokeWidth="2"
                  strokeLinecap="round"
                  opacity="0.5"
                />

                {/* Central Core */}
                <circle
                  cx="0"
                  cy="-6"
                  r="4.5"
                  fill="url(#yugmastra-gradient)"
                  filter="url(#yugmastra-glow-filter)"
                />
                <circle cx="0" cy="-6" r="3" fill="#ffffff" opacity="0.95"/>
                <circle cx="0" cy="-6" r="1.5" fill="url(#yugmastra-accent)" opacity="0.7"/>
              </g>

              {/* Corner Brackets - Minimal */}
              <g opacity="0.55" stroke="url(#yugmastra-gradient)" strokeWidth="2.5" strokeLinecap="round">
                <path d="M 27 27 L 27 30 M 27 27 L 30 27" />
                <path d="M 73 27 L 73 30 M 73 27 L 70 27" />
                <path d="M 27 73 L 27 70 M 27 73 L 30 73" />
                <path d="M 73 73 L 73 70 M 73 73 L 70 73" />
              </g>
            </>
          ) : (
            <>
              {/* Minimal Version - Clean Y */}
              <g transform="translate(50, 50)">
                <path
                  d="M -13 -22 L 0 -6 L 13 -22 M 0 -6 L 0 24"
                  stroke="url(#yugmastra-gradient)"
                  strokeWidth="6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                <circle cx="0" cy="-6" r="3.5" fill="url(#yugmastra-gradient)" />
                <circle cx="0" cy="-6" r="2" fill="#ffffff" opacity="0.9" />
              </g>
            </>
          )}
        </svg>
      </div>

      {/* Logo Text */}
      {showText && (
        <div className="flex flex-col leading-none">
          <span
            className={cn(
              'font-bold tracking-tight bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 bg-clip-text text-transparent',
              textSizeClasses[size]
            )}
          >
            YUGMĀSTRA
          </span>
          <span className="text-[0.6em] text-muted-foreground tracking-wider">
            AUTONOMOUS CYBER DEFENSE
          </span>
        </div>
      )}
    </div>
  );
}

// Compact version for small spaces
export function LogoIcon({ className, size = 'md' }: Pick<LogoProps, 'className' | 'size'>) {
  return <Logo className={className} size={size} showText={false} />;
}

// Animated version for loading/splash
export function LogoAnimated({ className }: { className?: string }) {
  return (
    <div className={cn('relative', className)}>
      <Logo size="xl" variant="default" />
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="h-full w-full animate-spin-slow">
          <svg viewBox="0 0 100 100" className="h-full w-full">
            <circle
              cx="50"
              cy="50"
              r="45"
              fill="none"
              stroke="url(#yugmastra-gradient)"
              strokeWidth="1"
              strokeDasharray="10 5"
              opacity="0.3"
            />
          </svg>
        </div>
      </div>
    </div>
  );
}
