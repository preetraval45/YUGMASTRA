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
    sm: 'h-6 w-6',
    md: 'h-8 w-8',
    lg: 'h-12 w-12',
    xl: 'h-16 w-16',
  };

  const textSizeClasses = {
    sm: 'text-lg',
    md: 'text-xl',
    lg: 'text-2xl',
    xl: 'text-3xl',
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
            <linearGradient id="yugmastra-glow" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#60a5fa" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#a78bfa" stopOpacity="0.8" />
            </linearGradient>
          </defs>

          {variant === 'default' ? (
            <>
              {/* Outer Glow */}
              <circle
                cx="50"
                cy="50"
                r="48"
                fill="none"
                stroke="url(#yugmastra-glow)"
                strokeWidth="2"
                opacity="0.3"
              />

              {/* Shield Background */}
              <path
                d="M50 10 L75 20 L78 50 C78 70, 65 85, 50 90 C35 85, 22 70, 22 50 L25 20 Z"
                fill="url(#yugmastra-gradient)"
                opacity="0.2"
              />

              {/* Stylized Y with Crossed Swords */}
              <g transform="translate(50, 50)">
                {/* Left Sword */}
                <path
                  d="M -20 -25 L -15 -20 L -15 10 L -17 12 L -13 12 L -15 10 L -15 -20 Z"
                  fill="url(#yugmastra-gradient)"
                  stroke="currentColor"
                  strokeWidth="0.5"
                  className="text-blue-400 dark:text-blue-300"
                />

                {/* Right Sword */}
                <path
                  d="M 20 -25 L 15 -20 L 15 10 L 13 12 L 17 12 L 15 10 L 15 -20 Z"
                  fill="url(#yugmastra-gradient)"
                  stroke="currentColor"
                  strokeWidth="0.5"
                  className="text-purple-400 dark:text-purple-300"
                />

                {/* Letter Y - Modern Angular Design */}
                <path
                  d="M -8 -20 L 0 -5 L 8 -20 M 0 -5 L 0 15"
                  stroke="url(#yugmastra-gradient)"
                  strokeWidth="4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  fill="none"
                />

                {/* Central Dot/Core */}
                <circle cx="0" cy="0" r="3" fill="url(#yugmastra-gradient)" />
                <circle cx="0" cy="0" r="2" fill="#fff" opacity="0.9" />
              </g>

              {/* Circuit Pattern */}
              <g opacity="0.3">
                <circle cx="30" cy="30" r="2" fill="url(#yugmastra-gradient)" />
                <circle cx="70" cy="30" r="2" fill="url(#yugmastra-gradient)" />
                <circle cx="30" cy="70" r="2" fill="url(#yugmastra-gradient)" />
                <circle cx="70" cy="70" r="2" fill="url(#yugmastra-gradient)" />
                <path d="M 30 30 L 70 30 M 30 70 L 70 70 M 30 30 L 30 70 M 70 30 L 70 70"
                  stroke="url(#yugmastra-gradient)"
                  strokeWidth="0.5"
                  strokeDasharray="2,2"
                />
              </g>
            </>
          ) : (
            <>
              {/* Minimal Version - Just Y with Gradient */}
              <g transform="translate(50, 50)">
                <path
                  d="M -15 -25 L 0 -5 L 15 -25 M 0 -5 L 0 25"
                  stroke="url(#yugmastra-gradient)"
                  strokeWidth="6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  fill="none"
                />
                <circle cx="0" cy="0" r="4" fill="url(#yugmastra-gradient)" />
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
