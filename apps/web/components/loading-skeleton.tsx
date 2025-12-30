"use client";

import React from 'react';
import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular' | 'rounded';
  animation?: 'pulse' | 'wave' | 'none';
  width?: string | number;
  height?: string | number;
}

export function Skeleton({
  className,
  variant = 'rectangular',
  animation = 'pulse',
  width,
  height,
}: SkeletonProps) {
  const baseClasses = 'bg-gradient-to-r from-gray-800 via-gray-700 to-gray-800 bg-[length:200%_100%]';

  const variantClasses = {
    text: 'h-4 rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-none',
    rounded: 'rounded-lg',
  };

  const animationClasses = {
    pulse: 'animate-pulse',
    wave: 'animate-shimmer',
    none: '',
  };

  const style: React.CSSProperties = {};
  if (width) style.width = typeof width === 'number' ? `${width}px` : width;
  if (height) style.height = typeof height === 'number' ? `${height}px` : height;

  return (
    <div
      className={cn(
        baseClasses,
        variantClasses[variant],
        animationClasses[animation],
        className
      )}
      style={style}
    />
  );
}

export function CardSkeleton() {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <Skeleton className="h-6 w-32" />
        <Skeleton variant="circular" className="w-8 h-8" />
      </div>
      <Skeleton className="h-10 w-24 mb-2" />
      <Skeleton className="h-4 w-full" />
    </div>
  );
}

export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-4 gap-4 p-4 bg-gray-800 rounded-t-lg">
        <Skeleton className="h-4" />
        <Skeleton className="h-4" />
        <Skeleton className="h-4" />
        <Skeleton className="h-4" />
      </div>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="grid grid-cols-4 gap-4 p-4 bg-gray-900/50">
          <Skeleton className="h-4" />
          <Skeleton className="h-4" />
          <Skeleton className="h-4" />
          <Skeleton className="h-4" />
        </div>
      ))}
    </div>
  );
}

export function GraphSkeleton() {
  return (
    <div className="relative w-full h-[600px] bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative">
          {/* Center node */}
          <Skeleton variant="circular" className="w-24 h-24" />

          {/* Orbiting nodes */}
          {Array.from({ length: 6 }).map((_, i) => {
            const angle = (i * 60 * Math.PI) / 180;
            const radius = 120;
            const x = Math.cos(angle) * radius;
            const y = Math.sin(angle) * radius;
            return (
              <Skeleton
                key={i}
                variant="circular"
                className="w-16 h-16 absolute"
                style={{
                  left: `calc(50% + ${x}px)`,
                  top: `calc(50% + ${y}px)`,
                  transform: 'translate(-50%, -50%)',
                }}
              />
            );
          })}
        </div>
      </div>

      <div className="absolute inset-0 flex items-center justify-center bg-black/50">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-gray-400">Loading Knowledge Graph...</p>
        </div>
      </div>
    </div>
  );
}

export function ChatSkeleton() {
  return (
    <div className="space-y-4">
      <div className="flex justify-end">
        <div className="bg-blue-600/20 rounded-lg p-4 max-w-[80%]">
          <Skeleton className="h-4 w-64 mb-2" />
          <Skeleton className="h-4 w-48" />
        </div>
      </div>

      <div className="flex justify-start">
        <div className="bg-gray-800 rounded-lg p-4 max-w-[80%]">
          <div className="flex items-center gap-2 mb-3">
            <Skeleton variant="circular" className="w-6 h-6" />
            <Skeleton className="h-4 w-20" />
          </div>
          <Skeleton className="h-4 w-full mb-2" />
          <Skeleton className="h-4 w-full mb-2" />
          <Skeleton className="h-4 w-3/4" />
        </div>
      </div>
    </div>
  );
}

export function PageSkeleton() {
  return (
    <div className="min-h-screen bg-black p-6">
      <div className="mb-8">
        <Skeleton className="h-10 w-64 mb-2" />
        <Skeleton className="h-6 w-96" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <CardSkeleton />
        <CardSkeleton />
        <CardSkeleton />
        <CardSkeleton />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <Skeleton className="h-6 w-48 mb-4" />
          <Skeleton className="h-64 w-full" />
        </div>
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <Skeleton className="h-6 w-48 mb-4" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    </div>
  );
}

export function ListSkeleton({ items = 5 }: { items?: number }) {
  return (
    <div className="space-y-3">
      {Array.from({ length: items }).map((_, i) => (
        <div key={i} className="flex items-center gap-4 p-4 bg-gray-800 rounded-lg">
          <Skeleton variant="circular" className="w-12 h-12" />
          <div className="flex-1">
            <Skeleton className="h-5 w-3/4 mb-2" />
            <Skeleton className="h-4 w-1/2" />
          </div>
          <Skeleton className="h-8 w-20" />
        </div>
      ))}
    </div>
  );
}
