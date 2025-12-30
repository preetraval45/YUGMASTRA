"use client";

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from 'lucide-react';
import { cn } from '@/lib/utils';

type ToastType = 'success' | 'error' | 'info' | 'warning';

interface Toast {
  id: string;
  type: ToastType;
  title: string;
  description?: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface ToastContextValue {
  toasts: Toast[];
  showToast: (toast: Omit<Toast, 'id'>) => void;
  hideToast: (id: string) => void;
  success: (title: string, description?: string) => void;
  error: (title: string, description?: string) => void;
  info: (title: string, description?: string) => void;
  warning: (title: string, description?: string) => void;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

export function useToast() {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within ToastProvider');
  }
  return context;
}

interface ToastProviderProps {
  children: ReactNode;
  maxToasts?: number;
}

export function ToastProvider({ children, maxToasts = 5 }: ToastProviderProps) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = useCallback((toast: Omit<Toast, 'id'>) => {
    const id = Math.random().toString(36).substring(7);
    const newToast: Toast = {
      ...toast,
      id,
      duration: toast.duration ?? 5000,
    };

    setToasts((prev) => {
      const updated = [...prev, newToast];
      // Limit number of toasts
      return updated.slice(-maxToasts);
    });

    // Auto dismiss
    if (newToast.duration && newToast.duration > 0) {
      setTimeout(() => {
        hideToast(id);
      }, newToast.duration);
    }
  }, [maxToasts]);

  const hideToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const success = useCallback((title: string, description?: string) => {
    showToast({ type: 'success', title, description });
  }, [showToast]);

  const error = useCallback((title: string, description?: string) => {
    showToast({ type: 'error', title, description, duration: 7000 });
  }, [showToast]);

  const info = useCallback((title: string, description?: string) => {
    showToast({ type: 'info', title, description });
  }, [showToast]);

  const warning = useCallback((title: string, description?: string) => {
    showToast({ type: 'warning', title, description });
  }, [showToast]);

  return (
    <ToastContext.Provider
      value={{ toasts, showToast, hideToast, success, error, info, warning }}
    >
      {children}
      <ToastContainer toasts={toasts} onClose={hideToast} />
    </ToastContext.Provider>
  );
}

function ToastContainer({
  toasts,
  onClose,
}: {
  toasts: Toast[];
  onClose: (id: string) => void;
}) {
  return (
    <div className="fixed top-4 right-4 z-[9999] flex flex-col gap-2 max-w-md">
      {toasts.map((toast) => (
        <ToastItem key={toast.id} toast={toast} onClose={onClose} />
      ))}
    </div>
  );
}

function ToastItem({ toast, onClose }: { toast: Toast; onClose: (id: string) => void }) {
  const icons = {
    success: CheckCircle,
    error: AlertCircle,
    info: Info,
    warning: AlertTriangle,
  };

  const colors = {
    success: 'bg-green-500/20 border-green-500/50 text-green-400',
    error: 'bg-red-500/20 border-red-500/50 text-red-400',
    info: 'bg-blue-500/20 border-blue-500/50 text-blue-400',
    warning: 'bg-yellow-500/20 border-yellow-500/50 text-yellow-400',
  };

  const Icon = icons[toast.type];

  return (
    <div
      className={cn(
        'flex items-start gap-3 p-4 rounded-lg border backdrop-blur-sm shadow-lg animate-in slide-in-from-right',
        colors[toast.type]
      )}
    >
      <Icon className="w-5 h-5 mt-0.5 flex-shrink-0" />

      <div className="flex-1 min-w-0">
        <h3 className="font-semibold text-white mb-1">{toast.title}</h3>
        {toast.description && (
          <p className="text-sm text-gray-300">{toast.description}</p>
        )}
        {toast.action && (
          <button
            onClick={toast.action.onClick}
            className="mt-2 text-sm font-medium underline hover:no-underline"
          >
            {toast.action.label}
          </button>
        )}
      </div>

      <button
        onClick={() => onClose(toast.id)}
        className="flex-shrink-0 text-gray-400 hover:text-white transition-colors"
      >
        <X className="w-4 h-4" />
      </button>
    </div>
  );
}

export function useToastHelpers() {
  const toast = useToast();

  return {
    ...toast,
    promiseToast: async <T,>(
      promise: Promise<T>,
      messages: {
        loading: string;
        success: string;
        error: string;
      }
    ): Promise<T> => {
      const id = Math.random().toString(36).substring(7);

      toast.info(messages.loading, undefined);

      try {
        const result = await promise;
        toast.hideToast(id);
        toast.success(messages.success);
        return result;
      } catch (error) {
        toast.hideToast(id);
        toast.error(
          messages.error,
          error instanceof Error ? error.message : 'Unknown error'
        );
        throw error;
      }
    },
  };
}
