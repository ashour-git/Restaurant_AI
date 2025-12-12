'use client';

import { clsx } from 'clsx';
import { AlertCircle, CheckCircle, Info, X, XCircle } from 'lucide-react';
import { useEffect, useState } from 'react';

type ToastVariant = 'success' | 'error' | 'warning' | 'info';

interface Toast {
  id: string;
  message: string;
  variant: ToastVariant;
  duration?: number;
}

interface ToastContextValue {
  toasts: Toast[];
  addToast: (message: string, variant?: ToastVariant, duration?: number) => void;
  removeToast: (id: string) => void;
}

const icons: Record<ToastVariant, React.ReactNode> = {
  success: <CheckCircle className="h-5 w-5 text-green-500" />,
  error: <XCircle className="h-5 w-5 text-red-500" />,
  warning: <AlertCircle className="h-5 w-5 text-yellow-500" />,
  info: <Info className="h-5 w-5 text-blue-500" />,
};

const variantStyles: Record<ToastVariant, string> = {
  success: 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-900/20',
  error: 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-900/20',
  warning: 'border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-900/20',
  info: 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-900/20',
};

function ToastItem({
  toast,
  onClose,
}: {
  toast: Toast;
  onClose: () => void;
}) {
  useEffect(() => {
    if (toast.duration) {
      const timer = setTimeout(onClose, toast.duration);
      return () => clearTimeout(timer);
    }
  }, [toast.duration, onClose]);

  return (
    <div
      className={clsx(
        'flex items-center gap-3 px-4 py-3 rounded-lg border shadow-lg',
        'animate-in slide-in-from-right-full duration-300',
        variantStyles[toast.variant]
      )}
    >
      {icons[toast.variant]}
      <p className="flex-1 text-sm font-medium text-slate-700 dark:text-slate-200">
        {toast.message}
      </p>
      <button
        onClick={onClose}
        className="p-1 rounded-full hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
      >
        <X className="h-4 w-4 text-slate-500" />
      </button>
    </div>
  );
}

// Simple toast store using custom hook
let toastListeners: Array<(toasts: Toast[]) => void> = [];
let toasts: Toast[] = [];

function emitChange() {
  toastListeners.forEach((listener) => listener([...toasts]));
}

export const toast = {
  success: (message: string, duration = 5000) => {
    const id = Date.now().toString();
    toasts = [...toasts, { id, message, variant: 'success', duration }];
    emitChange();
    if (duration) setTimeout(() => toast.dismiss(id), duration);
  },
  error: (message: string, duration = 5000) => {
    const id = Date.now().toString();
    toasts = [...toasts, { id, message, variant: 'error', duration }];
    emitChange();
    if (duration) setTimeout(() => toast.dismiss(id), duration);
  },
  warning: (message: string, duration = 5000) => {
    const id = Date.now().toString();
    toasts = [...toasts, { id, message, variant: 'warning', duration }];
    emitChange();
    if (duration) setTimeout(() => toast.dismiss(id), duration);
  },
  info: (message: string, duration = 5000) => {
    const id = Date.now().toString();
    toasts = [...toasts, { id, message, variant: 'info', duration }];
    emitChange();
    if (duration) setTimeout(() => toast.dismiss(id), duration);
  },
  dismiss: (id: string) => {
    toasts = toasts.filter((t) => t.id !== id);
    emitChange();
  },
};

export function useToasts() {
  const [currentToasts, setCurrentToasts] = useState<Toast[]>(toasts);

  useEffect(() => {
    toastListeners.push(setCurrentToasts);
    return () => {
      toastListeners = toastListeners.filter((l) => l !== setCurrentToasts);
    };
  }, []);

  return currentToasts;
}

export function ToastContainer() {
  const currentToasts = useToasts();

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {currentToasts.map((t) => (
        <ToastItem
          key={t.id}
          toast={t}
          onClose={() => toast.dismiss(t.id)}
        />
      ))}
    </div>
  );
}
