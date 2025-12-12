'use client';

import { clsx } from 'clsx';

type BadgeVariant = 'default' | 'success' | 'warning' | 'error' | 'info' | 'secondary';

interface BadgeProps {
  children: React.ReactNode;
  variant?: BadgeVariant;
  className?: string;
  dot?: boolean;
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300',
  success: 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400',
  warning: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400',
  error: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400',
  info: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
  secondary: 'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-400',
};

const dotColors: Record<BadgeVariant, string> = {
  default: 'bg-slate-500',
  success: 'bg-green-500',
  warning: 'bg-yellow-500',
  error: 'bg-red-500',
  info: 'bg-blue-500',
  secondary: 'bg-purple-500',
};

export function Badge({
  children,
  variant = 'default',
  className,
  dot = false,
}: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium',
        variantStyles[variant],
        className
      )}
    >
      {dot && (
        <span
          className={clsx('w-1.5 h-1.5 rounded-full', dotColors[variant])}
        />
      )}
      {children}
    </span>
  );
}

// Specific badge types for order status
export function OrderStatusBadge({ status }: { status: string }) {
  const statusConfig: Record<string, { variant: BadgeVariant; label: string }> = {
    pending: { variant: 'warning', label: 'Pending' },
    confirmed: { variant: 'info', label: 'Confirmed' },
    preparing: { variant: 'secondary', label: 'Preparing' },
    ready: { variant: 'success', label: 'Ready' },
    completed: { variant: 'success', label: 'Completed' },
    cancelled: { variant: 'error', label: 'Cancelled' },
  };

  const config = statusConfig[status.toLowerCase()] || {
    variant: 'default' as BadgeVariant,
    label: status,
  };

  return (
    <Badge variant={config.variant} dot>
      {config.label}
    </Badge>
  );
}

// Inventory status badge
export function StockBadge({ quantity, threshold = 10 }: { quantity: number; threshold?: number }) {
  if (quantity === 0) {
    return <Badge variant="error">Out of Stock</Badge>;
  }
  if (quantity <= threshold) {
    return <Badge variant="warning">Low Stock ({quantity})</Badge>;
  }
  return <Badge variant="success">In Stock ({quantity})</Badge>;
}
