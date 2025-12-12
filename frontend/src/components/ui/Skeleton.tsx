'use client';

import { clsx } from 'clsx';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string | number;
  height?: string | number;
  count?: number;
}

export function Skeleton({
  className,
  variant = 'rectangular',
  width,
  height,
  count = 1,
}: SkeletonProps) {
  const baseStyles = 'animate-pulse bg-slate-200 dark:bg-slate-700';

  const variantStyles = {
    text: 'rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-lg',
  };

  const style = {
    width: typeof width === 'number' ? `${width}px` : width,
    height: typeof height === 'number' ? `${height}px` : height,
  };

  if (count > 1) {
    return (
      <div className="space-y-2">
        {Array.from({ length: count }).map((_, i) => (
          <div
            key={i}
            className={clsx(baseStyles, variantStyles[variant], className)}
            style={style}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      className={clsx(baseStyles, variantStyles[variant], className)}
      style={style}
    />
  );
}

// Pre-built skeleton patterns
export function StatCardSkeleton() {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
      <div className="flex items-start justify-between">
        <div className="space-y-3">
          <Skeleton width={100} height={14} />
          <Skeleton width={80} height={32} />
        </div>
        <Skeleton variant="rectangular" width={48} height={48} />
      </div>
      <div className="mt-4 flex items-center gap-2">
        <Skeleton width={60} height={14} />
        <Skeleton width={80} height={14} />
      </div>
    </div>
  );
}

export function TableRowSkeleton({ columns = 5 }: { columns?: number }) {
  return (
    <tr className="border-b border-slate-200 dark:border-slate-700">
      {Array.from({ length: columns }).map((_, i) => (
        <td key={i} className="px-4 py-3">
          <Skeleton width="80%" height={16} />
        </td>
      ))}
    </tr>
  );
}

export function CardSkeleton() {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-xl p-6 border border-slate-200 dark:border-slate-700">
      <Skeleton width="60%" height={20} className="mb-4" />
      <div className="space-y-3">
        <Skeleton height={14} />
        <Skeleton height={14} />
        <Skeleton width="80%" height={14} />
      </div>
    </div>
  );
}

export function ChatMessageSkeleton() {
  return (
    <div className="flex gap-3">
      <Skeleton variant="circular" width={40} height={40} />
      <div className="flex-1 space-y-2">
        <Skeleton width="70%" height={16} />
        <Skeleton width="50%" height={16} />
      </div>
    </div>
  );
}
