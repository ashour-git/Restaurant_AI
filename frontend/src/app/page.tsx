'use client';

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import { useMenuItems, useOrders } from '@/hooks/useApi';
import { analyticsApi } from '@/lib/api';
import { useQuery } from '@tanstack/react-query';
import { clsx } from 'clsx';
import {
  ArrowDownRight,
  ArrowUpRight,
  DollarSign,
  RefreshCw,
  ShoppingCart,
  TrendingUp,
  Users,
} from 'lucide-react';
import { OrderStatusBadge, StatCardSkeleton } from '@/components/ui';

interface StatCardProps {
  title: string;
  value: string;
  change?: number;
  icon: React.ComponentType<{ className?: string }>;
  isLoading?: boolean;
}

function StatCard({ title, value, change, icon: Icon, isLoading }: StatCardProps) {
  if (isLoading) {
    return <StatCardSkeleton />;
  }

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {change !== undefined && (
          <p className="text-xs text-muted-foreground">
            <span
              className={clsx(
                'font-medium',
                change >= 0 ? 'text-green-500' : 'text-red-500'
              )}
            >
              {change >= 0 ? '+' : ''}
              {change}%
            </span>{' '}
            vs last week
          </p>
        )}
      </CardContent>
    </Card>
  );
}

export default function DashboardPage() {
  // Fetch dashboard data
  const {
    data: salesData,
    isLoading: salesLoading,
    refetch: refetchSales,
  } = useQuery({
    queryKey: ['salesSummary'],
    queryFn: () =>
      analyticsApi
        .getSalesReport({
          start_date: new Date(
            Date.now() - 7 * 24 * 60 * 60 * 1000
          ).toISOString(),
          end_date: new Date().toISOString(),
        })
        .then((res) => res.data)
        .catch(() => null),
    retry: false,
  });

  const { data: orders, isLoading: ordersLoading } = useOrders();
  const { data: menuItems, isLoading: menuLoading } = useMenuItems();

  // Calculate stats from real data or use defaults
  const stats = [
    {
      title: 'Total Revenue',
      value: salesData?.total_revenue
        ? `$${Number(salesData.total_revenue).toLocaleString()}`
        : '$0',
      change: 12.5,
      icon: DollarSign,
    },
    {
      title: 'Orders Today',
      value: orders?.length?.toString() || '0',
      change: 8.2,
      icon: ShoppingCart,
    },
    {
      title: 'Menu Items',
      value: menuItems?.length?.toString() || '0',
      icon: TrendingUp,
    },
    {
      title: 'Active Tables',
      value: '12',
      change: 5.1,
      icon: Users,
    },
  ];

  const isLoading = salesLoading || ordersLoading || menuLoading;

  // Get recent orders for display
  const recentOrders = orders?.slice(0, 5) || [];

  // Get top menu items
  const topItems = menuItems?.slice(0, 5) || [];

  return (
    <div className="space-y-4">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Dashboard</h1>
          <p className="text-muted-foreground">
            Welcome back! Here&apos;s what&apos;s happening at your restaurant.
          </p>
        </div>
        <Button
          onClick={() => refetchSales()}
          variant="outline"
          size="sm"
          isLoading={isLoading}
        >
          <RefreshCw
            className={clsx('h-4 w-4', isLoading && 'animate-spin')}
          />
          <span className="ml-2">Refresh</span>
        </Button>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => (
          <StatCard key={stat.title} {...stat} isLoading={isLoading} />
        ))}
      </div>

      {/* Main Content Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Recent Orders</CardTitle>
          </CardHeader>
          <CardContent>
            {ordersLoading ? (
              <div className="space-y-4">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between py-3 border-b border-slate-100 dark:border-slate-700 animate-pulse"
                  >
                    <div className="flex items-center space-x-4">
                      <div className="h-10 w-10 bg-slate-200 dark:bg-slate-700 rounded-lg" />
                      <div className="space-y-2">
                        <div className="h-4 w-24 bg-slate-200 dark:bg-slate-700 rounded" />
                        <div className="h-3 w-16 bg-slate-200 dark:bg-slate-700 rounded" />
                      </div>
                    </div>
                    <div className="h-6 w-20 bg-slate-200 dark:bg-slate-700 rounded-full" />
                  </div>
                ))}
              </div>
            ) : recentOrders.length > 0 ? (
              <div className="space-y-4">
                {recentOrders.map((order: any) => (
                  <div
                    key={order.id}
                    className="flex items-center justify-between py-3 border-b border-slate-100 dark:border-slate-700 last:border-0"
                  >
                    <div className="flex items-center space-x-4">
                      <div className="h-10 w-10 bg-slate-100 dark:bg-slate-700 rounded-lg flex items-center justify-center">
                        <ShoppingCart className="h-5 w-5 text-slate-500" />
                      </div>
                      <div>
                        <p className="font-medium">Order #{order.id}</p>
                        <p className="text-sm text-muted-foreground">
                          {order.table_number
                            ? `Table ${order.table_number}`
                            : 'Takeaway'}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium">
                        ${Number(order.total || 0).toFixed(2)}
                      </p>
                      <OrderStatusBadge status={order.status || 'pending'} />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <ShoppingCart className="h-12 w-12 mx-auto mb-3 opacity-50" />
                <p>No orders yet today</p>
                <p className="text-sm">New orders will appear here</p>
              </div>
            )}
          </CardContent>
        </Card>
        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>Top Menu Items</CardTitle>
          </CardHeader>
          <CardContent>
            {menuLoading ? (
              <div className="space-y-4">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between py-2 animate-pulse"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="h-4 w-6 bg-slate-200 dark:bg-slate-700 rounded" />
                      <div className="h-4 w-32 bg-slate-200 dark:bg-slate-700 rounded" />
                    </div>
                    <div className="h-4 w-16 bg-slate-200 dark:bg-slate-700 rounded" />
                  </div>
                ))}
              </div>
            ) : topItems.length > 0 ? (
              <div className="space-y-4">
                {topItems.map((item: any, i: number) => (
                  <div
                    key={item.id}
                    className="flex items-center justify-between py-2"
                  >
                    <div className="flex items-center space-x-3">
                      <span className="text-sm font-medium text-muted-foreground">
                        #{i + 1}
                      </span>
                      <span className="font-medium">{item.name}</span>
                    </div>
                    <span className="text-sm font-medium">
                      ${Number(item.price || 0).toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <p>No menu items yet</p>
                <Button
                  variant="link"
                  size="sm"
                  onClick={() => (window.location.href = '/menu')}
                >
                  Add your first item →
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
      {/* AI Insights Card */}
      <Card>
        <CardHeader>
          <CardTitle>AI-Powered Insights</CardTitle>
          <CardDescription>
            Get personalized recommendations and forecasts for your restaurant
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button onClick={() => (window.location.href = '/assistant')}>
            Chat with AI Assistant
            <ArrowUpRight className="h-4 w-4 ml-2" />
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
