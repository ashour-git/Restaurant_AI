'use client';

import { useAnalytics } from '@/hooks/useApi';
import { clsx } from 'clsx';
import {
    ArrowDown,
    ArrowUp,
    BarChart2,
    DollarSign,
    ShoppingCart,
    TrendingUp,
    Users
} from 'lucide-react';
import { useState } from 'react';
import {
    Area,
    AreaChart,
    Bar,
    BarChart,
    CartesianGrid,
    Cell,
    Legend,
    Pie,
    PieChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis
} from 'recharts';

// Mock data for charts - in production this would come from API
const revenueData = [
  { day: 'Mon', revenue: 4000, orders: 24 },
  { day: 'Tue', revenue: 3000, orders: 18 },
  { day: 'Wed', revenue: 5000, orders: 30 },
  { day: 'Thu', revenue: 4500, orders: 27 },
  { day: 'Fri', revenue: 6000, orders: 36 },
  { day: 'Sat', revenue: 8000, orders: 48 },
  { day: 'Sun', revenue: 7000, orders: 42 },
];

const categoryData = [
  { name: 'Main Courses', value: 40, color: '#3b82f6' },
  { name: 'Appetizers', value: 25, color: '#10b981' },
  { name: 'Beverages', value: 20, color: '#f59e0b' },
  { name: 'Desserts', value: 15, color: '#ef4444' },
];

const hourlyData = Array.from({ length: 24 }, (_, i) => ({
  hour: `${i}:00`,
  orders: Math.floor(Math.random() * 20) + (i >= 11 && i <= 14 ? 15 : i >= 18 && i <= 21 ? 20 : 5),
}));

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState<'today' | 'week' | 'month'>('week');
  const { data: analytics, isLoading } = useAnalytics({ period: timeRange });

  const stats = [
    {
      title: 'Total Revenue',
      value: `$${analytics?.total_revenue?.toLocaleString() || '0'}`,
      change: '+12.5%',
      trend: 'up',
      icon: DollarSign,
      color: 'text-green-500',
      bg: 'bg-green-100 dark:bg-green-900/30',
    },
    {
      title: 'Total Orders',
      value: analytics?.total_orders?.toLocaleString() || '0',
      change: '+8.2%',
      trend: 'up',
      icon: ShoppingCart,
      color: 'text-blue-500',
      bg: 'bg-blue-100 dark:bg-blue-900/30',
    },
    {
      title: 'Average Order Value',
      value: `$${analytics?.avg_order_value?.toFixed(2) || '0.00'}`,
      change: '+3.1%',
      trend: 'up',
      icon: TrendingUp,
      color: 'text-purple-500',
      bg: 'bg-purple-100 dark:bg-purple-900/30',
    },
    {
      title: 'Total Customers',
      value: analytics?.total_customers?.toLocaleString() || '0',
      change: '-2.4%',
      trend: 'down',
      icon: Users,
      color: 'text-amber-500',
      bg: 'bg-amber-100 dark:bg-amber-900/30',
    },
  ];

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <BarChart2 className="h-5 w-5 sm:h-7 sm:w-7 text-blue-500" />
            Analytics
          </h1>
          <p className="text-sm text-slate-500">Track your restaurant performance</p>
        </div>

        {/* Time Range Selector */}
        <div className="flex bg-slate-100 dark:bg-slate-700 rounded-lg p-1 self-start sm:self-auto">
          {(['today', 'week', 'month'] as const).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={clsx(
                'px-3 sm:px-4 py-1.5 sm:py-2 text-xs sm:text-sm font-medium rounded-md transition-colors',
                timeRange === range
                  ? 'bg-white dark:bg-slate-800 text-slate-900 dark:text-white shadow'
                  : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300'
              )}
            >
              {range.charAt(0).toUpperCase() + range.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        {stats.map((stat) => (
          <div
            key={stat.title}
            className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-3 sm:p-5"
          >
            <div className="flex items-start justify-between">
              <div className={clsx('h-8 w-8 sm:h-12 sm:w-12 rounded-lg flex items-center justify-center', stat.bg)}>
                <stat.icon className={clsx('h-4 w-4 sm:h-6 sm:w-6', stat.color)} />
              </div>
              <div
                className={clsx(
                  'flex items-center gap-0.5 sm:gap-1 text-xs sm:text-sm font-medium',
                  stat.trend === 'up' ? 'text-green-500' : 'text-red-500'
                )}
              >
                {stat.trend === 'up' ? (
                  <ArrowUp className="h-3 w-3 sm:h-4 sm:w-4" />
                ) : (
                  <ArrowDown className="h-3 w-3 sm:h-4 sm:w-4" />
                )}
                <span className="hidden xs:inline">{stat.change}</span>
              </div>
            </div>
            <div className="mt-2 sm:mt-4">
              <p className="text-lg sm:text-2xl font-bold text-slate-900 dark:text-white">{stat.value}</p>
              <p className="text-xs sm:text-sm text-slate-500 truncate">{stat.title}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
        {/* Revenue Chart */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 sm:p-6">
          <h3 className="text-base sm:text-lg font-semibold text-slate-900 dark:text-white mb-3 sm:mb-4">
            Revenue Overview
          </h3>
          <div className="h-60 sm:h-80">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={revenueData}>
                <defs>
                  <linearGradient id="colorRevenue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="day" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="revenue"
                  stroke="#3b82f6"
                  fillOpacity={1}
                  fill="url(#colorRevenue)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Category Distribution */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Sales by Category
          </h3>
          <div className="h-80 flex items-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={categoryData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {categoryData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Orders by Hour */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Orders by Hour
          </h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hourlyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="hour" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: 'none',
                    borderRadius: '8px',
                    color: '#fff',
                  }}
                />
                <Bar dataKey="orders" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Selling Items */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
            Top Selling Items
          </h3>
          <div className="space-y-4">
            {[
              { name: 'Grilled Salmon', sales: 156, revenue: 4680 },
              { name: 'Margherita Pizza', sales: 142, revenue: 2130 },
              { name: 'Caesar Salad', sales: 128, revenue: 1280 },
              { name: 'Beef Burger', sales: 115, revenue: 1725 },
              { name: 'Tiramisu', sales: 98, revenue: 784 },
            ].map((item, index) => (
              <div key={item.name} className="flex items-center gap-4">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg flex items-center justify-center text-white font-bold text-sm">
                  {index + 1}
                </div>
                <div className="flex-1">
                  <p className="font-medium text-slate-900 dark:text-white">{item.name}</p>
                  <p className="text-sm text-slate-500">{item.sales} orders</p>
                </div>
                <p className="font-semibold text-green-500">${item.revenue}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* AI Insights */}
      <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl p-6 text-white">
        <h3 className="text-lg font-semibold mb-4">AI Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white/20 backdrop-blur-sm rounded-lg p-4">
            <p className="text-sm font-medium mb-2">Peak Hours</p>
            <p className="text-2xl font-bold">12:00 - 14:00</p>
            <p className="text-xs opacity-80">Lunch rush optimization recommended</p>
          </div>
          <div className="bg-white/20 backdrop-blur-sm rounded-lg p-4">
            <p className="text-sm font-medium mb-2">Predicted Revenue</p>
            <p className="text-2xl font-bold">$45,200</p>
            <p className="text-xs opacity-80">Next week forecast</p>
          </div>
          <div className="bg-white/20 backdrop-blur-sm rounded-lg p-4">
            <p className="text-sm font-medium mb-2">Stock Alert</p>
            <p className="text-2xl font-bold">3 Items</p>
            <p className="text-xs opacity-80">Reorder needed before weekend</p>
          </div>
        </div>
      </div>
    </div>
  );
}
