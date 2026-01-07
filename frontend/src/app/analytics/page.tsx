'use client';

import { useAnalytics, useTopItems, useMLHealth } from '@/hooks/useApi';
import { mlApi } from '@/lib/api';
import { clsx } from 'clsx';
import {
    AlertTriangle,
    ArrowDown,
    ArrowUp,
    BarChart2,
    Brain,
    DollarSign,
    Lightbulb,
    Loader2,
    Package,
    ShoppingCart,
    Sparkles,
    Target,
    TrendingDown,
    TrendingUp,
    Users,
    Zap
} from 'lucide-react';
import { useEffect, useState } from 'react';
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
        <div className="flex items-center gap-2 mb-4">
          <Brain className="h-6 w-6" />
          <h3 className="text-lg font-semibold">AI-Powered Business Insights</h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white/20 backdrop-blur-sm rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Target className="h-4 w-4" />
              <p className="text-sm font-medium">Peak Hours</p>
            </div>
            <p className="text-2xl font-bold">12:00 - 14:00</p>
            <p className="text-xs opacity-80">Lunch rush optimization recommended</p>
          </div>
          <div className="bg-white/20 backdrop-blur-sm rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="h-4 w-4" />
              <p className="text-sm font-medium">Predicted Revenue</p>
            </div>
            <p className="text-2xl font-bold">$45,200</p>
            <p className="text-xs opacity-80">Next week forecast</p>
          </div>
          <div className="bg-white/20 backdrop-blur-sm rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Package className="h-4 w-4" />
              <p className="text-sm font-medium">Stock Alert</p>
            </div>
            <p className="text-2xl font-bold">3 Items</p>
            <p className="text-xs opacity-80">Reorder needed before weekend</p>
          </div>
        </div>
      </div>

      {/* Actionable Recommendations */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
        <div className="flex items-center gap-2 mb-6">
          <Lightbulb className="h-6 w-6 text-yellow-500" />
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
            Actionable Recommendations
          </h3>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Revenue Opportunity */}
          <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="bg-green-100 dark:bg-green-800 p-2 rounded-lg">
                <DollarSign className="h-5 w-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <h4 className="font-semibold text-green-800 dark:text-green-300 mb-1">
                  Increase Revenue by 15%
                </h4>
                <p className="text-sm text-green-700 dark:text-green-400 mb-2">
                  Bundle top-selling &quot;Fish & Chips&quot; with beverages. Customers who order mains are 3x more likely to add drinks.
                </p>
                <div className="flex items-center gap-1 text-xs text-green-600 dark:text-green-500">
                  <Zap className="h-3 w-3" />
                  <span>Est. +$2,400/month</span>
                </div>
              </div>
            </div>
          </div>

          {/* Customer Retention */}
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="bg-blue-100 dark:bg-blue-800 p-2 rounded-lg">
                <Users className="h-5 w-5 text-blue-600 dark:text-blue-400" />
              </div>
              <div>
                <h4 className="font-semibold text-blue-800 dark:text-blue-300 mb-1">
                  Re-engage 23 Customers
                </h4>
                <p className="text-sm text-blue-700 dark:text-blue-400 mb-2">
                  These customers haven&apos;t ordered in 30+ days but spent $50+ previously. Send a 15% discount offer.
                </p>
                <div className="flex items-center gap-1 text-xs text-blue-600 dark:text-blue-500">
                  <Sparkles className="h-3 w-3" />
                  <span>High-value recovery</span>
                </div>
              </div>
            </div>
          </div>

          {/* Menu Optimization */}
          <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="bg-purple-100 dark:bg-purple-800 p-2 rounded-lg">
                <BarChart2 className="h-5 w-5 text-purple-600 dark:text-purple-400" />
              </div>
              <div>
                <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-1">
                  Optimize Menu Pricing
                </h4>
                <p className="text-sm text-purple-700 dark:text-purple-400 mb-2">
                  &quot;Grilled Salmon&quot; has high demand but low margin. A $2 price increase would add $312/month with minimal impact.
                </p>
                <div className="flex items-center gap-1 text-xs text-purple-600 dark:text-purple-500">
                  <TrendingUp className="h-3 w-3" />
                  <span>Price elasticity: Low</span>
                </div>
              </div>
            </div>
          </div>

          {/* Staffing */}
          <div className="bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-800 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="bg-orange-100 dark:bg-orange-800 p-2 rounded-lg">
                <Users className="h-5 w-5 text-orange-600 dark:text-orange-400" />
              </div>
              <div>
                <h4 className="font-semibold text-orange-800 dark:text-orange-300 mb-1">
                  Staffing Opportunity
                </h4>
                <p className="text-sm text-orange-700 dark:text-orange-400 mb-2">
                  Saturday 6-8 PM sees 40% higher orders. Add one more server during peak to reduce wait times by 25%.
                </p>
                <div className="flex items-center gap-1 text-xs text-orange-600 dark:text-orange-500">
                  <Target className="h-3 w-3" />
                  <span>Customer satisfaction boost</span>
                </div>
              </div>
            </div>
          </div>

          {/* Inventory Alert */}
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="bg-red-100 dark:bg-red-800 p-2 rounded-lg">
                <AlertTriangle className="h-5 w-5 text-red-600 dark:text-red-400" />
              </div>
              <div>
                <h4 className="font-semibold text-red-800 dark:text-red-300 mb-1">
                  Inventory Warning
                </h4>
                <p className="text-sm text-red-700 dark:text-red-400 mb-2">
                  Based on forecasted demand, &quot;Fresh Salmon&quot; and &quot;Romaine Lettuce&quot; will run out by Friday. Reorder now.
                </p>
                <div className="flex items-center gap-1 text-xs text-red-600 dark:text-red-500">
                  <Package className="h-3 w-3" />
                  <span>Prevent stockout</span>
                </div>
              </div>
            </div>
          </div>

          {/* Trend Alert */}
          <div className="bg-cyan-50 dark:bg-cyan-900/20 border border-cyan-200 dark:border-cyan-800 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="bg-cyan-100 dark:bg-cyan-800 p-2 rounded-lg">
                <TrendingDown className="h-5 w-5 text-cyan-600 dark:text-cyan-400" />
              </div>
              <div>
                <h4 className="font-semibold text-cyan-800 dark:text-cyan-300 mb-1">
                  Declining Item Alert
                </h4>
                <p className="text-sm text-cyan-700 dark:text-cyan-400 mb-2">
                  &quot;Vegetable Soup&quot; orders dropped 35% this month. Consider refreshing the recipe or replacing with a seasonal option.
                </p>
                <div className="flex items-center gap-1 text-xs text-cyan-600 dark:text-cyan-500">
                  <BarChart2 className="h-3 w-3" />
                  <span>Review menu performance</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <button
          onClick={() => window.location.href = '/assistant'}
          className="bg-gradient-to-br from-purple-500 to-blue-500 text-white rounded-xl p-4 flex flex-col items-center gap-2 hover:opacity-90 transition-opacity"
        >
          <Brain className="h-8 w-8" />
          <span className="text-sm font-medium">Ask AI Assistant</span>
        </button>
        <button
          onClick={() => window.location.href = '/inventory'}
          className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl p-4 flex flex-col items-center gap-2 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
        >
          <Package className="h-8 w-8 text-orange-500" />
          <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Check Inventory</span>
        </button>
        <button
          onClick={() => window.location.href = '/customers'}
          className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl p-4 flex flex-col items-center gap-2 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
        >
          <Users className="h-8 w-8 text-blue-500" />
          <span className="text-sm font-medium text-slate-700 dark:text-slate-300">View Customers</span>
        </button>
        <button
          onClick={() => window.location.href = '/menu'}
          className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl p-4 flex flex-col items-center gap-2 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
        >
          <BarChart2 className="h-8 w-8 text-green-500" />
          <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Manage Menu</span>
        </button>
      </div>
    </div>
  );
}
