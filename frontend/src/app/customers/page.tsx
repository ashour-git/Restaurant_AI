'use client';

import { useCustomer, useCustomers } from '@/hooks/useApi';
import { clsx } from 'clsx';
import {
    AlertTriangle,
    Award,
    DollarSign,
    Mail,
    Phone,
    Search,
    ShoppingCart,
    Sparkles,
    Star,
    TrendingDown,
    TrendingUp,
    User,
    Users,
    Zap
} from 'lucide-react';
import { useMemo, useState } from 'react';

// Customer segment definitions based on RFM analysis
const getCustomerSegment = (customer: any) => {
  const daysSinceOrder = customer.last_order_date 
    ? Math.floor((Date.now() - new Date(customer.last_order_date).getTime()) / (1000 * 60 * 60 * 24))
    : 999;
  const totalSpent = Number(customer.total_spent || 0);
  const orderCount = customer.order_count || 0;
  
  // Champions: Recent, frequent, high spenders
  if (daysSinceOrder < 30 && orderCount >= 5 && totalSpent >= 200) {
    return { segment: 'Champion', color: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400', icon: Award, risk: 'low' };
  }
  // Loyal: Regular customers with good spend
  if (daysSinceOrder < 45 && orderCount >= 3 && totalSpent >= 100) {
    return { segment: 'Loyal', color: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400', icon: Star, risk: 'low' };
  }
  // At Risk: Haven't ordered recently but were valuable
  if (daysSinceOrder >= 30 && daysSinceOrder < 60 && totalSpent >= 50) {
    return { segment: 'At Risk', color: 'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400', icon: AlertTriangle, risk: 'medium' };
  }
  // Lost: Long time no order
  if (daysSinceOrder >= 60) {
    return { segment: 'Lost', color: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400', icon: TrendingDown, risk: 'high' };
  }
  // Promising: New with potential
  if (orderCount >= 2) {
    return { segment: 'Promising', color: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400', icon: TrendingUp, risk: 'low' };
  }
  // New: First-time customers
  return { segment: 'New', color: 'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400', icon: Sparkles, risk: 'medium' };
};

export default function CustomersPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCustomerId, setSelectedCustomerId] = useState<number | null>(null);
  const [filterSegment, setFilterSegment] = useState<string>('all');

  const { data: customers, isLoading } = useCustomers(
    searchQuery ? { search: searchQuery } : undefined
  );

  const { data: selectedCustomer } = useCustomer(selectedCustomerId || 0);

  // Calculate segment statistics
  const segmentStats = useMemo(() => {
    if (!customers) return { champions: 0, loyal: 0, atRisk: 0, lost: 0, promising: 0, new: 0 };
    
    const stats = { champions: 0, loyal: 0, atRisk: 0, lost: 0, promising: 0, new: 0 };
    customers.forEach((c: any) => {
      const { segment } = getCustomerSegment(c);
      if (segment === 'Champion') stats.champions++;
      else if (segment === 'Loyal') stats.loyal++;
      else if (segment === 'At Risk') stats.atRisk++;
      else if (segment === 'Lost') stats.lost++;
      else if (segment === 'Promising') stats.promising++;
      else stats.new++;
    });
    return stats;
  }, [customers]);

  // Filter customers by segment
  const filteredCustomers = useMemo(() => {
    if (!customers) return [];
    if (filterSegment === 'all') return customers;
    return customers.filter((c: any) => {
      const { segment } = getCustomerSegment(c);
      return segment.toLowerCase().replace(' ', '-') === filterSegment;
    });
  }, [customers, filterSegment]);

  const formatDate = (timestamp: string) => {
    if (!timestamp) return '--';
    return new Date(timestamp).toLocaleDateString();
  };

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Users className="h-5 w-5 sm:h-7 sm:w-7 text-purple-500" />
            Customers
          </h1>
          <p className="text-sm text-slate-500">
            {customers?.length || 0} customers in your database
          </p>
        </div>
      </div>

      {/* Segment Overview Cards */}
      <div className="grid grid-cols-3 sm:grid-cols-6 gap-2 sm:gap-3">
        <button
          onClick={() => setFilterSegment('all')}
          className={clsx(
            'p-2 sm:p-3 rounded-lg border text-center transition-all',
            filterSegment === 'all' 
              ? 'bg-slate-100 dark:bg-slate-700 border-slate-400 dark:border-slate-500'
              : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700'
          )}
        >
          <p className="text-lg sm:text-xl font-bold text-slate-900 dark:text-white">{customers?.length || 0}</p>
          <p className="text-[10px] sm:text-xs text-slate-500">All</p>
        </button>
        <button
          onClick={() => setFilterSegment('champion')}
          className={clsx(
            'p-2 sm:p-3 rounded-lg border text-center transition-all',
            filterSegment === 'champion'
              ? 'bg-yellow-100 dark:bg-yellow-900/30 border-yellow-400'
              : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-yellow-50 dark:hover:bg-yellow-900/10'
          )}
        >
          <p className="text-lg sm:text-xl font-bold text-yellow-600 dark:text-yellow-400">{segmentStats.champions}</p>
          <p className="text-[10px] sm:text-xs text-slate-500">Champions</p>
        </button>
        <button
          onClick={() => setFilterSegment('loyal')}
          className={clsx(
            'p-2 sm:p-3 rounded-lg border text-center transition-all',
            filterSegment === 'loyal'
              ? 'bg-green-100 dark:bg-green-900/30 border-green-400'
              : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-green-50 dark:hover:bg-green-900/10'
          )}
        >
          <p className="text-lg sm:text-xl font-bold text-green-600 dark:text-green-400">{segmentStats.loyal}</p>
          <p className="text-[10px] sm:text-xs text-slate-500">Loyal</p>
        </button>
        <button
          onClick={() => setFilterSegment('at-risk')}
          className={clsx(
            'p-2 sm:p-3 rounded-lg border text-center transition-all',
            filterSegment === 'at-risk'
              ? 'bg-orange-100 dark:bg-orange-900/30 border-orange-400'
              : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-orange-50 dark:hover:bg-orange-900/10'
          )}
        >
          <p className="text-lg sm:text-xl font-bold text-orange-600 dark:text-orange-400">{segmentStats.atRisk}</p>
          <p className="text-[10px] sm:text-xs text-slate-500">At Risk</p>
        </button>
        <button
          onClick={() => setFilterSegment('promising')}
          className={clsx(
            'p-2 sm:p-3 rounded-lg border text-center transition-all',
            filterSegment === 'promising'
              ? 'bg-blue-100 dark:bg-blue-900/30 border-blue-400'
              : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-blue-50 dark:hover:bg-blue-900/10'
          )}
        >
          <p className="text-lg sm:text-xl font-bold text-blue-600 dark:text-blue-400">{segmentStats.promising}</p>
          <p className="text-[10px] sm:text-xs text-slate-500">Promising</p>
        </button>
        <button
          onClick={() => setFilterSegment('lost')}
          className={clsx(
            'p-2 sm:p-3 rounded-lg border text-center transition-all',
            filterSegment === 'lost'
              ? 'bg-red-100 dark:bg-red-900/30 border-red-400'
              : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 hover:bg-red-50 dark:hover:bg-red-900/10'
          )}
        >
          <p className="text-lg sm:text-xl font-bold text-red-600 dark:text-red-400">{segmentStats.lost}</p>
          <p className="text-[10px] sm:text-xs text-slate-500">Lost</p>
        </button>
      </div>

      {/* Search */}
      <div className="relative w-full sm:max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search customers..."
          className="w-full pl-10 pr-4 py-2.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div className="flex flex-col lg:grid lg:grid-cols-3 gap-4 sm:gap-6">
        {/* Customer List */}
        <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden max-h-[50vh] lg:max-h-none overflow-y-auto">
          {isLoading ? (
            <div className="p-4 sm:p-8">
              <div className="animate-pulse space-y-4">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="flex items-center gap-3 sm:gap-4">
                    <div className="h-10 w-10 sm:h-12 sm:w-12 bg-slate-200 dark:bg-slate-700 rounded-full" />
                    <div className="flex-1 space-y-2">
                      <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-1/3" />
                      <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded w-1/4" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (filteredCustomers || []).length > 0 ? (
            <div className="divide-y divide-slate-200 dark:divide-slate-700">
              {(filteredCustomers || []).map((customer: any) => {
                const segmentInfo = getCustomerSegment(customer);
                const SegmentIcon = segmentInfo.icon;
                return (
                  <button
                    key={customer.id}
                    onClick={() => setSelectedCustomerId(customer.id)}
                    className={clsx(
                      'w-full flex items-center gap-3 sm:gap-4 p-3 sm:p-4 text-left hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors',
                      selectedCustomerId === customer.id && 'bg-blue-50 dark:bg-blue-900/20'
                    )}
                  >
                    <div className="h-10 w-10 sm:h-12 sm:w-12 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center text-white font-semibold flex-shrink-0">
                      {customer.name?.charAt(0) || 'C'}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <p className="font-medium text-sm sm:text-base text-slate-900 dark:text-white truncate">
                          {customer.name || 'Unknown Customer'}
                        </p>
                        <span className={clsx(
                          'px-1.5 py-0.5 rounded text-[10px] font-medium flex items-center gap-1',
                          segmentInfo.color
                        )}>
                          <SegmentIcon className="h-3 w-3" />
                          <span className="hidden sm:inline">{segmentInfo.segment}</span>
                        </span>
                      </div>
                      <p className="text-xs sm:text-sm text-slate-500 truncate">
                        {customer.email || customer.phone || 'No contact info'}
                      </p>
                    </div>
                    <div className="text-right hidden sm:block">
                      <p className="font-medium text-slate-900 dark:text-white">
                        ${Number(customer.total_spent || 0).toFixed(2)}
                      </p>
                      <p className="text-xs text-slate-500">
                        {customer.order_count || 0} orders
                      </p>
                    </div>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500">
              <Users className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No customers found</p>
              <p className="text-sm">Customers will appear here after making orders</p>
            </div>
          )}
        </div>

        {/* Customer Detail */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 sm:p-6">
          {selectedCustomer ? (() => {
              const segmentInfo = getCustomerSegment(selectedCustomer);
              const SegmentIcon = segmentInfo.icon;
              return (
            <div className="space-y-4 sm:space-y-6">
              {/* Customer Segment Badge */}
              <div className={clsx(
                'flex items-center justify-center gap-2 py-2 px-3 rounded-lg',
                segmentInfo.color
              )}>
                <SegmentIcon className="h-4 w-4" />
                <span className="font-medium text-sm">{segmentInfo.segment} Customer</span>
                {segmentInfo.risk === 'high' && (
                  <span className="text-xs opacity-80">• Needs attention</span>
                )}
              </div>

              {/* Profile */}
              <div className="text-center">
                <div className="h-16 w-16 sm:h-20 sm:w-20 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center text-white text-xl sm:text-2xl font-bold mx-auto mb-3 sm:mb-4">
                  {selectedCustomer.name?.charAt(0) || 'C'}
                </div>
                <h3 className="text-lg sm:text-xl font-semibold text-slate-900 dark:text-white">
                  {selectedCustomer.name || 'Unknown Customer'}
                </h3>
                <p className="text-xs sm:text-sm text-slate-500">
                  Customer since {formatDate(selectedCustomer.created_at)}
                </p>
              </div>

              {/* Contact Info */}
              <div className="space-y-2 sm:space-y-3">
                {selectedCustomer.email && (
                  <div className="flex items-center gap-2 sm:gap-3 text-sm">
                    <Mail className="h-4 w-4 text-slate-400 flex-shrink-0" />
                    <span className="text-slate-600 dark:text-slate-300 truncate">
                      {selectedCustomer.email}
                    </span>
                  </div>
                )}
                {selectedCustomer.phone && (
                  <div className="flex items-center gap-2 sm:gap-3 text-sm">
                    <Phone className="h-4 w-4 text-slate-400 flex-shrink-0" />
                    <span className="text-slate-600 dark:text-slate-300">
                      {selectedCustomer.phone}
                    </span>
                  </div>
                )}
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-3 sm:gap-4">
                <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3 sm:p-4 text-center">
                  <ShoppingCart className="h-5 w-5 sm:h-6 sm:w-6 text-blue-500 mx-auto mb-1 sm:mb-2" />
                  <p className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white">
                    {selectedCustomer.order_count || 0}
                  </p>
                  <p className="text-xs text-slate-500">Total Orders</p>
                </div>
                <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-3 sm:p-4 text-center">
                  <DollarSign className="h-5 w-5 sm:h-6 sm:w-6 text-green-500 mx-auto mb-1 sm:mb-2" />
                  <p className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white">
                    ${Number(selectedCustomer.total_spent || 0).toFixed(0)}
                  </p>
                  <p className="text-xs text-slate-500">Total Spent</p>
                </div>
              </div>

              {/* Loyalty */}
              <div className="bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Star className="h-5 w-5 text-yellow-500" />
                  <span className="font-medium text-slate-900 dark:text-white">
                    Loyalty Points
                  </span>
                </div>
                <p className="text-2xl font-bold text-yellow-600">
                  {selectedCustomer.loyalty_points || 0}
                </p>
                <p className="text-xs text-slate-500">
                  {Math.floor((selectedCustomer.loyalty_points || 0) / 100)} rewards available
                </p>
              </div>

              {/* Recent Orders */}
              {selectedCustomer.recent_orders && selectedCustomer.recent_orders.length > 0 && (
                <div>
                  <h4 className="font-medium text-slate-900 dark:text-white mb-3">
                    Recent Orders
                  </h4>
                  <div className="space-y-2">
                    {selectedCustomer.recent_orders.slice(0, 3).map((order: any) => (
                      <div key={order.id} className="flex justify-between text-sm">
                        <span className="text-slate-600 dark:text-slate-400">
                          #{order.id}
                        </span>
                      <span className="font-medium">
                          ${Number(order.total || 0).toFixed(2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
              );
            })() : (
            <div className="text-center py-12 text-slate-500">
              <User className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>Select a customer</p>
              <p className="text-sm">Click on a customer to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
