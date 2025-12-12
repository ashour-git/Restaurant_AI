'use client';

import { useCustomer, useCustomers } from '@/hooks/useApi';
import { clsx } from 'clsx';
import {
    DollarSign,
    Mail,
    Phone,
    Search,
    ShoppingCart,
    Star,
    User,
    Users
} from 'lucide-react';
import { useState } from 'react';

export default function CustomersPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCustomerId, setSelectedCustomerId] = useState<number | null>(null);

  const { data: customers, isLoading } = useCustomers(
    searchQuery ? { search: searchQuery } : undefined
  );

  const { data: selectedCustomer } = useCustomer(selectedCustomerId || 0);

  const formatDate = (timestamp: string) => {
    if (!timestamp) return '--';
    return new Date(timestamp).toLocaleDateString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Users className="h-7 w-7 text-purple-500" />
            Customers
          </h1>
          <p className="text-slate-500">
            {customers?.length || 0} customers in your database
          </p>
        </div>
      </div>

      {/* Search */}
      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search customers by name, email, or phone..."
          className="w-full pl-10 pr-4 py-2.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Customer List */}
        <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          {isLoading ? (
            <div className="p-8">
              <div className="animate-pulse space-y-4">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div key={i} className="flex items-center gap-4">
                    <div className="h-12 w-12 bg-slate-200 dark:bg-slate-700 rounded-full" />
                    <div className="flex-1 space-y-2">
                      <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-1/3" />
                      <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded w-1/4" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (customers || []).length > 0 ? (
            <div className="divide-y divide-slate-200 dark:divide-slate-700">
              {(customers || []).map((customer: any) => (
                <button
                  key={customer.id}
                  onClick={() => setSelectedCustomerId(customer.id)}
                  className={clsx(
                    'w-full flex items-center gap-4 p-4 text-left hover:bg-slate-50 dark:hover:bg-slate-700/50 transition-colors',
                    selectedCustomerId === customer.id && 'bg-blue-50 dark:bg-blue-900/20'
                  )}
                >
                  <div className="h-12 w-12 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center text-white font-semibold">
                    {customer.name?.charAt(0) || 'C'}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-slate-900 dark:text-white truncate">
                      {customer.name || 'Unknown Customer'}
                    </p>
                    <p className="text-sm text-slate-500 truncate">
                      {customer.email || customer.phone || 'No contact info'}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="font-medium text-slate-900 dark:text-white">
                      ${Number(customer.total_spent || 0).toFixed(2)}
                    </p>
                    <p className="text-xs text-slate-500">
                      {customer.order_count || 0} orders
                    </p>
                  </div>
                </button>
              ))}
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
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-6">
          {selectedCustomer ? (
            <div className="space-y-6">
              {/* Profile */}
              <div className="text-center">
                <div className="h-20 w-20 bg-gradient-to-br from-purple-500 to-blue-500 rounded-full flex items-center justify-center text-white text-2xl font-bold mx-auto mb-4">
                  {selectedCustomer.name?.charAt(0) || 'C'}
                </div>
                <h3 className="text-xl font-semibold text-slate-900 dark:text-white">
                  {selectedCustomer.name || 'Unknown Customer'}
                </h3>
                <p className="text-sm text-slate-500">
                  Customer since {formatDate(selectedCustomer.created_at)}
                </p>
              </div>

              {/* Contact Info */}
              <div className="space-y-3">
                {selectedCustomer.email && (
                  <div className="flex items-center gap-3 text-sm">
                    <Mail className="h-4 w-4 text-slate-400" />
                    <span className="text-slate-600 dark:text-slate-300">
                      {selectedCustomer.email}
                    </span>
                  </div>
                )}
                {selectedCustomer.phone && (
                  <div className="flex items-center gap-3 text-sm">
                    <Phone className="h-4 w-4 text-slate-400" />
                    <span className="text-slate-600 dark:text-slate-300">
                      {selectedCustomer.phone}
                    </span>
                  </div>
                )}
              </div>

              {/* Stats */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-4 text-center">
                  <ShoppingCart className="h-6 w-6 text-blue-500 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
                    {selectedCustomer.order_count || 0}
                  </p>
                  <p className="text-xs text-slate-500">Total Orders</p>
                </div>
                <div className="bg-slate-50 dark:bg-slate-700/50 rounded-lg p-4 text-center">
                  <DollarSign className="h-6 w-6 text-green-500 mx-auto mb-2" />
                  <p className="text-2xl font-bold text-slate-900 dark:text-white">
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
          ) : (
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
