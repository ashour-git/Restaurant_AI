'use client';

import { OrderStatusBadge } from '@/components/ui';
import { toast } from '@/components/ui/Toast';
import { useOrders, useUpdateOrderStatus } from '@/hooks/useApi';
import { clsx } from 'clsx';
import {
    ChefHat,
    Clock,
    Eye,
    RefreshCw,
    Search,
    X
} from 'lucide-react';
import { useState } from 'react';

const ORDER_STATUSES = [
  { value: 'all', label: 'All Orders' },
  { value: 'pending', label: 'Pending' },
  { value: 'confirmed', label: 'Confirmed' },
  { value: 'preparing', label: 'Preparing' },
  { value: 'ready', label: 'Ready' },
  { value: 'completed', label: 'Completed' },
  { value: 'cancelled', label: 'Cancelled' },
];

export default function OrdersPage() {
  const [statusFilter, setStatusFilter] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedOrder, setSelectedOrder] = useState<any>(null);

  const { data: orders, isLoading, refetch } = useOrders(
    statusFilter !== 'all' ? { status: statusFilter } : undefined
  );
  const updateStatus = useUpdateOrderStatus();

  const filteredOrders = (orders || []).filter((order: any) => {
    if (!searchQuery) return true;
    return (
      order.id.toString().includes(searchQuery) ||
      order.customer_name?.toLowerCase().includes(searchQuery.toLowerCase())
    );
  });

  const handleStatusChange = async (orderId: number, newStatus: string) => {
    try {
      await updateStatus.mutateAsync({ id: orderId, status: newStatus });
      toast.success(`Order #${orderId} status updated to ${newStatus}`);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to update status');
    }
  };

  const formatTime = (timestamp: string) => {
    if (!timestamp) return '--';
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const formatDate = (timestamp: string) => {
    if (!timestamp) return '--';
    const date = new Date(timestamp);
    return date.toLocaleDateString();
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <ChefHat className="h-7 w-7 text-orange-500" />
            Orders
          </h1>
          <p className="text-slate-500">
            Manage and track all restaurant orders
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 dark:bg-slate-800 dark:border-slate-700 dark:text-slate-300"
        >
          <RefreshCw className={clsx('h-4 w-4', isLoading && 'animate-spin')} />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        {/* Search */}
        <div className="relative flex-1 max-w-xs">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search by order # or customer..."
            className="w-full pl-10 pr-4 py-2.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Status Filter */}
        <div className="flex gap-2 overflow-x-auto">
          {ORDER_STATUSES.map((status) => (
            <button
              key={status.value}
              onClick={() => setStatusFilter(status.value)}
              className={clsx(
                'px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors',
                statusFilter === status.value
                  ? 'bg-blue-600 text-white'
                  : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-50'
              )}
            >
              {status.label}
            </button>
          ))}
        </div>
      </div>

      {/* Orders Table */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        {isLoading ? (
          <div className="p-8">
            <div className="animate-pulse space-y-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="flex items-center gap-4">
                  <div className="h-10 w-20 bg-slate-200 dark:bg-slate-700 rounded" />
                  <div className="h-10 flex-1 bg-slate-200 dark:bg-slate-700 rounded" />
                  <div className="h-10 w-24 bg-slate-200 dark:bg-slate-700 rounded" />
                  <div className="h-10 w-20 bg-slate-200 dark:bg-slate-700 rounded" />
                </div>
              ))}
            </div>
          </div>
        ) : filteredOrders.length > 0 ? (
          <table className="w-full">
            <thead className="bg-slate-50 dark:bg-slate-900/50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Order
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Table / Customer
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Items
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Total
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Time
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-slate-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
              {filteredOrders.map((order: any) => (
                <tr key={order.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/50">
                  <td className="px-4 py-4 whitespace-nowrap">
                    <span className="font-semibold text-slate-900 dark:text-white">
                      #{order.id}
                    </span>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <div>
                      <p className="font-medium text-slate-900 dark:text-white">
                        {order.table_number ? `Table ${order.table_number}` : 'Takeaway'}
                      </p>
                      {order.customer_name && (
                        <p className="text-sm text-slate-500">{order.customer_name}</p>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-sm text-slate-600 dark:text-slate-400">
                    {order.items?.length || 0} items
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <span className="font-semibold text-slate-900 dark:text-white">
                      ${Number(order.total || 0).toFixed(2)}
                    </span>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <OrderStatusBadge status={order.status || 'pending'} />
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <div className="flex items-center gap-1 text-sm text-slate-500">
                      <Clock className="h-4 w-4" />
                      {formatTime(order.created_at)}
                    </div>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-right">
                    <button
                      onClick={() => setSelectedOrder(order)}
                      className="p-2 text-slate-500 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="text-center py-12 text-slate-500">
            <ChefHat className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No orders found</p>
            <p className="text-sm">Orders will appear here when customers place them</p>
          </div>
        )}
      </div>

      {/* Order Detail Modal */}
      {selectedOrder && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => setSelectedOrder(null)}
          />
          <div className="relative bg-white dark:bg-slate-800 rounded-xl shadow-xl max-w-lg w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
                Order #{selectedOrder.id}
              </h2>
              <button
                onClick={() => setSelectedOrder(null)}
                className="p-1 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700"
              >
                <X className="h-5 w-5 text-slate-500" />
              </button>
            </div>

            <div className="p-4 space-y-4">
              {/* Order Info */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-slate-500">Table</p>
                  <p className="font-medium text-slate-900 dark:text-white">
                    {selectedOrder.table_number ? `Table ${selectedOrder.table_number}` : 'Takeaway'}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-slate-500">Status</p>
                  <OrderStatusBadge status={selectedOrder.status || 'pending'} />
                </div>
                <div>
                  <p className="text-sm text-slate-500">Created</p>
                  <p className="font-medium text-slate-900 dark:text-white">
                    {formatDate(selectedOrder.created_at)} {formatTime(selectedOrder.created_at)}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-slate-500">Total</p>
                  <p className="font-bold text-lg text-blue-600">
                    ${Number(selectedOrder.total || 0).toFixed(2)}
                  </p>
                </div>
              </div>

              {/* Items */}
              <div>
                <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Items</p>
                <div className="space-y-2">
                  {(selectedOrder.items || []).map((item: any, i: number) => (
                    <div key={i} className="flex justify-between py-2 border-b border-slate-100 dark:border-slate-700 last:border-0">
                      <div>
                        <p className="font-medium text-slate-900 dark:text-white">
                          {item.menu_item?.name || `Item #${item.menu_item_id}`}
                        </p>
                        <p className="text-sm text-slate-500">Qty: {item.quantity}</p>
                      </div>
                      <p className="font-medium">
                        ${(Number(item.unit_price || 0) * item.quantity).toFixed(2)}
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Status Actions */}
              <div>
                <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Update Status</p>
                <div className="flex flex-wrap gap-2">
                  {['confirmed', 'preparing', 'ready', 'completed', 'cancelled'].map((status) => (
                    <button
                      key={status}
                      onClick={() => handleStatusChange(selectedOrder.id, status)}
                      disabled={updateStatus.isPending || selectedOrder.status === status}
                      className={clsx(
                        'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
                        selectedOrder.status === status
                          ? 'bg-blue-100 text-blue-700 cursor-not-allowed'
                          : 'bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-700 dark:text-slate-300 dark:hover:bg-slate-600'
                      )}
                    >
                      {status.charAt(0).toUpperCase() + status.slice(1)}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
