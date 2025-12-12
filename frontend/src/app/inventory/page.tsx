'use client';

import { useInventory, useUpdateInventory } from '@/hooks/useApi';
import { clsx } from 'clsx';
import {
    AlertTriangle,
    ArrowDown,
    ArrowUp,
    Box,
    Edit2,
    Package,
    Save,
    Search,
    X,
} from 'lucide-react';
import { useState } from 'react';

export default function InventoryPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState<'all' | 'low' | 'out'>('all');
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editQuantity, setEditQuantity] = useState<number>(0);

  const { data: inventory, isLoading, refetch } = useInventory();
  const updateInventory = useUpdateInventory();

  // Filter inventory based on search and status
  const filteredInventory = (inventory || []).filter((item: any) => {
    const matchesSearch =
      item.name?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.category?.toLowerCase().includes(searchQuery.toLowerCase());

    if (filterStatus === 'low') {
      return matchesSearch && item.quantity <= item.reorder_point && item.quantity > 0;
    } else if (filterStatus === 'out') {
      return matchesSearch && item.quantity === 0;
    }
    return matchesSearch;
  });

  // Calculate stats
  const stats = {
    total: (inventory || []).length,
    lowStock: (inventory || []).filter(
      (item: any) => item.quantity <= item.reorder_point && item.quantity > 0
    ).length,
    outOfStock: (inventory || []).filter((item: any) => item.quantity === 0).length,
  };

  const handleEdit = (item: any) => {
    setEditingId(item.id);
    setEditQuantity(item.quantity);
  };

  const handleSave = async (id: number) => {
    try {
      await updateInventory.mutateAsync({ id, data: { quantity: editQuantity } });
      setEditingId(null);
      refetch();
    } catch (error) {
      console.error('Failed to update inventory:', error);
    }
  };

  const getStockStatus = (item: any) => {
    if (item.quantity === 0) {
      return { label: 'Out of Stock', color: 'text-red-500 bg-red-100 dark:bg-red-900/30' };
    } else if (item.quantity <= item.reorder_point) {
      return { label: 'Low Stock', color: 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30' };
    }
    return { label: 'In Stock', color: 'text-green-600 bg-green-100 dark:bg-green-900/30' };
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Package className="h-7 w-7 text-amber-500" />
            Inventory
          </h1>
          <p className="text-slate-500">Track and manage your stock levels</p>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-center">
              <Box className="h-6 w-6 text-blue-500" />
            </div>
            <div>
              <p className="text-2xl font-bold text-slate-900 dark:text-white">{stats.total}</p>
              <p className="text-sm text-slate-500">Total Items</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg flex items-center justify-center">
              <AlertTriangle className="h-6 w-6 text-yellow-500" />
            </div>
            <div>
              <p className="text-2xl font-bold text-yellow-600">{stats.lowStock}</p>
              <p className="text-sm text-slate-500">Low Stock</p>
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-5">
          <div className="flex items-center gap-3">
            <div className="h-12 w-12 bg-red-100 dark:bg-red-900/30 rounded-lg flex items-center justify-center">
              <X className="h-6 w-6 text-red-500" />
            </div>
            <div>
              <p className="text-2xl font-bold text-red-500">{stats.outOfStock}</p>
              <p className="text-sm text-slate-500">Out of Stock</p>
            </div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search inventory items..."
            className="w-full pl-10 pr-4 py-2.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>
        <div className="flex gap-2">
          {(['all', 'low', 'out'] as const).map((status) => (
            <button
              key={status}
              onClick={() => setFilterStatus(status)}
              className={clsx(
                'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
                filterStatus === status
                  ? 'bg-blue-500 text-white'
                  : 'bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-600'
              )}
            >
              {status === 'all' ? 'All Items' : status === 'low' ? 'Low Stock' : 'Out of Stock'}
            </button>
          ))}
        </div>
      </div>

      {/* Inventory Table */}
      <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
        {isLoading ? (
          <div className="p-8">
            <div className="animate-pulse space-y-4">
              {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="h-14 bg-slate-200 dark:bg-slate-700 rounded" />
              ))}
            </div>
          </div>
        ) : filteredInventory.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-slate-50 dark:bg-slate-700/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                    Item
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">
                    Category
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-slate-500 uppercase tracking-wider">
                    Quantity
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-slate-500 uppercase tracking-wider">
                    Reorder Point
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-slate-500 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-4 py-3 text-center text-xs font-medium text-slate-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {filteredInventory.map((item: any) => {
                  const status = getStockStatus(item);
                  const isEditing = editingId === item.id;

                  return (
                    <tr key={item.id} className="hover:bg-slate-50 dark:hover:bg-slate-700/50">
                      <td className="px-4 py-4">
                        <div className="flex items-center gap-3">
                          <div className="h-10 w-10 bg-slate-100 dark:bg-slate-700 rounded-lg flex items-center justify-center">
                            <Package className="h-5 w-5 text-slate-500" />
                          </div>
                          <div>
                            <p className="font-medium text-slate-900 dark:text-white">
                              {item.name}
                            </p>
                            <p className="text-xs text-slate-500">{item.unit}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-4 text-sm text-slate-600 dark:text-slate-400">
                        {item.category || '--'}
                      </td>
                      <td className="px-4 py-4 text-center">
                        {isEditing ? (
                          <div className="flex items-center justify-center gap-2">
                            <button
                              onClick={() => setEditQuantity((prev) => Math.max(0, prev - 1))}
                              className="p-1 bg-slate-100 dark:bg-slate-700 rounded hover:bg-slate-200 dark:hover:bg-slate-600"
                            >
                              <ArrowDown className="h-4 w-4" />
                            </button>
                            <input
                              type="number"
                              value={editQuantity}
                              onChange={(e) => setEditQuantity(Math.max(0, Number(e.target.value)))}
                              className="w-20 text-center py-1 border rounded dark:bg-slate-700 dark:border-slate-600"
                            />
                            <button
                              onClick={() => setEditQuantity((prev) => prev + 1)}
                              className="p-1 bg-slate-100 dark:bg-slate-700 rounded hover:bg-slate-200 dark:hover:bg-slate-600"
                            >
                              <ArrowUp className="h-4 w-4" />
                            </button>
                          </div>
                        ) : (
                          <span
                            className={clsx(
                              'font-medium',
                              item.quantity <= item.reorder_point
                                ? 'text-red-500'
                                : 'text-slate-900 dark:text-white'
                            )}
                          >
                            {item.quantity}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-4 text-center text-sm text-slate-600 dark:text-slate-400">
                        {item.reorder_point}
                      </td>
                      <td className="px-4 py-4 text-center">
                        <span className={clsx('text-xs font-medium px-2 py-1 rounded-full', status.color)}>
                          {status.label}
                        </span>
                      </td>
                      <td className="px-4 py-4 text-center">
                        {isEditing ? (
                          <div className="flex items-center justify-center gap-2">
                            <button
                              onClick={() => handleSave(item.id)}
                              className="p-2 bg-green-100 text-green-600 rounded-lg hover:bg-green-200 transition-colors"
                            >
                              <Save className="h-4 w-4" />
                            </button>
                            <button
                              onClick={() => setEditingId(null)}
                              className="p-2 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 transition-colors"
                            >
                              <X className="h-4 w-4" />
                            </button>
                          </div>
                        ) : (
                          <button
                            onClick={() => handleEdit(item)}
                            className="p-2 bg-blue-100 dark:bg-blue-900/30 text-blue-600 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-900/50 transition-colors"
                          >
                            <Edit2 className="h-4 w-4" />
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12 text-slate-500">
            <Package className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>No inventory items found</p>
          </div>
        )}
      </div>
    </div>
  );
}
