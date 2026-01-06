'use client';

import { toast } from '@/components/ui/Toast';
import { useCreateOrder, useMenuItems, useRecommendations } from '@/hooks/useApi';
import { clsx } from 'clsx';
import {
    CreditCard,
    DollarSign,
    Loader2,
    Minus,
    Plus,
    ShoppingCart,
    Sparkles,
    Trash2,
    UtensilsCrossed,
} from 'lucide-react';
import { useState } from 'react';

interface CartItem {
  id: number;
  name: string;
  price: number;
  quantity: number;
}

export default function POSPage() {
  const [activeCategory, setActiveCategory] = useState('All');
  const [cart, setCart] = useState<CartItem[]>([]);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [tableNumber, setTableNumber] = useState<string>('');

  // Fetch real menu items
  const { data: menuItems, isLoading: menuLoading } = useMenuItems();
  const createOrder = useCreateOrder();
  const getRecommendationsMutation = useRecommendations();

  // Extract unique categories from menu items
  const categories: string[] = ['All', ...Array.from(new Set<string>(
    (menuItems || []).map((item: any) => (item.category as string) || 'Other')
  ))];

  // Filter items by category
  const filteredItems = activeCategory === 'All'
    ? (menuItems || [])
    : (menuItems || []).filter((item: any) => item.category === activeCategory);

  const addToCart = (item: any) => {
    setCart((prev) => {
      const existing = prev.find((i) => i.id === item.id);
      if (existing) {
        return prev.map((i) =>
          i.id === item.id ? { ...i, quantity: i.quantity + 1 } : i
        );
      }
      return [...prev, { id: item.id, name: item.name, price: Number(item.price), quantity: 1 }];
    });
  };

  const updateQuantity = (id: number, delta: number) => {
    setCart((prev) =>
      prev
        .map((item) =>
          item.id === id
            ? { ...item, quantity: Math.max(0, item.quantity + delta) }
            : item
        )
        .filter((item) => item.quantity > 0)
    );
  };

  const removeFromCart = (id: number) => {
    setCart((prev) => prev.filter((item) => item.id !== id));
  };

  const clearCart = () => {
    setCart([]);
    setRecommendations([]);
  };

  const subtotal = cart.reduce((sum, item) => sum + item.price * item.quantity, 0);
  const tax = subtotal * 0.08;
  const total = subtotal + tax;

  const fetchRecommendations = async () => {
    if (cart.length === 0) return;

    try {
      const itemIds = cart.map((item) => item.id);
      const result = await getRecommendationsMutation.mutateAsync({
        item_ids: itemIds,
        top_k: 3
      });

      if (result.recommendations) {
        setRecommendations(result.recommendations.map((r: any) => r.item_name || r.name));
      }
    } catch (error) {
      // Silently fail - recommendations are optional
      console.log('Could not fetch recommendations');
    }
  };

  const handlePayment = async (paymentMethod: 'card' | 'cash') => {
    if (cart.length === 0) return;

    try {
      const orderData = {
        table_number: tableNumber ? parseInt(tableNumber) : null,
        items: cart.map((item) => ({
          menu_item_id: item.id,
          quantity: item.quantity,
          unit_price: item.price,
        })),
        subtotal,
        tax,
        total,
        payment_method: paymentMethod,
      };

      await createOrder.mutateAsync(orderData);
      toast.success(`Order placed successfully! Total: $${total.toFixed(2)}`);
      clearCart();
      setTableNumber('');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to create order');
    }
  };

  return (
    <div className="flex flex-col lg:flex-row h-auto lg:h-[calc(100vh-8rem)] gap-4 lg:gap-6">
      {/* Menu Items */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <div>
            <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white">
              Point of Sale
            </h1>
            <p className="text-sm text-slate-500">
              {menuItems?.length || 0} items available
            </p>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-sm text-slate-500">Table #:</label>
            <input
              type="number"
              value={tableNumber}
              onChange={(e) => setTableNumber(e.target.value)}
              placeholder="--"
              className="w-16 px-2 py-1 text-center border border-slate-200 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-slate-800 dark:border-slate-700"
            />
          </div>
        </div>

        {/* Categories */}
        <div className="flex gap-2 mb-4 overflow-x-auto pb-2">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setActiveCategory(category)}
              className={clsx(
                'px-4 py-2 rounded-lg text-sm font-medium whitespace-nowrap transition-colors',
                activeCategory === category
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 border border-slate-200 dark:border-slate-700'
              )}
            >
              {category}
            </button>
          ))}
        </div>

        {/* Menu Grid */}
        <div className="flex-1 overflow-y-auto lg:max-h-none max-h-[50vh]">
          {menuLoading ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 sm:gap-4">
              {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                <div key={i} className="bg-white dark:bg-slate-800 rounded-xl p-3 sm:p-4 border border-slate-200 dark:border-slate-700 animate-pulse">
                  <div className="h-16 sm:h-24 bg-slate-200 dark:bg-slate-700 rounded-lg mb-2 sm:mb-3" />
                  <div className="h-4 bg-slate-200 dark:bg-slate-700 rounded w-3/4 mb-2" />
                  <div className="h-5 bg-slate-200 dark:bg-slate-700 rounded w-1/2" />
                </div>
              ))}
            </div>
          ) : filteredItems.length > 0 ? (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 sm:gap-4">
              {filteredItems.map((item: any) => (
                <button
                  key={item.id}
                  onClick={() => addToCart(item)}
                  className="bg-white dark:bg-slate-800 rounded-xl p-3 sm:p-4 border border-slate-200 dark:border-slate-700 text-left hover:shadow-md hover:border-blue-300 transition-all group"
                >
                  <div className="h-16 sm:h-24 bg-gradient-to-br from-slate-100 to-slate-200 dark:from-slate-700 dark:to-slate-600 rounded-lg mb-2 sm:mb-3 flex items-center justify-center group-hover:from-blue-50 group-hover:to-blue-100 transition-colors overflow-hidden relative">
                    {item.image_url ? (
                      <img 
                        src={item.image_url} 
                        alt={item.name} 
                        className="w-full h-full object-cover"
                        loading="lazy"
                        onError={(e) => { 
                          (e.target as HTMLImageElement).src = 'https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=400&h=300&fit=crop'; 
                        }}
                      />
                    ) : (
                      <UtensilsCrossed className="h-6 w-6 sm:h-8 sm:w-8 text-slate-400 group-hover:text-blue-500 transition-colors" />
                    )}
                  </div>
                  <h3 className="font-medium text-sm sm:text-base text-slate-900 dark:text-white truncate">
                    {item.name}
                  </h3>
                  <p className="text-xs sm:text-sm text-slate-500 truncate mb-1 hidden sm:block">
                    {item.description || 'Delicious menu item'}
                  </p>
                  <p className="text-base sm:text-lg font-bold text-blue-600">
                    ${Number(item.price).toFixed(2)}
                  </p>
                </button>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-64 text-slate-500">
              <UtensilsCrossed className="h-12 w-12 mb-3 opacity-50" />
              <p>No menu items found</p>
              <p className="text-sm">Add items through the Menu page</p>
            </div>
          )}
        </div>
      </div>

      {/* Cart */}
      <div className="w-full lg:w-96 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 flex flex-col shadow-sm mt-4 lg:mt-0 max-h-[60vh] lg:max-h-none">
        <div className="p-3 sm:p-4 border-b border-slate-200 dark:border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShoppingCart className="h-4 w-4 sm:h-5 sm:w-5 text-slate-500" />
            <h2 className="text-base sm:text-lg font-semibold text-slate-900 dark:text-white">
              Current Order
            </h2>
            <span className="lg:hidden text-xs bg-blue-100 text-blue-600 px-2 py-0.5 rounded-full">
              {cart.length}
            </span>
          </div>
          {cart.length > 0 && (
            <button
              onClick={clearCart}
              className="text-sm text-red-500 hover:text-red-600"
            >
              Clear
            </button>
          )}
        </div>

        {/* Cart Items */}
        <div className="flex-1 overflow-y-auto p-4 space-y-3">
          {cart.length === 0 ? (
            <div className="text-center py-12 text-slate-500">
              <ShoppingCart className="h-12 w-12 mx-auto mb-3 opacity-50" />
              <p>No items in cart</p>
              <p className="text-sm">Click menu items to add them</p>
            </div>
          ) : (
            cart.map((item) => (
              <div
                key={item.id}
                className="flex items-center justify-between py-3 border-b border-slate-100 dark:border-slate-700 last:border-0"
              >
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-slate-900 dark:text-white truncate">
                    {item.name}
                  </p>
                  <p className="text-sm text-slate-500">
                    ${item.price.toFixed(2)} × {item.quantity} = ${(item.price * item.quantity).toFixed(2)}
                  </p>
                </div>
                <div className="flex items-center gap-1 ml-2">
                  <button
                    onClick={() => updateQuantity(item.id, -1)}
                    className="p-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 transition-colors"
                  >
                    <Minus className="h-3 w-3" />
                  </button>
                  <span className="w-8 text-center font-medium text-sm">
                    {item.quantity}
                  </span>
                  <button
                    onClick={() => updateQuantity(item.id, 1)}
                    className="p-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 dark:bg-slate-700 dark:hover:bg-slate-600 transition-colors"
                  >
                    <Plus className="h-3 w-3" />
                  </button>
                  <button
                    onClick={() => removeFromCart(item.id)}
                    className="p-1.5 rounded-lg text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors ml-1"
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        {/* AI Recommendations */}
        {cart.length > 0 && (
          <div className="p-4 border-t border-slate-200 dark:border-slate-700 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20">
            <button
              onClick={fetchRecommendations}
              disabled={getRecommendationsMutation.isPending}
              className="flex items-center text-sm text-blue-600 font-medium mb-2 hover:text-blue-700"
            >
              {getRecommendationsMutation.isPending ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Sparkles className="h-4 w-4 mr-1" />
              )}
              Get AI Recommendations
            </button>
            {recommendations.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {recommendations.map((rec, i) => (
                  <span
                    key={i}
                    className="px-2 py-1 bg-white dark:bg-slate-800 rounded-full text-xs text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-600"
                  >
                    + {rec}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Totals */}
        <div className="p-4 border-t border-slate-200 dark:border-slate-700 space-y-2 bg-slate-50 dark:bg-slate-800/50">
          <div className="flex justify-between text-sm">
            <span className="text-slate-500">Subtotal</span>
            <span className="font-medium">${subtotal.toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-500">Tax (8%)</span>
            <span className="font-medium">${tax.toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-xl font-bold pt-2 border-t border-slate-200 dark:border-slate-600">
            <span>Total</span>
            <span className="text-blue-600">${total.toFixed(2)}</span>
          </div>
        </div>

        {/* Payment Buttons */}
        <div className="p-4 border-t border-slate-200 dark:border-slate-700 space-y-2">
          <button
            onClick={() => handlePayment('card')}
            disabled={cart.length === 0 || createOrder.isPending}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {createOrder.isPending ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <CreditCard className="h-5 w-5" />
            )}
            <span>Pay with Card</span>
          </button>
          <button
            onClick={() => handlePayment('cash')}
            disabled={cart.length === 0 || createOrder.isPending}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <DollarSign className="h-5 w-5" />
            <span>Pay with Cash</span>
          </button>
        </div>
      </div>
    </div>
  );
}
