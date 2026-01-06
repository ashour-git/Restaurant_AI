/**
 * Custom hooks for data fetching using React Query.
 * Provides typed, cached data access with automatic refetching.
 */

import {
    analyticsApi,
    customersApi,
    inventoryApi,
    menuApi,
    mlApi,
    ordersApi,
} from '@/lib/api';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';

// ============== Menu Hooks ==============

export function useCategories() {
  return useQuery({
    queryKey: ['categories'],
    queryFn: () => menuApi.getCategories().then((res) => res.data),
  });
}

export function useMenuItems(params?: {
  category_id?: number;
  available_only?: boolean;
}) {
  return useQuery({
    queryKey: ['menuItems', params],
    queryFn: () => menuApi.getMenuItems(params).then((res) => res.data),
  });
}

export function useMenuItem(id: number) {
  return useQuery({
    queryKey: ['menuItem', id],
    queryFn: () => menuApi.getMenuItem(id).then((res) => res.data),
    enabled: !!id,
  });
}

export function useCreateMenuItem() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: any) => menuApi.createMenuItem(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['menuItems'] });
    },
  });
}

export function useUpdateMenuItem() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: any }) =>
      menuApi.updateMenuItem(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['menuItems'] });
    },
  });
}

export function useDeleteMenuItem() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => menuApi.deleteMenuItem(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['menuItems'] });
    },
  });
}

// ============== Orders Hooks ==============

export function useOrders(params?: { status?: string; date?: string }) {
  return useQuery({
    queryKey: ['orders', params],
    queryFn: () => ordersApi.getOrders(params).then((res) => res.data),
    refetchInterval: 10000, // Refetch every 10 seconds for live updates
  });
}

export function useOrder(id: number) {
  return useQuery({
    queryKey: ['order', id],
    queryFn: () => ordersApi.getOrder(id).then((res) => res.data),
    enabled: !!id,
  });
}

export function useCreateOrder() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: any) => ordersApi.createOrder(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });
}

export function useUpdateOrderStatus() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, status }: { id: number; status: string }) =>
      ordersApi.updateOrderStatus(id, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['orders'] });
    },
  });
}

// ============== Customers Hooks ==============

export function useCustomers(params?: { search?: string }) {
  return useQuery({
    queryKey: ['customers', params],
    queryFn: () => customersApi.getCustomers(params).then((res) => res.data),
  });
}

export function useCustomer(id: number) {
  return useQuery({
    queryKey: ['customer', id],
    queryFn: () => customersApi.getCustomer(id).then((res) => res.data),
    enabled: !!id,
  });
}

export function useCreateCustomer() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: any) => customersApi.createCustomer(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['customers'] });
    },
  });
}

// ============== Inventory Hooks ==============

export function useInventory() {
  return useQuery({
    queryKey: ['inventory'],
    queryFn: () => inventoryApi.getItems().then((res) => res.data),
  });
}

export function useLowStockAlerts() {
  return useQuery({
    queryKey: ['lowStockAlerts'],
    queryFn: () => inventoryApi.getLowStock().then((res) => res.data),
    refetchInterval: 60000, // Check every minute
  });
}

export function useUpdateInventory() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ id, data }: { id: number; data: any }) =>
      inventoryApi.updateItem(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['inventory'] });
      queryClient.invalidateQueries({ queryKey: ['lowStockAlerts'] });
    },
  });
}

// ============== Analytics Hooks ==============

export function useDashboardStats() {
  return useQuery({
    queryKey: ['dashboardStats'],
    queryFn: () => analyticsApi.getDashboard().then((res) => res.data),
    refetchInterval: 30000, // Refresh every 30 seconds
  });
}

export function useSalesReport(params: { start_date: string; end_date: string }) {
  return useQuery({
    queryKey: ['salesReport', params],
    queryFn: () => analyticsApi.getSalesReport(params).then((res) => res.data),
    enabled: !!params.start_date && !!params.end_date,
  });
}

export function useTopItems(limit?: number) {
  return useQuery({
    queryKey: ['topItems', limit],
    queryFn: () => analyticsApi.getTopItems({ limit }).then((res) => res.data),
  });
}

export function useAnalytics(params?: { period?: string }) {
  return useQuery({
    queryKey: ['analytics', params],
    queryFn: () => analyticsApi.getDashboard().then((res) => res.data),
    refetchInterval: 60000, // Refresh every minute
  });
}

// ============== ML Hooks ==============

export function useMLHealth() {
  return useQuery({
    queryKey: ['mlHealth'],
    queryFn: () => mlApi.getHealth().then((res) => res.data),
    refetchInterval: 60000,
  });
}

export function useForecast() {
  return useMutation({
    mutationFn: (data: { item_id?: number; days_ahead?: number }) =>
      mlApi.getForecast(data).then((res) => res.data),
  });
}

export function useRecommendations() {
  return useMutation({
    mutationFn: (data: { item_ids: number[]; top_k?: number }) =>
      mlApi.getRecommendations(data).then((res) => res.data),
  });
}

export function useChat() {
  return useMutation({
    mutationFn: (data: { message: string; use_rag?: boolean }) =>
      mlApi.chat(data).then((res) => res.data),
  });
}

export function useMenuSearch(query: string) {
  return useQuery({
    queryKey: ['menuSearch', query],
    queryFn: () => mlApi.searchMenu(query).then((res) => res.data),
    enabled: query.length > 2,
  });
}
