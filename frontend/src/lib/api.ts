import axios, { AxiosError, AxiosRequestConfig } from 'axios';

// API URL configuration - uses Fly.io backend in production
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'https://restaurant-ai-ortjww.fly.dev/api/v1';

// Check if we're in a browser environment
const isBrowser = typeof window !== 'undefined';

// Determine if backend is available (for graceful degradation)
let backendAvailable = true;

// Create axios instance with optimized defaults
export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

// Simple in-memory cache for GET requests
const cache = new Map<string, { data: any; timestamp: number }>();
const CACHE_TTL = 5000; // 5 seconds cache

function getCacheKey(config: AxiosRequestConfig): string {
  return `${config.method}:${config.url}:${JSON.stringify(config.params || {})}`;
}

// Request interceptor for auth token and caching
api.interceptors.request.use((config) => {
  // Add auth token
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }

  // Check cache for GET requests
  if (config.method === 'get') {
    const cacheKey = getCacheKey(config);
    const cached = cache.get(cacheKey);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
      // Return cached response by throwing a special error
      const error = new Error('CACHE_HIT') as any;
      error.cachedData = cached.data;
      error.config = config;
      return Promise.reject(error);
    }
  }

  return config;
});

// Response interceptor for caching and error handling
api.interceptors.response.use(
  (response) => {
    backendAvailable = true;
    // Cache successful GET responses
    if (response.config.method === 'get') {
      const cacheKey = getCacheKey(response.config);
      cache.set(cacheKey, { data: response, timestamp: Date.now() });
    }
    return response;
  },
  (error) => {
    // Handle cache hits
    if (error.message === 'CACHE_HIT') {
      return Promise.resolve(error.cachedData);
    }

    // Handle network errors gracefully
    if (error.code === 'ERR_NETWORK' || error.code === 'ECONNREFUSED') {
      backendAvailable = false;
      console.warn('Backend unavailable, using demo mode');
      return Promise.reject(new Error('Backend unavailable'));
    }

    // Handle auth errors
    if (error.response?.status === 401 && isBrowser) {
      localStorage.removeItem('auth_token');
      // Don't redirect on API routes, only on protected pages
      if (!window.location.pathname.startsWith('/login')) {
        window.location.href = '/login';
      }
    }

    return Promise.reject(error);
  }
);

// Export backend availability check
export const isBackendAvailable = () => backendAvailable;

// Retry helper for failed requests
async function withRetry<T>(
  fn: () => Promise<T>,
  retries = 2,
  delay = 1000
): Promise<T> {
  try {
    return await fn();
  } catch (error) {
    if (retries > 0 && (error as AxiosError)?.code !== 'ECONNABORTED') {
      await new Promise(resolve => setTimeout(resolve, delay));
      return withRetry(fn, retries - 1, delay * 2);
    }
    throw error;
  }
}

// Invalidate cache for a specific pattern
export function invalidateCache(pattern?: string): void {
  if (!pattern) {
    cache.clear();
    return;
  }
  for (const key of cache.keys()) {
    if (key.includes(pattern)) {
      cache.delete(key);
    }
  }
}

// Auth API
export const authApi = {
  login: (email: string, password: string) => {
    const formData = new URLSearchParams();
    formData.append('username', email);
    formData.append('password', password);
    return api.post('/auth/login', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
  },
  register: (data: { email: string; password: string; first_name: string; last_name: string }) =>
    api.post('/auth/register', data),
  me: () => api.get('/auth/me'),
  refresh: () => api.post('/auth/refresh'),
  logout: () => {
    localStorage.removeItem('auth_token');
    invalidateCache();
  },
};

// Menu API
export const menuApi = {
  getCategories: () => api.get('/menu/categories'),
  getSubcategories: () => api.get('/menu/subcategories'),
  getMenuItems: (params?: { category_id?: number; available_only?: boolean }) =>
    api.get('/menu/items', { params }),
  getMenuItem: (id: number) => api.get(`/menu/items/${id}`),
  createMenuItem: (data: any) => api.post('/menu/items', data),
  updateMenuItem: (id: number, data: any) => api.put(`/menu/items/${id}`, data),
  deleteMenuItem: (id: number) => api.delete(`/menu/items/${id}`),
};

// Orders API
export const ordersApi = {
  getOrders: (params?: { status?: string; date?: string }) =>
    api.get('/orders', { params }),
  getOrder: (id: number) => api.get(`/orders/${id}`),
  createOrder: (data: any) => api.post('/orders', data),
  updateOrderStatus: (id: number, status: string) =>
    api.patch(`/orders/${id}/status`, { status }),
  cancelOrder: (id: number) => api.delete(`/orders/${id}`),
};

// Customers API
export const customersApi = {
  getCustomers: (params?: { search?: string }) =>
    api.get('/customers', { params }),
  getCustomer: (id: number) => api.get(`/customers/${id}`),
  createCustomer: (data: any) => api.post('/customers', data),
  updateCustomer: (id: number, data: any) => api.put(`/customers/${id}`, data),
};

// Inventory API
export const inventoryApi = {
  getItems: () => api.get('/inventory'),
  getItem: (id: number) => api.get(`/inventory/${id}`),
  createItem: (data: any) => api.post('/inventory', data),
  updateItem: (id: number, data: any) => api.put(`/inventory/${id}`, data),
  getLowStock: () => api.get('/inventory/alerts/low-stock'),
};

// Analytics API
export const analyticsApi = {
  getDashboard: (params?: { period?: string }) => 
    api.get('/analytics/dashboard', { params: { period: params?.period || 'week' } }),
  getPublicDashboard: () => api.get('/analytics/dashboard/public'),
  getSalesSummary: (period?: string) =>
    api.get('/analytics/sales/summary', { params: { period: period || 'week' } }),
  getSalesReport: (params: { start_date: string; end_date: string }) =>
    api.get('/analytics/sales/daily', { params }),
  getTopItems: (params?: { limit?: number }) =>
    api.get('/analytics/items/top-selling', { params }),
  getItemsByCategory: () => api.get('/analytics/items/by-category'),
};

// ML API
export const mlApi = {
  getHealth: () => api.get('/ml/health'),
  getForecast: (data: { item_id?: number; days_ahead?: number }) =>
    api.post('/ml/forecast', data),
  getRecommendations: (data: { item_ids: number[]; top_k?: number }) =>
    api.post('/ml/recommend', data),
  chat: (data: { message: string; use_rag?: boolean }) =>
    api.post('/ml/chat', data),
  searchMenu: (query: string) =>
    api.get('/ml/menu-search', { params: { query } }),
};
