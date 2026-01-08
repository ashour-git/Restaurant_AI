'use client';

import { toast } from '@/components/ui/Toast';
import { useCategories, useSubcategories, useCreateMenuItem, useDeleteMenuItem, useMenuItems, useUpdateMenuItem } from '@/hooks/useApi';
import { clsx } from 'clsx';
import {
    AlertTriangle,
    Check,
    Download,
    Edit2,
    FileSpreadsheet,
    ImageIcon,
    Loader2,
    MoreVertical,
    Plus,
    Search,
    Trash2,
    Upload,
    UtensilsCrossed,
    X,
} from 'lucide-react';
import { useCallback, useRef, useState } from 'react';

interface MenuItem {
  id: number;
  name: string;
  description: string;
  price: number;
  cost: number;
  category: string;
  subcategory_id: number;
  is_active: boolean;
  image_url: string | null;
}

interface MenuItemFormData {
  name: string;
  description: string;
  price: string;
  cost: string;
  subcategory_id: string;
  is_active: boolean;
  image_url: string;
}

interface BulkImportItem {
  name: string;
  description: string;
  price: number;
  cost: number;
  category: string;
  is_active: boolean;
  image_url: string;
  isValid: boolean;
  errors: string[];
}

const defaultFormData: MenuItemFormData = {
  name: '',
  description: '',
  price: '',
  cost: '',
  subcategory_id: '',
  is_active: true,
  image_url: '',
};

export default function MenuPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState('All');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingItem, setEditingItem] = useState<MenuItem | null>(null);
  const [formData, setFormData] = useState<MenuItemFormData>(defaultFormData);
  const [activeDropdown, setActiveDropdown] = useState<number | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isUploadingImage, setIsUploadingImage] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Bulk import state
  const [isBulkModalOpen, setIsBulkModalOpen] = useState(false);
  const [bulkItems, setBulkItems] = useState<BulkImportItem[]>([]);
  const [bulkImporting, setBulkImporting] = useState(false);
  const [bulkProgress, setBulkProgress] = useState({ current: 0, total: 0 });

  // API hooks
  const { data: menuItems, isLoading } = useMenuItems();
  const { data: categories } = useCategories();
  const { data: subcategories } = useSubcategories();
  const createMenuItem = useCreateMenuItem();
  const updateMenuItem = useUpdateMenuItem();
  const deleteMenuItem = useDeleteMenuItem();

  // Extract unique categories from items or use fetched categories
  const allCategories: string[] = ['All', ...Array.from(new Set<string>(
    (menuItems || []).map((item: any) => (item.category as string) || 'Other')
  ))];

  // Filter items
  const filteredItems = (menuItems || []).filter((item: any) => {
    const matchesSearch = item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.description?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = activeCategory === 'All' || item.category === activeCategory;
    return matchesSearch && matchesCategory;
  });

  const openCreateModal = () => {
    setEditingItem(null);
    setFormData(defaultFormData);
    setIsModalOpen(true);
  };

  const openEditModal = (item: MenuItem) => {
    setEditingItem(item);
    setFormData({
      name: item.name,
      description: item.description || '',
      price: String(item.price),
      cost: String(item.cost || ''),
      subcategory_id: String(item.subcategory_id || ''),
      is_active: item.is_active !== false,
      image_url: item.image_url || '',
    });
    setIsModalOpen(true);
    setActiveDropdown(null);
  };

  const closeModal = () => {
    setIsModalOpen(false);
    setEditingItem(null);
    setFormData(defaultFormData);
    setIsSubmitting(false);
    setImagePreview(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.subcategory_id) {
      toast.error('Please select a category');
      return;
    }

    setIsSubmitting(true);
    
    const data = {
      name: formData.name,
      description: formData.description || null,
      price: parseFloat(formData.price),
      cost: parseFloat(formData.cost) || parseFloat(formData.price) * 0.4,
      subcategory_id: parseInt(formData.subcategory_id),
      is_active: formData.is_active,
      image_url: formData.image_url || null,
    };

    try {
      if (editingItem) {
        await updateMenuItem.mutateAsync({ id: editingItem.id, data });
        toast.success('Menu item updated successfully');
      } else {
        await createMenuItem.mutateAsync(data);
        toast.success('Menu item created successfully');
      }
      closeModal();
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to save menu item');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = async (id: number) => {
    if (!confirm('Are you sure you want to delete this item?')) return;

    try {
      await deleteMenuItem.mutateAsync(id);
      toast.success('Menu item deleted');
      setActiveDropdown(null);
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to delete item');
    }
  };

  // ========== BULK IMPORT FUNCTIONS ==========
  const downloadTemplate = () => {
    const template = `name,description,price,cost,category,is_active,image_url
"Margherita Pizza","Classic pizza with tomato, mozzarella, and basil",14.99,5.50,"Main Course",true,""
"Caesar Salad","Fresh romaine lettuce with Caesar dressing",9.99,3.00,"Starters",true,""
"Chocolate Cake","Rich dark chocolate layer cake",8.99,2.50,"Desserts",true,""`;
    
    const blob = new Blob([template], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'menu_import_template.csv';
    a.click();
    window.URL.revokeObjectURL(url);
    toast.success('Template downloaded!');
  };

  const parseCSV = (text: string): BulkImportItem[] => {
    const lines = text.trim().split('\n');
    if (lines.length < 2) return [];
    
    const headers = lines[0].toLowerCase().split(',').map(h => h.replace(/"/g, '').trim());
    const items: BulkImportItem[] = [];
    
    for (let i = 1; i < lines.length; i++) {
      const values: string[] = [];
      let current = '';
      let inQuotes = false;
      
      for (const char of lines[i]) {
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          values.push(current.trim());
          current = '';
        } else {
          current += char;
        }
      }
      values.push(current.trim());
      
      const errors: string[] = [];
      const name = values[headers.indexOf('name')] || '';
      const description = values[headers.indexOf('description')] || '';
      const priceStr = values[headers.indexOf('price')] || '0';
      const costStr = values[headers.indexOf('cost')] || '0';
      const category = values[headers.indexOf('category')] || '';
      const isActiveStr = values[headers.indexOf('is_active')] || 'true';
      const imageUrl = values[headers.indexOf('image_url')] || '';
      
      const price = parseFloat(priceStr);
      const cost = parseFloat(costStr);
      const isActive = isActiveStr.toLowerCase() !== 'false';
      
      if (!name) errors.push('Name is required');
      if (isNaN(price) || price <= 0) errors.push('Invalid price');
      if (!category) errors.push('Category is required');
      
      items.push({
        name,
        description,
        price: isNaN(price) ? 0 : price,
        cost: isNaN(cost) ? price * 0.4 : cost,
        category,
        is_active: isActive,
        image_url: imageUrl,
        isValid: errors.length === 0,
        errors,
      });
    }
    
    return items;
  };

  const handleFileUpload = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      const items = parseCSV(text);
      setBulkItems(items);
      
      const validCount = items.filter(i => i.isValid).length;
      toast.info(`Found ${items.length} items (${validCount} valid)`);
    };
    reader.readAsText(file);
    e.target.value = ''; // Reset input
  }, []);

  const handleBulkImport = async () => {
    const validItems = bulkItems.filter(i => i.isValid);
    if (validItems.length === 0) {
      toast.error('No valid items to import');
      return;
    }

    setBulkImporting(true);
    setBulkProgress({ current: 0, total: validItems.length });

    // Get subcategory mapping from categories
    const categoryMap: Record<string, number> = {};
    (subcategories || []).forEach((sub: any) => {
      const catName = sub.name.toLowerCase();
      categoryMap[catName] = sub.id;
    });

    let successCount = 0;
    let failCount = 0;

    for (let i = 0; i < validItems.length; i++) {
      const item = validItems[i];
      setBulkProgress({ current: i + 1, total: validItems.length });

      // Find matching subcategory
      const catLower = item.category.toLowerCase();
      let subcategoryId = categoryMap[catLower];
      
      // Try partial match if exact not found
      if (!subcategoryId) {
        const matchKey = Object.keys(categoryMap).find(k => 
          k.includes(catLower) || catLower.includes(k)
        );
        subcategoryId = matchKey ? categoryMap[matchKey] : Object.values(categoryMap)[0];
      }

      try {
        await createMenuItem.mutateAsync({
          name: item.name,
          description: item.description || null,
          price: item.price,
          cost: item.cost || item.price * 0.4,
          subcategory_id: subcategoryId,
          is_active: item.is_active,
          image_url: item.image_url || null,
        });
        successCount++;
      } catch (error) {
        failCount++;
        console.error(`Failed to import ${item.name}:`, error);
      }
    }

    setBulkImporting(false);
    setIsBulkModalOpen(false);
    setBulkItems([]);
    
    if (failCount === 0) {
      toast.success(`Successfully imported ${successCount} items!`);
    } else {
      toast.warning(`Imported ${successCount} items, ${failCount} failed`);
    }
  };

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white">
            Menu Management
          </h1>
          <p className="text-sm text-slate-500">
            {menuItems?.length || 0} items in your menu
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsBulkModalOpen(true)}
            className="flex items-center justify-center gap-2 px-4 py-2 border border-slate-200 dark:border-slate-600 text-slate-700 dark:text-slate-300 rounded-lg font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
          >
            <Upload className="h-4 w-4" />
            <span className="hidden sm:inline">Import</span>
          </button>
          <button
            onClick={openCreateModal}
            className="flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors"
          >
            <Plus className="h-5 w-5" />
            <span className="sm:inline">Add Item</span>
          </button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col gap-3 sm:gap-4">
        {/* Search */}
        <div className="relative w-full sm:max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search menu items..."
            className="w-full pl-10 pr-4 py-2.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
        </div>

        {/* Categories */}
        <div className="flex gap-2 overflow-x-auto pb-1 -mx-3 px-3 sm:mx-0 sm:px-0">
          {allCategories.map((category) => (
            <button
              key={category}
              onClick={() => setActiveCategory(category)}
              className={clsx(
                'px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs sm:text-sm font-medium whitespace-nowrap transition-colors',
                activeCategory === category
                  ? 'bg-blue-600 text-white'
                  : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700'
              )}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      {/* Menu Items Grid */}
      {isLoading ? (
        <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 sm:gap-4">
          {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
            <div key={i} className="bg-white dark:bg-slate-800 rounded-xl p-3 sm:p-4 border border-slate-200 dark:border-slate-700 animate-pulse">
              <div className="h-20 sm:h-32 bg-slate-200 dark:bg-slate-700 rounded-lg mb-2 sm:mb-3" />
              <div className="h-4 sm:h-5 bg-slate-200 dark:bg-slate-700 rounded w-3/4 mb-2" />
              <div className="h-3 sm:h-4 bg-slate-200 dark:bg-slate-700 rounded w-1/2" />
            </div>
          ))}
        </div>
      ) : filteredItems.length > 0 ? (
        <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 sm:gap-4">
          {filteredItems.map((item: any) => (
            <div
              key={item.id}
              className={clsx(
                'bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden',
                !item.is_active && 'opacity-60'
              )}
            >
              {/* Food Image */}
              <div className="h-32 bg-gradient-to-br from-slate-100 to-slate-200 dark:from-slate-700 dark:to-slate-600 flex items-center justify-center relative overflow-hidden">
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
                  <UtensilsCrossed className="h-10 w-10 text-slate-400" />
                )}
                {!item.is_active && (
                  <span className="absolute top-2 left-2 px-2 py-0.5 bg-red-500 text-white text-xs rounded-full">
                    Unavailable
                  </span>
                )}
                {/* Actions dropdown */}
                <div className="absolute top-2 right-2">
                  <button
                    onClick={() => setActiveDropdown(activeDropdown === item.id ? null : item.id)}
                    className="p-1.5 bg-white/80 dark:bg-slate-800/80 rounded-lg hover:bg-white dark:hover:bg-slate-700"
                  >
                    <MoreVertical className="h-4 w-4 text-slate-600" />
                  </button>
                  {activeDropdown === item.id && (
                    <div className="absolute right-0 mt-1 w-36 bg-white dark:bg-slate-800 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 py-1 z-10">
                      <button
                        onClick={() => openEditModal(item)}
                        className="w-full px-3 py-2 text-left text-sm text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2"
                      >
                        <Edit2 className="h-4 w-4" />
                        Edit
                      </button>
                      <button
                        onClick={() => handleDelete(item.id)}
                        className="w-full px-3 py-2 text-left text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 flex items-center gap-2"
                      >
                        <Trash2 className="h-4 w-4" />
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {/* Content */}
              <div className="p-4">
                <div className="flex items-start justify-between mb-1">
                  <h3 className="font-semibold text-slate-900 dark:text-white">
                    {item.name}
                  </h3>
                </div>
                <p className="text-sm text-slate-500 mb-2 line-clamp-2">
                  {item.description || 'No description'}
                </p>
                <div className="flex items-center justify-between">
                  <span className="text-lg font-bold text-blue-600">
                    ${Number(item.price).toFixed(2)}
                  </span>
                  <span className="text-xs px-2 py-1 bg-slate-100 dark:bg-slate-700 rounded-full text-slate-600 dark:text-slate-400">
                    {item.category || 'Other'}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 text-slate-500">
          <UtensilsCrossed className="h-12 w-12 mx-auto mb-3 opacity-50" />
          <p>No menu items found</p>
          <button
            onClick={openCreateModal}
            className="mt-4 text-blue-600 hover:text-blue-700 font-medium"
          >
            Add your first menu item
          </button>
        </div>
      )}

      {/* Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={closeModal}
          />
          <div className="relative bg-white dark:bg-slate-800 rounded-xl shadow-xl max-w-md w-full mx-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700">
              <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
                {editingItem ? 'Edit Menu Item' : 'Add Menu Item'}
              </h2>
              <button
                onClick={closeModal}
                className="p-1 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700"
              >
                <X className="h-5 w-5 text-slate-500" />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="p-4 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Name *
                </label>
                <input
                  type="text"
                  required
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-3 py-2 border border-slate-200 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., Margherita Pizza"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Description
                </label>
                <textarea
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  rows={2}
                  className="w-full px-3 py-2 border border-slate-200 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Describe the dish..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  Category *
                </label>
                <select
                  required
                  value={formData.subcategory_id}
                  onChange={(e) => setFormData({ ...formData, subcategory_id: e.target.value })}
                  className="w-full px-3 py-2 border border-slate-200 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="">Select a category</option>
                  {(subcategories || []).map((sub: any) => (
                    <option key={sub.id} value={sub.id}>
                      {sub.name} {sub.category_name ? `(${sub.category_name})` : ''}
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                    Price ($) *
                  </label>
                  <input
                    type="number"
                    required
                    min="0"
                    step="0.01"
                    value={formData.price}
                    onChange={(e) => setFormData({ ...formData, price: e.target.value })}
                    className="w-full px-3 py-2 border border-slate-200 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="19.99"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                    Cost ($)
                  </label>
                  <input
                    type="number"
                    min="0"
                    step="0.01"
                    value={formData.cost}
                    onChange={(e) => setFormData({ ...formData, cost: e.target.value })}
                    className="w-full px-3 py-2 border border-slate-200 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="8.00"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                  <span className="flex items-center gap-1">
                    <ImageIcon className="h-4 w-4" />
                    Menu Item Image
                  </span>
                </label>
                
                {/* Image Preview */}
                {(imagePreview || formData.image_url) && (
                  <div className="mb-3 relative">
                    <img
                      src={imagePreview || formData.image_url}
                      alt="Preview"
                      className="w-full h-32 object-cover rounded-lg border border-slate-200 dark:border-slate-600"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                    <button
                      type="button"
                      onClick={() => {
                        setImagePreview(null);
                        setFormData({ ...formData, image_url: '' });
                        if (fileInputRef.current) fileInputRef.current.value = '';
                      }}
                      className="absolute top-2 right-2 p-1 bg-red-500 text-white rounded-full hover:bg-red-600"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                )}
                
                {/* Upload Button */}
                <div className="flex gap-2">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={async (e) => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      
                      // Validate file size (max 5MB)
                      if (file.size > 5 * 1024 * 1024) {
                        toast.error('Image must be less than 5MB');
                        return;
                      }
                      
                      setIsUploadingImage(true);
                      
                      // Create preview
                      const reader = new FileReader();
                      reader.onload = (event) => {
                        const base64 = event.target?.result as string;
                        setImagePreview(base64);
                        // For now, store as data URL (in production, upload to cloud storage)
                        setFormData({ ...formData, image_url: base64 });
                        setIsUploadingImage(false);
                      };
                      reader.onerror = () => {
                        toast.error('Failed to read image');
                        setIsUploadingImage(false);
                      };
                      reader.readAsDataURL(file);
                    }}
                    className="hidden"
                    id="image-upload"
                  />
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploadingImage}
                    className="flex-1 flex items-center justify-center gap-2 px-3 py-2 border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-lg hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors disabled:opacity-50"
                  >
                    {isUploadingImage ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Upload className="h-4 w-4" />
                    )}
                    <span className="text-sm text-slate-600 dark:text-slate-400">
                      {isUploadingImage ? 'Processing...' : 'Upload Image'}
                    </span>
                  </button>
                </div>
                
                {/* Or use URL */}
                <div className="mt-2">
                  <p className="text-xs text-slate-500 mb-1">Or paste image URL:</p>
                  <input
                    type="url"
                    value={formData.image_url.startsWith('data:') ? '' : formData.image_url}
                    onChange={(e) => {
                      setFormData({ ...formData, image_url: e.target.value });
                      setImagePreview(null);
                    }}
                    className="w-full px-3 py-2 text-sm border border-slate-200 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="https://images.unsplash.com/..."
                  />
                </div>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="is_active"
                  checked={formData.is_active}
                  onChange={(e) => setFormData({ ...formData, is_active: e.target.checked })}
                  className="h-4 w-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
                />
                <label htmlFor="is_active" className="text-sm text-slate-700 dark:text-slate-300">
                  Available for ordering
                </label>
              </div>

              <div className="flex gap-3 pt-4">
                <button
                  type="button"
                  onClick={closeModal}
                  disabled={isSubmitting}
                  className="flex-1 px-4 py-2 border border-slate-200 dark:border-slate-600 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isSubmitting}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isSubmitting && (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  )}
                  {editingItem ? 'Update' : 'Create'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Bulk Import Modal */}
      {isBulkModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div
            className="absolute inset-0 bg-black/50"
            onClick={() => !bulkImporting && setIsBulkModalOpen(false)}
          />
          <div className="relative bg-white dark:bg-slate-800 rounded-xl shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-hidden flex flex-col">
            <div className="flex items-center justify-between p-4 border-b border-slate-200 dark:border-slate-700">
              <div className="flex items-center gap-2">
                <FileSpreadsheet className="h-5 w-5 text-blue-600" />
                <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
                  Bulk Import Menu Items
                </h2>
              </div>
              <button
                onClick={() => !bulkImporting && setIsBulkModalOpen(false)}
                disabled={bulkImporting}
                className="p-1 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 disabled:opacity-50"
              >
                <X className="h-5 w-5 text-slate-500" />
              </button>
            </div>

            <div className="p-4 flex-1 overflow-y-auto">
              {bulkItems.length === 0 ? (
                <div className="space-y-4">
                  <div className="text-center py-8 border-2 border-dashed border-slate-200 dark:border-slate-600 rounded-xl">
                    <Upload className="h-12 w-12 text-slate-400 mx-auto mb-3" />
                    <p className="text-slate-600 dark:text-slate-400 mb-2">
                      Upload a CSV file with your menu items
                    </p>
                    <p className="text-sm text-slate-500 mb-4">
                      Include: name, description, price, cost, category, is_active, image_url
                    </p>
                    <div className="flex items-center justify-center gap-3">
                      <label className="cursor-pointer px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors">
                        <input
                          type="file"
                          accept=".csv"
                          onChange={handleFileUpload}
                          className="hidden"
                        />
                        Select CSV File
                      </label>
                      <button
                        onClick={downloadTemplate}
                        className="flex items-center gap-2 px-4 py-2 border border-slate-200 dark:border-slate-600 text-slate-700 dark:text-slate-300 rounded-lg font-medium hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
                      >
                        <Download className="h-4 w-4" />
                        Template
                      </button>
                    </div>
                  </div>
                  
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-4">
                    <h3 className="font-medium text-blue-700 dark:text-blue-400 mb-2">CSV Format Guide</h3>
                    <ul className="text-sm text-blue-600 dark:text-blue-400 space-y-1">
                      <li>• First row must contain headers</li>
                      <li>• Required: name, price, category</li>
                      <li>• Use quotes for text with commas</li>
                      <li>• is_active: true or false</li>
                    </ul>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Summary */}
                  <div className="flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                    <div className="flex items-center gap-4">
                      <span className="text-sm text-slate-600 dark:text-slate-400">
                        Total: <strong>{bulkItems.length}</strong> items
                      </span>
                      <span className="text-sm text-green-600">
                        <Check className="h-4 w-4 inline mr-1" />
                        Valid: <strong>{bulkItems.filter(i => i.isValid).length}</strong>
                      </span>
                      {bulkItems.some(i => !i.isValid) && (
                        <span className="text-sm text-amber-600">
                          <AlertTriangle className="h-4 w-4 inline mr-1" />
                          Errors: <strong>{bulkItems.filter(i => !i.isValid).length}</strong>
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => setBulkItems([])}
                      className="text-sm text-slate-500 hover:text-slate-700"
                    >
                      Clear
                    </button>
                  </div>

                  {/* Progress bar */}
                  {bulkImporting && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-slate-600 dark:text-slate-400">Importing...</span>
                        <span className="font-medium">{bulkProgress.current} / {bulkProgress.total}</span>
                      </div>
                      <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-blue-600 transition-all duration-300"
                          style={{ width: `${(bulkProgress.current / bulkProgress.total) * 100}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* Items preview */}
                  <div className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
                    <table className="w-full text-sm">
                      <thead className="bg-slate-50 dark:bg-slate-700">
                        <tr>
                          <th className="text-left px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Status</th>
                          <th className="text-left px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Name</th>
                          <th className="text-left px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Price</th>
                          <th className="text-left px-3 py-2 font-medium text-slate-600 dark:text-slate-400">Category</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100 dark:divide-slate-700">
                        {bulkItems.slice(0, 10).map((item, i) => (
                          <tr key={i} className={clsx(
                            !item.isValid && "bg-red-50 dark:bg-red-900/10"
                          )}>
                            <td className="px-3 py-2">
                              {item.isValid ? (
                                <Check className="h-4 w-4 text-green-500" />
                              ) : (
                                <span title={item.errors.join(', ')}>
                                  <AlertTriangle className="h-4 w-4 text-amber-500" />
                                </span>
                              )}
                            </td>
                            <td className="px-3 py-2 text-slate-900 dark:text-white font-medium">
                              {item.name || <span className="text-red-500 italic">Missing</span>}
                            </td>
                            <td className="px-3 py-2 text-slate-600 dark:text-slate-400">
                              ${item.price.toFixed(2)}
                            </td>
                            <td className="px-3 py-2 text-slate-600 dark:text-slate-400">
                              {item.category || <span className="text-red-500 italic">Missing</span>}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    {bulkItems.length > 10 && (
                      <div className="px-3 py-2 bg-slate-50 dark:bg-slate-700 text-sm text-slate-500 text-center">
                        And {bulkItems.length - 10} more items...
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {bulkItems.length > 0 && (
              <div className="p-4 border-t border-slate-200 dark:border-slate-700 flex gap-3">
                <button
                  onClick={() => setBulkItems([])}
                  disabled={bulkImporting}
                  className="flex-1 px-4 py-2 border border-slate-200 dark:border-slate-600 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick={handleBulkImport}
                  disabled={bulkImporting || !bulkItems.some(i => i.isValid)}
                  className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {bulkImporting ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Upload className="h-4 w-4" />
                  )}
                  Import {bulkItems.filter(i => i.isValid).length} Items
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
