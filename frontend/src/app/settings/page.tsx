'use client';

import { clsx } from 'clsx';
import {
    Bell,
    Building2,
    CreditCard,
    Globe,
    Key,
    Moon,
    Palette,
    Save,
    Settings,
    Shield,
    Sun,
    Users,
} from 'lucide-react';
import { useState } from 'react';

interface SettingSection {
  id: string;
  label: string;
  icon: React.ReactNode;
}

const settingSections: SettingSection[] = [
  { id: 'general', label: 'General', icon: <Settings className="h-5 w-5" /> },
  { id: 'restaurant', label: 'Restaurant', icon: <Building2 className="h-5 w-5" /> },
  { id: 'appearance', label: 'Appearance', icon: <Palette className="h-5 w-5" /> },
  { id: 'notifications', label: 'Notifications', icon: <Bell className="h-5 w-5" /> },
  { id: 'users', label: 'Users & Roles', icon: <Users className="h-5 w-5" /> },
  { id: 'integrations', label: 'Integrations', icon: <Globe className="h-5 w-5" /> },
  { id: 'payment', label: 'Payment', icon: <CreditCard className="h-5 w-5" /> },
  { id: 'security', label: 'Security', icon: <Shield className="h-5 w-5" /> },
];

export default function SettingsPage() {
  const [activeSection, setActiveSection] = useState('general');
  const [theme, setTheme] = useState<'light' | 'dark' | 'system'>('system');
  const [notificationsEnabled, setNotificationsEnabled] = useState(true);
  const [emailNotifications, setEmailNotifications] = useState(true);
  const [lowStockAlerts, setLowStockAlerts] = useState(true);
  const [isSaving, setIsSaving] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 1000));
    setIsSaving(false);
  };

  const renderContent = () => {
    switch (activeSection) {
      case 'general':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-4">
                Language & Region
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Language
                  </label>
                  <select className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <option value="en">English</option>
                    <option value="ar">العربية</option>
                    <option value="es">Español</option>
                    <option value="fr">Français</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Timezone
                  </label>
                  <select className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <option value="UTC">UTC</option>
                    <option value="America/New_York">Eastern Time</option>
                    <option value="America/Los_Angeles">Pacific Time</option>
                    <option value="Europe/London">London</option>
                    <option value="Asia/Riyadh">Riyadh</option>
                  </select>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-4">
                Currency & Format
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Currency
                  </label>
                  <select className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <option value="USD">USD ($)</option>
                    <option value="EUR">EUR (€)</option>
                    <option value="SAR">SAR (ر.س)</option>
                    <option value="AED">AED (د.إ)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Date Format
                  </label>
                  <select className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <option value="MM/DD/YYYY">MM/DD/YYYY</option>
                    <option value="DD/MM/YYYY">DD/MM/YYYY</option>
                    <option value="YYYY-MM-DD">YYYY-MM-DD</option>
                  </select>
                </div>
              </div>
            </div>
          </div>
        );

      case 'restaurant':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-4">
                Restaurant Information
              </h3>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Restaurant Name
                  </label>
                  <input
                    type="text"
                    defaultValue="Smart Restaurant"
                    className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Address
                  </label>
                  <textarea
                    rows={2}
                    defaultValue="123 Main Street, City, Country"
                    className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Phone
                    </label>
                    <input
                      type="tel"
                      defaultValue="+1 (555) 123-4567"
                      className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Email
                    </label>
                    <input
                      type="email"
                      defaultValue="info@smartrestaurant.com"
                      className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-4">
                Operating Hours
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Opening Time
                  </label>
                  <input
                    type="time"
                    defaultValue="09:00"
                    className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                    Closing Time
                  </label>
                  <input
                    type="time"
                    defaultValue="22:00"
                    className="w-full px-3 py-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          </div>
        );

      case 'appearance':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-4">
                Theme
              </h3>
              <div className="grid grid-cols-3 gap-4">
                {(['light', 'dark', 'system'] as const).map((t) => (
                  <button
                    key={t}
                    onClick={() => setTheme(t)}
                    className={clsx(
                      'flex flex-col items-center gap-2 p-4 rounded-xl border-2 transition-colors',
                      theme === t
                        ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                        : 'border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600'
                    )}
                  >
                    {t === 'light' && <Sun className="h-8 w-8 text-yellow-500" />}
                    {t === 'dark' && <Moon className="h-8 w-8 text-blue-500" />}
                    {t === 'system' && <Settings className="h-8 w-8 text-slate-500" />}
                    <span className="text-sm font-medium capitalize">{t}</span>
                  </button>
                ))}
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-4">
                Accent Color
              </h3>
              <div className="flex gap-3">
                {['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'].map((color) => (
                  <button
                    key={color}
                    className="w-10 h-10 rounded-full border-4 border-white dark:border-slate-700 shadow-lg"
                    style={{ backgroundColor: color }}
                  />
                ))}
              </div>
            </div>
          </div>
        );

      case 'notifications':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-4">
                Notification Preferences
              </h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">
                      Push Notifications
                    </p>
                    <p className="text-sm text-slate-500">
                      Receive notifications in the browser
                    </p>
                  </div>
                  <button
                    onClick={() => setNotificationsEnabled(!notificationsEnabled)}
                    className={clsx(
                      'w-12 h-6 rounded-full transition-colors relative',
                      notificationsEnabled ? 'bg-blue-500' : 'bg-slate-300 dark:bg-slate-600'
                    )}
                  >
                    <span
                      className={clsx(
                        'absolute top-1 w-4 h-4 bg-white rounded-full transition-transform',
                        notificationsEnabled ? 'right-1' : 'left-1'
                      )}
                    />
                  </button>
                </div>

                <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">
                      Email Notifications
                    </p>
                    <p className="text-sm text-slate-500">
                      Receive daily summary emails
                    </p>
                  </div>
                  <button
                    onClick={() => setEmailNotifications(!emailNotifications)}
                    className={clsx(
                      'w-12 h-6 rounded-full transition-colors relative',
                      emailNotifications ? 'bg-blue-500' : 'bg-slate-300 dark:bg-slate-600'
                    )}
                  >
                    <span
                      className={clsx(
                        'absolute top-1 w-4 h-4 bg-white rounded-full transition-transform',
                        emailNotifications ? 'right-1' : 'left-1'
                      )}
                    />
                  </button>
                </div>

                <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                  <div>
                    <p className="font-medium text-slate-900 dark:text-white">
                      Low Stock Alerts
                    </p>
                    <p className="text-sm text-slate-500">
                      Get notified when inventory is low
                    </p>
                  </div>
                  <button
                    onClick={() => setLowStockAlerts(!lowStockAlerts)}
                    className={clsx(
                      'w-12 h-6 rounded-full transition-colors relative',
                      lowStockAlerts ? 'bg-blue-500' : 'bg-slate-300 dark:bg-slate-600'
                    )}
                  >
                    <span
                      className={clsx(
                        'absolute top-1 w-4 h-4 bg-white rounded-full transition-transform',
                        lowStockAlerts ? 'right-1' : 'left-1'
                      )}
                    />
                  </button>
                </div>
              </div>
            </div>
          </div>
        );

      case 'security':
        return (
          <div className="space-y-6">
            <div>
              <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-4">
                API Keys
              </h3>
              <div className="p-4 bg-slate-50 dark:bg-slate-700/50 rounded-lg">
                <div className="flex items-center gap-3 mb-2">
                  <Key className="h-5 w-5 text-slate-500" />
                  <span className="font-medium text-slate-900 dark:text-white">
                    Groq API Key
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="password"
                    defaultValue="gsk_KIZsT188aZSWfSMoo7C7WGdyb..."
                    className="flex-1 px-3 py-2 bg-white dark:bg-slate-700 border border-slate-200 dark:border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm"
                  />
                  <button className="px-4 py-2 text-sm font-medium text-blue-500 hover:text-blue-600">
                    Update
                  </button>
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-4">
                Password
              </h3>
              <button className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors">
                Change Password
              </button>
            </div>

            <div>
              <h3 className="text-lg font-medium text-slate-900 dark:text-white mb-4">
                Two-Factor Authentication
              </h3>
              <div className="p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg">
                <p className="text-sm text-yellow-800 dark:text-yellow-200">
                  Two-factor authentication is not enabled. Enable it for extra security.
                </p>
                <button className="mt-2 px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition-colors text-sm">
                  Enable 2FA
                </button>
              </div>
            </div>
          </div>
        );

      default:
        return (
          <div className="text-center py-12 text-slate-500">
            <Settings className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p>This section is coming soon</p>
          </div>
        );
    }
  };

  return (
    <div className="space-y-4 sm:space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-slate-900 dark:text-white flex items-center gap-2">
            <Settings className="h-5 w-5 sm:h-7 sm:w-7 text-slate-500" />
            Settings
          </h1>
          <p className="text-sm text-slate-500">Manage your restaurant configuration</p>
        </div>
        <button
          onClick={handleSave}
          disabled={isSaving}
          className={clsx(
            'flex items-center justify-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg transition-colors self-start sm:self-auto',
            isSaving ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-600'
          )}
        >
          <Save className="h-4 w-4" />
          {isSaving ? 'Saving...' : 'Save Changes'}
        </button>
      </div>

      <div className="flex flex-col lg:grid lg:grid-cols-4 gap-4 sm:gap-6">
        {/* Sidebar - Horizontal scroll on mobile */}
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-3 sm:p-4 lg:order-1 order-1">
          <nav className="flex lg:flex-col gap-1 overflow-x-auto lg:overflow-visible pb-2 lg:pb-0 -mx-1 px-1 lg:mx-0 lg:px-0">
            {settingSections.map((section) => (
              <button
                key={section.id}
                onClick={() => setActiveSection(section.id)}
                className={clsx(
                  'flex items-center gap-2 sm:gap-3 px-3 py-2 sm:py-2.5 rounded-lg text-left transition-colors whitespace-nowrap lg:whitespace-normal lg:w-full',
                  activeSection === section.id
                    ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600'
                    : 'text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700'
                )}
              >
                <span className="flex-shrink-0">{section.icon}</span>
                <span className="font-medium text-sm sm:text-base">{section.label}</span>
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="lg:col-span-3 bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 sm:p-6 lg:order-2 order-2">
          {renderContent()}
        </div>
      </div>
    </div>
  );
}
