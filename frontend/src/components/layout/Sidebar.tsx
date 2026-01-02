'use client';

import { clsx } from 'clsx';
import {
    BarChart3,
    ChefHat,
    LayoutDashboard,
    Package,
    Settings,
    ShoppingCart,
    Sparkles,
    Users,
    UtensilsCrossed,
    X
} from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useSidebar } from './SidebarContext';

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'POS', href: '/pos', icon: ShoppingCart },
  { name: 'Menu', href: '/menu', icon: UtensilsCrossed },
  { name: 'Orders', href: '/orders', icon: ChefHat },
  { name: 'Customers', href: '/customers', icon: Users },
  { name: 'Inventory', href: '/inventory', icon: Package },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
  { name: 'AI Assistant', href: '/assistant', icon: Sparkles, highlight: true },
  { name: 'Settings', href: '/settings', icon: Settings },
];

export default function Sidebar() {
  const pathname = usePathname();
  const { isOpen, isMobile, close } = useSidebar();

  const handleNavClick = () => {
    if (isMobile) {
      close();
    }
  };

  return (
    <>
      {isMobile && isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={close}
        />
      )}

      <aside
        className={clsx(
          'bg-white dark:bg-slate-800 border-r border-slate-200 dark:border-slate-700 flex flex-col z-50',
          'transition-transform duration-300 ease-in-out',
          isMobile
            ? 'fixed inset-y-0 left-0 w-72'
            : 'relative w-64 shrink-0',
          isMobile && !isOpen && '-translate-x-full',
          isMobile && isOpen && 'translate-x-0'
        )}
      >
        <div className="h-16 flex items-center justify-between px-6 border-b border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-3">
            <div className="h-9 w-9 bg-gradient-to-br from-orange-400 to-orange-600 rounded-lg flex items-center justify-center shadow-sm">
              <ChefHat className="h-5 w-5 text-white" />
            </div>
            <span className="text-xl font-bold text-slate-800 dark:text-white">
              RestaurantAI
            </span>
          </div>
          {isMobile && (
            <button
              onClick={close}
              className="p-2 rounded-lg text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-700"
            >
              <X className="h-5 w-5" />
            </button>
          )}
        </div>

        <nav className="flex-1 p-4 space-y-1 overflow-y-auto">
          {navigation.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;
            return (
              <Link
                key={item.name}
                href={item.href}
                onClick={handleNavClick}
                className={clsx(
                  'flex items-center px-4 py-2.5 text-sm font-medium rounded-lg transition-all duration-200',
                  isActive
                    ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 shadow-sm'
                    : 'text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-700/50',
                  item.highlight && !isActive && 'text-purple-600 dark:text-purple-400'
                )}
              >
                <Icon className={clsx(
                  'h-5 w-5 mr-3 shrink-0',
                  item.highlight && !isActive && 'text-purple-500'
                )} />
                {item.name}
                {item.highlight && (
                  <span className="ml-auto px-1.5 py-0.5 text-[10px] font-semibold bg-purple-100 text-purple-600 dark:bg-purple-900/50 dark:text-purple-400 rounded">
                    AI
                  </span>
                )}
              </Link>
            );
          })}
        </nav>

        <div className="p-4 border-t border-slate-200 dark:border-slate-700">
          <div className="px-4 py-3 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg">
            <p className="text-xs font-medium text-blue-700 dark:text-blue-400">
              Powered by AI
            </p>
            <p className="text-xs text-blue-600/70 dark:text-blue-400/70 mt-0.5">
              Groq + Llama 3.3
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}
