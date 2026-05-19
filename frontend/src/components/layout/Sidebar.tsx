import { NavLink } from 'react-router-dom';
import { LayoutDashboard, ShoppingCart, FolderOpen, UtensilsCrossed, Users, User, X, KanbanSquare } from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';

const navItems = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/orders', label: 'Orders', icon: ShoppingCart },
  { to: '/categories', label: 'Categories', icon: FolderOpen },
  { to: '/menu-items', label: 'Menu Items', icon: UtensilsCrossed },
  { to: '/users', label: 'Users', icon: Users, adminOnly: true },
  { to: '/profile', label: 'Profile', icon: User },
  { to: '/custom-views', label: 'Order Views', icon: KanbanSquare },
];

interface Props {
  open: boolean;
  onClose: () => void;
}

export default function Sidebar({ open, onClose }: Props) {
  const { user } = useAuth();

  return (
    <>
      {/* Overlay */}
      {open && <div className="fixed inset-0 bg-black/50 z-30 lg:hidden" onClick={onClose} />}

      <aside className={`fixed top-0 left-0 z-40 h-full w-64 bg-white shadow-lg transform transition-transform duration-200 lg:translate-x-0 ${open ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex items-center justify-between p-4 border-b">
          <h1 className="text-xl font-bold text-gray-900">Admin</h1>
          <button onClick={onClose} className="lg:hidden">
            <X className="w-5 h-5" />
          </button>
        </div>

        <nav className="p-4 space-y-1">
          {navItems.map((item) => {
            if (item.adminOnly && user?.role !== 'admin') return null;
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={onClose}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                    isActive ? 'bg-blue-50 text-blue-700 font-medium' : 'text-gray-600 hover:bg-gray-50'
                  }`
                }
              >
                <Icon className="w-5 h-5" />
                {item.label}
              </NavLink>
            );
          })}
        </nav>
      </aside>
    </>
  );
}
