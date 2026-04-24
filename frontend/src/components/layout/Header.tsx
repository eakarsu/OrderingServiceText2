import { Menu, LogOut } from 'lucide-react';
import { useAuth } from '../../contexts/AuthContext';
import Badge from '../ui/Badge';

interface Props {
  onMenuClick: () => void;
}

export default function Header({ onMenuClick }: Props) {
  const { user, logout } = useAuth();

  return (
    <header className="bg-white shadow-sm border-b px-4 py-3 flex items-center justify-between">
      <button onClick={onMenuClick} className="lg:hidden">
        <Menu className="w-6 h-6" />
      </button>

      <div className="hidden lg:block" />

      <div className="flex items-center gap-4">
        <div className="text-right hidden sm:block">
          <div className="text-sm font-medium text-gray-900">
            {user?.first_name} {user?.last_name}
          </div>
          <Badge value={user?.role || ''} />
        </div>
        <button onClick={logout} className="p-2 text-gray-500 hover:text-gray-700 rounded-lg hover:bg-gray-100" title="Logout">
          <LogOut className="w-5 h-5" />
        </button>
      </div>
    </header>
  );
}
