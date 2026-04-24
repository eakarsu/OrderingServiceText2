import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ShoppingCart, DollarSign, Clock, Users, FolderOpen, UtensilsCrossed } from 'lucide-react';
import api from '../../api/client';
import Badge from '../../components/ui/Badge';
import LoadingSkeleton from '../../components/ui/LoadingSkeleton';

interface Stats {
  total_orders: number;
  orders_today: number;
  revenue: number;
  by_status: Record<string, number>;
  recent_orders: any[];
  active_categories: number;
  active_menu_items: number;
  active_users: number;
}

export default function DashboardPage() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    api.get('/dashboard/stats')
      .then((res) => setStats(res.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingSkeleton variant="card" count={6} />;

  const cards = [
    { label: 'Total Orders', value: stats?.total_orders ?? 0, icon: ShoppingCart, color: 'bg-blue-500' },
    { label: 'Today\'s Orders', value: stats?.orders_today ?? 0, icon: Clock, color: 'bg-green-500' },
    { label: 'Revenue', value: `$${(stats?.revenue ?? 0).toFixed(2)}`, icon: DollarSign, color: 'bg-yellow-500' },
    { label: 'Active Users', value: stats?.active_users ?? 0, icon: Users, color: 'bg-purple-500' },
    { label: 'Categories', value: stats?.active_categories ?? 0, icon: FolderOpen, color: 'bg-indigo-500' },
    { label: 'Menu Items', value: stats?.active_menu_items ?? 0, icon: UtensilsCrossed, color: 'bg-pink-500' },
  ];

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {cards.map((c) => {
          const Icon = c.icon;
          return (
            <div key={c.label} className="bg-white rounded-lg shadow p-5 flex items-center gap-4">
              <div className={`${c.color} p-3 rounded-lg`}>
                <Icon className="w-6 h-6 text-white" />
              </div>
              <div>
                <p className="text-sm text-gray-500">{c.label}</p>
                <p className="text-2xl font-bold text-gray-900">{c.value}</p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Status breakdown */}
      <div className="bg-white rounded-lg shadow p-5">
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Orders by Status</h2>
        <div className="flex flex-wrap gap-3">
          {Object.entries(stats?.by_status ?? {}).map(([status, count]) => (
            <div key={status} className="flex items-center gap-2 px-3 py-2 bg-gray-50 rounded-lg">
              <Badge value={status} />
              <span className="font-medium">{count}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Recent orders */}
      <div className="bg-white rounded-lg shadow p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-semibold text-gray-900">Recent Orders</h2>
          <button onClick={() => navigate('/orders')} className="text-sm text-blue-600 hover:underline">View all</button>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-gray-50">
                <th className="p-2 text-left">ID</th>
                <th className="p-2 text-left">Phone</th>
                <th className="p-2 text-left">Status</th>
                <th className="p-2 text-left">Total</th>
                <th className="p-2 text-left">Date</th>
              </tr>
            </thead>
            <tbody>
              {(stats?.recent_orders ?? []).map((o: any) => (
                <tr key={o.id} onClick={() => navigate(`/orders/${o.id}`)} className="border-b hover:bg-gray-50 cursor-pointer">
                  <td className="p-2">#{o.id}</td>
                  <td className="p-2">{o.phone_number}</td>
                  <td className="p-2"><Badge value={o.status} /></td>
                  <td className="p-2">{o.order_data?.total_price || '-'}</td>
                  <td className="p-2">{new Date(o.orderdate).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
