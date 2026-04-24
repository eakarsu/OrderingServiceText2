import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Trash2 } from 'lucide-react';
import api from '../../api/client';
import Badge from '../../components/ui/Badge';
import ConfirmDialog from '../../components/ui/ConfirmDialog';
import LoadingSkeleton from '../../components/ui/LoadingSkeleton';
import toast from 'react-hot-toast';

const STATUSES = ['pending', 'confirmed', 'preparing', 'ready', 'delivered', 'cancelled'];

export default function OrderDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [order, setOrder] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [status, setStatus] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    api.get(`/orders/${id}`)
      .then((res) => { setOrder(res.data); setStatus(res.data.status); })
      .catch(() => toast.error('Order not found'))
      .finally(() => setLoading(false));
  }, [id]);

  const handleSave = async () => {
    try {
      const res = await api.put(`/orders/${id}`, { status });
      setOrder(res.data);
      setEditing(false);
      toast.success('Order updated');
    } catch { toast.error('Update failed'); }
  };

  const handleDelete = async () => {
    try {
      await api.delete(`/orders/${id}`);
      toast.success('Order deleted');
      navigate('/orders');
    } catch { toast.error('Delete failed'); }
  };

  if (loading) return <LoadingSkeleton variant="detail" />;
  if (!order) return <div className="text-center py-12 text-gray-500">Order not found</div>;

  const od = order.order_data || {};

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="flex items-center gap-4">
        <button onClick={() => navigate('/orders')} className="p-2 hover:bg-gray-100 rounded-lg">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <h1 className="text-2xl font-bold text-gray-900">Order #{order.id}</h1>
        <Badge value={order.status} />
      </div>

      <div className="bg-white rounded-lg shadow p-6 space-y-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-500">Phone</p>
            <p className="font-medium">{order.phone_number}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Date</p>
            <p className="font-medium">{new Date(order.orderdate).toLocaleString()}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Delivery</p>
            <p className="font-medium capitalize">{od.pickup_or_delivery || '-'}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Payment</p>
            <p className="font-medium capitalize">{od.payment_type || '-'}</p>
          </div>
          {od.address && (
            <div className="col-span-2">
              <p className="text-sm text-gray-500">Address</p>
              <p className="font-medium">{od.address}</p>
            </div>
          )}
        </div>

        <div>
          <p className="text-sm text-gray-500 mb-2">Items</p>
          <div className="border rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b">
                  <th className="p-2 text-left">Item</th>
                  <th className="p-2 text-left">Qty</th>
                  <th className="p-2 text-left">Price</th>
                  <th className="p-2 text-left">Custom</th>
                </tr>
              </thead>
              <tbody>
                {(od.menu_items_ordered || []).map((item: any, i: number) => (
                  <tr key={i} className="border-b">
                    <td className="p-2">{item.item}</td>
                    <td className="p-2">{item.quantity}</td>
                    <td className="p-2">{item.price}</td>
                    <td className="p-2 text-gray-500">{item.custom || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-right mt-2 font-bold text-lg">{od.total_price || '-'}</div>
        </div>

        {/* Status edit */}
        <div className="flex items-center gap-3 pt-4 border-t">
          {editing ? (
            <>
              <select value={status} onChange={(e) => setStatus(e.target.value)} className="border rounded-lg px-3 py-2 text-sm">
                {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
              <button onClick={handleSave} className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700">Save</button>
              <button onClick={() => { setEditing(false); setStatus(order.status); }} className="px-4 py-2 text-gray-600 text-sm hover:bg-gray-100 rounded-lg">Cancel</button>
            </>
          ) : (
            <>
              <button onClick={() => setEditing(true)} className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700">Edit Status</button>
              <button onClick={() => setConfirmDelete(true)} className="flex items-center gap-1 px-4 py-2 text-red-600 text-sm hover:bg-red-50 rounded-lg">
                <Trash2 className="w-4 h-4" /> Delete
              </button>
            </>
          )}
        </div>
      </div>

      <ConfirmDialog open={confirmDelete} title="Delete Order" message="This action cannot be undone."
        variant="danger" confirmLabel="Delete" onConfirm={handleDelete} onCancel={() => setConfirmDelete(false)} />
    </div>
  );
}
