import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Trash2 } from 'lucide-react';
import api from '../../api/client';
import ConfirmDialog from '../../components/ui/ConfirmDialog';
import LoadingSkeleton from '../../components/ui/LoadingSkeleton';
import toast from 'react-hot-toast';

export default function MenuItemDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [item, setItem] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState(false);
  const [form, setForm] = useState({ name: '', description: '', price: '', is_available: true });
  const [confirmDelete, setConfirmDelete] = useState(false);

  useEffect(() => {
    api.get(`/menu-items/${id}`)
      .then((res) => {
        setItem(res.data);
        setForm({ name: res.data.name, description: res.data.description || '', price: String(res.data.price), is_available: res.data.is_available });
      })
      .catch(() => toast.error('Item not found'))
      .finally(() => setLoading(false));
  }, [id]);

  const handleSave = async () => {
    try {
      const res = await api.put(`/menu-items/${id}`, { name: form.name, description: form.description, price: Number(form.price), is_available: form.is_available });
      setItem(res.data);
      setEditing(false);
      toast.success('Item updated');
    } catch { toast.error('Update failed'); }
  };

  const handleDelete = async () => {
    try {
      await api.delete(`/menu-items/${id}`);
      toast.success('Item deleted');
      navigate('/menu-items');
    } catch { toast.error('Delete failed'); }
  };

  if (loading) return <LoadingSkeleton variant="detail" />;
  if (!item) return <div className="text-center py-12 text-gray-500">Item not found</div>;

  return (
    <div className="space-y-6 max-w-2xl">
      <div className="flex items-center gap-4">
        <button onClick={() => navigate(-1)} className="p-2 hover:bg-gray-100 rounded-lg">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <h1 className="text-2xl font-bold text-gray-900">{item.name}</h1>
      </div>

      <div className="bg-white rounded-lg shadow p-6 space-y-4">
        {editing ? (
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
              <input type="text" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })}
                className="w-full px-3 py-2 border rounded-lg text-sm" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
              <textarea value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })}
                className="w-full px-3 py-2 border rounded-lg text-sm" rows={3} />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Price</label>
                <input type="number" step="0.01" value={form.price} onChange={(e) => setForm({ ...form, price: e.target.value })}
                  className="w-full px-3 py-2 border rounded-lg text-sm" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Available</label>
                <select value={String(form.is_available)} onChange={(e) => setForm({ ...form, is_available: e.target.value === 'true' })}
                  className="w-full px-3 py-2 border rounded-lg text-sm">
                  <option value="true">Yes</option>
                  <option value="false">No</option>
                </select>
              </div>
            </div>
            <div className="flex gap-3">
              <button onClick={handleSave} className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700">Save</button>
              <button onClick={() => setEditing(false)} className="px-4 py-2 text-gray-600 text-sm hover:bg-gray-100 rounded-lg">Cancel</button>
            </div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-500">Category</p>
                <p className="font-medium">{item.category_name}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Price</p>
                <p className="font-medium text-lg">${item.price.toFixed(2)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Available</p>
                <p className={`font-medium ${item.is_available ? 'text-green-600' : 'text-red-600'}`}>
                  {item.is_available ? 'Yes' : 'No'}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-500">Created</p>
                <p className="font-medium">{new Date(item.created_at).toLocaleDateString()}</p>
              </div>
            </div>
            {item.description && (
              <div>
                <p className="text-sm text-gray-500">Description</p>
                <p className="text-gray-700">{item.description}</p>
              </div>
            )}
            <div className="flex gap-3 pt-4 border-t">
              <button onClick={() => setEditing(true)} className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700">Edit</button>
              <button onClick={() => setConfirmDelete(true)} className="flex items-center gap-1 px-4 py-2 text-red-600 text-sm hover:bg-red-50 rounded-lg">
                <Trash2 className="w-4 h-4" /> Delete
              </button>
            </div>
          </>
        )}
      </div>

      <ConfirmDialog open={confirmDelete} title="Delete Item" message={`Delete "${item.name}"?`}
        variant="danger" confirmLabel="Delete" onConfirm={handleDelete} onCancel={() => setConfirmDelete(false)} />
    </div>
  );
}
