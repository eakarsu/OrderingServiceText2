import { useEffect, useState, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Plus } from 'lucide-react';
import api from '../../api/client';
import DataTable from '../../components/ui/DataTable';
import SearchBar from '../../components/ui/SearchBar';
import toast from 'react-hot-toast';

export default function CategoryMenuPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [category, setCategory] = useState<any>(null);
  const [data, setData] = useState({ items: [], total: 0, page: 1, page_size: 10, total_pages: 1 });
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('name');
  const [sortDir, setSortDir] = useState('asc');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ name: '', description: '', price: '' });

  useEffect(() => {
    api.get(`/categories-mgmt/${id}`).then((res) => setCategory(res.data)).catch(() => {});
  }, [id]);

  const fetchItems = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get('/menu-items', { params: { page, page_size: pageSize, search, category_id: id, sort_by: sortBy, sort_dir: sortDir } });
      setData(res.data);
    } catch {} finally { setLoading(false); }
  }, [id, page, pageSize, search, sortBy, sortDir]);

  useEffect(() => { fetchItems(); }, [fetchItems]);

  const handleSort = (key: string) => {
    if (sortBy === key) setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    else { setSortBy(key); setSortDir('asc'); }
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api.post('/menu-items', { category_id: Number(id), name: form.name, description: form.description, price: Number(form.price) || 0 });
      setShowForm(false);
      setForm({ name: '', description: '', price: '' });
      fetchItems();
      toast.success('Item created');
    } catch (err: any) { toast.error(err.response?.data?.detail || 'Create failed'); }
  };

  const columns = [
    { key: 'name', label: 'Name', sortable: true },
    { key: 'price', label: 'Price', sortable: true, render: (i: any) => `$${i.price.toFixed(2)}` },
    { key: 'description', label: 'Description', render: (i: any) => <span className="text-gray-500 truncate max-w-xs block">{i.description || '-'}</span> },
    { key: 'is_available', label: 'Available', render: (i: any) => i.is_available ? <span className="text-green-600">Yes</span> : <span className="text-red-600">No</span> },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <button onClick={() => navigate('/categories')} className="p-2 hover:bg-gray-100 rounded-lg">
          <ArrowLeft className="w-5 h-5" />
        </button>
        <h1 className="text-2xl font-bold text-gray-900">{category?.name || 'Category'}</h1>
        <span className="text-gray-500">({data.total} items)</span>
      </div>

      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
        <SearchBar value={search} onChange={(v) => { setSearch(v); setPage(1); }} placeholder="Search items..." />
        <button onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-1 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm">
          <Plus className="w-4 h-4" /> Add Item
        </button>
      </div>

      {showForm && (
        <form onSubmit={handleCreate} className="bg-white rounded-lg shadow p-4 flex flex-col sm:flex-row gap-3">
          <input type="text" required placeholder="Item name" value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            className="flex-1 px-3 py-2 border rounded-lg text-sm" />
          <input type="text" placeholder="Description" value={form.description}
            onChange={(e) => setForm({ ...form, description: e.target.value })}
            className="flex-1 px-3 py-2 border rounded-lg text-sm" />
          <input type="number" step="0.01" placeholder="Price" value={form.price}
            onChange={(e) => setForm({ ...form, price: e.target.value })}
            className="w-24 px-3 py-2 border rounded-lg text-sm" />
          <button type="submit" className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm">Create</button>
        </form>
      )}

      <div className="bg-white rounded-lg shadow p-4">
        <DataTable
          data={data.items} columns={columns}
          total={data.total} page={data.page} pageSize={data.page_size} totalPages={data.total_pages}
          loading={loading} sortBy={sortBy} sortDir={sortDir} selectedIds={selectedIds}
          onPageChange={setPage} onPageSizeChange={(s) => { setPageSize(s); setPage(1); }}
          onSort={handleSort} onSelect={setSelectedIds}
          onRowClick={(i: any) => navigate(`/menu-items/${i.id}`)}
        />
      </div>
    </div>
  );
}
