import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../api/client';
import DataTable from '../../components/ui/DataTable';
import SearchBar from '../../components/ui/SearchBar';
import BulkActions from '../../components/ui/BulkActions';
import ConfirmDialog from '../../components/ui/ConfirmDialog';
import toast from 'react-hot-toast';

export default function MenuItemsPage() {
  const navigate = useNavigate();
  const [data, setData] = useState({ items: [], total: 0, page: 1, page_size: 10, total_pages: 1 });
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('name');
  const [sortDir, setSortDir] = useState('asc');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get('/menu-items', { params: { page, page_size: pageSize, search, sort_by: sortBy, sort_dir: sortDir } });
      setData(res.data);
    } catch { toast.error('Failed to load menu items'); }
    finally { setLoading(false); }
  }, [page, pageSize, search, sortBy, sortDir]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleSort = (key: string) => {
    if (sortBy === key) setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    else { setSortBy(key); setSortDir('asc'); }
  };

  const handleBulkDelete = async () => {
    try {
      await api.post('/menu-items/bulk-delete', { ids: selectedIds });
      toast.success('Items deleted');
      setSelectedIds([]);
      setConfirmDelete(false);
      fetchData();
    } catch { toast.error('Delete failed'); }
  };

  const columns = [
    { key: 'name', label: 'Name', sortable: true },
    { key: 'category_name', label: 'Category' },
    { key: 'price', label: 'Price', sortable: true, render: (i: any) => `$${i.price.toFixed(2)}` },
    { key: 'is_available', label: 'Available', render: (i: any) => i.is_available ? <span className="text-green-600">Yes</span> : <span className="text-red-600">No</span> },
    { key: 'created_at', label: 'Created', sortable: true, render: (i: any) => new Date(i.created_at).toLocaleDateString() },
  ];

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-gray-900">Menu Items</h1>
      <SearchBar value={search} onChange={(v) => { setSearch(v); setPage(1); }} placeholder="Search menu items..." />
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
      <BulkActions count={selectedIds.length} onDelete={() => setConfirmDelete(true)} />
      <ConfirmDialog open={confirmDelete} title="Delete Items" message={`Delete ${selectedIds.length} items?`}
        variant="danger" confirmLabel="Delete" onConfirm={handleBulkDelete} onCancel={() => setConfirmDelete(false)} />
    </div>
  );
}
