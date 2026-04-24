import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Download } from 'lucide-react';
import api from '../../api/client';
import DataTable from '../../components/ui/DataTable';
import SearchBar from '../../components/ui/SearchBar';
import BulkActions from '../../components/ui/BulkActions';
import ConfirmDialog from '../../components/ui/ConfirmDialog';
import Badge from '../../components/ui/Badge';
import toast from 'react-hot-toast';

const STATUSES = ['pending', 'confirmed', 'preparing', 'ready', 'delivered', 'cancelled'];

export default function OrdersPage() {
  const navigate = useNavigate();
  const [data, setData] = useState({ items: [], total: 0, page: 1, page_size: 10, total_pages: 1 });
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [status, setStatus] = useState('');
  const [sortBy, setSortBy] = useState('orderdate');
  const [sortDir, setSortDir] = useState('desc');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get('/orders', { params: { page, page_size: pageSize, search, status, sort_by: sortBy, sort_dir: sortDir } });
      setData(res.data);
    } catch { toast.error('Failed to load orders'); }
    finally { setLoading(false); }
  }, [page, pageSize, search, status, sortBy, sortDir]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleSort = (key: string) => {
    if (sortBy === key) setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    else { setSortBy(key); setSortDir('asc'); }
  };

  const handleBulkDelete = async () => {
    try {
      await api.post('/orders/bulk-delete', { ids: selectedIds });
      toast.success('Orders deleted');
      setSelectedIds([]);
      setConfirmDelete(false);
      fetchData();
    } catch { toast.error('Delete failed'); }
  };

  const handleBulkStatus = async (newStatus: string) => {
    try {
      await api.put('/orders/bulk-update', { ids: selectedIds, updates: { status: newStatus } });
      toast.success(`Updated to ${newStatus}`);
      setSelectedIds([]);
      fetchData();
    } catch { toast.error('Update failed'); }
  };

  const exportCsv = () => { window.open(`/api/orders/export/csv?search=${search}&status=${status}`, '_blank'); };
  const exportPdf = () => { window.open(`/api/orders/export/pdf?search=${search}&status=${status}`, '_blank'); };

  const columns = [
    { key: 'id', label: 'ID', sortable: true, render: (o: any) => `#${o.id}` },
    { key: 'phone_number', label: 'Phone', sortable: true },
    { key: 'status', label: 'Status', sortable: true, render: (o: any) => <Badge value={o.status} /> },
    { key: 'total', label: 'Total', render: (o: any) => o.order_data?.total_price || '-' },
    { key: 'items', label: 'Items', render: (o: any) => `${o.order_data?.menu_items_ordered?.length || 0} items` },
    { key: 'orderdate', label: 'Date', sortable: true, render: (o: any) => new Date(o.orderdate).toLocaleDateString() },
  ];

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <h1 className="text-2xl font-bold text-gray-900">Orders</h1>
        <div className="flex gap-2">
          <button onClick={exportCsv} className="flex items-center gap-1 px-3 py-2 text-sm bg-green-600 text-white rounded-lg hover:bg-green-700">
            <Download className="w-4 h-4" /> CSV
          </button>
          <button onClick={exportPdf} className="flex items-center gap-1 px-3 py-2 text-sm bg-red-600 text-white rounded-lg hover:bg-red-700">
            <Download className="w-4 h-4" /> PDF
          </button>
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-3">
        <SearchBar value={search} onChange={(v) => { setSearch(v); setPage(1); }} placeholder="Search orders..." />
        <select value={status} onChange={(e) => { setStatus(e.target.value); setPage(1); }}
          className="border rounded-lg px-3 py-2 text-sm">
          <option value="">All Statuses</option>
          {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      <div className="bg-white rounded-lg shadow p-4">
        <DataTable
          data={data.items} columns={columns}
          total={data.total} page={data.page} pageSize={data.page_size} totalPages={data.total_pages}
          loading={loading} sortBy={sortBy} sortDir={sortDir} selectedIds={selectedIds}
          onPageChange={setPage} onPageSizeChange={(s) => { setPageSize(s); setPage(1); }}
          onSort={handleSort} onSelect={setSelectedIds}
          onRowClick={(o: any) => navigate(`/orders/${o.id}`)}
        />
      </div>

      <BulkActions count={selectedIds.length} onDelete={() => setConfirmDelete(true)} onStatusUpdate={handleBulkStatus} statusOptions={STATUSES} />
      <ConfirmDialog open={confirmDelete} title="Delete Orders" message={`Delete ${selectedIds.length} orders?`}
        variant="danger" confirmLabel="Delete" onConfirm={handleBulkDelete} onCancel={() => setConfirmDelete(false)} />
    </div>
  );
}
