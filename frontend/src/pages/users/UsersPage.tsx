import { useEffect, useState, useCallback } from 'react';
import api from '../../api/client';
import DataTable from '../../components/ui/DataTable';
import SearchBar from '../../components/ui/SearchBar';
import BulkActions from '../../components/ui/BulkActions';
import ConfirmDialog from '../../components/ui/ConfirmDialog';
import Badge from '../../components/ui/Badge';
import toast from 'react-hot-toast';

export default function UsersPage() {
  const [data, setData] = useState({ items: [], total: 0, page: 1, page_size: 10, total_pages: 1 });
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [role, setRole] = useState('');
  const [sortBy, setSortBy] = useState('created_at');
  const [sortDir, setSortDir] = useState('desc');
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [editUser, setEditUser] = useState<any>(null);
  const [editForm, setEditForm] = useState({ role: '', is_active: true });

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.get('/users', { params: { page, page_size: pageSize, search, role, sort_by: sortBy, sort_dir: sortDir } });
      setData(res.data);
    } catch { toast.error('Failed to load users'); }
    finally { setLoading(false); }
  }, [page, pageSize, search, role, sortBy, sortDir]);

  useEffect(() => { fetchData(); }, [fetchData]);

  const handleSort = (key: string) => {
    if (sortBy === key) setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    else { setSortBy(key); setSortDir('asc'); }
  };

  const handleBulkDelete = async () => {
    try {
      await api.post('/users/bulk-delete', { ids: selectedIds });
      toast.success('Users deleted');
      setSelectedIds([]);
      setConfirmDelete(false);
      fetchData();
    } catch { toast.error('Delete failed'); }
  };

  const handleEditSave = async () => {
    if (!editUser) return;
    try {
      await api.put(`/users/${editUser.id}`, editForm);
      toast.success('User updated');
      setEditUser(null);
      fetchData();
    } catch { toast.error('Update failed'); }
  };

  const columns = [
    { key: 'name', label: 'Name', sortable: true, render: (u: any) => `${u.first_name} ${u.last_name}` },
    { key: 'email', label: 'Email', sortable: true },
    { key: 'role', label: 'Role', sortable: true, render: (u: any) => <Badge value={u.role} /> },
    { key: 'is_active', label: 'Status', render: (u: any) => <Badge value={u.is_active ? 'active' : 'inactive'} /> },
    { key: 'created_at', label: 'Created', sortable: true, render: (u: any) => new Date(u.created_at).toLocaleDateString() },
  ];

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-gray-900">Users</h1>
      <div className="flex flex-col sm:flex-row gap-3">
        <SearchBar value={search} onChange={(v) => { setSearch(v); setPage(1); }} placeholder="Search users..." />
        <select value={role} onChange={(e) => { setRole(e.target.value); setPage(1); }}
          className="border rounded-lg px-3 py-2 text-sm">
          <option value="">All Roles</option>
          <option value="admin">Admin</option>
          <option value="manager">Manager</option>
          <option value="staff">Staff</option>
        </select>
      </div>

      <div className="bg-white rounded-lg shadow p-4">
        <DataTable
          data={data.items} columns={columns}
          total={data.total} page={data.page} pageSize={data.page_size} totalPages={data.total_pages}
          loading={loading} sortBy={sortBy} sortDir={sortDir} selectedIds={selectedIds}
          onPageChange={setPage} onPageSizeChange={(s) => { setPageSize(s); setPage(1); }}
          onSort={handleSort} onSelect={setSelectedIds}
          onRowClick={(u: any) => { setEditUser(u); setEditForm({ role: u.role, is_active: u.is_active }); }}
        />
      </div>

      {/* Edit modal */}
      {editUser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50" onClick={() => setEditUser(null)}>
          <div className="bg-white rounded-lg shadow-xl p-6 max-w-md w-full mx-4 space-y-4" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-semibold">Edit {editUser.first_name} {editUser.last_name}</h3>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Role</label>
              <select value={editForm.role} onChange={(e) => setEditForm({ ...editForm, role: e.target.value })}
                className="w-full border rounded-lg px-3 py-2 text-sm">
                <option value="admin">Admin</option>
                <option value="manager">Manager</option>
                <option value="staff">Staff</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Status</label>
              <select value={String(editForm.is_active)} onChange={(e) => setEditForm({ ...editForm, is_active: e.target.value === 'true' })}
                className="w-full border rounded-lg px-3 py-2 text-sm">
                <option value="true">Active</option>
                <option value="false">Inactive</option>
              </select>
            </div>
            <div className="flex justify-end gap-3">
              <button onClick={() => setEditUser(null)} className="px-4 py-2 text-sm text-gray-700 bg-gray-100 rounded-lg">Cancel</button>
              <button onClick={handleEditSave} className="px-4 py-2 text-sm text-white bg-blue-600 rounded-lg hover:bg-blue-700">Save</button>
            </div>
          </div>
        </div>
      )}

      <BulkActions count={selectedIds.length} onDelete={() => setConfirmDelete(true)} />
      <ConfirmDialog open={confirmDelete} title="Delete Users" message={`Delete ${selectedIds.length} users?`}
        variant="danger" confirmLabel="Delete" onConfirm={handleBulkDelete} onCancel={() => setConfirmDelete(false)} />
    </div>
  );
}
