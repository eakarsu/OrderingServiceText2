import { useEffect, useState } from 'react';
import api from '../api/client';

const COLUMN_STYLES = {
  'new':        { bg: '#eff6ff', label: 'New',         accent: '#3b82f6' },
  'preparing':  { bg: '#fffbeb', label: 'Preparing',   accent: '#f59e0b' },
  'ready':      { bg: '#ecfdf5', label: 'Ready',       accent: '#10b981' },
  'picked-up':  { bg: '#f3f4f6', label: 'Picked Up',   accent: '#6b7280' },
};

export default function OrderKanban() {
  const [columns, setColumns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.get('/custom-views/order-kanban')
      .then((res) => setColumns(res.data.columns || []))
      .catch((e) => setErr(e?.message || 'Failed to load'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-4 text-gray-500">Loading kanban…</div>;
  if (err)     return <div className="p-4 text-red-600">Kanban error: {err}</div>;

  return (
    <div data-testid="order-kanban">
      <h2 className="text-lg font-semibold mb-3 text-gray-800">Order Kanban</h2>
      <div style={{ display: 'flex', gap: '12px', overflowX: 'auto', paddingBottom: '8px' }}>
        {columns.map((col) => {
          const style = COLUMN_STYLES[col.status] || { bg: '#f9fafb', label: col.status, accent: '#9ca3af' };
          return (
            <div
              key={col.status}
              style={{
                flex: '1 1 0',
                minWidth: '220px',
                background: style.bg,
                borderTop: `3px solid ${style.accent}`,
                borderRadius: '8px',
                padding: '12px',
              }}
            >
              <div className="flex items-center justify-between mb-3">
                <span className="font-semibold text-gray-800">{style.label}</span>
                <span
                  className="text-xs font-medium px-2 py-0.5 rounded-full"
                  style={{ background: style.accent, color: 'white' }}
                >
                  {col.orders.length}
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                {col.orders.length === 0 && (
                  <div className="text-xs text-gray-400 italic">No orders</div>
                )}
                {col.orders.map((o) => (
                  <div
                    key={o.id}
                    className="bg-white rounded shadow-sm border border-gray-200 p-2 text-sm"
                  >
                    <div className="flex justify-between">
                      <span className="font-medium">#{o.id}</span>
                      <span className="text-xs text-gray-500">{o.item_count} items</span>
                    </div>
                    <div className="text-xs text-gray-600 truncate">{o.phone_number || '—'}</div>
                    <div className="text-xs text-gray-700 mt-1">${(o.total || 0).toFixed(2)}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
