import { useEffect, useState } from 'react';
import api from '../api/client';

export default function ReceiptPDF() {
  const [orders, setOrders] = useState([]);
  const [orderId, setOrderId] = useState('');
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [status, setStatus] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.get('/custom-views/order-options')
      .then((res) => {
        const list = res.data.orders || [];
        setOrders(list);
        if (list.length) setOrderId(String(list[0].id));
      })
      .catch((e) => setErr(e?.message || 'Failed to load orders'))
      .finally(() => setLoading(false));
  }, []);

  const download = async () => {
    if (!orderId) return;
    setDownloading(true);
    setStatus(null);
    setErr(null);
    try {
      const res = await api.get(`/custom-views/receipt/${orderId}`, {
        responseType: 'blob',
      });
      const blob = new Blob([res.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `receipt-${orderId}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
      setStatus(`Receipt #${orderId} downloaded (${blob.size} bytes).`);
    } catch (e) {
      setErr(e?.response?.data?.detail || e?.message || 'Download failed');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div data-testid="receipt-pdf" className="bg-white border border-gray-200 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-3 text-gray-800">Order Receipt PDF</h2>
      {loading && <div className="text-gray-500">Loading orders…</div>}
      {!loading && (
        <>
          <label className="block text-sm text-gray-600 mb-1">Pick an order</label>
          <select
            value={orderId}
            onChange={(e) => setOrderId(e.target.value)}
            className="w-full border border-gray-300 rounded p-2 text-sm mb-3"
          >
            {orders.length === 0 && <option value="">No orders</option>}
            {orders.map((o) => (
              <option key={o.id} value={o.id}>{o.label}</option>
            ))}
          </select>
          <button
            type="button"
            onClick={download}
            disabled={!orderId || downloading}
            data-testid="receipt-pdf-download"
            className="px-4 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 disabled:bg-gray-300 text-sm font-medium"
          >
            {downloading ? 'Generating…' : 'Download Receipt PDF'}
          </button>
          {status && (
            <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded text-sm text-green-800">
              {status}
            </div>
          )}
          {err && (
            <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">
              {err}
            </div>
          )}
        </>
      )}
    </div>
  );
}
