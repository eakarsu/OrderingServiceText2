import { useEffect, useState } from 'react';
import api from '../api/client';

export default function SMSBroadcast() {
  const [customers, setCustomers] = useState([]);
  const [selected, setSelected] = useState(new Set());
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [result, setResult] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.get('/custom-views/customers')
      .then((res) => setCustomers(res.data.customers || []))
      .catch((e) => setErr(e?.message || 'Failed to load customers'))
      .finally(() => setLoading(false));
  }, []);

  const toggle = (phone) => {
    const next = new Set(selected);
    if (next.has(phone)) next.delete(phone); else next.add(phone);
    setSelected(next);
  };

  const toggleAll = () => {
    if (selected.size === customers.length) setSelected(new Set());
    else setSelected(new Set(customers.map((c) => c.phone)));
  };

  const send = async () => {
    setSending(true);
    setResult(null);
    setErr(null);
    try {
      const res = await api.post('/custom-views/sms-broadcast', {
        phone_numbers: Array.from(selected),
        message,
      });
      setResult(res.data);
    } catch (e) {
      setErr(e?.response?.data?.detail || e?.message || 'Send failed');
    } finally {
      setSending(false);
    }
  };

  return (
    <div data-testid="sms-broadcast" className="bg-white border border-gray-200 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-3 text-gray-800">SMS Broadcast</h2>
      {loading && <div className="text-gray-500">Loading customers…</div>}
      {!loading && (
        <>
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-600">
              {selected.size} of {customers.length} selected
            </span>
            <button
              type="button"
              onClick={toggleAll}
              className="text-xs px-2 py-1 rounded border border-gray-300 hover:bg-gray-50"
            >
              {selected.size === customers.length ? 'Clear all' : 'Select all'}
            </button>
          </div>
          <div className="max-h-48 overflow-y-auto border border-gray-200 rounded mb-3">
            {customers.length === 0 && (
              <div className="p-3 text-sm text-gray-400 italic">No customers yet.</div>
            )}
            {customers.map((c) => (
              <label
                key={c.phone}
                className="flex items-center gap-2 px-3 py-1.5 text-sm hover:bg-gray-50 border-b border-gray-100 cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={selected.has(c.phone)}
                  onChange={() => toggle(c.phone)}
                />
                <span className="font-mono">{c.phone}</span>
                <span className="text-xs text-gray-500 ml-auto">{c.order_count} orders</span>
              </label>
            ))}
          </div>
          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Type your broadcast message…"
            rows={4}
            className="w-full border border-gray-300 rounded p-2 text-sm mb-3"
          />
          <button
            type="button"
            onClick={send}
            disabled={sending || selected.size === 0 || !message.trim()}
            data-testid="sms-broadcast-send"
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed text-sm font-medium"
          >
            {sending ? 'Sending…' : `Send to ${selected.size}`}
          </button>
          {result && (
            <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded text-sm text-green-800">
              Sent {result.sent} messages.{' '}
              <span className="text-xs text-green-700">
                First SID: {(result.sids || [])[0] || '—'}
              </span>
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
