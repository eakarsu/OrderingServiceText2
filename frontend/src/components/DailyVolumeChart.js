import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer,
} from 'recharts';
import api from '../api/client';

export default function DailyVolumeChart() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);

  useEffect(() => {
    api.get('/custom-views/daily-volume')
      .then((res) => {
        const days = (res.data.days || []).map((d) => ({
          date: d.date.slice(5),  // MM-DD
          count: d.count,
        }));
        setData(days);
      })
      .catch((e) => setErr(e?.message || 'Failed to load'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-4 text-gray-500">Loading chart…</div>;
  if (err)     return <div className="p-4 text-red-600">Chart error: {err}</div>;

  return (
    <div data-testid="daily-volume-chart" className="bg-white border border-gray-200 rounded-lg p-4">
      <h2 className="text-lg font-semibold mb-3 text-gray-800">Daily Order Volume (Last 30 Days)</h2>
      <div style={{ width: '100%', height: 280 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis dataKey="date" tick={{ fontSize: 10 }} interval={3} />
            <YAxis allowDecimals={false} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="count" fill="#3b82f6" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
