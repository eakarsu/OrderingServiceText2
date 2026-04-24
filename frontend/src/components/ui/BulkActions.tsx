import { Trash2 } from 'lucide-react';

interface Props {
  count: number;
  onDelete: () => void;
  onStatusUpdate?: (status: string) => void;
  statusOptions?: string[];
}

export default function BulkActions({ count, onDelete, onStatusUpdate, statusOptions }: Props) {
  if (count === 0) return null;

  return (
    <div className="fixed bottom-4 left-1/2 -translate-x-1/2 bg-gray-900 text-white rounded-lg shadow-xl px-6 py-3 flex items-center gap-4 z-40">
      <span className="text-sm font-medium">{count} selected</span>
      {onStatusUpdate && statusOptions && (
        <select
          onChange={(e) => e.target.value && onStatusUpdate(e.target.value)}
          className="bg-gray-700 text-white text-sm rounded px-2 py-1"
          defaultValue=""
        >
          <option value="" disabled>Update Status</option>
          {statusOptions.map((s) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      )}
      <button onClick={onDelete} className="flex items-center gap-1 text-red-400 hover:text-red-300 text-sm">
        <Trash2 className="w-4 h-4" /> Delete Selected
      </button>
    </div>
  );
}
