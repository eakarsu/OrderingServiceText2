import { ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';
import LoadingSkeleton from './LoadingSkeleton';
import EmptyState from './EmptyState';
import Pagination from './Pagination';

interface Column<T> {
  key: string;
  label: string;
  sortable?: boolean;
  render?: (item: T) => React.ReactNode;
}

interface Props<T> {
  data: T[];
  columns: Column<T>[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
  loading: boolean;
  sortBy: string;
  sortDir: string;
  selectedIds: number[];
  onPageChange: (p: number) => void;
  onPageSizeChange: (s: number) => void;
  onSort: (key: string) => void;
  onSelect: (ids: number[]) => void;
  onRowClick?: (item: T) => void;
  idKey?: string;
}

export default function DataTable<T extends Record<string, any>>({
  data, columns, total, page, pageSize, totalPages, loading,
  sortBy, sortDir, selectedIds, onPageChange, onPageSizeChange,
  onSort, onSelect, onRowClick, idKey = 'id',
}: Props<T>) {
  const allSelected = data.length > 0 && data.every((d) => selectedIds.includes(d[idKey]));

  const toggleAll = () => {
    if (allSelected) {
      onSelect([]);
    } else {
      onSelect(data.map((d) => d[idKey]));
    }
  };

  const toggleOne = (id: number) => {
    onSelect(
      selectedIds.includes(id) ? selectedIds.filter((s) => s !== id) : [...selectedIds, id]
    );
  };

  if (loading) return <LoadingSkeleton variant="table-row" count={pageSize} />;

  if (data.length === 0) return <EmptyState />;

  return (
    <div>
      {/* Mobile card view */}
      <div className="block lg:hidden space-y-3">
        {data.map((item) => (
          <div
            key={item[idKey]}
            onClick={() => onRowClick?.(item)}
            className="bg-white rounded-lg shadow p-4 cursor-pointer hover:shadow-md transition-shadow"
          >
            <div className="flex items-center gap-3 mb-2">
              <input
                type="checkbox"
                checked={selectedIds.includes(item[idKey])}
                onChange={(e) => { e.stopPropagation(); toggleOne(item[idKey]); }}
                className="w-4 h-4 rounded"
              />
              <div className="font-medium text-gray-900">
                {columns[0].render ? columns[0].render(item) : String(item[columns[0].key] ?? '')}
              </div>
            </div>
            {columns.slice(1).map((col) => (
              <div key={col.key} className="flex justify-between text-sm py-1">
                <span className="text-gray-500">{col.label}</span>
                <span>{col.render ? col.render(item) : String(item[col.key] ?? '')}</span>
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Desktop table view */}
      <div className="hidden lg:block overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-gray-50">
              <th className="p-3 w-10">
                <input
                  type="checkbox"
                  checked={allSelected}
                  onChange={toggleAll}
                  className="w-4 h-4 rounded"
                />
              </th>
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`p-3 text-left font-medium text-gray-600 ${col.sortable ? 'cursor-pointer select-none hover:text-gray-900' : ''}`}
                  onClick={() => col.sortable && onSort(col.key)}
                >
                  <div className="flex items-center gap-1">
                    {col.label}
                    {col.sortable && (
                      sortBy === col.key
                        ? sortDir === 'asc' ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />
                        : <ArrowUpDown className="w-3 h-3 text-gray-300" />
                    )}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.map((item) => (
              <tr
                key={item[idKey]}
                onClick={() => onRowClick?.(item)}
                className="border-b hover:bg-gray-50 cursor-pointer transition-colors"
              >
                <td className="p-3" onClick={(e) => e.stopPropagation()}>
                  <input
                    type="checkbox"
                    checked={selectedIds.includes(item[idKey])}
                    onChange={() => toggleOne(item[idKey])}
                    className="w-4 h-4 rounded"
                  />
                </td>
                {columns.map((col) => (
                  <td key={col.key} className="p-3">
                    {col.render ? col.render(item) : String(item[col.key] ?? '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Pagination
        page={page}
        totalPages={totalPages}
        pageSize={pageSize}
        total={total}
        onPageChange={onPageChange}
        onPageSizeChange={onPageSizeChange}
      />
    </div>
  );
}
