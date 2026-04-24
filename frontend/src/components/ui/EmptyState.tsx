import { InboxIcon } from 'lucide-react';

interface Props {
  title?: string;
  message?: string;
  action?: { label: string; onClick: () => void };
}

export default function EmptyState({ title = 'No data found', message = 'Try adjusting your search or filters.', action }: Props) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-gray-500">
      <InboxIcon className="w-16 h-16 mb-4 text-gray-300" />
      <h3 className="text-lg font-medium text-gray-700">{title}</h3>
      <p className="text-sm mt-1">{message}</p>
      {action && (
        <button
          onClick={action.onClick}
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
        >
          {action.label}
        </button>
      )}
    </div>
  );
}
