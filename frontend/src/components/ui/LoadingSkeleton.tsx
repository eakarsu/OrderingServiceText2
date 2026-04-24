import { clsx } from 'clsx';

interface Props {
  variant?: 'table-row' | 'card' | 'detail' | 'line';
  count?: number;
}

function SkeletonLine({ className }: { className?: string }) {
  return <div className={clsx('animate-pulse bg-gray-200 rounded', className)} />;
}

export default function LoadingSkeleton({ variant = 'table-row', count = 5 }: Props) {
  if (variant === 'card') {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {Array.from({ length: count }).map((_, i) => (
          <div key={i} className="bg-white rounded-lg shadow p-4 space-y-3">
            <SkeletonLine className="h-6 w-3/4" />
            <SkeletonLine className="h-4 w-full" />
            <SkeletonLine className="h-4 w-1/2" />
          </div>
        ))}
      </div>
    );
  }

  if (variant === 'detail') {
    return (
      <div className="bg-white rounded-lg shadow p-6 space-y-4">
        <SkeletonLine className="h-8 w-1/3" />
        <SkeletonLine className="h-4 w-2/3" />
        <SkeletonLine className="h-4 w-1/2" />
        <SkeletonLine className="h-4 w-3/4" />
        <SkeletonLine className="h-4 w-1/3" />
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="flex gap-4 items-center">
          <SkeletonLine className="h-4 w-8" />
          <SkeletonLine className="h-4 w-1/4" />
          <SkeletonLine className="h-4 w-1/3" />
          <SkeletonLine className="h-4 w-1/6" />
          <SkeletonLine className="h-4 w-1/6" />
        </div>
      ))}
    </div>
  );
}
