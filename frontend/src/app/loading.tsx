import { Loader2 } from 'lucide-react';

export default function Loading() {
  return (
    <div className="flex items-center justify-center h-full">
      <div className="text-center">
        <Loader2 className="h-10 w-10 text-blue-500 animate-spin mx-auto" />
        <p className="mt-4 text-slate-500">Loading...</p>
      </div>
    </div>
  );
}
