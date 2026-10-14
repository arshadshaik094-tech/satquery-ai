import { useEffect, useState } from "react";
import { History as HistoryIcon } from "lucide-react";
import { PageHeader } from "./SingleImageAnalysis";
import { getDashboardHistory } from "../lib/api";

export default function History() {
  const [items, setItems] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getDashboardHistory(100).then((r) => setItems(r.items)).catch(() => setError(true));
  }, []);

  return (
    <div className="p-6">
      <PageHeader
        title="Analysis History"
        subtitle="Every query you've run this session, persisted in the backend's SQLite store (backend/satquery.db)."
      />

      <div className="card p-4 mt-5">
        {error && (
          <p className="text-sm text-accent-red">Couldn't reach the backend.</p>
        )}
        {!error && !items && <p className="text-sm text-ink-400">Loading…</p>}
        {items && items.length === 0 && (
          <div className="py-10 text-center flex flex-col items-center gap-2">
            <HistoryIcon size={22} className="text-ink-300" />
            <p className="text-sm text-ink-400">No history yet — run any analysis to see it logged here.</p>
          </div>
        )}
        {items && items.length > 0 && (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-ink-500 border-b border-line">
                <th className="pb-2 font-medium">Query / Task</th>
                <th className="pb-2 font-medium">Type</th>
                <th className="pb-2 font-medium">Confidence</th>
                <th className="pb-2 font-medium">Time (s)</th>
                <th className="pb-2 font-medium">Date</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={r.id} className="border-b border-line/60 last:border-0">
                  <td className="py-2.5 pr-3 max-w-xs truncate text-ink-800">{r.query}</td>
                  <td className="py-2.5"><span className="chip bg-brand-50 text-brand-700">{r.task_type}</span></td>
                  <td className="py-2.5 tabular-nums">{(r.confidence * 100).toFixed(0)}%</td>
                  <td className="py-2.5 tabular-nums">{r.processing_time_s}</td>
                  <td className="py-2.5 text-ink-400">{new Date(r.created_at * 1000).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
