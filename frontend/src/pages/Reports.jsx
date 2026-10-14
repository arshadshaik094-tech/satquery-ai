import { useEffect, useState } from "react";
import { FileText, Download, FileBarChart2 } from "lucide-react";
import { PageHeader } from "./SingleImageAnalysis";
import { getDashboardHistory, getDashboardOverview } from "../lib/api";

export default function Reports() {
  const [items, setItems] = useState([]);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    getDashboardHistory(100).then((r) => setItems(Array.isArray(r) ? r : (r.items ?? []))).catch(() => {});
    getDashboardOverview().then(setStats).catch(() => {});
  }, []);

  function downloadCSV() {
    const header = "id,task_type,query,answer,confidence,processing_time_s,created_at\n";
    const rows = items
      .map((r) =>
        [r.id, r.task_type, csvEscape(r.query), csvEscape(r.answer),
          r.confidence, r.processing_time_s, r.created_at].join(",")
      ).join("\n");
    triggerDownload(new Blob([header + rows], { type: "text/csv" }), "satquery_history.csv");
  }

  function downloadJSON() {
    triggerDownload(new Blob([JSON.stringify(items, null, 2)], { type: "application/json" }), "satquery_history.json");
  }

  function downloadPDF() {
    const w = window.open("", "_blank");
    const rows = items.slice(0, 50).map((r) => `
      <tr>
        <td>${r.task_type}</td>
        <td>${(r.query || "").slice(0, 60)}</td>
        <td>${(r.answer || "").slice(0, 80)}</td>
        <td>${(r.confidence * 100).toFixed(0)}%</td>
        <td>${new Date(r.created_at * 1000).toLocaleString()}</td>
      </tr>`).join("");
    w.document.write(`<html><head><title>SatQuery AI Report</title>
      <style>
        body{font-family:Arial,sans-serif;padding:32px;color:#1e293b}
        h1{font-size:20px;margin-bottom:4px}p{color:#64748b;font-size:13px;margin-bottom:20px}
        .stats{display:flex;gap:24px;margin-bottom:24px;flex-wrap:wrap}
        .stat{background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px 20px}
        .stat-value{font-size:24px;font-weight:700;color:#0F172A}
        .stat-label{font-size:11px;color:#94a3b8;margin-top:2px}
        table{width:100%;border-collapse:collapse;font-size:12px}
        th{background:#0F172A;color:white;padding:8px 10px;text-align:left}
        td{padding:7px 10px;border-bottom:1px solid #e2e8f0}
        tr:nth-child(even) td{background:#f8fafc}
        .footer{margin-top:24px;font-size:11px;color:#94a3b8}
      </style></head><body>
      <h1>SatQuery AI — Analysis Report</h1>
      <p>Generated: ${new Date().toLocaleString()} · SIH 2026</p>
      ${stats ? `<div class="stats">
        <div class="stat"><div class="stat-value">${stats.total_analyses}</div><div class="stat-label">Total Analyses</div></div>
        <div class="stat"><div class="stat-value">${stats.change_detections}</div><div class="stat-label">Change Detections</div></div>
        <div class="stat"><div class="stat-value">${stats.vqa_queries}</div><div class="stat-label">VQA Queries</div></div>
        <div class="stat"><div class="stat-value">${stats.avg_accuracy_pct}%</div><div class="stat-label">Avg. Confidence</div></div>
      </div>` : ""}
      <table><thead><tr><th>Task</th><th>Query</th><th>Answer</th><th>Confidence</th><th>Time</th></tr></thead>
      <tbody>${rows}</tbody></table>
      <div class="footer">SatQuery AI · Agentic Multimodal Remote Sensing VQA · SIH 2026 · All data from real model outputs</div>
      </body></html>`);
    w.document.close();
    w.print();
  }

  return (
    <div className="p-6">
      <PageHeader
        title="Reports & Downloads"
        subtitle="Export your real analysis history from the backend SQLite log — no placeholder data."
      />

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-5 max-w-2xl">
        <ReportCard icon={<FileText size={18} className="text-brand-600" />}
          title="Export CSV" subtitle={`${items.length} records`}
          onClick={downloadCSV} disabled={items.length === 0} />
        <ReportCard icon={<FileText size={18} className="text-accent-violet" />}
          title="Export JSON" subtitle={`${items.length} records`}
          onClick={downloadJSON} disabled={items.length === 0} />
        <ReportCard icon={<FileBarChart2 size={18} className="text-green-600" />}
          title="Print / PDF" subtitle="Formatted summary"
          onClick={downloadPDF} disabled={items.length === 0} />
      </div>

      {items.length === 0 && (
        <p className="text-xs text-ink-400 mt-4">No records yet — run any analysis first.</p>
      )}

      {items.length > 0 && (
        <div className="mt-6 card p-4 overflow-x-auto">
          <h3 className="font-semibold text-sm text-ink-900 mb-3">Preview (last 10 records)</h3>
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-line">
                {["Task", "Query", "Answer", "Confidence", "Date"].map((h) => (
                  <th key={h} className="text-left py-2 px-2 text-ink-500 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {items.slice(0, 10).map((r) => (
                <tr key={r.id} className="border-b border-line hover:bg-surface">
                  <td className="py-2 px-2">
                    <span className="chip bg-brand-50 text-brand-700">{r.task_type}</span>
                  </td>
                  <td className="py-2 px-2 text-ink-700 max-w-[160px] truncate">{r.query}</td>
                  <td className="py-2 px-2 text-ink-700 max-w-[180px] truncate">{r.answer}</td>
                  <td className="py-2 px-2 text-ink-700">{(r.confidence * 100).toFixed(0)}%</td>
                  <td className="py-2 px-2 text-ink-400 whitespace-nowrap">
                    {new Date(r.created_at * 1000).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function ReportCard({ icon, title, subtitle, onClick, disabled }) {
  return (
    <button onClick={onClick} disabled={disabled}
      className="card p-4 flex items-center gap-3 text-left hover:border-brand-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
      <div className="w-9 h-9 rounded-lg bg-surface grid place-items-center shrink-0">{icon}</div>
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-ink-900">{title}</p>
        <p className="text-xs text-ink-400">{subtitle}</p>
      </div>
      <Download size={15} className="text-ink-400 shrink-0" />
    </button>
  );
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

function csvEscape(v) {
  return `"${String(v ?? "").replace(/"/g, '""')}"`;
}
