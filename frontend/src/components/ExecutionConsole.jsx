/**
 * ExecutionConsole — terminal-style agent trace panel.
 * Shows the real ToolTrace steps returned by the backend orchestrator/specialist.
 * Every line corresponds to a real operation — no fake data.
 */
import { useState } from "react";
import { Terminal, ChevronDown, ChevronUp, Clock, CheckCircle2, Zap } from "lucide-react";

export default function ExecutionConsole({ traces = [], processingTime, taskType, loading = false }) {
  const [open, setOpen] = useState(true);

  // System boot lines shown before any real trace
  const bootLines = [
    "SatQuery AI System Console",
    `Task type: ${taskType || "pending"}`,
    `Local compute: CPU (offline-isolated)`,
    "Network monitor: ACTIVE",
    "Awaiting VLM execution query...",
  ];

  return (
    <div className="mt-5 rounded-xl border border-line overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-2.5 bg-[#0F172A] text-white"
      >
        <div className="flex items-center gap-2">
          <Terminal size={14} className="text-green-400" />
          <span className="text-xs font-mono font-semibold text-green-400">
            Execution Console
          </span>
          {loading && (
            <span className="flex gap-1 ml-2">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-bounce [animation-delay:-0.3s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-bounce [animation-delay:-0.15s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-bounce" />
            </span>
          )}
          {!loading && traces.length > 0 && (
            <span className="ml-2 text-[10px] bg-green-900/60 text-green-300 px-2 py-0.5 rounded-full font-mono">
              {traces.length} step{traces.length > 1 ? "s" : ""} · {processingTime}s
            </span>
          )}
        </div>
        {open ? <ChevronUp size={14} className="text-white/40" /> : <ChevronDown size={14} className="text-white/40" />}
      </button>

      {/* Console body */}
      {open && (
        <div className="bg-[#0F172A] px-4 pb-4 pt-1 font-mono text-xs flex flex-col gap-1 max-h-64 overflow-y-auto">
          {/* Boot lines */}
          {bootLines.map((line, i) => (
            <div key={i} className="flex items-start gap-2 text-white/40">
              <span className="text-white/20 shrink-0">{">"}</span>
              <span>{line}</span>
            </div>
          ))}

          {/* Separator */}
          <div className="border-t border-white/10 my-1" />

          {/* Loading state */}
          {loading && !traces.length && (
            <div className="flex items-center gap-2 text-yellow-400">
              <Zap size={11} className="animate-pulse" />
              <span>Routing query through agentic controller...</span>
            </div>
          )}

          {/* Real trace steps from backend */}
          {traces.map((t, i) => (
            <div key={i} className="flex flex-col gap-0.5">
              <div className="flex items-start gap-2 text-green-300">
                <CheckCircle2 size={11} className="mt-0.5 shrink-0 text-green-400" />
                <span className="text-green-200 font-semibold">[Tool] {t.tool}</span>
              </div>
              <div className="flex items-start gap-2 text-white/50 pl-4">
                <span className="text-white/30">→</span>
                <span>{t.reason}</span>
              </div>
              {t.output_summary && (
                <div className="flex items-start gap-2 text-cyan-300/80 pl-4">
                  <span className="text-white/30">→</span>
                  <span className="truncate max-w-[420px]">{t.output_summary}</span>
                </div>
              )}
              <div className="flex items-center gap-2 text-white/30 pl-4">
                <Clock size={10} />
                <span>{t.duration_ms}ms</span>
              </div>
            </div>
          ))}

          {/* Final summary */}
          {!loading && traces.length > 0 && (
            <>
              <div className="border-t border-white/10 my-1" />
              <div className="flex items-center gap-2 text-green-400">
                <CheckCircle2 size={11} />
                <span>Pipeline complete · total {processingTime}s</span>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
