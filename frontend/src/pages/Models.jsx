import { useEffect, useState } from "react";
import { BrainCircuit, Cpu, Server } from "lucide-react";
import { PageHeader } from "./SingleImageAnalysis";
import { getModelsStatus } from "../lib/api";

export default function Models() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    getModelsStatus().then(setData).catch(() => setError(true));
  }, []);

  return (
    <div className="p-6">
      <PageHeader
        title="Model & System Status"
        subtitle="Live configuration pulled directly from the backend's settings — the exact model IDs actually in use."
      />

      {error && (
        <div className="card p-4 mt-5 text-sm text-accent-red bg-red-50 border-accent-red/20">
          Couldn't reach the backend.
        </div>
      )}

      {data && (
        <>
          <div className="card p-4 mt-5">
            <div className="flex items-center gap-2 mb-3">
              <BrainCircuit size={15} className="text-brand-600" />
              <h3 className="font-semibold text-sm">Active Models</h3>
            </div>
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-ink-500 border-b border-line">
                  <th className="pb-2 font-medium">Model</th>
                  <th className="pb-2 font-medium">HuggingFace / library ID</th>
                  <th className="pb-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {data.active_models.map((m) => (
                  <tr key={m.name} className="border-b border-line/60 last:border-0">
                    <td className="py-2.5 font-medium text-ink-800">{m.name}</td>
                    <td className="py-2.5 font-mono text-[11px] text-ink-500">{m.id}</td>
                    <td className="py-2.5">
                      <span className="chip bg-teal-50 text-accent-teal">{m.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-5">
            <div className="card p-4">
              <div className="flex items-center gap-2 mb-2">
                <Cpu size={15} className="text-accent-violet" />
                <h3 className="font-semibold text-sm">Inference Mode</h3>
              </div>
              <p className="text-xs text-ink-600 leading-relaxed">
                <span className="font-mono bg-surface px-1.5 py-0.5 rounded">
                  MODEL_WEIGHTS_MODE={data.weights_mode}
                </span>
                <br /><br />
                {data.weights_mode === "off"
                  ? "Running in demo mode — outputs come from real classical-CV image statistics (LAB-diff, HSV water masks, k-means clustering), not fabricated numbers. Set MODEL_WEIGHTS_MODE=on and provide internet access to enable real deep-model inference."
                  : "Deep models are enabled. Each response is tagged with which engine actually produced it."}
              </p>
            </div>
            <div className="card p-4">
              <div className="flex items-center gap-2 mb-2">
                <Server size={15} className="text-accent-amber" />
                <h3 className="font-semibold text-sm">LLM Orchestration</h3>
              </div>
              <p className="text-xs text-ink-600 leading-relaxed">
                Provider priority: <span className="font-mono bg-surface px-1.5 py-0.5 rounded">{data.llm_priority}</span>
                <br /><br />
                Groq (llama3-70b-8192) tried first if <span className="font-mono">GROQ_API_KEY</span> is set;
                falls back to OpenAI GPT-4o if <span className="font-mono">OPENAI_API_KEY</span> is set; otherwise
                falls back to keyword-based intent routing (still real, just not LLM-based).
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
