import { useEffect, useState } from "react";
import { Settings as SettingsIcon, KeyRound } from "lucide-react";
import { PageHeader } from "./SingleImageAnalysis";
import { getModelsStatus, API_BASE } from "../lib/api";

export default function Settings() {
  const [data, setData] = useState(null);

  useEffect(() => {
    getModelsStatus().then(setData).catch(() => {});
  }, []);

  return (
    <div className="p-6 max-w-2xl">
      <PageHeader
        title="Settings"
        subtitle="Real configuration is set via environment variables on the backend — this page reflects, not fakes, current state."
      />

      <div className="card p-4 mt-5 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <SettingsIcon size={15} className="text-brand-600" />
          <h3 className="font-semibold text-sm">Connection</h3>
        </div>
        <Row label="API Base URL" value={API_BASE} mono />
        <Row label="Weights mode" value={data?.weights_mode ?? "—"} mono />
        <Row label="LLM provider priority" value={data?.llm_priority ?? "—"} mono />
      </div>

      <div className="card p-4 mt-4 flex flex-col gap-3">
        <div className="flex items-center gap-2">
          <KeyRound size={15} className="text-accent-amber" />
          <h3 className="font-semibold text-sm">API Keys</h3>
        </div>
        <p className="text-xs text-ink-600 leading-relaxed">
          Groq / OpenAI keys are never entered in this UI or sent to the frontend — they live only in{" "}
          <code className="font-mono bg-surface px-1 py-0.5 rounded">backend/.env</code> on your machine
          (<code className="font-mono">GROQ_API_KEY</code>, <code className="font-mono">OPENAI_API_KEY</code>).
          This keeps them out of the browser and out of version control.
        </p>
      </div>
    </div>
  );
}

function Row({ label, value, mono }) {
  return (
    <div className="flex items-center justify-between text-sm py-1.5 border-b border-line/60 last:border-0">
      <span className="text-ink-500">{label}</span>
      <span className={`text-ink-800 ${mono ? "font-mono text-xs" : ""}`}>{value}</span>
    </div>
  );
}
