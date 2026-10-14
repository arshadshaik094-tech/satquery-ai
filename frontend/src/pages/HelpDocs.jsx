import { HelpCircle, Terminal } from "lucide-react";
import { PageHeader } from "./SingleImageAnalysis";

const STEPS = [
  { cmd: "cd backend && python -m venv .venv && source .venv/bin/activate", note: "Create a virtualenv" },
  { cmd: "pip install -r requirements.txt", note: "Install real deps: torch, transformers, timm, segment-anything, groq, openai, fastapi…" },
  { cmd: "cp .env.example .env  # then add GROQ_API_KEY / OPENAI_API_KEY", note: "Configure LLM keys (optional but recommended)" },
  { cmd: "uvicorn app.main:app --reload --port 8000", note: "Start the real FastAPI backend" },
  { cmd: "cd ../frontend && npm install && npm run dev", note: "Start the React dashboard (default: http://localhost:5173)" },
];

export default function HelpDocs() {
  return (
    <div className="p-6 max-w-2xl">
      <PageHeader title="Help & Docs" subtitle="Getting the real system running end-to-end on your machine." />

      <div className="card p-4 mt-5">
        <div className="flex items-center gap-2 mb-3">
          <Terminal size={15} className="text-brand-600" />
          <h3 className="font-semibold text-sm">Run locally</h3>
        </div>
        <div className="flex flex-col gap-3">
          {STEPS.map((s, i) => (
            <div key={i} className="flex flex-col gap-1">
              <code className="font-mono text-[11px] bg-navy-950 text-teal-300 rounded-lg px-3 py-2 block overflow-x-auto">
                {s.cmd}
              </code>
              <p className="text-[11px] text-ink-400">{s.note}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card p-4 mt-4">
        <div className="flex items-center gap-2 mb-2">
          <HelpCircle size={15} className="text-accent-violet" />
          <h3 className="font-semibold text-sm">Fine-tuning specialist heads</h3>
        </div>
        <p className="text-xs text-ink-600 leading-relaxed">
          Training scripts for each specialist head are in <code className="font-mono bg-surface px-1 rounded">backend/train/</code>{" "}
          (EuroSAT classifier, LEVIR-CD change head, SEN12MS fusion head). Until you run them and point the
          corresponding <code className="font-mono">*_WEIGHTS</code> env var at the checkpoint, the backend
          transparently uses classical-CV fallbacks — see the Models page for live status.
        </p>
      </div>
    </div>
  );
}
