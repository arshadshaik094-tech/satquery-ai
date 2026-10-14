import { useState } from "react";
import { Search, Camera, Aperture, RotateCcw } from "lucide-react";
import UploadDropzone from "../components/UploadDropzone";
import ExecutionConsole from "../components/ExecutionConsole";
import { runVQA } from "../lib/api";
import { useAnalysis } from "../context/AnalysisContext";

export default function SingleImageAnalysis() {
  const { setImages } = useAnalysis();
  const [imgPath, setImgPath] = useState(null);
  const [question, setQuestion] = useState("Is there a river in the image?");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  function handleUploaded(res) {
    setImgPath(res.path);
    setImages({ image_t1_path: res.path, image_t2_path: null });
    setResult(null);
  }

  async function analyze() {
    if (!imgPath || !question.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runVQA({ image_path: imgPath, question });
      setResult(res);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-6 max-w-4xl">
      <PageHeader
        title="Single Image Analysis"
        subtitle="BLIP-1 (Salesforce/blip-vqa-base) for open-ended VQA + ResNet-50 (timm, EuroSAT) for scene classification."
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mt-5">
        <div className="card p-4">
          <p className="text-xs font-medium text-ink-500 mb-2">1. Upload imagery</p>
          <UploadDropzone label="Satellite / aerial image" onUploaded={handleUploaded} />
        </div>

        <div className="card p-4 flex flex-col gap-3">
          <p className="text-xs font-medium text-ink-500">2. Ask a question</p>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            className="w-full text-sm bg-surface border border-line rounded-lg p-3 outline-none focus:border-brand-400 resize-none"
            placeholder="e.g. What land-cover types are visible?"
          />
          <div className="flex flex-wrap gap-1.5">
            {[
              "Is there a river in the image?",
              "Describe the land-cover and major objects visible.",
              "What is the dominant land-use type?",
            ].map((q) => (
              <button key={q} onClick={() => setQuestion(q)}
                className="text-[11px] text-brand-700 bg-brand-50 hover:bg-brand-100 rounded-full px-2.5 py-1">
                {q}
              </button>
            ))}
          </div>
          <button onClick={analyze} disabled={!imgPath || loading}
            className="btn-primary flex items-center justify-center gap-2 mt-auto">
            {loading ? <RotateCcw size={14} className="animate-spin" /> : <Search size={14} />}
            {loading ? "Analyzing…" : "Run Analysis"}
          </button>
        </div>
      </div>

      {error && (
        <div className="card p-4 mt-5 text-sm text-accent-red bg-red-50 border-accent-red/20">{error}</div>
      )}

      {/* Execution Console — shows real trace from backend */}
      {(loading || result) && (
        <ExecutionConsole
          traces={result?.tools_used || []}
          processingTime={result?.processing_time_s}
          taskType="describe_image"
          loading={loading}
        />
      )}

      {result && (
        <div className="card p-5 mt-5 flex flex-col gap-4">
          {/* VQA */}
          <div className="flex items-center gap-2">
            <Camera size={15} className="text-brand-600" />
            <h3 className="font-semibold text-sm">VQA Answer</h3>
            <EngineChip engine={result.vqa?.engine} />
          </div>
          <p className="text-sm text-ink-800 leading-relaxed">{result.vqa?.answer}</p>

          {/* Confidence bar */}
          <ConfidenceBar value={result.vqa?.confidence} />
          {result.vqa?.note && <NoteBox text={result.vqa.note} />}

          {/* Land-cover */}
          <div className="border-t border-line pt-4 flex items-center gap-2">
            <Aperture size={15} className="text-accent-violet" />
            <h3 className="font-semibold text-sm">Land-cover Classification</h3>
            <EngineChip engine={result.landcover?.engine} />
          </div>

          {result.landcover?.label === "unsupervised-cluster-summary" ? (
            <div className="grid grid-cols-5 gap-2">
              {Object.entries(result.landcover.clusters || {}).map(([k, v]) => (
                <div key={k} className="text-center">
                  <div className="h-14 rounded-md bg-brand-400" style={{ opacity: 0.3 + v * 1.4 }} />
                  <p className="text-[10px] text-ink-500 mt-1">Cluster {k}</p>
                  <p className="text-[10px] font-semibold text-ink-800">{(v * 100).toFixed(0)}%</p>
                </div>
              ))}
            </div>
          ) : (
            <>
              <p className="text-sm text-ink-800">
                <span className="font-semibold">{result.landcover?.label}</span>
              </p>
              {/* All-class probability bars when available */}
              {result.landcover?.all_probs && (
                <div className="flex flex-col gap-1.5">
                  {Object.entries(result.landcover.all_probs)
                    .sort(([, a], [, b]) => b - a)
                    .slice(0, 5)
                    .map(([cls, prob]) => (
                      <div key={cls} className="flex items-center gap-2">
                        <span className="text-[11px] text-ink-500 w-36 shrink-0">{cls}</span>
                        <div className="flex-1 bg-surface rounded-full h-2 border border-line overflow-hidden">
                          <div
                            className="h-2 rounded-full bg-brand-500 transition-all"
                            style={{ width: `${(prob * 100).toFixed(1)}%` }}
                          />
                        </div>
                        <span className="text-[11px] text-ink-500 w-10 text-right">
                          {(prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                </div>
              )}
              <ConfidenceBar value={result.landcover?.confidence} label="Classification confidence" />
            </>
          )}
          {result.landcover?.note && <NoteBox text={result.landcover.note} />}
        </div>
      )}
    </div>
  );
}

export function PageHeader({ title, subtitle }) {
  return (
    <div>
      <h2 className="text-xl font-bold text-ink-900">{title}</h2>
      <p className="text-sm text-ink-500 mt-0.5">{subtitle}</p>
    </div>
  );
}

export function EngineChip({ engine }) {
  if (!engine) return null;
  const isDeep = !engine.includes("classical");
  return (
    <span className={`chip ${isDeep ? "bg-teal-50 text-accent-teal" : "bg-amber-50 text-accent-amber"}`}
      style={{ background: isDeep ? "#E7FBF7" : "#FEF3E2" }}>
      {isDeep ? "Deep model" : "Classical-CV fallback"}: {engine}
    </span>
  );
}

export function NoteBox({ text }) {
  return (
    <div className="text-[11px] text-ink-500 bg-surface border border-line rounded-lg p-2.5 leading-relaxed">
      ℹ️ {text}
    </div>
  );
}

export function ConfidenceBar({ value, label = "Confidence" }) {
  if (value == null) return null;
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? "#10B981" : pct >= 60 ? "#F59E0B" : "#EF4444";
  return (
    <div className="flex items-center gap-3">
      <span className="text-[11px] text-ink-500 w-20 shrink-0">{label}</span>
      <div className="flex-1 bg-surface rounded-full h-2 border border-line overflow-hidden">
        <div className="h-2 rounded-full transition-all" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="text-[11px] font-semibold text-ink-700 w-10 text-right">{pct}%</span>
    </div>
  );
}
