import { useState } from "react";
import { MessageCircleQuestion, RotateCcw } from "lucide-react";
import UploadDropzone from "../components/UploadDropzone";
import ExecutionConsole from "../components/ExecutionConsole";
import { PageHeader, EngineChip, NoteBox, ConfidenceBar } from "./SingleImageAnalysis";
import { runGrounding } from "../lib/api";
import { useAnalysis } from "../context/AnalysisContext";

export default function GroundingVQA() {
  const { setImages } = useAnalysis();
  const [imgPath, setImgPath] = useState(null);
  const [expr, setExpr] = useState("Highlight the water body.");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function analyze() {
    if (!imgPath || !expr.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runGrounding({ image_path: imgPath, referring_expression: expr });
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
        title="Grounding (VQA)"
        subtitle="SAM (ViT-B, Meta AI) prompted from LLM-parsed referring expressions + real land-cover heuristics."
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mt-5">
        <div className="card p-4">
          <p className="text-xs font-medium text-ink-500 mb-2">1. Upload imagery</p>
          <UploadDropzone label="Satellite / aerial image"
            onUploaded={(res) => { setImgPath(res.path); setImages({ image_t1_path: res.path, image_t2_path: null }); }} />
        </div>
        <div className="card p-4 flex flex-col gap-3">
          <p className="text-xs font-medium text-ink-500">2. What to highlight</p>
          <input value={expr} onChange={(e) => setExpr(e.target.value)}
            className="w-full text-sm bg-surface border border-line rounded-lg p-3 outline-none focus:border-brand-400"
            placeholder="e.g. Highlight the water body." />
          <div className="flex flex-wrap gap-1.5">
            {["Highlight the water body.", "Highlight built-up areas.", "Highlight vegetation."].map((q) => (
              <button key={q} onClick={() => setExpr(q)}
                className="text-[11px] text-brand-700 bg-brand-50 hover:bg-brand-100 rounded-full px-2.5 py-1">
                {q}
              </button>
            ))}
          </div>
          <button onClick={analyze} disabled={!imgPath || loading}
            className="btn-primary flex items-center justify-center gap-2 mt-auto">
            {loading ? <RotateCcw size={14} className="animate-spin" /> : <MessageCircleQuestion size={14} />}
            {loading ? "Grounding…" : "Run Grounding"}
          </button>
        </div>
      </div>

      {error && (
        <div className="card p-4 mt-5 text-sm text-accent-red bg-red-50 border-accent-red/20">{error}</div>
      )}

      {(loading || result) && (
        <ExecutionConsole
          traces={result?.tools_used || []}
          processingTime={result?.processing_time_s}
          taskType="ground_referring_expression"
          loading={loading}
        />
      )}

      {result && (
        <div className="card p-5 mt-5 flex flex-col gap-4">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-semibold text-sm">Grounded Region: {result.target}</h3>
            <EngineChip engine={result.engine} />
          </div>

          {/* Not detected case */}
          {result.coverage_percent === 0 ? (
            <div className="rounded-xl bg-amber-50 border border-amber-200 p-4 text-sm text-amber-800">
              ⚠️ No <strong>{result.target}</strong> region detected in this image. Try a different image or expression.
            </div>
          ) : (
            <>
              {result.overlay_image_b64 && (
                <img src={`data:image/png;base64,${result.overlay_image_b64}`}
                  alt="Grounding overlay" className="rounded-xl border border-line w-full" />
              )}

              <div className="grid grid-cols-2 gap-4">
                <div className="card p-3 bg-surface">
                  <p className="text-xs text-ink-500 mb-1">Coverage</p>
                  <p className="text-xl font-bold text-ink-900">{result.coverage_percent ?? "n/a"}%</p>
                  <p className="text-[11px] text-ink-400">of image frame</p>
                </div>
                <div className="card p-3 bg-surface">
                  <p className="text-xs text-ink-500 mb-1">Target region</p>
                  <p className="text-xl font-bold text-ink-900 capitalize">{result.target}</p>
                  <p className="text-[11px] text-ink-400">detected</p>
                </div>
              </div>

              <ConfidenceBar value={result.confidence} label="Grounding confidence" />
            </>
          )}
          {result.note && result.coverage_percent > 0 && <NoteBox text={result.note} />}
        </div>
      )}
    </div>
  );
}
