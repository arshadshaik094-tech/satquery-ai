import { useState } from "react";
import { GitCompareArrows, RotateCcw, Satellite } from "lucide-react";
import UploadDropzone from "../components/UploadDropzone";
import SatelliteTimePicker from "../components/SatelliteTimePicker";
import ExecutionConsole from "../components/ExecutionConsole";
import { PageHeader, EngineChip, NoteBox } from "./SingleImageAnalysis";
import { runChangeDetection } from "../lib/api";
import { useAnalysis } from "../context/AnalysisContext";

export default function ChangeDetection() {
  const { setImages } = useAnalysis();
  const [t1, setT1] = useState(null);
  const [t2, setT2] = useState(null);
  const [t1Preview, setT1Preview] = useState(null);
  const [t2Preview, setT2Preview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("upload"); // "upload" | "satellite"
  const [captureInfo, setCaptureInfo] = useState(null);

  async function analyze() {
    if (!t1 || !t2) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runChangeDetection({ image_t1_path: t1, image_t2_path: t2 });
      setResult(res);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }

  function handleSatelliteCapture(data) {
    setT1(data.beforePath);
    setT2(data.afterPath);
    setT1Preview(`http://localhost:8000/${data.beforePath}`);
    setT2Preview(`http://localhost:8000/${data.afterPath}`);
    setCaptureInfo(data);
    setImages({ image_t1_path: data.beforePath, image_t2_path: data.afterPath });
    setLoading(true);
    setError(null);
    setResult(null);
    setTimeout(() => {
      runChangeDetection({ image_t1_path: data.beforePath, image_t2_path: data.afterPath })
        .then((res) => { setResult(res); setLoading(false); })
        .catch((e) => { setError(e?.response?.data?.detail || e.message); setLoading(false); });
    }, 500);
  }

  return (
    <div className="p-6 max-w-4xl">
      <PageHeader
        title="Change Detection (Bi-Temporal)"
        subtitle="Siamese MobileNet-V3 (timm), intended fine-tune target: LEVIR-CD (637 building-change pairs, 0.5m)."
      />

      {/* Mode selector */}
      <div className="flex gap-2 mt-5 mb-4">
        <button
          onClick={() => setMode("upload")}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
            mode === "upload" ? "bg-brand-600 text-white" : "bg-surface text-ink-600 hover:bg-brand-50"
          }`}
        >
          Upload Images
        </button>
        <button
          onClick={() => setMode("satellite")}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
            mode === "satellite" ? "bg-brand-600 text-white" : "bg-surface text-ink-600 hover:bg-brand-50"
          }`}
        >
          <Satellite size={14} />
          Satellite Time-Travel
        </button>
      </div>

      {mode === "upload" ? (
        <div className="grid grid-cols-2 gap-4">
          <UploadDropzone label="T1 (Earlier)" compact
            onUploaded={(res) => { setT1(res.path); setT1Preview(null); setImages({ image_t1_path: res.path }); }} />
          <UploadDropzone label="T2 (Later)" compact
            onUploaded={(res) => { setT2(res.path); setT2Preview(null); setImages({ image_t2_path: res.path }); }} />
        </div>
      ) : (
        <>
          <SatelliteTimePicker onCapture={handleSatelliteCapture} />

          {/* Show captured images in preview boxes */}
          {(t1Preview || t2Preview) && (
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="relative rounded-xl border-2 border-brand-400 overflow-hidden bg-surface" style={{ height: "200px" }}>
                {t1Preview ? (
                  <>
                    <img src={t1Preview} alt="Before" className="w-full h-full object-cover" />
                    <div className="absolute bottom-2 left-2 right-2 bg-black/60 backdrop-blur-sm rounded px-2 py-1.5">
                      <p className="text-white text-xs font-semibold">Before: {captureInfo?.dateBefore}</p>
                    </div>
                  </>
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-ink-400">
                    <Satellite size={24} />
                  </div>
                )}
              </div>
              <div className="relative rounded-xl border-2 border-brand-400 overflow-hidden bg-surface" style={{ height: "200px" }}>
                {t2Preview ? (
                  <>
                    <img src={t2Preview} alt="After" className="w-full h-full object-cover" />
                    <div className="absolute bottom-2 left-2 right-2 bg-black/60 backdrop-blur-sm rounded px-2 py-1.5">
                      <p className="text-white text-xs font-semibold">After: {captureInfo?.dateAfter}</p>
                    </div>
                  </>
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-ink-400">
                    <Satellite size={24} />
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}

      {captureInfo && (
        <div className="card p-3 mt-4 text-xs bg-brand-50 border-brand-200">
          <p className="font-semibold text-brand-900">📡 Captured from EOX Sentinel-2</p>
          <p className="text-brand-700 mt-1">
            {captureInfo.location} · {captureInfo.dateBefore} → {captureInfo.dateAfter}
          </p>
          <p className="text-brand-600 mt-1 text-[11px]">✓ Running change detection automatically…</p>
        </div>
      )}

      {mode === "upload" && (
        <button onClick={analyze} disabled={!t1 || !t2 || loading}
          className="btn-primary flex items-center justify-center gap-2 mt-4 w-full sm:w-auto">
          {loading ? <RotateCcw size={14} className="animate-spin" /> : <GitCompareArrows size={14} />}
          {loading ? "Detecting change…" : "Run Change Detection"}
        </button>
      )}

      {error && (
        <div className="card p-4 mt-5 text-sm text-accent-red bg-red-50 border-accent-red/20">{error}</div>
      )}

      {(loading || result) && (
        <ExecutionConsole
          traces={result?.tools_used || []}
          processingTime={result?.processing_time_s}
          taskType="detect_change"
          loading={loading}
        />
      )}

      {result && (
        <div className="card p-5 mt-5 flex flex-col gap-4">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-semibold text-sm">Change Map</h3>
            <EngineChip engine={result.engine} />
          </div>

          {result.overlay_image_b64 && (
            <img src={`data:image/png;base64,${result.overlay_image_b64}`}
              alt="Change heatmap overlay" className="rounded-xl border border-line w-full" />
          )}

          <div className="flex items-center gap-6">
            <div>
              <p className="text-2xl font-bold text-ink-900">{result.change_percent}%</p>
              <p className="text-xs text-ink-500">of frame changed</p>
            </div>
            <div>
              <p className={`text-lg font-bold capitalize ${
                result.trend === "increased" ? "text-accent-red" :
                result.trend === "unchanged" ? "text-ink-500" : "text-accent-amber"
              }`}>{result.trend}</p>
              <p className="text-xs text-ink-500">trend</p>
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-[11px] text-ink-500">Change intensity</span>
            <div className="w-full bg-surface rounded-full h-3 border border-line overflow-hidden">
              <div className="h-3 rounded-full transition-all"
                style={{
                  width: `${Math.min(result.change_percent, 100)}%`,
                  background: result.change_percent > 20 ? "#EF4444" : result.change_percent > 8 ? "#F59E0B" : "#10B981"
                }} />
            </div>
            <span className="text-[11px] text-ink-400">{result.change_percent}% of image area affected</span>
          </div>

          {result.note && <NoteBox text={result.note} />}
        </div>
      )}
    </div>
  );
}
