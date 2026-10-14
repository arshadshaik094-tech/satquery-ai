import { useState } from "react";
import { Layers, RotateCcw } from "lucide-react";
import UploadDropzone from "../components/UploadDropzone";
import ExecutionConsole from "../components/ExecutionConsole";
import { PageHeader, EngineChip, NoteBox } from "./SingleImageAnalysis";
import { runFusion } from "../lib/api";
import { useAnalysis } from "../context/AnalysisContext";

export default function OpticalSarAnalysis() {
  const { setImages } = useAnalysis();
  const [optical, setOptical] = useState(null);
  const [sar, setSar] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function analyze() {
    if (!optical || !sar) return;
    setLoading(true);
    setError(null);
    try {
      const res = await runFusion({ optical_path: optical, sar_path: sar });
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
        title="Optical + SAR Analysis"
        subtitle="Dual-branch ResNet-18 (torchvision), intended fine-tune target: SEN12MS (co-registered Sentinel-1/2 patches)."
      />

      <div className="grid grid-cols-2 gap-4 mt-5">
        <UploadDropzone label="Optical (Sentinel-2 style)" compact
          onUploaded={(res) => { setOptical(res.path); setImages({ optical_path: res.path }); }} />
        <UploadDropzone label="SAR (Sentinel-1 style)" compact
          onUploaded={(res) => { setSar(res.path); setImages({ sar_path: res.path }); }} />
      </div>

      <button onClick={analyze} disabled={!optical || !sar || loading}
        className="btn-primary flex items-center justify-center gap-2 mt-4 w-full sm:w-auto">
        {loading ? <RotateCcw size={14} className="animate-spin" /> : <Layers size={14} />}
        {loading ? "Fusing modalities…" : "Run Optical+SAR Fusion"}
      </button>

      {error && (
        <div className="card p-4 mt-5 text-sm text-accent-red bg-red-50 border-accent-red/20">{error}</div>
      )}

      {(loading || result) && (
        <ExecutionConsole
          traces={result?.tools_used || []}
          processingTime={result?.processing_time_s}
          taskType="fuse_optical_sar"
          loading={loading}
        />
      )}

      {result && (
        <div className="card p-5 mt-5 flex flex-col gap-4">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-semibold text-sm">Fusion Result</h3>
            <EngineChip engine={result.engine} />
          </div>

          {result.overlay_image_b64 && (
            <img src={`data:image/png;base64,${result.overlay_image_b64}`}
              alt="Fusion overlay" className="rounded-xl border border-line w-full" />
          )}

          {/* Visual percentage bars for water and built-up */}
          <div className="flex flex-col gap-3">
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[12px] font-medium text-blue-600">🔵 Water coverage</span>
                <span className="text-[12px] font-bold text-ink-800">{result.water_percent}%</span>
              </div>
              <div className="w-full bg-surface rounded-full h-3 border border-line overflow-hidden">
                <div className="h-3 rounded-full transition-all bg-blue-400"
                  style={{ width: `${Math.min(result.water_percent, 100)}%` }} />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[12px] font-medium text-red-500">🔴 Built-up coverage</span>
                <span className="text-[12px] font-bold text-ink-800">{result.built_up_percent}%</span>
              </div>
              <div className="w-full bg-surface rounded-full h-3 border border-line overflow-hidden">
                <div className="h-3 rounded-full transition-all bg-red-400"
                  style={{ width: `${Math.min(result.built_up_percent, 100)}%` }} />
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[12px] font-medium text-green-600">🟢 Other / vegetation</span>
                <span className="text-[12px] font-bold text-ink-800">
                  {Math.max(0, 100 - result.water_percent - result.built_up_percent).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-surface rounded-full h-3 border border-line overflow-hidden">
                <div className="h-3 rounded-full transition-all bg-green-400"
                  style={{ width: `${Math.max(0, 100 - result.water_percent - result.built_up_percent)}%` }} />
              </div>
            </div>
          </div>

          {result.note && <NoteBox text={result.note} />}
        </div>
      )}
    </div>
  );
}
