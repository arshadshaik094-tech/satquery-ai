import { useState } from "react";
import { Droplets, RotateCcw, AlertTriangle } from "lucide-react";
import UploadDropzone from "../components/UploadDropzone";
import ExecutionConsole from "../components/ExecutionConsole";
import { PageHeader, EngineChip, NoteBox } from "./SingleImageAnalysis";
import { runFloodDetection } from "../lib/api";

export default function FloodDetection() {
  const [imagePath, setImagePath] = useState(null);
  const [loading, setLoading]     = useState(false);
  const [result, setResult]       = useState(null);
  const [error, setError]         = useState(null);

  async function analyze() {
    if (!imagePath) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await runFloodDetection({ image_path: imagePath, resolution_m: 10.0 });
      setResult(res);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  }

  const alertLevel =
    result?.flood_percent > 40 ? "high" :
    result?.flood_percent > 15 ? "medium" : "low";

  const alertColors = {
    high:   { bg: "bg-red-50",    border: "border-red-300",    text: "text-red-700",    badge: "bg-red-500"    },
    medium: { bg: "bg-amber-50",  border: "border-amber-300",  text: "text-amber-700",  badge: "bg-amber-500"  },
    low:    { bg: "bg-blue-50",   border: "border-blue-300",   text: "text-blue-700",   badge: "bg-blue-500"   },
  };
  const ac = alertColors[alertLevel];

  return (
    <div className="p-6 max-w-4xl">
      <PageHeader
        title="Flood & Water Extent Detection"
        subtitle="Detects water/flood pixels using NDWI-proxy + HSV heuristics. Outputs affected area in sq.km with confidence score."
      />

      {/* Info banner */}
      <div className="mt-5 flex items-start gap-3 p-3 rounded-xl bg-blue-50 border border-blue-200">
        <Droplets size={16} className="text-blue-600 mt-0.5 shrink-0" />
        <div className="text-xs text-blue-800 leading-relaxed">
          Upload a satellite image (optical/RGB). The model detects water/flood-covered areas,
          highlights them in blue, and calculates the affected area in sq.km.
          Works best with <strong>Sentinel-2</strong> or similar multispectral imagery.
        </div>
      </div>

      {/* Upload */}
      <div className="mt-5 max-w-md">
        <UploadDropzone
          label="Satellite Image (RGB)"
          onUploaded={(res) => { setImagePath(res.path); setResult(null); }}
        />
      </div>

      <button
        onClick={analyze}
        disabled={!imagePath || loading}
        className="btn-primary flex items-center justify-center gap-2 mt-4 w-full sm:w-auto"
      >
        {loading
          ? <><RotateCcw size={14} className="animate-spin" /> Detecting flood extent…</>
          : <><Droplets size={14} /> Detect Flood Areas</>
        }
      </button>

      {error && (
        <div className="card p-4 mt-5 text-sm text-accent-red bg-red-50 border-accent-red/20">
          {error}
        </div>
      )}

      {(loading || result) && (
        <ExecutionConsole
          traces={result?.tools_used || []}
          processingTime={result?.processing_time_s}
          taskType="flood_detection"
          loading={loading}
        />
      )}

      {result && (
        <div className="card p-5 mt-5 flex flex-col gap-5">

          {/* Alert level banner */}
          <div className={`flex items-center gap-3 p-3 rounded-xl border ${ac.bg} ${ac.border}`}>
            <AlertTriangle size={16} className={ac.text} />
            <div className="flex-1">
              <p className={`text-sm font-bold ${ac.text}`}>
                {alertLevel === "high"   ? "⚠ High Flood Coverage Detected" :
                 alertLevel === "medium" ? "⚡ Moderate Water Coverage"      :
                                          "ℹ Low Water Coverage"}
              </p>
              <p className={`text-xs mt-0.5 ${ac.text} opacity-80`}>
                Based on pixel analysis of the uploaded satellite image
              </p>
            </div>
            <span className={`text-white text-xs font-bold px-3 py-1.5 rounded-full ${ac.badge}`}>
              {alertLevel.toUpperCase()}
            </span>
          </div>

          {/* Key stats */}
          <div className="grid grid-cols-3 gap-4">
            <div className="rounded-xl bg-surface border border-line p-4 text-center">
              <p className="text-2xl font-black text-blue-600">
                {result.flood_area_km2.toLocaleString()}
              </p>
              <p className="text-xs text-ink-500 mt-1">sq.km affected</p>
            </div>
            <div className="rounded-xl bg-surface border border-line p-4 text-center">
              <p className="text-2xl font-black text-ink-900">
                {result.flood_percent}%
              </p>
              <p className="text-xs text-ink-500 mt-1">of image area</p>
            </div>
            <div className="rounded-xl bg-surface border border-line p-4 text-center">
              <p className="text-2xl font-black text-accent-teal">
                {Math.round(result.confidence * 100)}%
              </p>
              <p className="text-xs text-ink-500 mt-1">confidence</p>
            </div>
          </div>

          {/* Overlay image */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <h3 className="font-semibold text-sm">Flood Extent Overlay</h3>
              <EngineChip engine={result.engine} />
            </div>
            <div className="relative">
              <img
                src={`data:image/png;base64,${result.overlay_image_b64}`}
                alt="Flood detection overlay"
                className="rounded-xl border border-line w-full"
              />
              {/* Legend */}
              <div className="absolute bottom-3 left-3 flex items-center gap-1.5 bg-black/60 backdrop-blur-sm rounded-lg px-2.5 py-1.5">
                <span className="w-3 h-3 rounded-sm bg-blue-500 shrink-0" />
                <span className="text-white text-[11px] font-medium">Flood / Water</span>
              </div>
            </div>
          </div>

          {/* Coverage bar */}
          <div className="flex flex-col gap-1.5">
            <div className="flex justify-between text-[11px] text-ink-500">
              <span>Flood coverage</span>
              <span>{result.flood_percent}% of image</span>
            </div>
            <div className="w-full bg-surface rounded-full h-3 border border-line overflow-hidden">
              <div
                className="h-3 rounded-full transition-all bg-blue-500"
                style={{ width: `${Math.min(result.flood_percent, 100)}%` }}
              />
            </div>
          </div>

          {result.note && <NoteBox text={result.note} />}

          <div className="text-[11px] text-ink-400 border-t border-line pt-3">
            Pixel count: {result.pixel_count.toLocaleString()} px · Resolution: 10m/px ·
            Engine: NDWI-proxy + HSV water heuristic
          </div>
        </div>
      )}
    </div>
  );
}
