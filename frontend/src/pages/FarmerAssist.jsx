import { useState } from "react";
import { Sprout, RotateCcw } from "lucide-react";
import { analyzeSoil, analyzePlant, getFarmerAdvice } from "../lib/api";
import { PageHeader, EngineChip, NoteBox, ConfidenceBar } from "./SingleImageAnalysis";

export default function FarmerAssist() {
  const [soilFile, setSoilFile]       = useState(null);
  const [plantFile, setPlantFile]     = useState(null);
  const [soilResult, setSoilResult]   = useState(null);
  const [plantResult, setPlantResult] = useState(null);
  const [advice, setAdvice]           = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);

  async function runAnalysis() {
    if (!soilFile) return;
    setLoading(true);
    setError(null);
    setSoilResult(null);
    setPlantResult(null);
    setAdvice(null);
    try {
      const soil = await analyzeSoil(soilFile);
      setSoilResult(soil);

      let plant = null;
      if (plantFile) {
        plant = await analyzePlant(plantFile);
        setPlantResult(plant);
      }

      const adv = await getFarmerAdvice(soil.soil_type, plant?.health_signal ?? null);
      setAdvice(adv);
    } catch (e) {
      setError(e?.response?.data?.detail || e.message || "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setSoilFile(null);
    setPlantFile(null);
    setSoilResult(null);
    setPlantResult(null);
    setAdvice(null);
    setError(null);
  }

  return (
    <div className="p-6 max-w-4xl">
      <PageHeader
        title="Farmer Assist"
        subtitle="Soil-type & texture analysis (EfficientNet-B0 / colour+GLCM heuristic) + plant health signal (ResNet-18 / Excess Green Index). Upload a soil photo to get started."
      />

      {/* Disclaimer */}
      <div className="card p-4 mt-5 text-xs text-ink-600 bg-amber-50 border border-accent-amber/30 leading-relaxed rounded-xl">
        ⚠️ Soil bulk density (g/cm³) cannot be measured from a photo — it requires a lab test.
        This tool provides a colour/texture-based soil-type hint and a plant health signal only.
        Use it as a starting point, not a substitute for a soil test or your local Krishi Vigyan Kendra (KVK).
      </div>

      {/* Upload cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-5">
        <div className="card p-4">
          <p className="text-xs font-medium text-ink-500 mb-2">1. Soil photo <span className="text-accent-red">*</span></p>
          <DropZone onFile={setSoilFile} label="Close-up photo of soil" icon={Sprout} />
        </div>
        <div className="card p-4">
          <p className="text-xs font-medium text-ink-500 mb-2">2. Plant / leaf photo <span className="text-ink-400">(optional)</span></p>
          <DropZone onFile={setPlantFile} label="Photo of crop leaf" icon={Sprout} />
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-3 mt-4">
        <button
          onClick={runAnalysis}
          disabled={!soilFile || loading}
          className="btn-primary flex items-center gap-2"
        >
          {loading
            ? <RotateCcw size={14} className="animate-spin" />
            : <Sprout size={14} />}
          {loading ? "Analyzing…" : "Run Farmer Assist"}
        </button>
        {(soilResult || error) && (
          <button onClick={reset} className="btn-ghost text-xs">
            Reset
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="card p-4 mt-5 text-sm text-accent-red bg-red-50 border border-accent-red/20 rounded-xl">
          {error}
        </div>
      )}

      {/* Soil result */}
      {soilResult && (
        <div className="card p-5 mt-5 flex flex-col gap-3">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-semibold text-sm text-ink-900">Soil Analysis</h3>
            <EngineChip engine={soilResult.engine} />
          </div>

          <p className="text-sm text-ink-800">
            <span className="font-semibold">{soilResult.soil_type}</span>
          </p>

          <ConfidenceBar value={soilResult.confidence} label="Confidence" />

          {soilResult.texture_apparent && (
            <div className="flex flex-col gap-0.5">
              <span className="text-[11px] text-ink-400 font-medium uppercase tracking-wide">Apparent texture</span>
              <p className="text-sm text-ink-700">{soilResult.texture_apparent}</p>
            </div>
          )}

          {soilResult.moisture_hint && (
            <div className="flex flex-col gap-0.5">
              <span className="text-[11px] text-ink-400 font-medium uppercase tracking-wide">Moisture hint</span>
              <p className="text-sm text-ink-700">{soilResult.moisture_hint}</p>
            </div>
          )}

          {soilResult.hsv_mean && (
            <div className="flex gap-4 text-xs text-ink-500">
              <span>H: {soilResult.hsv_mean.h}</span>
              <span>S: {soilResult.hsv_mean.s}</span>
              <span>V: {soilResult.hsv_mean.v}</span>
            </div>
          )}

          {soilResult.note && <NoteBox text={soilResult.note} />}
        </div>
      )}

      {/* Plant health result */}
      {plantResult && (
        <div className="card p-5 mt-4 flex flex-col gap-3">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-semibold text-sm text-ink-900">Plant Health</h3>
            <EngineChip engine={plantResult.engine} />
          </div>

          <p className="text-sm text-ink-800">
            {plantResult.crop
              ? <><span className="font-semibold">{plantResult.crop}</span> — {plantResult.condition}</>
              : plantResult.health_signal}
          </p>

          <ConfidenceBar value={plantResult.confidence} label="Confidence" />

          {plantResult.greenness_index_exg != null && (
            <div className="grid grid-cols-3 gap-3 text-xs">
              <div className="bg-surface border border-line rounded-lg p-2.5 text-center">
                <p className="text-ink-400">Greenness (ExG)</p>
                <p className="font-semibold text-ink-800 mt-0.5">{plantResult.greenness_index_exg}</p>
              </div>
              <div className="bg-surface border border-line rounded-lg p-2.5 text-center">
                <p className="text-ink-400">Discoloration</p>
                <p className="font-semibold text-ink-800 mt-0.5">{plantResult.discoloration_percent}%</p>
              </div>
              <div className="bg-surface border border-line rounded-lg p-2.5 text-center">
                <p className="text-ink-400">Necrotic spots</p>
                <p className="font-semibold text-ink-800 mt-0.5">{plantResult.necrotic_spot_percent}%</p>
              </div>
            </div>
          )}

          {plantResult.note && <NoteBox text={plantResult.note} />}
        </div>
      )}

      {/* Advisory */}
      {advice && (
        <div className="card p-5 mt-4 flex flex-col gap-3">
          <h3 className="font-semibold text-sm text-ink-900">General Pointers</h3>
          <ul className="flex flex-col gap-2">
            {advice.pointers.map((p, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-ink-700">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-500 mt-1.5 shrink-0" />
                {p}
              </li>
            ))}
          </ul>
          <p className="text-[11px] text-ink-400 leading-relaxed border-t border-line pt-2 mt-1">
            {advice.disclaimer}
          </p>
        </div>
      )}
    </div>
  );
}

function DropZone({ onFile, label, icon: Icon }) {
  const [preview, setPreview] = useState(null);
  return (
    <label className="relative cursor-pointer rounded-xl border-2 border-dashed border-line hover:border-brand-400 bg-surface h-40 flex flex-col items-center justify-center text-center overflow-hidden block transition-colors">
      <input
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (!f) return;
          setPreview(URL.createObjectURL(f));
          onFile(f);
        }}
      />
      {preview ? (
        <img src={preview} alt={label} className="absolute inset-0 w-full h-full object-cover" />
      ) : (
        <>
          <Icon size={20} className="text-ink-400 mb-2" />
          <p className="text-sm font-medium text-ink-700">{label}</p>
          <p className="text-xs text-ink-400 mt-1">Click to choose a photo</p>
        </>
      )}
    </label>
  );
}
