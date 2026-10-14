import { useState } from "react";
import { Sprout, RotateCcw } from "lucide-react";
// Reuse the shared helpers already in your project from SingleImageAnalysis.jsx
import { PageHeader, EngineChip, NoteBox } from "./SingleImageAnalysis";
import { API_BASE } from "../lib/api";

// --- Add these to your existing src/lib/api.js instead of importing from here ---
async function analyzeSoil(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/farmer/soil`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "Soil analysis failed");
  return res.json();
}
async function analyzePlant(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/farmer/plant`, { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "Plant analysis failed");
  return res.json();
}
async function getAdvice(soil_type, health_signal) {
  const res = await fetch(`${API_BASE}/api/farmer/advise`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ soil_type, health_signal }),
  });
  if (!res.ok) throw new Error("Advisory failed");
  return res.json();
}
// --- end api additions ---

export default function FarmerAssist() {
  const [soilFile, setSoilFile] = useState(null);
  const [plantFile, setPlantFile] = useState(null);
  const [soilResult, setSoilResult] = useState(null);
  const [plantResult, setPlantResult] = useState(null);
  const [advice, setAdvice] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function runAnalysis() {
    if (!soilFile) return;
    setLoading(true);
    setError(null);
    try {
      const soil = await analyzeSoil(soilFile);
      setSoilResult(soil);
      let plant = null;
      if (plantFile) {
        plant = await analyzePlant(plantFile);
        setPlantResult(plant);
      }
      const adv = await getAdvice(soil.soil_type, plant?.health_signal);
      setAdvice(adv);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-6 max-w-4xl">
      <PageHeader
        title="Farmer Assist"
        subtitle="Soil-type & texture hinting (EfficientNet-B0, colour/GLCM texture analysis) + plant health signal (ResNet-18, PlantVillage-targeted). Does not estimate soil density — see note below."
      />

      <div className="card p-4 mt-5 text-xs text-ink-600 bg-amber-50 border-accent-amber/20 leading-relaxed">
        ⚠️ Soil bulk density (g/cm³) cannot be measured from a photo — it needs a lab test. This tool
        gives a colour/texture-based soil-type hint and a plant health signal only, both clearly
        confidence-capped. Use it as a starting point, not a substitute for a soil test or your local
        Krishi Vigyan Kendra (KVK).
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div className="card p-4">
          <p className="text-xs font-medium text-ink-500 mb-2">1. Soil photo (required)</p>
          <UploadDropzoneLocal onFile={setSoilFile} label="Photo of soil, close-up" />
        </div>
        <div className="card p-4">
          <p className="text-xs font-medium text-ink-500 mb-2">2. Plant/leaf photo (optional)</p>
          <UploadDropzoneLocal onFile={setPlantFile} label="Photo of crop leaf" />
        </div>
      </div>

      <button
        onClick={runAnalysis}
        disabled={!soilFile || loading}
        className="btn-primary flex items-center justify-center gap-2 mt-4"
      >
        {loading ? <RotateCcw size={14} className="animate-spin" /> : <Sprout size={14} />}
        {loading ? "Analyzing…" : "Run Farmer Assist"}
      </button>

      {error && <div className="card p-4 mt-5 text-sm text-accent-red bg-red-50 border-accent-red/20">{error}</div>}

      {soilResult && (
        <div className="card p-5 mt-5 flex flex-col gap-3">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-semibold text-sm">Soil Analysis</h3>
            <EngineChip engine={soilResult.engine} />
          </div>
          <p className="text-sm text-ink-800">
            <span className="font-semibold">{soilResult.soil_type}</span> · confidence{" "}
            {(soilResult.confidence * 100).toFixed(0)}%
          </p>
          {soilResult.texture_apparent && (
            <p className="text-xs text-ink-600">Apparent texture: {soilResult.texture_apparent}</p>
          )}
          {soilResult.moisture_hint && (
            <p className="text-xs text-ink-600">Moisture hint: {soilResult.moisture_hint}</p>
          )}
          <NoteBox text={soilResult.note} />
        </div>
      )}

      {plantResult && (
        <div className="card p-5 mt-4 flex flex-col gap-3">
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className="font-semibold text-sm">Plant Health</h3>
            <EngineChip engine={plantResult.engine} />
          </div>
          <p className="text-sm text-ink-800">
            {plantResult.crop ? `${plantResult.crop} — ${plantResult.condition}` : plantResult.health_signal}
          </p>
          <NoteBox text={plantResult.note} />
        </div>
      )}

      {advice && (
        <div className="card p-5 mt-4 flex flex-col gap-2">
          <h3 className="font-semibold text-sm">General Pointers</h3>
          <ul className="list-disc list-inside text-sm text-ink-700 flex flex-col gap-1">
            {advice.pointers.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
          <p className="text-[11px] text-ink-400 mt-1">{advice.disclaimer}</p>
        </div>
      )}
    </div>
  );
}

function UploadDropzoneLocal({ onFile, label }) {
  const [preview, setPreview] = useState(null);
  return (
    <label className="relative cursor-pointer rounded-xl border-2 border-dashed border-line hover:border-brand-400 bg-surface h-40 flex flex-col items-center justify-center text-center overflow-hidden block">
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
          <Sprout size={20} className="text-ink-400 mb-2" />
          <p className="text-sm font-medium text-ink-700">{label}</p>
          <p className="text-xs text-ink-400 mt-1">Click to choose a photo</p>
        </>
      )}
    </label>
  );
}
