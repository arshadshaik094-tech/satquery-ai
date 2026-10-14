import { useState } from "react";
import { Sparkles, Calendar, CheckCircle, XCircle } from "lucide-react";

const YEARS = ["2018", "2019", "2020", "2021", "2022", "2023"];

export default function SatelliteTimePicker({ onCapture }) {
  const [location, setLocation] = useState("");
  const [yearBefore, setYearBefore] = useState("2018");
  const [yearAfter, setYearAfter] = useState("2023");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Geocode confirmation step
  const [foundPlace, setFoundPlace] = useState(null); // { lat, lon, display_name }
  const [allResults, setAllResults] = useState([]);   // show alternatives
  const [confirmed, setConfirmed] = useState(false);

  async function handleSearch() {
    if (!location.trim()) return;
    setError(null);
    setFoundPlace(null);
    setConfirmed(false);
    setAllResults([]);

    try {
      const res = await fetch(
        `http://localhost:8000/api/geodata/geocode?place=${encodeURIComponent(location)}`
      );
      const data = await res.json();
      const results = data.results || [];

      if (results.length === 0) {
        setError('Not found. Try adding city/state — e.g. "Ellenki College Hyderabad Telangana".');
        return;
      }

      // Show top result + alternatives
      setFoundPlace({
        lat: parseFloat(results[0].lat),
        lon: parseFloat(results[0].lon),
        display_name: results[0].display_name,
      });
      setAllResults(results.slice(0, 5));
    } catch (e) {
      setError("Search failed. Check connection.");
    }
  }

  function selectResult(r) {
    setFoundPlace({
      lat: parseFloat(r.lat),
      lon: parseFloat(r.lon),
      display_name: r.display_name,
    });
    setConfirmed(false);
  }

  async function handleGo() {
    if (!foundPlace) return;
    setLoading(true);
    setError(null);

    try {
      const { lat, lon } = foundPlace;

      // Use small bbox (1.5km) so building/campus fills the frame
      const [resBefore, resAfter] = await Promise.all([
        fetch(
          `http://localhost:8000/api/geodata/capture-region?date=${yearBefore}-06-01&lat=${lat}&lon=${lon}&bbox_size_km=1.5`,
          { method: "POST" }
        ),
        fetch(
          `http://localhost:8000/api/geodata/capture-region?date=${yearAfter}-06-01&lat=${lat}&lon=${lon}&bbox_size_km=1.5`,
          { method: "POST" }
        ),
      ]);

      const dataBefore = await resBefore.json();
      const dataAfter  = await resAfter.json();

      onCapture({
        beforePath: dataBefore.image_path,
        afterPath:  dataAfter.image_path,
        location:   foundPlace.display_name,
        dateBefore: yearBefore,
        dateAfter:  yearAfter,
      });
    } catch (e) {
      setError(e.message || "Failed to fetch images.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="card p-6 flex flex-col gap-5 bg-gradient-to-br from-white to-brand-50/30 border-brand-200">
      {/* Header */}
      <div className="text-center">
        <div className="inline-flex items-center gap-2 mb-2">
          <Sparkles size={18} className="text-brand-600" />
          <h3 className="font-bold text-base text-ink-900">Satellite Time Machine</h3>
        </div>
        <p className="text-sm text-ink-600">Compare any place on Earth across time</p>
        <p className="text-xs text-ink-400 mt-1">Sentinel-2 · 10m resolution · 2018–2023</p>
      </div>

      {/* Step 1 — Search */}
      <div>
        <label className="text-sm font-semibold text-ink-800 mb-2 block">
          1️⃣ Search your location
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="College, landmark, city, forest..."
            value={location}
            onChange={(e) => { setLocation(e.target.value); setFoundPlace(null); setConfirmed(false); }}
            onKeyDown={(e) => e.key === "Enter" && handleSearch()}
            className="input flex-1 text-base"
            disabled={loading}
          />
          <button
            onClick={handleSearch}
            disabled={!location.trim() || loading}
            className="btn-primary px-4 shrink-0"
          >
            Search
          </button>
        </div>
        <p className="text-xs text-ink-400 mt-1.5">
          Include city for best results — <span className="text-ink-600">"Ellenki College Hyderabad"</span>
        </p>
        {/* Quick chips */}
        <div className="flex flex-wrap gap-2 mt-2">
          {["Amazon Rainforest Brazil", "Dubai Palm Jumeirah", "Aral Sea Kazakhstan", "Las Vegas Strip", "Three Gorges Dam China"].map((p) => (
            <button key={p} onClick={() => { setLocation(p); setFoundPlace(null); setConfirmed(false); }}
              disabled={loading}
              className="text-[11px] px-2 py-1 rounded-full bg-brand-50 border border-brand-200 text-brand-700 hover:bg-brand-100 transition-colors">
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* Geocode results — confirm or pick alternative */}
      {allResults.length > 0 && !confirmed && (
        <div className="border border-line rounded-xl overflow-hidden">
          <p className="text-xs font-semibold text-ink-500 px-3 py-2 bg-surface border-b border-line">
            Is this the right place? Pick one:
          </p>
          {allResults.map((r, i) => (
            <button
              key={i}
              onClick={() => { selectResult(r); setConfirmed(true); }}
              className={`w-full text-left px-3 py-2.5 text-xs hover:bg-brand-50 transition-colors border-b border-line last:border-0 flex items-start gap-2 ${
                foundPlace?.display_name === r.display_name ? "bg-brand-50 text-brand-800 font-medium" : "text-ink-700"
              }`}
            >
              <CheckCircle size={13} className="mt-0.5 shrink-0 text-brand-400" />
              <span className="leading-snug">{r.display_name}</span>
            </button>
          ))}
        </div>
      )}

      {confirmed && foundPlace && (
        <div className="flex items-start gap-2 bg-brand-50 border border-brand-200 rounded-lg px-3 py-2.5">
          <CheckCircle size={14} className="text-brand-600 mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-xs font-semibold text-brand-900 truncate">{foundPlace.display_name}</p>
            <p className="text-[11px] text-brand-600 mt-0.5">{foundPlace.lat.toFixed(5)}, {foundPlace.lon.toFixed(5)}</p>
          </div>
          <button onClick={() => { setConfirmed(false); setFoundPlace(null); }}
            className="text-ink-400 hover:text-accent-red shrink-0">
            <XCircle size={14} />
          </button>
        </div>
      )}

      {/* Step 2 — Years */}
      <div>
        <label className="text-sm font-semibold text-ink-800 mb-2 block">
          2️⃣ Pick two years to compare
        </label>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-ink-500 mb-1 flex items-center gap-1">
              <Calendar size={11} /> Before Year
            </label>
            <select value={yearBefore} onChange={(e) => setYearBefore(e.target.value)}
              className="input w-full" disabled={loading}>
              {YEARS.filter((y) => y < yearAfter).map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-ink-500 mb-1 flex items-center gap-1">
              <Calendar size={11} /> After Year
            </label>
            <select value={yearAfter} onChange={(e) => setYearAfter(e.target.value)}
              className="input w-full" disabled={loading}>
              {YEARS.filter((y) => y > yearBefore).map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Step 3 — Go */}
      <button
        onClick={handleGo}
        disabled={!confirmed || !foundPlace || loading}
        className="btn-primary py-3 text-base font-semibold flex items-center justify-center gap-2 shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? (
          <>
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            Fetching Sentinel-2 imagery…
          </>
        ) : (
          <>
            <Sparkles size={16} />
            3️⃣ Get Images &amp; Run Change Detection
          </>
        )}
      </button>

      {error && (
        <div className="text-sm text-accent-red bg-red-50 p-3 rounded-lg border border-red-200">
          ❌ {error}
        </div>
      )}

      <p className="text-xs text-ink-400 text-center border-t border-line pt-3">
        Powered by EOX Sentinel-2 cloudless · Free · No API key · 10m resolution
      </p>
    </div>
  );
}

