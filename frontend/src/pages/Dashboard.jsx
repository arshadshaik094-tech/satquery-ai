import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  BarChart3, GitCompareArrows, MessageCircleQuestion, Gauge, ArrowUpRight, Flame, Map,
} from "lucide-react";
import StatCard from "../components/StatCard";
import { getDashboardOverview, getEonetEvents, getGdacsEvents } from "../lib/api";

// Leaflet CSS — must be imported before MapContainer
import "leaflet/dist/leaflet.css";
import { MapContainer, TileLayer, WMSTileLayer, CircleMarker, Popup } from "react-leaflet";

// Fix default marker icon path issue with Vite
import L from "leaflet";
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

const EONET_CATEGORY_COLORS = {
  Wildfires: "#EF4444",
  "Severe Storms": "#F59E0B",
  Volcanoes: "#F97316",
  Earthquakes: "#8B5CF6",
  Floods: "#3B82F6",
};

const GDACS_ALERT_COLORS = { Red: "#EF4444", Orange: "#F97316", Green: "#22C55E" };
const GDACS_TYPE_COLORS = {
  FL: "#3B82F6",   // Floods — blue
  EQ: "#8B5CF6",   // Earthquakes — purple
  TC: "#F59E0B",   // Tropical Cyclones — amber
  VO: "#F97316",   // Volcanoes — orange
  WF: "#EF4444",   // Wildfires — red
  DR: "#A16207",   // Drought — brown
};
const GDACS_TYPE_LABELS = { FL: "Flood", EQ: "Earthquake", TC: "Cyclone", VO: "Volcano", WF: "Wildfire", DR: "Drought" };

function eventColor(ev) {
  const cat = ev.categories?.[0]?.title || "";
  return EONET_CATEGORY_COLORS[cat] || "#6B7280";
}

function gdacsColor(feat) {
  return GDACS_ALERT_COLORS[feat.properties?.alertlevel] || "#6B7280";
}

function getCoords(ev) {
  const geom = ev.geometry?.[0];
  if (!geom) return null;
  if (geom.type === "Point") return [geom.coordinates[1], geom.coordinates[0]];
  return null;
}

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [events, setEvents] = useState(null);
  const [gdacs, setGdacs] = useState(null);
  const [eventsError, setEventsError] = useState(null);
  const [statsError, setStatsError] = useState(null);
  const [showBhuvan, setShowBhuvan] = useState(false);

  useEffect(() => {
    getDashboardOverview().then(setStats).catch(() => setStatsError(true));
    getEonetEvents(20, "open").then(setEvents).catch(() => setEventsError(true));
    getGdacsEvents(30).then(setGdacs).catch(() => {}); // GDACS failure is non-fatal
  }, []);

  const recentChange = stats?.recent?.find((r) =>
    ["detect_change", "change_trend"].includes(r.task_type)
  );

  const mappableEvents = (events?.events || []).filter(getCoords);
  const mappableGdacs = (gdacs?.features || []).filter(f => f.geometry?.coordinates);

  return (
    <div className="p-6 flex flex-col gap-6">
      <div>
        <h2 className="text-xl font-bold text-ink-900">Dashboard Overview</h2>
        <p className="text-sm text-ink-500 mt-0.5">
          Monitor your analyses, recent activity and system insights.
        </p>
      </div>

      {statsError && (
        <div className="card p-4 text-sm text-accent-red bg-red-50 border-accent-red/20">
          Couldn't reach the backend for live stats. Start it with{" "}
          <code className="font-mono text-xs bg-white px-1.5 py-0.5 rounded">
            uvicorn app.main:app --reload
          </code>{" "}
          in <code className="font-mono text-xs">backend/</code>.
        </div>
      )}

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Analyses"
          value={stats ? stats.total_analyses.toLocaleString() : "—"}
          delta={stats && stats.total_analyses > 0 ? "Real, from your session" : "Run an analysis to populate"}
          icon={BarChart3} />
        <StatCard label="Change Detections"
          value={stats ? stats.change_detections.toLocaleString() : "—"}
          delta="From /api/change-detection & orchestrator"
          icon={GitCompareArrows} />
        <StatCard label="VQA Queries"
          value={stats ? stats.vqa_queries.toLocaleString() : "—"}
          delta="From /api/vqa & orchestrator calls"
          icon={MessageCircleQuestion} />
        <StatCard label="Avg. Confidence"
          value={stats ? `${stats.avg_accuracy_pct}%` : "—"}
          delta="Model self-reported confidence, averaged"
          icon={Gauge} />
      </div>

      {/* Live NASA EONET Map */}
      <div className="card p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Map size={15} className="text-brand-600" />
            <h3 className="font-semibold text-sm text-ink-900">Live Earth Events Map</h3>
            <span className="chip bg-brand-50 text-brand-700 text-[10px]">NASA EONET + GDACS · real-time</span>
          </div>
          <div className="flex items-center gap-3">
            {/* Bhuvan LULC toggle */}
            <button
              onClick={() => setShowBhuvan(v => !v)}
              className={`flex items-center gap-1.5 text-[11px] px-2.5 py-1 rounded-full border transition-colors ${
                showBhuvan
                  ? "bg-orange-500 border-orange-500 text-white"
                  : "bg-surface border-line text-ink-500 hover:border-orange-400 hover:text-orange-600"
              }`}
              title="Toggle Bhuvan ISRO Land Use / Land Cover layer"
            >
              <span className="font-bold">🇮🇳</span> Bhuvan LULC
            </button>
            {(mappableEvents.length > 0 || mappableGdacs.length > 0) && (
              <span className="text-xs text-ink-400">
                {mappableEvents.length} EONET + {mappableGdacs.length} GDACS events
              </span>
            )}
          </div>
        </div>
        <p className="text-[11px] text-ink-400 mb-3">
          Pins show real-time disaster alerts. Note: Satellite basemap imagery may not reflect current event conditions.
        </p>

        {eventsError && (
          <div className="rounded-xl bg-surface border border-line h-64 grid place-items-center text-xs text-ink-400">
            Couldn't load NASA EONET events — check network.
          </div>
        )}

        {!eventsError && (
          <div className="rounded-xl overflow-hidden border border-line" style={{ height: "360px" }}>
            <MapContainer
              center={[20, 0]}
              zoom={2}
              style={{ height: "100%", width: "100%" }}
              scrollWheelZoom={false}
            >
              {/* Esri World Imagery — real satellite basemap, keyless */}
              <TileLayer
                url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                attribution="Tiles &copy; Esri &mdash; Source: Esri, Maxar, GeoEye, Earthstar Geographics"
                maxZoom={18}
              />

              {/* Bhuvan ISRO LULC layer — toggled on/off */}
              {showBhuvan && (
                <WMSTileLayer
                  url="https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
                  layers="lulc50k"
                  format="image/png"
                  transparent={true}
                  opacity={0.65}
                  version="1.1.1"
                  attribution="Bhuvan &copy; ISRO/NRSC — Land Use Land Cover 50K"
                />
              )}
              {/* NASA EONET event pins */}
              {mappableEvents.map((ev) => {
                const coords = getCoords(ev);
                const color = eventColor(ev);
                return (
                  <CircleMarker
                    key={ev.id}
                    center={coords}
                    radius={8}
                    pathOptions={{ color, fillColor: color, fillOpacity: 0.8, weight: 2 }}
                  >
                    <Popup>
                      <div className="text-xs max-w-[200px]">
                        <p className="font-semibold leading-snug">{ev.title}</p>
                        <p className="text-gray-500 mt-0.5">{ev.categories?.[0]?.title}</p>
                        <p className="text-gray-400 mt-0.5">
                          {ev.geometry?.[0]?.date
                            ? new Date(ev.geometry[0].date).toLocaleDateString()
                            : ""}
                        </p>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}

              {/* GDACS disaster event pins */}
              {mappableGdacs.map((feat) => {
                // GDACS coordinates come as string "lon lat", not [lon, lat]
                const coordStr = feat.geometry.coordinates;
                let lon, lat;
                if (typeof coordStr === 'string') {
                  [lon, lat] = coordStr.split(' ').map(Number);
                } else if (Array.isArray(coordStr)) {
                  [lon, lat] = coordStr;
                }
                
                const props = feat.properties || {};
                const alertLevel = props.alertlevel || "Green";
                const eventType = props.eventtype || "Unknown";
                const radius = alertLevel === "Red" ? 10 : alertLevel === "Orange" ? 8 : 6;
                const color = gdacsColor(feat);
                
                return (
                  <CircleMarker
                    key={feat.id || `gdacs-${lat}-${lon}`}
                    center={[lat, lon]}
                    radius={radius}
                    pathOptions={{ color, fillColor: color, fillOpacity: 0.8, weight: 2 }}
                  >
                    <Popup>
                      <div className="text-xs max-w-[200px]">
                        <p className="font-semibold leading-snug">{props.name || "Unnamed Event"}</p>
                        <p className="text-gray-500 mt-0.5">
                          {GDACS_TYPE_LABELS[eventType] || eventType} · {alertLevel} Alert
                        </p>
                        <p className="text-gray-400 mt-0.5">
                          {props.fromdate ? new Date(props.fromdate).toLocaleDateString() : ""}
                        </p>
                        <p className="text-gray-400 text-[10px] mt-1">Source: GDACS</p>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>
          </div>
        )}

        {/* Legend */}
        {(mappableEvents.length > 0 || mappableGdacs.length > 0) && (
          <div className="mt-3 space-y-2">
            <div className="flex flex-wrap gap-3">
              <span className="text-[10px] text-ink-400 font-semibold mr-1">EONET Categories:</span>
              {Object.entries(EONET_CATEGORY_COLORS).map(([cat, color]) => (
                <div key={cat} className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: color }} />
                  <span className="text-[11px] text-ink-500">{cat}</span>
                </div>
              ))}
            </div>
            {mappableGdacs.length > 0 && (
              <div className="flex flex-wrap gap-3">
                <span className="text-[10px] text-ink-400 font-semibold mr-1">GDACS Alert Levels:</span>
                {Object.entries(GDACS_ALERT_COLORS).map(([level, color]) => (
                  <div key={level} className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: color }} />
                    <span className="text-[11px] text-ink-500">{level}</span>
                  </div>
                ))}
              </div>
            )}
            {/* Data source badges */}
            <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-line mt-1">
              <span className="text-[10px] text-ink-400 font-semibold">Data Sources:</span>
              {[
                { label: "NASA EONET", color: "bg-blue-50 text-blue-700 border-blue-200" },
                { label: "GDACS · UN/EC", color: "bg-red-50 text-red-700 border-red-200" },
                { label: "Bhuvan · ISRO/NRSC", color: "bg-orange-50 text-orange-700 border-orange-200" },
                { label: "MOSDAC · ISRO/SAC", color: "bg-green-50 text-green-700 border-green-200" },
                { label: "Esri World Imagery", color: "bg-surface text-ink-500 border-line" },
              ].map(({ label, color }) => (
                <span key={label} className={`text-[10px] px-2 py-0.5 rounded-full border font-medium ${color}`}>
                  {label}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Recent analysis */}
        <div className="lg:col-span-2 card p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-sm text-ink-900">Recent Analysis</h3>
            <Link to="/change-detection" className="text-xs text-brand-600 font-medium flex items-center gap-0.5 hover:text-brand-700">
              Run new <ArrowUpRight size={12} />
            </Link>
          </div>
          {recentChange ? (
            <>
              <div className="rounded-xl bg-[#0F172A] h-20 grid place-items-center text-xs text-white/40 border border-line font-mono px-4 text-center">
                Images not retained server-side after request — re-run on Change Detection page for live overlay.
              </div>
              <div className="mt-3 flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-ink-800">Change Detection · {recentChange.answer}</p>
                  <p className="text-xs text-ink-400 mt-0.5">{new Date(recentChange.created_at * 1000).toLocaleString()}</p>
                </div>
                <Link to="/history" className="btn-ghost !py-1.5 text-xs">View Details</Link>
              </div>
            </>
          ) : (
            <EmptyState text="No analyses run yet." cta="Go to Change Detection" to="/change-detection" />
          )}
        </div>

        {/* Active Events - merged EONET + GDACS */}
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-3">
            <Flame size={15} className="text-accent-amber" />
            <h3 className="font-semibold text-sm text-ink-900">Active Events</h3>
          </div>
          <p className="text-[11px] text-ink-400 mb-3">NASA EONET + GDACS — open events</p>
          {eventsError && <p className="text-xs text-accent-red">Couldn't reach NASA EONET.</p>}
          {!eventsError && !events && !gdacs && <p className="text-xs text-ink-400">Loading live events…</p>}
          {(events || gdacs) && (
            <ul className="flex flex-col gap-2 max-h-64 overflow-y-auto">
              {/* Show 5 EONET events */}
              {events?.events?.slice(0, 5).map((ev) => (
                <li key={ev.id} className="flex items-start gap-2 text-xs">
                  <span className="w-2 h-2 rounded-full mt-1 shrink-0"
                    style={{ background: eventColor(ev) }} />
                  <div>
                    <p className="text-ink-800 font-medium leading-snug">{ev.title}</p>
                    <p className="text-ink-400">{ev.categories?.[0]?.title} · EONET</p>
                  </div>
                </li>
              ))}
              {/* Show 5 GDACS events */}
              {gdacs?.features?.slice(0, 5).map((feat) => {
                const props = feat.properties || {};
                return (
                  <li key={feat.id || props.name} className="flex items-start gap-2 text-xs">
                    <span className="w-2 h-2 rounded-full mt-1 shrink-0"
                      style={{ background: gdacsColor(feat) }} />
                    <div>
                      <p className="text-ink-800 font-medium leading-snug">{props.name || "Unnamed"}</p>
                      <p className="text-ink-400">
                        {GDACS_TYPE_LABELS[props.eventtype] || props.eventtype} · {props.alertlevel} · GDACS
                      </p>
                    </div>
                  </li>
                );
              })}
              {(!events?.events?.length && !gdacs?.features?.length) && (
                <p className="text-xs text-ink-400">No open events right now.</p>
              )}
            </ul>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Recent queries */}
        <div className="card p-4">
          <h3 className="font-semibold text-sm text-ink-900 mb-3">Recent Queries</h3>
          {stats?.recent?.length ? (
            <ul className="flex flex-col divide-y divide-line">
              {stats.recent.slice(0, 5).map((r) => (
                <li key={r.id} className="py-2.5 flex items-center justify-between gap-3">
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-ink-800 truncate">{r.query}</p>
                    <span className="chip bg-brand-50 text-brand-700 mt-1">{r.task_type}</span>
                  </div>
                  <span className="text-[11px] text-ink-400 shrink-0">{timeAgo(r.created_at)}</span>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState text="No queries yet — ask the assistant on the right." />
          )}
          <Link to="/history" className="btn-ghost w-full text-center mt-3 text-xs">View All</Link>
        </div>

        {/* Quick start */}
        <div className="card p-4">
          <h3 className="font-semibold text-sm text-ink-900 mb-3">Quick Start</h3>
          <div className="flex flex-col gap-2">
            {[
              ["Single Image", "Describe the land-cover and major objects visible.", "/single-image"],
              ["Grounding", "Highlight the water body referred to in the query.", "/grounding"],
              ["Change Detection", "What changed between these two dates?", "/change-detection"],
              ["Optical + SAR", "Identify built-up and water-covered regions.", "/optical-sar"],
            ].map(([tag, q, to]) => (
              <Link key={q} to={to}
                className="flex items-start gap-2.5 p-2.5 rounded-lg border border-line hover:border-brand-300 hover:bg-brand-50/40 transition-colors">
                <span className="chip bg-surface border border-line text-ink-500 shrink-0 mt-0.5">{tag}</span>
                <span className="text-xs text-ink-700 leading-snug">{q}</span>
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ text, cta, to }) {
  return (
    <div className="py-8 flex flex-col items-center text-center gap-2">
      <p className="text-xs text-ink-400">{text}</p>
      {cta && <Link to={to} className="btn-primary !py-1.5 !px-3 text-xs">{cta}</Link>}
    </div>
  );
}

function timeAgo(unixSeconds) {
  const s = Math.floor(Date.now() / 1000 - unixSeconds);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
