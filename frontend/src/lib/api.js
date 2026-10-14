import axios from "axios";

// Points at the real FastAPI backend (see backend/README.md to run it).
// Override with VITE_API_BASE_URL in frontend/.env if backend runs elsewhere.
export const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

const client = axios.create({ baseURL: API_BASE, timeout: 60000 });

// ---- Health / dashboard ----
export const getHealth = () => client.get("/api/health").then((r) => r.data);
export const getDashboardOverview = () =>
  client.get("/api/dashboard/overview").then((r) => r.data);
export const getDashboardHistory = (limit = 20) =>
  client.get(`/api/dashboard/history?limit=${limit}`).then((r) => r.data);
export const getModelsStatus = () =>
  client.get("/api/dashboard/models").then((r) => r.data);

// ---- Upload ----
export const uploadImage = (file) => {
  const form = new FormData();
  form.append("file", file);
  return client
    .post("/api/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    })
    .then((r) => r.data);
};
export const fileUrl = (path) =>
  path ? `${API_BASE}/uploads/${path.split("/").pop()}` : null;

// ---- Orchestrator (agentic — single entry point) ----
export const runOrchestrator = (payload) =>
  client.post("/api/orchestrate", payload).then((r) => r.data);

// ---- Direct specialist calls (bypass orchestrator, for dedicated tool pages) ----
// Matches backend/app/routers/specialist.py exactly (prefix "/api", no "/specialist").
export const runVQA = (payload) =>
  client.post("/api/vqa", payload).then((r) => r.data); // returns { vqa, landcover }
export const runChangeDetection = (payload) =>
  client.post("/api/change-detection", payload).then((r) => r.data);
export const runFusion = (payload) =>
  client.post("/api/fusion", payload).then((r) => r.data);
export const runGrounding = (payload) =>
  client.post("/api/grounding", payload).then((r) => r.data);

export const runFloodDetection = (payload) =>
  client.post("/api/flood-detection", payload).then((r) => r.data);

// ---- Farmer Assist ----
export const analyzeSoil = (file) => {
  const form = new FormData();
  form.append("file", file);
  return client.post("/api/farmer/soil", form, {
    headers: { "Content-Type": "multipart/form-data" },
  }).then((r) => r.data);
};
export const analyzePlant = (file) => {
  const form = new FormData();
  form.append("file", file);
  return client.post("/api/farmer/plant", form, {
    headers: { "Content-Type": "multipart/form-data" },
  }).then((r) => r.data);
};
export const getFarmerAdvice = (soil_type, health_signal) =>
  client.post("/api/farmer/advise", { soil_type, health_signal }).then((r) => r.data);

// ---- Batch ----
export const createBatchTask = (task_type, payload) =>
  client.post("/api/batch", { task_type, payload }).then((r) => r.data);
export const getBatchTask = (id) =>
  client.get(`/api/batch/${id}`).then((r) => r.data);
export const listBatchTasks = () =>
  client.get("/api/batch").then((r) => r.data);

// ---- Live geodata (real, keyless public sources) ----
export const getImageryTileUrl = (lat, lon, zoom = 15) =>
  client
    .get("/api/geodata/imagery-tile-url", { params: { lat, lon, zoom } })
    .then((r) => r.data);
export const getEonetEvents = (limit = 20, status = "open", category) =>
  client
    .get("/api/geodata/events", { params: { limit, status, category } })
    .then((r) => r.data);
export const geocodePlace = (place) =>
  client.get("/api/geodata/geocode", { params: { place } }).then((r) => r.data);
export const getNdviTileUrl = (lat, lon, date, zoom = 6) =>
  client
    .get("/api/geodata/ndvi-tile-url", { params: { lat, lon, date, zoom } })
    .then((r) => r.data);
export const getGdacsEvents = (daysBack = 30) =>
  client.get("/api/geodata/gdacs-events", { params: { days_back: daysBack } }).then((r) => r.data);

export default client;
