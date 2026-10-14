import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import Topbar from "./components/Topbar";
import { AnalysisProvider } from "./context/AnalysisContext";

import Dashboard from "./pages/Dashboard";
import SingleImageAnalysis from "./pages/SingleImageAnalysis";
import ChangeDetection from "./pages/ChangeDetection";
import OpticalSarAnalysis from "./pages/OpticalSarAnalysis";
import GroundingVQA from "./pages/GroundingVQA";
import FloodDetection from "./pages/FloodDetection";
import BatchProcessing from "./pages/BatchProcessing";
import History from "./pages/History";
import Datasets from "./pages/Datasets";
import Models from "./pages/Models";
import Reports from "./pages/Reports";
import FarmerAssist from "./pages/FarmerAssist";
import Settings from "./pages/Settings";
import HelpDocs from "./pages/HelpDocs";

const TITLES = {
  "/": ["Dashboard", "Overview of your remote-sensing analyses"],
  "/single-image": ["Single Image Analysis", "VQA + land-cover classification"],
  "/farmer-assist": ["Farmer Assist", "Soil type & plant health from photos"],
  "/change-detection": ["Change Detection", "Bi-temporal analysis"],
  "/optical-sar": ["Optical + SAR Analysis", "Cross-modal fusion"],
  "/grounding": ["Grounding (VQA)", "Referring-expression segmentation"],
  "/flood-detection": ["Flood & Water Detection", "Detect flood extent in sq.km"],
  "/batch": ["Batch Processing", "Queue multiple analyses"],
  "/history": ["Analysis History", ""],
  "/datasets": ["Datasets", ""],
  "/models": ["Model & System Status", ""],
  "/reports": ["Reports & Downloads", ""],
  "/settings": ["Settings", ""],
  "/help": ["Help & Docs", ""],
};

function Shell() {
  const location = useLocation();
  const [title, subtitle] = TITLES[location.pathname] || ["SatQuery AI", ""];

  return (
    <div className="flex min-h-screen bg-surface">
      <Sidebar />
      <div className="flex-1 min-w-0 flex flex-col">
        <Topbar title={title} subtitle={subtitle} />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/single-image" element={<SingleImageAnalysis />} />
            <Route path="/farmer-assist" element={<FarmerAssist />} />
            <Route path="/change-detection" element={<ChangeDetection />} />
            <Route path="/optical-sar" element={<OpticalSarAnalysis />} />
            <Route path="/grounding" element={<GroundingVQA />} />
            <Route path="/flood-detection" element={<FloodDetection />} />
            <Route path="/batch" element={<BatchProcessing />} />
            <Route path="/history" element={<History />} />
            <Route path="/datasets" element={<Datasets />} />
            <Route path="/models" element={<Models />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/help" element={<HelpDocs />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AnalysisProvider>
        <Shell />
      </AnalysisProvider>
    </BrowserRouter>
  );
}
