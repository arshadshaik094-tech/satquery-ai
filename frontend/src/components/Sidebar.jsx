import { NavLink } from "react-router-dom";
import {
  LayoutDashboard, ImageIcon, GitCompareArrows, Layers, MessageCircleQuestion,
  ListChecks, History as HistoryIcon, Database, BrainCircuit, FileText,
  Settings as SettingsIcon, HelpCircle, Satellite, Droplets, Sprout,
} from "lucide-react";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard, end: true },
  { to: "/single-image", label: "Single Image Analysis", icon: ImageIcon },
  { to: "/farmer-assist", label: "Farmer Assist", icon: Sprout },
  { to: "/change-detection", label: "Change Detection", icon: GitCompareArrows },
  { to: "/flood-detection", label: "Flood Detection", icon: Droplets },
  { to: "/optical-sar", label: "Optical + SAR Analysis", icon: Layers },
  { to: "/grounding", label: "Grounding (VQA)", icon: MessageCircleQuestion },
  { to: "/batch", label: "Batch Processing", icon: ListChecks },
  { to: "/history", label: "History", icon: HistoryIcon },
  { to: "/datasets", label: "Datasets", icon: Database },
  { to: "/models", label: "Models", icon: BrainCircuit },
  { to: "/reports", label: "Reports", icon: FileText },
];

const BOTTOM_NAV = [
  { to: "/settings", label: "Settings", icon: SettingsIcon },
  { to: "/help", label: "Help & Docs", icon: HelpCircle },
];

function Item({ to, label, icon: Icon, end }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-colors focus-ring ${
          isActive
            ? "bg-brand-500 text-white shadow-pop"
            : "text-white/65 hover:bg-white/8 hover:text-white"
        }`
      }
    >
      <Icon size={17} strokeWidth={2} className="shrink-0" />
      <span className="truncate">{label}</span>
    </NavLink>
  );
}

export default function Sidebar() {
  return (
    <aside className="w-[248px] shrink-0 bg-navy-900 h-screen sticky top-0 flex flex-col py-5 px-3">
      <div className="flex items-center gap-2.5 px-2.5 pb-6">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-400 to-brand-600 grid place-items-center shadow-pop">
          <Satellite size={17} className="text-white" strokeWidth={2.4} />
        </div>
        <span className="text-white font-bold text-[17px] tracking-tight">SatQuery AI</span>
      </div>

      <nav className="flex-1 flex flex-col gap-1 overflow-y-auto">
        {NAV.map((n) => (
          <Item key={n.to} {...n} />
        ))}
      </nav>

      <div className="flex flex-col gap-1 pt-3 border-t border-white/10 mt-3">
        {BOTTOM_NAV.map((n) => (
          <Item key={n.to} {...n} />
        ))}
      </div>
    </aside>
  );
}
