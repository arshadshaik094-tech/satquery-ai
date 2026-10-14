import { useEffect, useState } from "react";
import { Menu, Bell, Sun } from "lucide-react";
import { getHealth } from "../lib/api";

export default function Topbar({ title, subtitle }) {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let mounted = true;
    getHealth()
      .then((d) => mounted && setHealth(d))
      .catch(() => mounted && setError(true));
    return () => (mounted = false);
  }, []);

  return (
    <header className="h-[64px] shrink-0 flex items-center justify-between px-6 border-b border-line bg-white/80 backdrop-blur sticky top-0 z-20">
      <div className="flex items-center gap-3">
        <button className="text-ink-500 hover:text-ink-900 md:hidden">
          <Menu size={20} />
        </button>
        <div>
          <h1 className="text-[15px] font-semibold text-ink-900 leading-tight">{title}</h1>
          {subtitle && <p className="text-xs text-ink-500 mt-0.5">{subtitle}</p>}
        </div>
      </div>

      <div className="flex items-center gap-3">
        <div
          className={`hidden sm:flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full border ${
            error
              ? "border-accent-red/30 bg-red-50 text-accent-red"
              : health
              ? "border-accent-teal/30 bg-teal-50 text-accent-teal"
              : "border-line bg-surface text-ink-400"
          }`}
          title={error ? "Backend unreachable — start the FastAPI server (see backend/README.md)" : "Backend connected"}
        >
          <span className={`w-1.5 h-1.5 rounded-full ${error ? "bg-accent-red" : health ? "bg-accent-teal" : "bg-ink-400"}`} />
          {error ? "Backend offline" : health ? `Backend live · ${health.weights_mode} weights` : "Connecting…"}
        </div>
        <button className="w-9 h-9 grid place-items-center rounded-full hover:bg-surface text-ink-500 focus-ring">
          <Sun size={18} />
        </button>
        <button className="w-9 h-9 grid place-items-center rounded-full hover:bg-surface text-ink-500 relative focus-ring">
          <Bell size={18} />
        </button>
        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-brand-500 to-accent-violet grid place-items-center text-white text-xs font-bold">
          A
        </div>
      </div>
    </header>
  );
}
