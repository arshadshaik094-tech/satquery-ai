import { useState, useRef, useEffect } from "react";
import { Send, Sparkles, Plus, ThumbsUp, ThumbsDown, Clock, Gauge, Wrench, Terminal, CheckCircle2 } from "lucide-react";
import { runOrchestrator, fileUrl } from "../lib/api";
import { useAnalysis } from "../context/AnalysisContext";

const SUGGESTIONS = [
  "Detect flood affected areas in this image.",
  "Highlight the water body in this satellite image.",
  "What changed between these two dates?",
  "Identify built-up and water-covered regions.",
  "Describe the land-cover visible here.",
  "Has the built-up area increased or decreased?",
];

export default function AssistantPanel() {
  const { activeImages } = useAnalysis();
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text:
        "Hi, I'm the SatQuery AI assistant. Upload imagery on any analysis page (or use the demo pair on Dashboard), then ask me anything — I'll route your question to the right specialist model automatically.",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  async function send(text) {
    const query = (text ?? input).trim();
    if (!query || loading) return;
    setInput("");
    setError(null);
    setMessages((m) => [...m, { role: "user", text: query }]);
    setLoading(true);

    const hasAny =
      activeImages.image_t1_path || (activeImages.optical_path && activeImages.sar_path);

    // If no image is loaded, still try — orchestrator will answer what it can
    // or give a helpful prompt to upload specific imagery
    try {
      const payload = hasAny
        ? { query, ...activeImages }
        : { query };
      const res = await runOrchestrator(payload);
      setMessages((m) => [...m, { role: "assistant", data: res }]);
    } catch (e) {
      // If backend says no image — give friendly guidance
      const detail = e?.response?.data?.detail || e.message;
      const isNoImage = detail?.toLowerCase().includes("not found") || detail?.toLowerCase().includes("upload");
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: isNoImage
            ? "To analyse a specific area, please upload a satellite image first on any analysis page, then ask your question here."
            : `Request failed: ${detail}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <aside className="w-[380px] shrink-0 h-screen sticky top-0 flex flex-col bg-white border-l border-line">
      <div className="h-[64px] shrink-0 flex items-center justify-between px-4 border-b border-line">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-brand-500 to-accent-violet grid place-items-center">
            <Sparkles size={14} className="text-white" />
          </div>
          <span className="font-semibold text-[14px] text-ink-900">SatQuery AI Assistant</span>
        </div>
        <button
          onClick={() =>
            setMessages([
              { role: "assistant", text: "New chat started. Ask me anything about your loaded imagery." },
            ])
          }
          className="text-xs font-medium text-brand-600 hover:text-brand-700 flex items-center gap-1"
        >
          <Plus size={13} /> New Chat
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3">
        {messages.map((m, i) => (
          <Bubble key={i} msg={m} />
        ))}
        {loading && (
          <div className="self-start max-w-[90%]">
            <div className="card px-3.5 py-3 text-xs text-ink-500 flex items-center gap-2">
              <span className="flex gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-bounce [animation-delay:-0.3s]" />
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-bounce [animation-delay:-0.15s]" />
                <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-bounce" />
              </span>
              Routing query to the right specialist model…
            </div>
          </div>
        )}
      </div>

      <div className="px-4 pt-2 pb-1 flex gap-1.5 flex-wrap">
        {SUGGESTIONS.slice(0, 2).map((s) => (
          <button
            key={s}
            onClick={() => send(s)}
            className="text-[11px] leading-tight text-brand-700 bg-brand-50 hover:bg-brand-100 rounded-full px-2.5 py-1.5 transition-colors text-left"
          >
            {s.length > 42 ? s.slice(0, 40) + "…" : s}
          </button>
        ))}
      </div>

      <div className="p-3 border-t border-line">
        <div className="flex items-center gap-2 bg-surface rounded-xl px-3 py-2 border border-line focus-within:border-brand-400 transition-colors">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            placeholder="Ask a follow-up…"
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-ink-400"
          />
          <button
            onClick={() => send()}
            disabled={loading || !input.trim()}
            className="w-8 h-8 shrink-0 grid place-items-center rounded-full bg-brand-500 hover:bg-brand-600 disabled:opacity-40 text-white transition-colors"
          >
            <Send size={14} />
          </button>
        </div>
      </div>
    </aside>
  );
}

function Bubble({ msg }) {
  if (msg.role === "user") {
    return (
      <div className="self-end max-w-[88%] bg-brand-500 text-white rounded-2xl rounded-br-sm px-3.5 py-2.5 text-[13px] leading-relaxed shadow-sm">
        {msg.text}
      </div>
    );
  }

  if (msg.text && !msg.data) {
    return (
      <div className="self-start max-w-[92%] card px-3.5 py-2.5 text-[13px] leading-relaxed text-ink-700">
        {msg.text}
      </div>
    );
  }

  const d = msg.data;
  const overlay = d.overlay_image_b64 || d.mask_image_b64;
  return (
    <div className="self-start max-w-[94%] flex flex-col gap-2">
      {overlay && (
        <img
          src={`data:image/png;base64,${overlay}`}
          alt="Analysis overlay"
          className="rounded-xl border border-line w-full"
        />
      )}
      {/* Mini execution trace */}
      {d.tools_used?.length > 0 && (
        <div className="bg-[#0F172A] rounded-xl px-3 py-2 flex flex-col gap-1">
          <div className="flex items-center gap-1.5 mb-1">
            <Terminal size={10} className="text-green-400" />
            <span className="text-[10px] font-mono text-green-400">Execution trace</span>
          </div>
          {d.tools_used.map((t, i) => (
            <div key={i} className="flex items-start gap-1.5">
              <CheckCircle2 size={10} className="text-green-400 mt-0.5 shrink-0" />
              <span className="text-[10px] font-mono text-green-200">{t.tool}</span>
              <span className="text-[10px] font-mono text-white/40 ml-auto shrink-0">{t.duration_ms}ms</span>
            </div>
          ))}
        </div>
      )}
      <div className="card px-3.5 py-3 text-[13px] leading-relaxed text-ink-800">
        {d.answer}
      </div>
      <div className="flex flex-wrap gap-1.5 text-[11px] text-ink-500">
        <span className="chip bg-brand-50 text-brand-700">
          <Gauge size={11} /> Confidence: {(d.confidence * 100).toFixed(0)}%
        </span>
        <span className="chip bg-surface text-ink-500 border border-line">
          <Clock size={11} /> {d.processing_time_s}s
        </span>
        {d.tools_used?.map((t, i) => (
          <span key={i} className="chip bg-violet-50 text-accent-violet" style={{ background: "#F1EEFE" }}>
            <Wrench size={11} /> {t.tool}
          </span>
        ))}
      </div>
      <div className="flex items-center gap-3 text-ink-400 px-0.5">
        <button className="hover:text-ink-700"><ThumbsUp size={13} /></button>
        <button className="hover:text-ink-700"><ThumbsDown size={13} /></button>
      </div>
    </div>
  );
}
