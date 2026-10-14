import { useState, useRef, useCallback } from "react";

export default function ImageCompareSlider({ beforeSrc, afterSrc, beforeLabel = "T1 (Earlier)", afterLabel = "T2 (Later)", height = "h-56" }) {
  const [pos, setPos] = useState(50);
  const wrapRef = useRef(null);
  const dragging = useRef(false);

  const updateFromClientX = useCallback((clientX) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    if (!rect) return;
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPos(Math.min(100, Math.max(0, pct)));
  }, []);

  if (!beforeSrc || !afterSrc) {
    return (
      <div className={`${height} rounded-xl bg-surface border border-dashed border-line grid place-items-center text-xs text-ink-400`}>
        Upload both T1 and T2 images to preview
      </div>
    );
  }

  return (
    <div
      ref={wrapRef}
      className={`relative ${height} rounded-xl overflow-hidden select-none border border-line bg-navy-950`}
      onMouseMove={(e) => dragging.current && updateFromClientX(e.clientX)}
      onMouseUp={() => (dragging.current = false)}
      onMouseLeave={() => (dragging.current = false)}
      onTouchMove={(e) => updateFromClientX(e.touches[0].clientX)}
    >
      <img src={afterSrc} alt={afterLabel} className="absolute inset-0 w-full h-full object-cover" />
      <div
        className="absolute inset-0"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      >
        <img src={beforeSrc} alt={beforeLabel} className="absolute inset-0 w-full h-full object-cover" />
      </div>

      <div className="absolute top-2 left-2 bg-black/55 text-white text-[10px] font-semibold px-2 py-1 rounded-md">
        {beforeLabel}
      </div>
      <div className="absolute top-2 right-2 bg-black/55 text-white text-[10px] font-semibold px-2 py-1 rounded-md">
        {afterLabel}
      </div>

      <div
        className="absolute top-0 bottom-0 w-0.5 bg-white cursor-ew-resize"
        style={{ left: `${pos}%` }}
        onMouseDown={() => (dragging.current = true)}
        onTouchStart={() => (dragging.current = true)}
      >
        <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-white shadow-md grid place-items-center">
          <div className="flex gap-0.5">
            <div className="w-0.5 h-3 bg-ink-400 rounded-full" />
            <div className="w-0.5 h-3 bg-ink-400 rounded-full" />
          </div>
        </div>
      </div>
    </div>
  );
}
