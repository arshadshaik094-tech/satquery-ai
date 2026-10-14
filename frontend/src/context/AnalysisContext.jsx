import { createContext, useContext, useState, useCallback } from "react";

const AnalysisContext = createContext(null);

const EMPTY = {
  image_t1_path: null,
  image_t2_path: null,
  optical_path: null,
  sar_path: null,
};

export function AnalysisProvider({ children }) {
  const [activeImages, setActiveImages] = useState(EMPTY);

  const setImages = useCallback((partial) => {
    setActiveImages((prev) => ({ ...prev, ...partial }));
  }, []);

  const clearImages = useCallback(() => setActiveImages(EMPTY), []);

  return (
    <AnalysisContext.Provider value={{ activeImages, setImages, clearImages }}>
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysis() {
  const ctx = useContext(AnalysisContext);
  if (!ctx) throw new Error("useAnalysis must be used within AnalysisProvider");
  return ctx;
}
