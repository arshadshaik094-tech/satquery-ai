import { useRef, useState } from "react";
import { UploadCloud, CheckCircle2, Loader2, ImageOff } from "lucide-react";
import { uploadImage, fileUrl } from "../lib/api";

export default function UploadDropzone({ label, onUploaded, compact }) {
  const inputRef = useRef(null);
  const [status, setStatus] = useState("idle"); // idle | uploading | done | error
  const [preview, setPreview] = useState(null);
  const [errMsg, setErrMsg] = useState("");

  async function handleFile(file) {
    if (!file) return;
    setStatus("uploading");
    setErrMsg("");
    setPreview(URL.createObjectURL(file));
    try {
      const res = await uploadImage(file);
      setStatus("done");
      onUploaded?.(res);
    } catch (e) {
      setStatus("error");
      setErrMsg(e?.response?.data?.detail || "Upload failed — is the backend running?");
    }
  }

  return (
    <div
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        handleFile(e.dataTransfer.files?.[0]);
      }}
      onClick={() => inputRef.current?.click()}
      className={`relative cursor-pointer rounded-xl border-2 border-dashed transition-colors flex flex-col items-center justify-center text-center overflow-hidden
        ${status === "error" ? "border-accent-red/40 bg-red-50" : "border-line hover:border-brand-400 bg-surface"}
        ${compact ? "h-40" : "h-56"}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp,.tif,.tiff"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      {preview ? (
        <>
          <img src={preview} alt={label} className="absolute inset-0 w-full h-full object-cover" />
          <div className="absolute inset-0 bg-gradient-to-t from-black/55 via-black/0 to-black/0" />
          <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">
            <span className="text-white text-xs font-medium drop-shadow">{label}</span>
            {status === "uploading" && <Loader2 size={16} className="text-white animate-spin" />}
            {status === "done" && <CheckCircle2 size={16} className="text-accent-teal" />}
          </div>
        </>
      ) : (
        <>
          <UploadCloud size={22} className="text-ink-400 mb-2" />
          <p className="text-sm font-medium text-ink-700">{label}</p>
          <p className="text-xs text-ink-400 mt-1">Click or drag a .png / .jpg / .tif file</p>
        </>
      )}
      {status === "error" && (
        <div className="absolute inset-x-2 bottom-2 bg-white rounded-lg px-2 py-1.5 text-[11px] text-accent-red flex items-center gap-1.5">
          <ImageOff size={12} /> {errMsg}
        </div>
      )}
    </div>
  );
}
