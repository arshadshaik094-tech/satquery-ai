import { useEffect, useState } from "react";
import { ListChecks, Plus, RotateCcw } from "lucide-react";
import UploadDropzone from "../components/UploadDropzone";
import { PageHeader } from "./SingleImageAnalysis";
import { createBatchTask, listBatchTasks, getBatchTask } from "../lib/api";

const TASK_TYPES = [
  { id: "vqa", label: "VQA (single image)" },
  { id: "single_image", label: "Land-cover Classification" },
  { id: "change_detection", label: "Change Detection" },
  { id: "fusion", label: "Optical+SAR Fusion" },
  { id: "grounding", label: "Grounding" },
];

export default function BatchProcessing() {
  const [taskType, setTaskType] = useState("single_image");
  const [imagePath, setImagePath] = useState(null);
  const [t2Path, setT2Path] = useState(null);
  const [question, setQuestion] = useState("Describe this image.");
  const [tasks, setTasks] = useState([]);
  const [creating, setCreating] = useState(false);

  async function refresh() {
    try {
      const res = await listBatchTasks();
      setTasks(res.items);
    } catch {
      /* backend offline — silently keep last known list */
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 2500);
    return () => clearInterval(id);
  }, []);

  async function submit() {
    setCreating(true);
    let payload = {};
    if (taskType === "vqa") payload = { image_path: imagePath, question };
    else if (taskType === "single_image") payload = { image_path: imagePath };
    else if (taskType === "change_detection") payload = { image_t1_path: imagePath, image_t2_path: t2Path };
    else if (taskType === "fusion") payload = { optical_path: imagePath, sar_path: t2Path };
    else if (taskType === "grounding") payload = { image_path: imagePath, referring_expression: question };

    try {
      await createBatchTask(taskType, payload);
      await refresh();
    } finally {
      setCreating(false);
    }
  }

  const needsSecondImage = ["change_detection", "fusion"].includes(taskType);
  const needsText = ["vqa", "grounding"].includes(taskType);

  return (
    <div className="p-6 max-w-5xl">
      <PageHeader
        title="Batch Processing"
        subtitle="Queue multiple specialist-model tasks; each runs as a real background job against the FastAPI backend (SQLite-backed queue)."
      />

      <div className="card p-4 mt-5 flex flex-col gap-3">
        <p className="text-xs font-medium text-ink-500">New batch task</p>
        <div className="flex flex-wrap gap-1.5">
          {TASK_TYPES.map((t) => (
            <button
              key={t.id}
              onClick={() => setTaskType(t.id)}
              className={`text-xs font-medium px-3 py-1.5 rounded-full border transition-colors ${
                taskType === t.id
                  ? "bg-brand-500 text-white border-brand-500"
                  : "border-line text-ink-600 hover:border-brand-300"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-1">
          <UploadDropzone
            label={needsSecondImage ? (taskType === "fusion" ? "Optical image" : "T1 (Earlier)") : "Image"}
            compact
            onUploaded={(res) => setImagePath(res.path)}
          />
          {needsSecondImage && (
            <UploadDropzone
              label={taskType === "fusion" ? "SAR image" : "T2 (Later)"}
              compact
              onUploaded={(res) => setT2Path(res.path)}
            />
          )}
        </div>

        {needsText && (
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="text-sm bg-surface border border-line rounded-lg p-2.5 outline-none focus:border-brand-400"
            placeholder={taskType === "vqa" ? "Question" : "Referring expression"}
          />
        )}

        <button
          onClick={submit}
          disabled={creating || !imagePath || (needsSecondImage && !t2Path)}
          className="btn-primary flex items-center justify-center gap-2 w-fit"
        >
          {creating ? <RotateCcw size={14} className="animate-spin" /> : <Plus size={14} />}
          Queue Task
        </button>
      </div>

      <div className="card p-4 mt-5">
        <div className="flex items-center gap-2 mb-3">
          <ListChecks size={15} className="text-brand-600" />
          <h3 className="font-semibold text-sm">Task Queue</h3>
        </div>
        {tasks.length === 0 ? (
          <p className="text-xs text-ink-400 py-6 text-center">No batch tasks yet. Queue one above.</p>
        ) : (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-ink-500 border-b border-line">
                <th className="pb-2 font-medium">Task</th>
                <th className="pb-2 font-medium">Type</th>
                <th className="pb-2 font-medium">Status</th>
                <th className="pb-2 font-medium">Progress</th>
              </tr>
            </thead>
            <tbody>
              {tasks.map((t) => (
                <tr key={t.id} className="border-b border-line/60 last:border-0">
                  <td className="py-2.5 font-mono text-[11px] text-ink-500">{t.id.slice(0, 8)}</td>
                  <td className="py-2.5">{t.task_type}</td>
                  <td className="py-2.5">
                    <StatusPill status={t.status} />
                  </td>
                  <td className="py-2.5 w-40">
                    <div className="h-1.5 bg-surface rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${t.status === "failed" ? "bg-accent-red" : "bg-brand-500"}`}
                        style={{ width: `${t.progress}%` }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

function StatusPill({ status }) {
  const map = {
    completed: "bg-teal-50 text-accent-teal",
    processing: "bg-blue-50 text-brand-600",
    queued: "bg-surface text-ink-500 border border-line",
    failed: "bg-red-50 text-accent-red",
  };
  return <span className={`chip ${map[status] || ""}`}>{status}</span>;
}
